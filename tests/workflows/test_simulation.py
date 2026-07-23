"""Tests for hand_probability and simulate_opening_hands workflows."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mtg_mcp_server.services.scryfall import CardNotFoundError
from mtg_mcp_server.types import Card, CardPrices
from mtg_mcp_server.workflows.simulation import (
    _bottom_cards_playability,
    _CardClass,
    _classify_card,
    _classify_tutor,
    _goldfish,
    _hypergeom_pmf,
    _keep_playability,
    _mana_production,
    _run_simulation,
    _Slot,
    _source_colors,
    _tutor_targets,
    hand_probability,
    simulate_opening_hands,
)

# ---------------------------------------------------------------------------
# hand_probability
# ---------------------------------------------------------------------------


class TestHandProbability:
    """Exact hypergeometric draw odds."""

    async def test_probability_at_least_one(self):
        """P(>= 1 copy) for a 10-card deck, 5 copies, 3 cards seen."""
        result = await hand_probability(deck_size=10, copies=5, cards_seen=3, min_count=1)
        assert result.data["probability"] == pytest.approx(11 / 12)

    async def test_probability_exact_count(self):
        """P(exactly 3 copies) for the same deck."""
        result = await hand_probability(
            deck_size=10, copies=5, cards_seen=3, min_count=3, max_count=3
        )
        assert result.data["probability"] == pytest.approx(10 / 120)

    async def test_probability_range(self):
        """P(1 <= copies <= 2) for the same deck."""
        result = await hand_probability(
            deck_size=10, copies=5, cards_seen=3, min_count=1, max_count=2
        )
        assert result.data["probability"] == pytest.approx(100 / 120)

    async def test_min_count_zero_covers_full_range(self):
        """min_count=0 with no max_count sums the full PMF to 1.0."""
        result = await hand_probability(deck_size=10, copies=5, cards_seen=3, min_count=0)
        assert result.data["probability"] == pytest.approx(1.0)

    async def test_expectation(self):
        """Expectation equals n * K / N."""
        result = await hand_probability(deck_size=10, copies=5, cards_seen=3, min_count=0)
        assert result.data["expectation"] == pytest.approx(3 * 5 / 10)

    async def test_defaults(self):
        """Default deck_size=99 and cards_seen=7 (Commander opening hand)."""
        result = await hand_probability()
        assert result.data["deck_size"] == 99
        assert result.data["cards_seen"] == 7

    async def test_copies_exceeds_deck_size_raises(self):
        with pytest.raises(ValueError, match="cannot exceed deck_size"):
            await hand_probability(deck_size=10, copies=11, cards_seen=3, min_count=1)

    async def test_max_count_less_than_min_count_raises(self):
        with pytest.raises(ValueError, match="cannot be less than min_count"):
            await hand_probability(deck_size=10, copies=5, cards_seen=3, min_count=3, max_count=1)

    async def test_negative_value_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            await hand_probability(deck_size=10, copies=-1, cards_seen=3, min_count=1)


# ---------------------------------------------------------------------------
# simulate_opening_hands -- mock data helpers
# ---------------------------------------------------------------------------


def _make_card(
    name: str,
    *,
    type_line: str = "Creature - Bear",
    mana_cost: str | None = "{1}{G}{G}",
    cmc: float = 3.0,
    oracle_text: str | None = None,
    produced_mana: list[str] | None = None,
) -> Card:
    return Card(
        id="test-id-" + name.lower().replace(" ", "-").replace(",", ""),
        name=name,
        mana_cost=mana_cost,
        cmc=cmc,
        type_line=type_line,
        oracle_text=oracle_text,
        produced_mana=produced_mana or [],
        prices=CardPrices(usd="1.00"),
        rarity="common",
    )


def _basic_deck(*, forests: int = 36, creatures: int = 63) -> list[str]:
    """36 basic Forests + N generic 3-CMC creatures (no ramp)."""
    entries = [f"{forests}x Forest"]
    entries.extend(f"Bear {i}" for i in range(creatures))
    return entries


def _make_bulk(cards: dict[str, Card]) -> AsyncMock:
    """Mock ScryfallBulkClient that returns Card objects by name."""

    async def get_card(name: str) -> Card | None:
        return cards.get(name.lower())

    mock = AsyncMock()
    mock.get_card = AsyncMock(side_effect=get_card)
    return mock


def _make_scryfall(cards: dict[str, Card]) -> AsyncMock:
    """Mock ScryfallClient fallback: raises CardNotFoundError for unknown names."""

    async def get_card_by_name(name: str, *, fuzzy: bool = False) -> Card:
        key = name.lower()
        if key in cards:
            return cards[key]
        raise CardNotFoundError(f"Card not found: '{name}'", status_code=404)

    mock = AsyncMock()
    mock.get_card_by_name = AsyncMock(side_effect=get_card_by_name)
    return mock


def _bear_cards(count: int) -> dict[str, Card]:
    return {f"bear {i}".lower(): _make_card(f"Bear {i}") for i in range(count)}


FOREST = _make_card("Forest", type_line="Basic Land - Forest", mana_cost=None, cmc=0.0)
SOL_RING = _make_card(
    "Sol Ring", type_line="Artifact", mana_cost="{1}", cmc=1.0, oracle_text="{T}: Add {C}{C}."
)
LLANOWAR_ELVES = _make_card(
    "Llanowar Elves",
    type_line="Creature - Elf Druid",
    mana_cost="{G}",
    cmc=1.0,
    oracle_text="{T}: Add {G}.",
)
RAMPANT_GROWTH = _make_card(
    "Rampant Growth",
    type_line="Sorcery",
    mana_cost="{1}{G}",
    cmc=2.0,
    oracle_text=(
        "Search your library for a basic land card, put it onto the battlefield "
        "tapped, then shuffle."
    ),
)
MDFC_LAND = _make_card(
    "Sea Gate Restoration // Sea Gate, Reborn",
    type_line="Sorcery // Land",
    mana_cost="{5}{U}{U}",
    cmc=7.0,
)
SIGNET = _make_card(
    "Fake Signet", type_line="Artifact", mana_cost="{2}", cmc=2.0, oracle_text="{T}: Add {W}."
)


def _slot_of(
    card: Card, *, extra: frozenset[str] = frozenset(), exclude: frozenset[str] = frozenset()
) -> _Slot:
    """Reduce a resolved ``Card`` to the ``_Slot`` the simulation engine consumes."""
    cls = _classify_card(card, extra_mana_sources=extra, exclude_cards=exclude)
    production = _mana_production(card, cls)
    return _Slot(cls, round(card.cmc), production)


def _big_deck(*, forests: int = 36, mdfc: int = 4, creatures: int = 59) -> dict[str, Card]:
    """A 99-card deck: 36 Forest, 4 MDFC lands, 59 generic creatures."""
    cards = {"forest": FOREST, "sea gate restoration // sea gate, reborn": MDFC_LAND}
    cards.update(_bear_cards(creatures))
    return cards


def _big_decklist(*, forests: int = 36, mdfc: int = 4, creatures: int = 59) -> list[str]:
    entries = [f"{forests}x Forest", f"{mdfc}x Sea Gate Restoration // Sea Gate, Reborn"]
    entries.extend(f"Bear {i}" for i in range(creatures))
    return entries


# ---------------------------------------------------------------------------
# simulate_opening_hands
# ---------------------------------------------------------------------------


class TestSimulateOpeningHands:
    """Monte Carlo opening hand simulation."""

    async def test_reproducible_with_seed(self):
        """Same seed produces identical results; a different seed diverges."""
        cards = _bear_cards(63)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)
        decklist = _basic_deck()

        result_a = await simulate_opening_hands(
            decklist, iterations=500, seed=42, bulk=bulk, scryfall=_make_scryfall(cards)
        )
        result_b = await simulate_opening_hands(
            decklist, iterations=500, seed=42, bulk=bulk, scryfall=_make_scryfall(cards)
        )
        assert result_a.data["keep_pct_by_hand_size"] == result_b.data["keep_pct_by_hand_size"]
        assert result_a.data["kept_land_distribution"] == result_b.data["kept_land_distribution"]

        result_c = await simulate_opening_hands(
            decklist, iterations=500, seed=7, bulk=bulk, scryfall=_make_scryfall(cards)
        )
        assert result_a.data["kept_land_distribution"] != result_c.data["kept_land_distribution"]

    async def test_kept_land_distribution_matches_hypergeometric(self):
        """No mulligan (min_lands=0, max_lands=7): distribution matches the exact PMF.

        Pinned to the legacy lands-only keep rule (``keep_rule="lands_v1"``,
        ``free_mulligan=False``): the hypergeometric PMF this test compares
        against only models raw land draws, not the playability rule's
        hand-development/gas checks.
        """
        cards = _bear_cards(63)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)
        decklist = _basic_deck()

        result = await simulate_opening_hands(
            decklist,
            iterations=10000,
            seed=42,
            min_lands=0,
            max_lands=7,
            keep_rule="lands_v1",
            free_mulligan=False,
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        distribution = result.data["kept_land_distribution"]
        for k in (2, 3, 4):
            expected = _hypergeom_pmf(99, 36, 7, k)
            actual = distribution.get(k, distribution.get(float(k), 0.0))
            assert abs(actual - expected) < 0.02, (k, actual, expected)

    async def test_mdfc_lands_classified_and_toggle(self):
        """MDFC lands classify as MDFC_LAND; count_mdfc_lands=False drops them from lands."""
        cards = _big_deck()
        bulk = _make_bulk(cards)
        decklist = _big_decklist()

        result = await simulate_opening_hands(
            decklist, iterations=200, seed=1, bulk=bulk, scryfall=_make_scryfall(cards)
        )
        mdfc_names = result.data["card_classes"]["MDFC_LAND"]
        assert "Sea Gate Restoration // Sea Gate, Reborn" in mdfc_names

        with_mdfc = await simulate_opening_hands(
            decklist,
            iterations=200,
            seed=1,
            count_mdfc_lands=True,
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        without_mdfc = await simulate_opening_hands(
            decklist,
            iterations=200,
            seed=1,
            count_mdfc_lands=False,
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        assert (
            with_mdfc.data["kept_land_distribution"] != without_mdfc.data["kept_land_distribution"]
        )

    async def test_extra_mana_sources_and_exclude_cards_overrides(self):
        """extra_mana_sources forces ROCK; exclude_cards forces OTHER."""
        weird_rock = _make_card("Weird Rock", type_line="Artifact", mana_cost="{4}", cmc=4.0)
        cards = _bear_cards(60)
        cards["forest"] = FOREST
        cards["sol ring"] = SOL_RING
        cards["weird rock"] = weird_rock
        bulk = _make_bulk(cards)
        decklist = [
            "36x Forest",
            "Sol Ring",
            "Weird Rock",
        ] + [f"Bear {i}" for i in range(60)]

        result = await simulate_opening_hands(
            decklist,
            iterations=100,
            seed=1,
            extra_mana_sources=["Weird Rock"],
            exclude_cards=["Sol Ring"],
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        assert "Weird Rock" in result.data["card_classes"]["ROCK"]
        assert "Sol Ring" in result.data["card_classes"]["OTHER"]
        assert "Sol Ring" not in result.data["card_classes"]["ROCK"]

    async def test_detects_rock_dork_and_ramp_spell(self):
        """Sol Ring -> ROCK, Llanowar Elves -> DORK, Rampant Growth -> RAMP_SPELL."""
        cards = _bear_cards(60)
        cards["forest"] = FOREST
        cards["sol ring"] = SOL_RING
        cards["llanowar elves"] = LLANOWAR_ELVES
        cards["rampant growth"] = RAMPANT_GROWTH
        bulk = _make_bulk(cards)
        decklist = [
            "36x Forest",
            "Sol Ring",
            "Llanowar Elves",
            "Rampant Growth",
        ] + [f"Bear {i}" for i in range(60)]

        result = await simulate_opening_hands(
            decklist, iterations=100, seed=1, bulk=bulk, scryfall=_make_scryfall(cards)
        )
        assert "Sol Ring" in result.data["card_classes"]["ROCK"]
        assert "Llanowar Elves" in result.data["card_classes"]["DORK"]
        assert "Rampant Growth" in result.data["card_classes"]["RAMP_SPELL"]

    async def test_forced_mulligan_to_four(self):
        """A land-light deck (7 lands / 99) forces mulligans down toward 4."""
        cards = _bear_cards(92)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)
        decklist = ["7x Forest"] + [f"Bear {i}" for i in range(92)]

        result = await simulate_opening_hands(
            decklist, iterations=300, seed=1, min_lands=2, bulk=bulk, scryfall=_make_scryfall(cards)
        )
        keep_pct = result.data["keep_pct_by_hand_size"]
        assert sum(keep_pct.values()) == pytest.approx(1.0)
        assert keep_pct[7] < 0.6
        assert keep_pct[4] > 0  # forced keeps at 4 cards must actually occur

    async def test_deck_too_small_raises(self):
        cards = {"forest": FOREST, "bear 1": _make_card("Bear 1")}
        bulk = _make_bulk(cards)
        with pytest.raises(ValueError, match="at least 7 cards"):
            await simulate_opening_hands(
                ["3x Forest", "Bear 1"], bulk=bulk, scryfall=_make_scryfall(cards)
            )

    async def test_iterations_out_of_range_raises(self):
        cards = _bear_cards(63)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)
        with pytest.raises(ValueError, match="iterations must be between"):
            await simulate_opening_hands(
                _basic_deck(), iterations=10, bulk=bulk, scryfall=_make_scryfall(cards)
            )

    async def test_unresolved_card_reported_and_run_succeeds(self):
        """An unknown card name lands in data['unresolved'] but the run completes."""
        cards = _bear_cards(62)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)
        decklist = ["36x Forest", "Totally Unknown Card"] + [f"Bear {i}" for i in range(62)]

        result = await simulate_opening_hands(
            decklist, iterations=100, seed=1, bulk=bulk, scryfall=_make_scryfall(cards)
        )
        assert "Totally Unknown Card" in result.data["unresolved"]


# ---------------------------------------------------------------------------
# _goldfish -- deterministic, hand-computed turn-by-turn mana
# ---------------------------------------------------------------------------


class TestGoldfishDeterministic:
    """Direct calls to ``_goldfish`` with a fixed hand/library: exact assertions."""

    def test_land_then_rock_turn_by_turn_mana(self):
        """2 lands + 1 rock (cmc 2, prod 1) in hand; a 3rd land drawn turn 3.

        Manual calculation:
        - T1: land #1 played (land #2 still in hand). Pool = land #1's
          production = 1 (active immediately). Rock costs 2 > pool 1, not
          castable. Spendable T1 = 1.
        - T2: land #2 played. Pool = land #1 (1) + land #2 (1) = 2. Rock
          (cmc 2) is castable: pool -= 2 -> 0. A ROCK produces immediately,
          so pool += its production (1) -> 1. Spendable T2 = 1. The rock was
          cast on turn 2, so ramp_by_turn2 = True.
        - T3: the drawn card is a land, played (3rd land drop). Pool = land#1
          (1) + land#2 (1) + land#3 (1) + rock (1, active since it was played
          on turn 2 < 3) = 4. No more castable cards in hand. Spendable T3 = 4.
        - T4/T5: library is exhausted, no new land or spell to play/cast.
          Active sources are unchanged (3 lands + 1 rock), so pool stays 4.
        """
        land = _Slot(_CardClass.LAND, 0, 1)
        rock = _Slot(_CardClass.ROCK, 2, 1)
        other = _Slot(_CardClass.OTHER, 3, 0)

        hand = [land, land, rock, other]
        library = [other, other, land]  # drawn on T1, T2, T3 respectively

        result = _goldfish(hand, library, count_mdfc_lands=True)

        assert result["spendable_by_turn"] == [1.0, 1.0, 4.0, 4.0, 4.0]
        assert result["ramp_by_turn2"] is True

    def test_dork_produces_only_from_the_following_turn(self):
        """A dork (cmc 1, prod 1) is cast turn 1 but only produces from turn 2.

        Manual calculation:
        - T1: land played. Pool = 1 (land, active immediately). The dork
          (cmc 1) is castable: pool -= 1 -> 0. Unlike a rock, a dork does
          NOT add its production back this turn (summoning sickness).
          Spendable T1 = 0.
        - T2: no land left to play, and the drawn cards are not lands. Pool =
          land (1, active) + dork (1, active now that its played turn 1 < 2)
          = 2. No more castable cards. Spendable T2 = 2.
        """
        land = _Slot(_CardClass.LAND, 0, 1)
        dork = _Slot(_CardClass.DORK, 1, 1)
        other = _Slot(_CardClass.OTHER, 3, 0)

        hand = [land, dork, other]
        library = [other, other, other]

        result = _goldfish(hand, library, count_mdfc_lands=True)

        assert result["spendable_by_turn"][0] == 0.0
        assert result["spendable_by_turn"][1] == 2.0
        assert result["ramp_by_turn2"] is True


# ---------------------------------------------------------------------------
# _keep_playability -- playability keep rule, direct calls
# ---------------------------------------------------------------------------

_KEEP_PLAYABILITY_KWARGS = {
    "min_lands": 2,
    "max_lands": 5,
    "count_mdfc_lands": True,
    "gas_cmc_threshold": 4,
}


class TestKeepPlayability:
    """Direct calls to ``_keep_playability``, defaults min_lands=2/max_lands=5/mdfc=True/gas=4."""

    def test_land_plus_rock_is_keepable(self):
        hand = [_slot_of(FOREST), _slot_of(SOL_RING)]
        hand += [_slot_of(_make_card(f"Bear {i}")) for i in range(5)]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) is None

    def test_slow_rock_only_hand_screws(self):
        hand = [_slot_of(FOREST), _slot_of(SIGNET)]
        hand += [_slot_of(_make_card(f"Bear {i}")) for i in range(5)]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) == "screw"

    def test_land_plus_two_rocks_is_keepable(self):
        hand = [_slot_of(FOREST), _slot_of(SOL_RING), _slot_of(SIGNET)]
        hand += [_slot_of(_make_card(f"Bear {i}")) for i in range(4)]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) is None

    def test_two_lands_no_ramp_is_keepable(self):
        hand = [_slot_of(FOREST), _slot_of(FOREST)]
        hand += [_slot_of(_make_card(f"Bear {i}")) for i in range(5)]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) is None

    def test_land_plus_dork_is_keepable(self):
        hand = [_slot_of(FOREST), _slot_of(LLANOWAR_ELVES)]
        hand += [_slot_of(_make_card(f"Bear {i}")) for i in range(5)]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) is None

    def test_zero_land_double_rock_hand_is_keepable(self):
        hand = [_Slot(_CardClass.ROCK, 0, 1), _Slot(_CardClass.ROCK, 0, 1)]
        hand += [_slot_of(_make_card(f"Bear {i}")) for i in range(5)]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) is None

    def test_six_lands_one_bear_floods(self):
        hand = [_slot_of(FOREST) for _ in range(6)] + [_slot_of(_make_card("Bear 0"))]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) == "flood"

    def test_three_lands_four_rocks_no_gas(self):
        hand = [_slot_of(FOREST) for _ in range(3)] + [_slot_of(SOL_RING) for _ in range(4)]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) == "no_gas"

    def test_two_lands_five_expensive_cards_no_gas(self):
        hand = [_slot_of(FOREST), _slot_of(FOREST)]
        hand += [_slot_of(_make_card(f"Big {i}", cmc=6.0)) for i in range(5)]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) == "no_gas"

    def test_mdfc_land_counted_toggle(self):
        hand = [_slot_of(FOREST), _slot_of(MDFC_LAND)]
        hand += [_slot_of(_make_card(f"Bear {i}")) for i in range(5)]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) is None
        no_mdfc_kwargs = dict(_KEEP_PLAYABILITY_KWARGS, count_mdfc_lands=False)
        assert _keep_playability(hand, **no_mdfc_kwargs) == "screw"


class TestKeepPlayabilityColorAndTutor:
    """Color screen (T8, T9) and tutor-as-gas (T15) extensions to the keep rule."""

    def _white_hand(self) -> list[_Slot]:
        white_land = _Slot(_CardClass.LAND, 0, 1, frozenset({"W"}))
        gas = _Slot(_CardClass.OTHER, 2, 0)
        return [white_land, white_land, gas, gas, gas, gas, gas]

    def test_color_screen_inert_without_commander_colors(self):
        assert _keep_playability(self._white_hand(), **_KEEP_PLAYABILITY_KWARGS) is None

    def test_color_screen_passes_when_all_colors_sourced(self):
        assert (
            _keep_playability(
                self._white_hand(), commander_colors=frozenset({"W"}), **_KEEP_PLAYABILITY_KWARGS
            )
            is None
        )

    def test_color_screen_fails_when_a_color_is_missing(self):
        assert (
            _keep_playability(
                self._white_hand(),
                commander_colors=frozenset({"W", "U"}),
                **_KEEP_PLAYABILITY_KWARGS,
            )
            == "colors"
        )

    def test_tutor_is_no_gas_without_flag_but_gas_with_flag(self):
        land = _Slot(_CardClass.LAND, 0, 1)
        tutor = _Slot(_CardClass.OTHER, 2, 0, frozenset(), frozenset(), True)
        big = _Slot(_CardClass.OTHER, 6, 0)
        hand = [land, land, tutor, big, big, big, big]
        assert _keep_playability(hand, **_KEEP_PLAYABILITY_KWARGS) == "no_gas"
        assert _keep_playability(hand, tutor_aware=True, **_KEEP_PLAYABILITY_KWARGS) is None


# ---------------------------------------------------------------------------
# _bottom_cards_playability -- London mulligan bottoming, direct calls
# ---------------------------------------------------------------------------

_BOTTOM_PLAYABILITY_KWARGS = {"count_mdfc_lands": True, "max_lands": 5, "gas_cmc_threshold": 4}


class TestBottomCardsPlayability:
    """Direct calls to ``_bottom_cards_playability`` with fixed hands."""

    def test_trims_a_land_when_flooded(self):
        hand = [_slot_of(FOREST) for _ in range(5)]
        hand += [_slot_of(SOL_RING), _slot_of(_make_card("Bear 0"))]
        remaining, bottomed = _bottom_cards_playability(hand, 1, **_BOTTOM_PLAYABILITY_KWARGS)
        assert len(bottomed) == 1
        assert bottomed[0].cls == _CardClass.LAND
        assert len(remaining) == len(hand) - 1

    def test_bottoms_most_expensive_gas_first(self):
        hand = [_slot_of(FOREST), _slot_of(FOREST)]
        hand += [_slot_of(_make_card(f"Spell {cmc}", cmc=float(cmc))) for cmc in (2, 3, 4, 5, 6)]
        remaining, bottomed = _bottom_cards_playability(hand, 2, **_BOTTOM_PLAYABILITY_KWARGS)
        assert [c.cmc for c in bottomed] == [6, 5]
        assert any(c.cmc == 2 for c in remaining)

    def test_sole_color_source_land_is_never_bottomed(self):
        red_land = _Slot(_CardClass.LAND, 0, 1, frozenset({"R"}))
        white_lands = [_Slot(_CardClass.LAND, 0, 1, frozenset({"W"})) for _ in range(5)]
        gas = _Slot(_CardClass.OTHER, 2, 0)
        hand = [*white_lands, red_land, gas]
        remaining, bottomed = _bottom_cards_playability(
            hand, 3, commander_colors=frozenset({"R", "W"}), **_BOTTOM_PLAYABILITY_KWARGS
        )
        assert red_land in remaining
        assert red_land not in bottomed
        assert all(c.cls == _CardClass.LAND for c in bottomed)


# ---------------------------------------------------------------------------
# simulate_opening_hands -- playability keep rule, end-to-end via mocks
# ---------------------------------------------------------------------------


class TestSimulateOpeningHandsV2:
    """End-to-end assertions on the playability keep rule via mocked resolution."""

    async def test_free_mulligan_keeps_hand_size_seven_through_first_redraw(self):
        """With free_mulligan=True, a hand kept after 0 or 1 mulligan is still 7 cards."""
        cards = _bear_cards(92)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)
        decklist = ["7x Forest"] + [f"Bear {i}" for i in range(92)]

        result = await simulate_opening_hands(
            decklist,
            iterations=300,
            seed=1,
            free_mulligan=True,
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        keep_pct = result.data["keep_pct_by_hand_size"]
        keep_pct_by_mull = result.data["keep_pct_by_mulligans"]
        assert sum(keep_pct.values()) == pytest.approx(1.0)
        assert sum(keep_pct_by_mull.values()) == pytest.approx(1.0)
        assert keep_pct[7] == pytest.approx(keep_pct_by_mull["0"] + keep_pct_by_mull["1"], abs=0.02)

        params = result.data["params"]
        assert params["keep_rule"] == "playability"
        assert params["free_mulligan"] is True
        assert params["gas_cmc_threshold"] == 4

    async def test_no_free_mulligan_hand_size_six_matches_one_mulligan(self):
        """With free_mulligan=False, a hand kept after 1 mulligan bottoms to size 6."""
        cards = _bear_cards(92)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)
        decklist = ["7x Forest"] + [f"Bear {i}" for i in range(92)]

        result = await simulate_opening_hands(
            decklist,
            iterations=300,
            seed=1,
            free_mulligan=False,
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        keep_pct = result.data["keep_pct_by_hand_size"]
        keep_pct_by_mull = result.data["keep_pct_by_mulligans"]
        assert sum(keep_pct.values()) == pytest.approx(1.0)
        assert sum(keep_pct_by_mull.values()) == pytest.approx(1.0)
        assert keep_pct[6] == pytest.approx(keep_pct_by_mull["1"], abs=0.02)

    async def test_forced_mulligan_all_land_deck_only_no_gas(self):
        """An all-land deck (max_lands=7, so flood never fires) is rejected purely on
        'no_gas', forcing every hand down to 4 cards after 4 rejected mulligans."""
        cards = {"forest": FOREST}
        bulk = _make_bulk(cards)
        decklist = ["99x Forest"]

        result = await simulate_opening_hands(
            decklist,
            iterations=200,
            seed=1,
            max_lands=7,
            free_mulligan=True,
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        assert result.data["keep_pct_by_hand_size"][4] == pytest.approx(1.0)
        assert result.data["keep_pct_by_mulligans"]["3plus"] == pytest.approx(1.0)
        assert result.data["mull_reasons"]["no_gas"] == 800

    async def test_mulligan_transparency_derives_from_keep_pct_by_mulligans(self):
        """keep_first_deal_pct / keep_via_free_mulligan_pct mirror keep_pct_by_mulligans."""
        cards = _bear_cards(63)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)
        decklist = _basic_deck()

        result = await simulate_opening_hands(
            decklist,
            iterations=500,
            seed=42,
            free_mulligan=True,
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        keep_pct_by_mull = result.data["keep_pct_by_mulligans"]
        assert result.data["keep_first_deal_pct"] == keep_pct_by_mull["0"]
        assert result.data["keep_via_free_mulligan_pct"] == keep_pct_by_mull["1"]
        assert "Kept on the first 7" in result.markdown
        assert "Saved by the Commander free mulligan" in result.markdown


# ---------------------------------------------------------------------------
# lands_v1 exact reproduction -- bit-for-bit match of the pre-playability output
# ---------------------------------------------------------------------------


class TestLandsV1ExactReproduction:
    """``keep_rule="lands_v1"`` + ``free_mulligan=False`` reproduces v1 output exactly."""

    def test_matches_pre_playability_reference(self):
        # Reference literals captured by running _run_simulation on commit 522b6cb
        # (pre-v2 code) with this exact deck, iterations=500, seed=42.
        deck = [_Slot(_CardClass.LAND, 0, 1)] * 36 + [_Slot(_CardClass.OTHER, 3, 0)] * 63

        result = _run_simulation(
            deck,
            iterations=500,
            seed=42,
            min_lands=2,
            max_lands=5,
            count_mdfc_lands=True,
            keep_rule="lands_v1",
            free_mulligan=False,
        )

        assert result["keep_pct_by_hand_size"] == {4: 0.008, 5: 0.05, 6: 0.14, 7: 0.802}
        assert result["kept_land_distribution"] == {
            1.0: 0.002,
            2.0: 0.406,
            3.0: 0.392,
            4.0: 0.162,
            5.0: 0.038,
        }


_DEMONIC_TUTOR = _make_card(
    "Demonic Tutor",
    type_line="Sorcery",
    mana_cost="{1}{B}",
    cmc=2.0,
    oracle_text="Search your library for a card and put that card into your hand, then shuffle.",
)
_VAMPIRIC_TUTOR = _make_card(
    "Vampiric Tutor",
    type_line="Instant",
    mana_cost="{B}",
    cmc=1.0,
    oracle_text=(
        "Search your library for a card, then shuffle and put that card on top of it. "
        "You lose 2 life."
    ),
)
_WORLDLY_TUTOR = _make_card(
    "Worldly Tutor",
    type_line="Instant",
    mana_cost="{G}",
    cmc=1.0,
    oracle_text=(
        "Search your library for a creature card, reveal that card, then shuffle and put "
        "that card on top of it."
    ),
)
_ENTOMB = _make_card(
    "Entomb",
    type_line="Instant",
    mana_cost="{B}",
    cmc=1.0,
    oracle_text="Search your library for a card, put that card into your graveyard, then shuffle.",
)
_ENLIGHTENED_TUTOR = _make_card(
    "Enlightened Tutor",
    type_line="Instant",
    mana_cost="{W}",
    cmc=1.0,
    oracle_text=(
        "Search your library for an artifact or enchantment card, reveal that card, then "
        "shuffle and put that card on top of it."
    ),
)
_GREEN_SUNS_ZENITH = _make_card(
    "Green Sun's Zenith",
    type_line="Sorcery",
    mana_cost="{X}{G}",
    cmc=1.0,
    oracle_text=(
        "Search your library for a green creature card with mana value X or less, put it "
        "onto the battlefield, then shuffle. Shuffle Green Sun's Zenith into its owner's library."
    ),
)
_CHORD_OF_CALLING = _make_card(
    "Chord of Calling",
    type_line="Instant",
    mana_cost="{X}{G}{G}{G}",
    cmc=3.0,
    oracle_text=(
        "Convoke. Flash. Search your library for a creature card, put it onto the "
        "battlefield, then shuffle."
    ),
)
_LAND_TAX = _make_card(
    "Land Tax",
    type_line="Enchantment",
    mana_cost="{W}",
    cmc=1.0,
    oracle_text=(
        "Whenever an opponent controls more lands than you, you may search your library "
        "for up to three basic land cards, reveal them, put them into your hand, then shuffle."
    ),
)


class TestClassifyTutor:
    """Static tutor classification (T12) and non-tutor exclusions (T13)."""

    @pytest.mark.parametrize(
        ("card", "expected"),
        [
            (_DEMONIC_TUTOR, ("any", "hand", "sorcery")),
            (_VAMPIRIC_TUTOR, ("any", "top", "instant")),
            (_WORLDLY_TUTOR, ("creature", "top", "instant")),
            (_ENTOMB, ("any", "graveyard", "instant")),
            (_ENLIGHTENED_TUTOR, ("artifact_enchantment", "top", "instant")),
            (_GREEN_SUNS_ZENITH, ("color_creature", "battlefield", "sorcery")),
            (_CHORD_OF_CALLING, ("creature", "battlefield", "instant")),
            (_LAND_TAX, ("basic_land", "hand", "sorcery")),
        ],
    )
    def test_classify_known_tutors(self, card: Card, expected: tuple[str, str, str]):
        cls = _classify_card(card, extra_mana_sources=frozenset(), exclude_cards=frozenset())
        assert _classify_tutor(card, cls) == expected

    def test_fetchland_and_ramp_spell_are_not_tutors(self):
        arid_mesa = _make_card(
            "Arid Mesa",
            type_line="Land",
            mana_cost=None,
            cmc=0.0,
            oracle_text=(
                "{T}, Pay 1 life, Sacrifice Arid Mesa: Search your library for a "
                "Mountain or Plains card, put it onto the battlefield, then shuffle."
            ),
        )
        for card in (arid_mesa, RAMPANT_GROWTH):
            cls = _classify_card(card, extra_mana_sources=frozenset(), exclude_cards=frozenset())
            assert _classify_tutor(card, cls) is None


class TestTutorTargetsAndAggregation:
    """Real target counting (T14) and Monte-Carlo tutor aggregation."""

    def test_creature_tutor_target_count(self):
        deck_cards = [_make_card(f"Beast {i}", type_line="Creature - Beast") for i in range(5)]
        deck_cards += [
            _make_card("Sol Ring", type_line="Artifact", mana_cost="{1}"),
            _make_card("Counterspell", type_line="Instant", mana_cost="{U}{U}"),
            _make_card("Forest", type_line="Basic Land - Forest", mana_cost=None),
        ]
        names, count = _tutor_targets("creature", deck_cards)
        assert count == 5
        assert all("Beast" in n for n in names)

    def test_any_tutor_counts_every_card(self):
        deck_cards = [_make_card(f"Beast {i}", type_line="Creature - Beast") for i in range(3)]
        _, count = _tutor_targets("any", deck_cards)
        assert count == 3

    def test_run_simulation_reports_tutor_in_hand(self):
        tutor = _Slot(_CardClass.OTHER, 2, 0, frozenset(), frozenset(), True)
        deck = [_Slot(_CardClass.LAND, 0, 1)] * 36 + [tutor] * 63
        stats = _run_simulation(
            deck,
            iterations=200,
            seed=42,
            min_lands=2,
            max_lands=5,
            count_mdfc_lands=True,
            tutor_aware=True,
            commander_colors=frozenset(),
        )
        assert stats["tutor_in_hand_pct"] > 0.9
        assert "colors" in stats["mull_reasons"]


class TestSourceColors:
    """Deck-aware color sourcing for the v3 color screen (T4-T7)."""

    def test_basic_land_reports_its_color(self):
        forest = _make_card(
            "Forest", type_line="Basic Land - Forest", mana_cost=None, cmc=0.0, produced_mana=["G"]
        )
        cls = _classify_card(forest, extra_mana_sources=frozenset(), exclude_cards=frozenset())
        assert _source_colors(forest, cls, deck_land_colors=frozenset()) == frozenset({"G"})

    def test_fetchland_named_subtypes_deck_aware(self):
        arid_mesa = _make_card(
            "Arid Mesa",
            type_line="Land",
            mana_cost=None,
            cmc=0.0,
            oracle_text=(
                "{T}, Pay 1 life, Sacrifice Arid Mesa: Search your library for a "
                "Mountain or Plains card, put it onto the battlefield, then shuffle."
            ),
            produced_mana=[],
        )
        cls = _classify_card(arid_mesa, extra_mana_sources=frozenset(), exclude_cards=frozenset())
        assert cls == _CardClass.LAND
        assert _source_colors(arid_mesa, cls, deck_land_colors=frozenset({"U"})) == frozenset(
            {"R", "W"}
        )

    def test_command_tower_reports_all_five(self):
        tower = _make_card(
            "Command Tower",
            type_line="Land",
            mana_cost=None,
            cmc=0.0,
            oracle_text="{T}: Add one mana of any color in your commander's color identity.",
            produced_mana=["W", "U", "B", "R", "G"],
        )
        cls = _classify_card(tower, extra_mana_sources=frozenset(), exclude_cards=frozenset())
        assert _source_colors(tower, cls, deck_land_colors=frozenset()) == frozenset(
            {"W", "U", "B", "R", "G"}
        )

    def test_colorless_rock_filtered_to_empty(self):
        rock = _make_card(
            "Sol Ring",
            type_line="Artifact",
            mana_cost="{1}",
            cmc=1.0,
            oracle_text="{T}: Add {C}{C}.",
            produced_mana=["C"],
        )
        cls = _classify_card(rock, extra_mana_sources=frozenset(), exclude_cards=frozenset())
        assert cls == _CardClass.ROCK
        assert _source_colors(rock, cls, deck_land_colors=frozenset()) == frozenset()

    def test_generic_fetch_ramp_uses_deck_colors(self):
        cls = _classify_card(
            RAMPANT_GROWTH, extra_mana_sources=frozenset(), exclude_cards=frozenset()
        )
        assert cls == _CardClass.RAMP_SPELL
        assert _source_colors(
            RAMPANT_GROWTH, cls, deck_land_colors=frozenset({"G", "W"})
        ) == frozenset({"G", "W"})


class TestPlayabilityExactReproduction:
    """Default playability rule reproduces its pre-v3 output bit-for-bit (T11).

    Guards the v3 color-screen/tutor extension against regressions: with no
    ``commander_colors``/``tutor_aware`` inputs, every pre-existing aggregate
    must stay numerically identical. Any drift here means a new field is being
    read outside its feature-flag guard.
    """

    def test_matches_playability_reference(self):
        # Reference literals captured on commit 6e59a52 (intact main, pre-v3) via a
        # throwaway _run_simulation run: this exact deck, iterations=500, seed=42,
        # playability defaults (min_lands=2, max_lands=5, mdfc=True, free_mull=True, gas=4).
        deck = [_Slot(_CardClass.LAND, 0, 1)] * 36 + [_Slot(_CardClass.OTHER, 3, 0)] * 63

        result = _run_simulation(
            deck,
            iterations=500,
            seed=42,
            min_lands=2,
            max_lands=5,
            count_mdfc_lands=True,
            keep_rule="playability",
            free_mulligan=True,
            gas_cmc_threshold=4,
        )

        assert result["keep_pct_by_hand_size"] == {7: 0.942, 6: 0.05, 5: 0.006, 4: 0.002}
        assert result["kept_land_distribution"] == {4.0: 0.188, 3.0: 0.356, 2.0: 0.408, 5.0: 0.048}
        assert result["avg_spendable_mana_by_turn"] == {1: 1.0, 2: 2.0, 3: 2.894, 4: 3.68, 5: 4.308}
        assert result["on_curve_pct_by_turn"] == {1: 1.0, 2: 1.0, 3: 0.894, 4: 0.748, 5: 0.548}
        assert result["ramp_by_turn2_pct"] == 0.0
        assert result["keep_pct_by_mulligans"] == {"0": 0.802, "1": 0.14, "2": 0.05, "3plus": 0.008}
        # Pre-existing mull-reason keys must not drift (a "colors" key may be added additively).
        assert result["mull_reasons"]["screw"] == 130
        assert result["mull_reasons"]["flood"] == 3
        assert result["mull_reasons"]["no_gas"] == 0


class TestHandComposition:
    """Kept-hand composition aggregation (T13): lands, ramp, tutors, gas, top-end, colors."""

    _DECK = (
        [_Slot(_CardClass.LAND, 0, 1, frozenset({"W"}))] * 20
        + [_Slot(_CardClass.LAND, 0, 1, frozenset({"U"}))] * 16
        + [_Slot(_CardClass.ROCK, 2, 1, frozenset())] * 5
        + [_Slot(_CardClass.OTHER, 2, 0, frozenset(), frozenset(), True)] * 2
        + [_Slot(_CardClass.OTHER, 3, 0)] * 50
        + [_Slot(_CardClass.OTHER, 7, 0)] * 6
    )

    def test_known_deck_exact_composition(self):
        # Reference literals captured via a throwaway _run_simulation run on this
        # exact 99-card mock deck (20 W lands, 16 U lands, 5 rocks, 2 tutors cmc 2,
        # 50 gas cmc 3, 6 bombs cmc 7), iterations=500, seed=42, playability defaults,
        # tutor_aware=True.
        result = _run_simulation(
            self._DECK,
            iterations=500,
            seed=42,
            min_lands=2,
            max_lands=5,
            count_mdfc_lands=True,
            keep_rule="playability",
            free_mulligan=True,
            gas_cmc_threshold=4,
            tutor_aware=True,
        )
        composition = result["hand_composition"]
        assert composition["avg_lands"] == 2.876
        assert composition["avg_rocks_dorks"] == 0.316
        assert composition["avg_tutors"] == 0.15
        assert composition["avg_gas"] == 3.376
        assert composition["avg_cmc6plus"] == 0.364
        assert composition["pct_no_gas"] == 0.0
        assert composition["pct_2plus_cmc6"] == 0.054
        assert composition["color_source_pct"] == {"W": 0.904, "U": 0.8}

    def test_avg_tutors_is_none_without_tutor_aware(self):
        """avg_tutors reports None (not a false 0) when tutor_aware is off."""
        result = _run_simulation(
            self._DECK,
            iterations=500,
            seed=42,
            min_lands=2,
            max_lands=5,
            count_mdfc_lands=True,
            keep_rule="playability",
            free_mulligan=True,
            gas_cmc_threshold=4,
            tutor_aware=False,
        )
        assert result["hand_composition"]["avg_tutors"] is None

    async def test_hand_composition_flows_through_simulate_opening_hands(self):
        """data['hand_composition'] and the markdown section are wired end-to-end."""
        cards = _bear_cards(63)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)

        result = await simulate_opening_hands(
            _basic_deck(), iterations=200, seed=1, bulk=bulk, scryfall=_make_scryfall(cards)
        )
        composition = result.data["hand_composition"]
        assert composition["avg_tutors"] is None
        # Module-level FOREST has no produced_mana: no card in this mock deck
        # reports a color, so the per-color breakdown is empty (not a fixture bug --
        # colorless sourcing is a real, documented edge case, see CHANGELOG).
        assert composition["color_source_pct"] == {}
        assert "## Hand Composition" in result.markdown
        assert "Reading tip" in result.markdown


class TestSimulateOpeningHandsV3:
    """End-to-end color-screen and tutor extensions via mocked resolution."""

    async def test_tutor_aware_reports_tutors_and_odds(self):
        cards = _bear_cards(62)
        cards["forest"] = FOREST
        cards["demonic tutor"] = _DEMONIC_TUTOR
        bulk = _make_bulk(cards)
        decklist = ["36x Forest", "Demonic Tutor", *[f"Bear {i}" for i in range(62)]]

        result = await simulate_opening_hands(
            decklist,
            iterations=200,
            seed=1,
            tutor_aware=True,
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        tutor_names = [t["name"] for t in result.data["tutors"]]
        assert "Demonic Tutor" in tutor_names
        assert isinstance(result.data["tutor_in_hand_pct"], float)
        assert result.data["params"]["tutor_aware"] is True

    async def test_commander_colors_parsed_into_params(self):
        cards = _bear_cards(63)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)

        result = await simulate_opening_hands(
            _basic_deck(),
            iterations=200,
            seed=1,
            commander_colors="mardu",
            bulk=bulk,
            scryfall=_make_scryfall(cards),
        )
        assert result.data["params"]["commander_colors"] == ["B", "R", "W"]

    async def test_invalid_commander_colors_raises(self):
        cards = _bear_cards(63)
        cards["forest"] = FOREST
        bulk = _make_bulk(cards)
        with pytest.raises(ValueError, match="color identity"):
            await simulate_opening_hands(
                _basic_deck(),
                iterations=200,
                commander_colors="notacolor",
                bulk=bulk,
                scryfall=_make_scryfall(cards),
            )
