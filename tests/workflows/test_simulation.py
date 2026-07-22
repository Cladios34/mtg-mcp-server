"""Tests for hand_probability and simulate_opening_hands workflows."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mtg_mcp_server.services.scryfall import CardNotFoundError
from mtg_mcp_server.types import Card, CardPrices
from mtg_mcp_server.workflows.simulation import (
    _CardClass,
    _goldfish,
    _hypergeom_pmf,
    _Slot,
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
) -> Card:
    return Card(
        id="test-id-" + name.lower().replace(" ", "-").replace(",", ""),
        name=name,
        mana_cost=mana_cost,
        cmc=cmc,
        type_line=type_line,
        oracle_text=oracle_text,
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
        """No mulligan (min_lands=0, max_lands=7): distribution matches the exact PMF."""
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
