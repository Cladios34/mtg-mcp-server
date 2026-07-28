"""Tests for deck_mechanic_map.

Regression origin (2026-07-27, Yuriko audit) — the most structurally damaging error of
that session. Every deliverable described the deck as a loop "cheap creature -> Yuriko
-> reveal": ONE destination. The deck actually ran 15 ninjutsu cards. The cheap creature
does not exist to deploy Yuriko, it exists to deploy whichever of the 15 answers the
turn; Yuriko is merely one of them.

No tool said so. deck_analysis returns categories per card, EDHREC returns popularity
per card. The shared mechanic was visible only to someone who thought to intersect a
type search with the decklist — and nothing prompted that intersection.

This tool makes the mechanic the unit of analysis instead of the card.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mtg_mcp_server.types import Card
from mtg_mcp_server.workflows.mechanic_map import deck_mechanic_map


def _card(
    name: str,
    *,
    type_line: str = "Creature — Ninja",
    oracle: str | None = None,
    mana_cost: str = "{1}{U}",
    cmc: float = 2.0,
    keywords: list[str] | None = None,
) -> Card:
    return Card(
        id=name.lower().replace(" ", "-"),
        name=name,
        type_line=type_line,
        oracle_text=oracle,
        mana_cost=mana_cost,
        cmc=cmc,
        keywords=keywords or [],
    )


YURIKO = _card(
    "Yuriko, the Tiger's Shadow",
    type_line="Legendary Creature — Human Ninja",
    oracle=(
        "Commander ninjutsu {1}{U}{B}\n"
        "Whenever a Ninja you control deals combat damage to a player, reveal the top "
        "card of your library and put that card into your hand."
    ),
    mana_cost="{1}{U}{B}",
    cmc=3.0,
    keywords=["Commander ninjutsu"],
)

# The reducer that produced two opposite errors in one audit.
SILVER_FUR = _card(
    "Silver-Fur Master",
    type_line="Creature — Rat Ninja",
    oracle=(
        "Ninjutsu {1}{U}\n"
        "Other Ninja spells you cast cost {1} less to cast. "
        "Ninjutsu abilities you activate cost {1} less to activate."
    ),
    mana_cost="{2}{U}",
    cmc=3.0,
)

# Coloured-only ninjutsu cost: the reducer CANNOT touch it (601.2f).
SKULLSNATCHER = _card(
    "Skullsnatcher",
    type_line="Creature — Rat Ninja",
    oracle="Ninjutsu {U}{B}\nWhenever this creature deals combat damage to a player, exile cards.",
    mana_cost="{1}{B}",
)

# Generic component present: the reducer DOES touch it.
MISTBLADE = _card(
    "Mistblade Shinobi",
    oracle="Ninjutsu {1}{U}\nWhenever this creature deals combat damage to a player, return a creature.",
)

CHANGELING_OUTCAST = _card(
    "Changeling Outcast",
    type_line="Creature — Shapeshifter",
    oracle="Changeling (This card is every creature type.)\nThis creature can't block and can't be blocked.",
    mana_cost="{U}",
    cmc=1.0,
    keywords=["Changeling"],
)

KAITO = _card(
    "Kaito, Bane of Nightmares",
    type_line="Legendary Planeswalker — Kaito",
    oracle=(
        "During your turn, as long as Kaito has one or more loyalty counters on him, "
        "he's a 3/4 Ninja creature."
    ),
    mana_cost="{2}{U}{B}",
    cmc=4.0,
)

PROSPEROUS_THIEF = _card(
    "Prosperous Thief",
    oracle=(
        "Ninjutsu {1}{U}\n"
        "Whenever one or more Ninja or Rogue you control deal combat damage to a "
        "player, add {U}{U}."
    ),
)

SOL_RING = _card(
    "Sol Ring", type_line="Artifact", oracle="{T}: Add {C}{C}.", mana_cost="{1}", cmc=1.0
)

_ALL = [
    YURIKO,
    SILVER_FUR,
    SKULLSNATCHER,
    MISTBLADE,
    CHANGELING_OUTCAST,
    KAITO,
    PROSPEROUS_THIEF,
    SOL_RING,
]


@pytest.fixture
def bulk() -> AsyncMock:
    client = AsyncMock()
    by_name = {c.name.lower(): c for c in _ALL}

    async def get_card(name: str) -> Card | None:
        return by_name.get(name.lower())

    client.get_card.side_effect = get_card
    return client


@pytest.fixture
def scryfall() -> AsyncMock:
    """Fallback that knows nothing — every bulk miss comes back unresolved."""
    client = AsyncMock()

    async def collection(names: list[str]) -> tuple[list[Card], list[str]]:
        return [], list(names)

    client.get_cards_collection.side_effect = collection
    return client


DECK = [c.name for c in _ALL if c.name != YURIKO.name]


def _mechanic(data, keyword: str) -> dict:
    return next(m for m in data["mechanics"] if m["keyword"].lower() == keyword.lower())


class TestMechanicDetection:
    """The mechanic, not the card, is the unit."""

    async def test_commander_keyword_is_mapped(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        ninjutsu = _mechanic(result.data, "Ninjutsu")
        assert ninjutsu["source"] == "commander_keyword"

    async def test_every_carrier_is_counted_not_just_the_commander(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        ninjutsu = _mechanic(result.data, "Ninjutsu")
        names = {c["name"] for c in ninjutsu["cards"]}
        # Yuriko is one destination among many, and the commander counts as one of them.
        assert {
            "Silver-Fur Master",
            "Skullsnatcher",
            "Mistblade Shinobi",
            "Prosperous Thief",
        } <= names
        assert ninjutsu["count"] == len(names)

    async def test_markdown_states_the_number_of_destinations(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        assert "Ninjutsu" in result.markdown
        assert str(_mechanic(result.data, "Ninjutsu")["count"]) in result.markdown

    async def test_activation_costs_are_reported(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        by_name = {c["name"]: c for c in _mechanic(result.data, "Ninjutsu")["cards"]}
        assert by_name["Skullsnatcher"]["activation_cost"] == "{U}{B}"
        assert by_name["Mistblade Shinobi"]["activation_cost"] == "{1}{U}"

    async def test_cost_tiers_group_by_activation_value(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        tiers = _mechanic(result.data, "Ninjutsu")["cost_tiers"]
        assert "Mistblade Shinobi" in tiers["2"]


class TestCostModifiers:
    """The reducer error, in both directions it was made."""

    async def test_reducer_is_detected(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        modifiers = _mechanic(result.data, "Ninjutsu")["cost_modifiers"]
        assert any(m["name"] == "Silver-Fur Master" for m in modifiers)

    async def test_reducer_reports_how_many_it_actually_reduces(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        mechanic = _mechanic(result.data, "Ninjutsu")
        modifier = next(m for m in mechanic["cost_modifiers"] if m["name"] == "Silver-Fur Master")
        # Not "all of them" and not a guess: the ones with a generic component.
        assert modifier["reduces"] < modifier["of"]
        assert "Skullsnatcher" in modifier["unaffected"]

    async def test_denominator_counts_only_what_was_tested(self, bulk, scryfall) -> None:
        """ "reduces X of Y" must have Y = cards actually examined.

        The reducer carries the keyword itself and is skipped, so quoting the full
        carrier count would claim a card was tested when it never was — the same
        species of wrong ratio ("4 instead of 9 of 15") this tool exists to prevent.
        """
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        mechanic = _mechanic(result.data, "Ninjutsu")
        modifier = next(m for m in mechanic["cost_modifiers"] if m["name"] == "Silver-Fur Master")
        assert modifier["of"] == modifier["reduces"] + len(modifier["unaffected"])
        # The reducer itself is a carrier, so the tested set is strictly smaller.
        assert modifier["of"] == modifier["carriers_total"] - 1

    async def test_coloured_only_cost_is_named_as_unaffected(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        modifier = next(
            m
            for m in _mechanic(result.data, "Ninjutsu")["cost_modifiers"]
            if m["name"] == "Silver-Fur Master"
        )
        assert "601.2f" in modifier["note"]

    async def test_markdown_carries_the_generic_only_warning(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        assert "601.2f" in result.markdown


class TestTypeSynergy:
    """Rule 702.73a, and the conditional case."""

    def _ninja(self, data) -> dict:
        return next(t for t in data["type_synergy"] if t["type"] == "Ninja")

    async def test_changelings_count_as_the_tribe(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        ninja = self._ninja(result.data)
        assert ninja["cards_by_changeling"] == ["Changeling Outcast"]

    async def test_conditional_creature_counts_with_its_condition(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        ninja = self._ninja(result.data)
        conditional = ninja["cards_conditional"]
        assert conditional[0]["name"] == "Kaito, Bane of Nightmares"
        assert conditional[0]["condition"]

    async def test_total_counts_every_route_to_the_type(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        ninja = self._ninja(result.data)
        assert ninja["total"] == (
            len(ninja["cards_typed"])
            + len(ninja["cards_by_changeling"])
            + len(ninja["cards_conditional"])
        )

    async def test_changeling_note_cites_the_rule(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        assert "702.73a" in result.markdown


class TestTriggerScope:
    """Per-source and per-combat must not be flattened into one another."""

    async def test_per_source_and_per_combat_are_distinguished(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        triggers = {t["name"]: t for t in result.data["triggers"]}
        assert triggers["Yuriko, the Tiger's Shadow"]["scope"] == "per_source"
        assert triggers["Prosperous Thief"]["scope"] == "per_combat"

    async def test_zero_power_caveat_travels_with_combat_damage_triggers(
        self, bulk, scryfall
    ) -> None:
        result = await deck_mechanic_map(DECK, YURIKO.name, bulk=bulk, scryfall=scryfall)
        triggers = {t["name"]: t for t in result.data["triggers"]}
        assert "510.1a" in triggers["Yuriko, the Tiger's Shadow"]["notes"]


class TestUnresolved:
    """A card no source can resolve is reported, never silently dropped."""

    async def test_unresolved_cards_are_listed(self, bulk, scryfall) -> None:
        result = await deck_mechanic_map(
            [*DECK, "Not A Real Card"], YURIKO.name, bulk=bulk, scryfall=scryfall
        )
        assert "Not A Real Card" in result.data["unresolved"]

    async def test_second_ability_is_reported_not_dropped(self, bulk, scryfall) -> None:
        """A card with two triggers reports the widest AND names the others.

        Adversarial review (2026-07-28) caught the field being computed and then
        left out of the payload: the reader saw a per_source card and never learned
        it also had an upkeep trigger.
        """
        two_abilities = _card(
            "Two Ability Ninja",
            oracle=(
                "Ninjutsu {1}{U}\n"
                "At the beginning of your upkeep, draw a card.\n"
                "Whenever a Ninja you control deals combat damage to a player, draw."
            ),
        )
        by_name = {c.name.lower(): c for c in [*_ALL, two_abilities]}
        bulk.get_card.side_effect = lambda name: by_name.get(name.lower())

        result = await deck_mechanic_map(
            [*DECK, two_abilities.name], YURIKO.name, bulk=bulk, scryfall=scryfall
        )
        entry = next(t for t in result.data["triggers"] if t["name"] == two_abilities.name)
        assert entry["scope"] == "per_source"
        assert "per_turn" in entry["other_scopes"]

    async def test_entry_trigger_naming_a_class_multiplies(self, bulk, scryfall) -> None:
        """Impact Tremors fires per creature, so it must not rank as a one-shot ETB.

        Adversarial review (2026-07-28): the entry branch matched before the
        multi-source branch, so every ETB payoff came back `on_entry` and dropped out
        of the trigger-reach section entirely — the one blindness this tool exists to
        remove.
        """
        tremors = _card(
            "Impact Tremors",
            oracle=(
                "Whenever a creature enters the battlefield under your control, "
                "Impact Tremors deals 1 damage to each opponent."
            ),
            type_line="Enchantment",
        )
        one_shot = _card(
            "Solitary Ninja",
            oracle="When this creature enters, draw a card.",
        )
        by_name = {c.name.lower(): c for c in [*_ALL, tremors, one_shot]}
        bulk.get_card.side_effect = lambda name: by_name.get(name.lower())

        result = await deck_mechanic_map(
            [*DECK, tremors.name, one_shot.name], YURIKO.name, bulk=bulk, scryfall=scryfall
        )
        triggers = {t["name"]: t for t in result.data["triggers"]}
        assert triggers[tremors.name]["scope"] == "per_source"
        assert triggers[tremors.name]["sources"] == "class"
        # The one-shot must NOT be swept up by the same widening.
        assert triggers[one_shot.name]["scope"] == "on_entry"

    async def test_reducer_scope_is_shown_not_asserted(self, bulk, scryfall) -> None:
        """A reduction whose scope is a chosen type cannot be credited to a mechanic.

        Adversarial review (2026-07-28): the count answered "does the amount fit
        these costs", and the markdown read it as "this card reduces them".
        """
        horn = _card(
            "Herald's Horn",
            oracle=(
                "As Herald's Horn enters, choose a creature type.\n"
                "Spells you cast of the chosen type cost {1} less to cast."
            ),
            type_line="Artifact",
        )
        by_name = {c.name.lower(): c for c in [*_ALL, horn]}
        bulk.get_card.side_effect = lambda name: by_name.get(name.lower())

        result = await deck_mechanic_map(
            [*DECK, horn.name], YURIKO.name, bulk=bulk, scryfall=scryfall
        )
        modifiers = [
            m
            for mech in result.data["mechanics"]
            for m in mech["cost_modifiers"]
            if m["name"] == horn.name
        ]
        assert modifiers, "the reducer should still be surfaced"
        assert modifiers[0]["depends_on_choice"] is True
        assert "chosen type" in modifiers[0]["clause"]
        assert "depends on that choice" in result.markdown
