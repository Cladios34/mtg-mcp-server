"""Tests for trigger scope derivation.

Regression origin (2026-07-27, Yuriko audit). Three cards in one deck read almost
identically and behave very differently:

- Yuriko: "Whenever a Ninja you control deals combat damage to a player" — fires once
  PER NINJA that connects.
- Ingenious Infiltrator: same shape, same per-source multiplication.
- Prosperous Thief: "Whenever one or more Ninja or Rogue..." — fires ONCE per combat,
  however many creatures connect.

A card whose effect multiplies by the number of attackers is not worth the same as one
that caps at one, and nothing in the card data flagged the difference. The first error
of that audit came from the neighbouring confusion: "attacks" is not "deals combat
damage" — an attacker that is blocked, or has 0 power, never deals any (510.1a).
"""

from __future__ import annotations

from mtg_mcp_server.types import Card
from mtg_mcp_server.utils.triggers import derive_trigger


def _card(oracle: str | None, **kwargs) -> Card:
    base = {"id": "x", "name": "Test", "type_line": "Creature — Ninja", "oracle_text": oracle}
    return Card.model_validate(base | kwargs)


class TestPerSourceVsPerCombat:
    """The distinction that changes a card's value outright."""

    def test_yuriko_multiplies_per_ninja(self) -> None:
        trigger = derive_trigger(
            _card(
                "Whenever a Ninja you control deals combat damage to a player, "
                "reveal the top card of your library and put that card into your hand."
            )
        )
        assert trigger.scope == "per_source"
        assert trigger.condition == "combat_damage_to_player"

    def test_one_or_more_caps_at_one_per_combat(self) -> None:
        trigger = derive_trigger(
            _card(
                "Whenever one or more Ninja or Rogue you control deal combat damage "
                "to a player, add {U}{U}."
            )
        )
        assert trigger.scope == "per_combat"

    def test_each_creature_multiplies(self) -> None:
        trigger = derive_trigger(
            _card("Whenever each other Ninja you control deals combat damage to a player, draw.")
        )
        assert trigger.scope == "per_source"

    def test_self_reference_is_a_single_source(self) -> None:
        trigger = derive_trigger(
            _card("Whenever this creature deals combat damage to a player, draw a card.")
        )
        assert trigger.scope == "per_source"
        assert trigger.sources == "self"


class TestAttacksVsDealsDamage:
    """The neighbouring confusion, and the 510.1a corollary."""

    def test_attacks_is_its_own_condition(self) -> None:
        trigger = derive_trigger(_card("Whenever this creature attacks, draw a card."))
        assert trigger.condition == "attacks"

    def test_combat_damage_carries_the_zero_power_caveat(self) -> None:
        trigger = derive_trigger(
            _card("Whenever this creature deals combat damage to a player, draw a card.")
        )
        assert trigger.notes is not None
        assert "510.1a" in trigger.notes

    def test_attacks_does_not_carry_the_damage_caveat(self) -> None:
        trigger = derive_trigger(_card("Whenever this creature attacks, draw a card."))
        assert trigger.notes is None

    def test_combat_damage_without_a_player_target(self) -> None:
        trigger = derive_trigger(_card("Whenever this creature deals combat damage, draw a card."))
        assert trigger.condition == "combat_damage"


class TestOtherScopes:
    def test_upkeep_is_per_turn(self) -> None:
        trigger = derive_trigger(_card("At the beginning of your upkeep, draw a card."))
        assert trigger.scope == "per_turn"
        assert trigger.condition == "phase_step"

    def test_enters_the_battlefield(self) -> None:
        trigger = derive_trigger(_card("When this creature enters, draw a card."))
        assert trigger.scope == "on_entry"
        assert trigger.condition == "enters"

    def test_each_opponent(self) -> None:
        trigger = derive_trigger(
            _card("Whenever this creature attacks, each opponent loses 1 life.")
        )
        assert trigger.scope == "per_opponent"

    def test_static_ability_has_no_trigger(self) -> None:
        trigger = derive_trigger(_card("Flying, first strike"))
        assert trigger.scope == "static"
        assert trigger.condition is None

    def test_no_oracle_text_is_static(self) -> None:
        assert derive_trigger(_card(None)).scope == "static"

    def test_dies_trigger(self) -> None:
        trigger = derive_trigger(_card("When this creature dies, draw a card."))
        assert trigger.condition == "dies"


# ---------------------------------------------------------------------------
# Adversarial review findings (2026-07-28)
# ---------------------------------------------------------------------------


class TestMultipleAbilities:
    """A card with several abilities has several triggers.

    Judging the whole oracle at once let whichever pattern matched first win. A
    commander carrying both an upkeep trigger and a per-Ninja combat trigger came out
    as `per_turn`, burying the multiplying trigger this module exists to surface.
    """

    def test_upkeep_clause_does_not_bury_the_combat_trigger(self) -> None:
        trigger = derive_trigger(
            _card(
                "At the beginning of your upkeep, draw a card.\n"
                "Whenever a Ninja you control deals combat damage to a player, "
                "reveal the top card of your library."
            )
        )
        assert trigger.scope == "per_source"
        assert "per_turn" in trigger.other_scopes

    def test_other_scopes_is_empty_for_a_single_ability(self) -> None:
        trigger = derive_trigger(_card("Whenever this creature attacks, draw a card."))
        assert trigger.other_scopes == ()


class TestEntersOrAttacks:
    """ "enters or attacks" is one clause with two conditions."""

    def test_both_halves_are_reported(self) -> None:
        trigger = derive_trigger(
            _card(
                "Whenever Sun Titan enters or attacks, return target permanent card "
                "from your graveyard to the battlefield.",
                name="Sun Titan",
            )
        )
        assert trigger.condition == "enters_or_attacks"
        # It fires every combat, not only once on arrival.
        assert trigger.scope == "per_combat"

    def test_plain_entry_trigger_is_unchanged(self) -> None:
        trigger = derive_trigger(_card("When this creature enters, draw a card."))
        assert trigger.scope == "on_entry"
        assert trigger.condition == "enters"


class TestSelfReferenceByName:
    """Older templating names the card instead of saying "this creature"."""

    def test_literal_card_name_is_recognised_as_self(self) -> None:
        trigger = derive_trigger(
            _card(
                "Whenever Kalonian Hydra attacks, double the number of +1/+1 counters "
                "on each creature you control.",
                name="Kalonian Hydra",
            )
        )
        assert trigger.scope == "per_source"
        assert trigger.sources == "self"

    def test_modern_templating_still_works(self) -> None:
        trigger = derive_trigger(
            _card("Whenever this creature deals combat damage to a player, draw a card.")
        )
        assert trigger.sources == "self"

    def test_a_class_of_sources_is_not_turned_into_self(self) -> None:
        trigger = derive_trigger(
            _card(
                "Whenever a Ninja you control deals combat damage to a player, draw.",
                name="Yuriko, the Tiger's Shadow",
            )
        )
        assert trigger.sources == "class"
