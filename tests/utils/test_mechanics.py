"""Tests for mechanic and mana-cost reasoning shared by the mechanic tools.

Regression origin (2026-07-27, Yuriko audit). Three distinct errors were shipped:

1. "Silver-Fur Master + Skullsnatcher -> ninjutsu at {0}". False: a generic reduction
   never touches coloured pips (rule 601.2f).
2. The correction then claimed the reducer affected 4 of the deck's ninjutsu cards.
   The real answer was 9 of 15 — wrong in the opposite direction.
3. Two changelings were not counted as Ninjas, though rule 702.73a makes them every
   creature type everywhere.

Each of these is a mechanical rule that should never have been left to judgement.
"""

from __future__ import annotations

from mtg_mcp_server.types import Card
from mtg_mcp_server.utils.mechanics import (
    apply_generic_reduction,
    creature_types,
    has_creature_type,
    keyword_activation_cost,
    parse_mana_cost,
    reduction_amount,
)


def _card(**kwargs) -> Card:
    """Build a Card with only the fields a mechanic check reads."""
    base = {"id": "x", "name": "Test Card", "type_line": "Creature — Ninja"}
    return Card.model_validate(base | kwargs)


# ---------------------------------------------------------------------------
# Mana cost parsing
# ---------------------------------------------------------------------------


class TestParseManaCost:
    def test_generic_and_coloured_are_separated(self) -> None:
        parsed = parse_mana_cost("{2}{U}{U}")
        assert parsed.generic == 2
        assert parsed.symbols == ["U", "U"]

    def test_pure_coloured_cost_has_no_generic_part(self) -> None:
        parsed = parse_mana_cost("{U}{B}")
        assert parsed.generic == 0
        assert parsed.symbols == ["U", "B"]

    def test_hybrid_symbol_is_not_generic(self) -> None:
        # {2/U} can be paid with 2 generic, but it is a single symbol, not a
        # generic component a cost reducer can shave.
        parsed = parse_mana_cost("{2/U}{U}")
        assert parsed.generic == 0
        assert parsed.symbols == ["2/U", "U"]

    def test_colorless_is_not_generic(self) -> None:
        # {C} demands colorless mana specifically; a generic reduction cannot touch it.
        parsed = parse_mana_cost("{2}{C}")
        assert parsed.generic == 2
        assert parsed.symbols == ["C"]

    def test_x_is_not_a_reducible_generic(self) -> None:
        parsed = parse_mana_cost("{X}{R}")
        assert parsed.generic == 0
        assert parsed.symbols == ["X", "R"]

    def test_empty_cost(self) -> None:
        parsed = parse_mana_cost("")
        assert parsed.generic == 0
        assert parsed.symbols == []
        assert parsed.render() == "{0}"


# ---------------------------------------------------------------------------
# Generic cost reduction — rule 601.2f
# ---------------------------------------------------------------------------


class TestGenericReduction:
    """The exact rule that produced two opposite errors in one audit."""

    def test_coloured_only_cost_is_untouched(self) -> None:
        result = apply_generic_reduction("{U}{B}", 1)
        assert result.reduced is False
        assert result.result == "{U}{B}"
        assert "generic" in result.reason.lower()

    def test_generic_part_is_shaved(self) -> None:
        result = apply_generic_reduction("{2}{U}{U}", 1)
        assert result.reduced is True
        assert result.result == "{1}{U}{U}"

    def test_reduction_stops_at_zero_generic(self) -> None:
        # {1}{U} minus {2} is {U}, never a negative or a shaved pip.
        result = apply_generic_reduction("{1}{U}", 2)
        assert result.reduced is True
        assert result.result == "{U}"

    def test_fully_generic_cost_can_reach_zero(self) -> None:
        result = apply_generic_reduction("{2}", 2)
        assert result.reduced is True
        assert result.result == "{0}"

    def test_zero_reduction_changes_nothing(self) -> None:
        assert apply_generic_reduction("{2}{U}", 0).reduced is False


# ---------------------------------------------------------------------------
# Keyword activation costs
# ---------------------------------------------------------------------------


class TestKeywordActivationCost:
    def test_ninjutsu_cost_is_extracted(self) -> None:
        card = _card(oracle_text="Ninjutsu {1}{U} (Return an unblocked attacker...)")
        assert keyword_activation_cost(card, "Ninjutsu") == "{1}{U}"

    def test_multi_symbol_cost(self) -> None:
        card = _card(oracle_text="Ninjutsu {2}{U}{B}\nWhenever this creature...")
        assert keyword_activation_cost(card, "Ninjutsu") == "{2}{U}{B}"

    def test_commander_ninjutsu_variant_is_matched(self) -> None:
        card = _card(oracle_text="Commander ninjutsu {1}{U}{B} (Return an unblocked...)")
        assert keyword_activation_cost(card, "Ninjutsu") == "{1}{U}{B}"

    def test_missing_keyword_returns_none(self) -> None:
        assert keyword_activation_cost(_card(oracle_text="Flying"), "Ninjutsu") is None

    def test_no_oracle_text(self) -> None:
        assert keyword_activation_cost(_card(oracle_text=None), "Ninjutsu") is None


# ---------------------------------------------------------------------------
# Creature types, including changelings — rule 702.73a
# ---------------------------------------------------------------------------


class TestCreatureTypes:
    def test_types_come_from_the_subtype_half(self) -> None:
        card = _card(type_line="Legendary Creature — Human Ninja")
        assert creature_types(card) == {"Human", "Ninja"}

    def test_no_subtype_half(self) -> None:
        assert creature_types(_card(type_line="Artifact")) == set()

    def test_changeling_keyword_matches_every_type(self) -> None:
        # Changeling Outcast: rule 702.73a, "this object is every creature type".
        card = _card(
            name="Changeling Outcast",
            type_line="Creature — Shapeshifter",
            keywords=["Changeling"],
        )
        match = has_creature_type(card, "Ninja")
        assert match.matches is True
        assert match.via == "changeling"

    def test_changeling_from_oracle_text_without_the_keyword(self) -> None:
        card = _card(
            type_line="Creature — Shapeshifter",
            oracle_text="Changeling (This card is every creature type.)",
        )
        assert has_creature_type(card, "Ninja").via == "changeling"

    def test_printed_type_wins_over_changeling_label(self) -> None:
        card = _card(type_line="Creature — Ninja")
        match = has_creature_type(card, "Ninja")
        assert match.matches is True
        assert match.via == "printed"

    def test_planeswalker_that_becomes_a_typed_creature_is_conditional(self) -> None:
        # Kaito, Bane of Nightmares: a Ninja creature during your turn, while it has
        # loyalty counters. It counts, but the condition must travel with the count.
        card = _card(
            name="Kaito, Bane of Nightmares",
            type_line="Legendary Planeswalker — Kaito",
            oracle_text=(
                "During your turn, as long as Kaito has one or more loyalty counters "
                "on him, he's a 3/4 Ninja creature."
            ),
        )
        match = has_creature_type(card, "Ninja")
        assert match.matches is True
        assert match.via == "conditional"
        assert match.condition is not None

    def test_unrelated_card_does_not_match(self) -> None:
        card = _card(type_line="Creature — Human Wizard", oracle_text="Flying")
        match = has_creature_type(card, "Ninja")
        assert match.matches is False
        assert match.via is None

    def test_type_match_is_case_insensitive(self) -> None:
        assert has_creature_type(_card(type_line="Creature — Ninja"), "ninja").matches is True


# ---------------------------------------------------------------------------
# Adversarial review findings (2026-07-28)
# ---------------------------------------------------------------------------


class TestVariableReduction:
    """A variable reduction has no single number, so none is quoted.

    Metalwork Colossus reads "costs {1} less to cast for each artifact you control".
    Reporting a flat -{1} is the same failure this module exists to prevent: a figure
    that is precise, plausible, and wrong.
    """

    def test_for_each_reduction_returns_none(self) -> None:
        card = _card(
            name="Metalwork Colossus",
            oracle_text="This spell costs {1} less to cast for each artifact you control.",
        )
        assert reduction_amount(card) is None

    def test_fixed_reduction_still_returns_its_amount(self) -> None:
        card = _card(oracle_text="Ninja spells you cast cost {1} less to cast.")
        assert reduction_amount(card) == 1

    def test_card_without_reduction_clause(self) -> None:
        assert reduction_amount(_card(oracle_text="Flying")) is None


class TestTypeRequiresACreature:
    """A creature subtype belongs to a creature.

    "Kindred Sorcery — Otter" is a sorcery sharing a tribe's name. Counting it as a
    tribe member inflates every tribal total with cards that never arrive as creatures.
    """

    def test_kindred_spell_is_not_a_tribe_member(self) -> None:
        card = _card(name="Fell", type_line="Kindred Sorcery — Otter")
        assert has_creature_type(card, "Otter").matches is False

    def test_real_creature_still_matches(self) -> None:
        assert has_creature_type(_card(type_line="Creature — Otter"), "Otter").matches is True

    def test_changeling_must_also_be_a_creature(self) -> None:
        spell = _card(
            type_line="Kindred Instant — Shapeshifter",
            keywords=["Changeling"],
            oracle_text="Changeling (This card is every creature type.)",
        )
        assert has_creature_type(spell, "Ninja").matches is False

    def test_planeswalker_conditional_grant_still_counts(self) -> None:
        # The one route open to a non-creature: it BECOMES a creature.
        card = _card(
            type_line="Legendary Planeswalker — Kaito",
            oracle_text="During your turn, Kaito is a 3/4 Ninja creature.",
        )
        assert has_creature_type(card, "Ninja").via == "conditional"


class TestKeywordPatternCacheIsBounded:
    """The cache key comes from a freely-chosen tool argument.

    An unbounded module-level dict would let any caller of cost_reduction_check grow
    server memory without limit by passing a fresh keyword each call.
    """

    def test_cache_has_a_max_size(self) -> None:
        from mtg_mcp_server.utils.mechanics import _keyword_cost_pattern

        assert _keyword_cost_pattern.cache_info().maxsize is not None

    def test_arbitrary_keywords_do_not_grow_it_without_bound(self) -> None:
        from mtg_mcp_server.utils.mechanics import _keyword_cost_pattern

        card = _card(oracle_text="Ninjutsu {1}{U}")
        for i in range(1000):
            keyword_activation_cost(card, f"made-up-keyword-{i}")
        assert _keyword_cost_pattern.cache_info().currsize <= 256

    def test_basic_prefix_variant_is_matched(self) -> None:
        card = _card(oracle_text="Basic landcycling {1}", type_line="Land")
        assert keyword_activation_cost(card, "Landcycling") == "{1}"
