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

import contextlib
import time

import pytest

from mtg_mcp_server.types import Card
from mtg_mcp_server.utils.mechanics import (
    MAX_KEYWORD_LENGTH,
    AlternativeCost,
    alternative_cast_cost,
    apply_generic_reduction,
    carries_keyword,
    creature_types,
    has_creature_type,
    has_reduction_clause,
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


class TestAmbiguousReduction:
    """Two different amounts in one oracle produce no quotable number."""

    def test_two_distinct_amounts_refuse_to_quote(self) -> None:
        """Adversarial review (2026-07-28): the scan returned the LARGEST {N} less
        found anywhere in the oracle, so a card reducing casts by {2} and activations
        by {1} reported 2 for both."""
        card = _card(
            name="Split Reducer",
            oracle_text=(
                "Ninja spells you cast cost {2} less to cast.\n"
                "Ninjutsu abilities you activate cost {1} less to activate."
            ),
        )
        assert reduction_amount(card) is None
        assert has_reduction_clause(card) is True

    def test_same_amount_twice_still_quotes_it(self) -> None:
        card = _card(
            name="Consistent Reducer",
            oracle_text=(
                "Ninja spells you cast cost {1} less to cast.\n"
                "Ninjutsu abilities you activate cost {1} less to activate."
            ),
        )
        assert reduction_amount(card) == 1

    def test_no_clause_is_not_an_unquotable_clause(self) -> None:
        card = _card(name="Plain Bear", oracle_text="Trample")
        assert reduction_amount(card) is None
        assert has_reduction_clause(card) is False


# ---------------------------------------------------------------------------
# Alternative casting costs
# ---------------------------------------------------------------------------


class TestAlternativeCastCost:
    """A card's mana value is not always what it costs to get it onto the board.

    Origin (2026-07-29): the opening-hand simulator judged castability on mana
    value alone, so a 9/9 with ``Warp {3}`` read as a 9-drop and the hands
    holding it were mulliganed as having no castable spell. 1280 hands in 10000
    were thrown away for holding a turn-3 play.
    """

    def test_warp_cost_beats_mana_value(self) -> None:
        # Bygone Colossus: {9} on the card, playable on turn 3 for {3}.
        card = _card(
            name="Bygone Colossus",
            type_line="Artifact Creature — Robot",
            mana_cost="{9}",
            cmc=9.0,
            keywords=["Warp"],
            oracle_text=(
                "Warp {3} (You may cast this card from your hand for its warp cost. "
                "Exile this creature at the beginning of the next end step, then you "
                "may cast it from exile on a later turn.)"
            ),
        )
        assert alternative_cast_cost(card) == AlternativeCost(value=3, keyword="Warp", cost="{3}")

    def test_evoke_cost_beats_mana_value(self) -> None:
        card = _card(
            name="Mulldrifter",
            type_line="Creature — Elemental",
            mana_cost="{4}{U}",
            cmc=5.0,
            keywords=["Flying", "Evoke"],
            oracle_text=(
                "Flying\nWhen this creature enters, draw two cards.\n"
                "Evoke {2}{U} (You may cast this spell for its evoke cost. If you do, "
                "it's sacrificed when it enters.)"
            ),
        )
        assert alternative_cast_cost(card) == AlternativeCost(
            value=3, keyword="Evoke", cost="{2}{U}"
        )

    def test_impending_cost_survives_its_counter_and_dash(self) -> None:
        # Impending prints as "Impending 4-{2}{W}{W}": a counter and a dash sit
        # between the keyword and the cost, which a plain keyword-then-cost
        # pattern walks straight past. Overlord of the Mistmoors is a {5}{W}{W}
        # card castable for 4.
        card = _card(
            name="Overlord of the Mistmoors",
            type_line="Enchantment Creature — Horror",
            mana_cost="{5}{W}{W}",
            cmc=7.0,
            keywords=["Impending"],
            oracle_text=(
                "Impending 4—{2}{W}{W} (If you cast this spell for its impending cost, it "
                "enters with four time counters and isn't a creature until the last is "
                "removed.)"
            ),
        )
        assert alternative_cast_cost(card) == AlternativeCost(
            value=4, keyword="Impending", cost="{2}{W}{W}"
        )

    def test_ability_named_after_the_keyword_is_not_a_cost(self) -> None:
        # Mutalith Vortex Beast's keyword is "Warp Vortex" -- the name of a
        # triggered ability, not the Warp keyword. Reading it as a discount
        # would make a 6-drop look like a turn-1 play.
        card = _card(
            name="Mutalith Vortex Beast",
            type_line="Creature — Beast",
            mana_cost="{4}{U}{R}",
            cmc=6.0,
            keywords=["Warp Vortex", "Trample"],
            oracle_text=(
                "Trample\nWarp Vortex — When this creature enters, flip a coin for each "
                "opponent you have. For each flip you win, draw a card. For each flip you "
                "lose, this creature deals 3 damage to that player."
            ),
        )
        assert alternative_cast_cost(card) is None

    def test_non_mana_alternative_cost_is_not_a_discount(self) -> None:
        # Grief's evoke cost is "exile a black card", not mana. Whether it is
        # payable depends on the rest of the hand, which this function cannot
        # see -- so it declines rather than guesses.
        card = _card(
            name="Grief",
            type_line="Creature — Elemental Incarnation",
            mana_cost="{2}{B}{B}",
            cmc=4.0,
            keywords=["Evoke", "Menace"],
            oracle_text=(
                "Menace\nWhen this creature enters, target opponent reveals their hand. "
                "You choose a nonland card from it. That player discards that card.\n"
                "Evoke—Exile a black card from your hand."
            ),
        )
        assert alternative_cast_cost(card) is None

    def test_alternative_costing_more_is_ignored(self) -> None:
        card = _card(
            name="Expensive Option",
            mana_cost="{1}{U}",
            cmc=2.0,
            keywords=["Warp"],
            oracle_text="Warp {6}{U}{U}",
        )
        assert alternative_cast_cost(card) is None

    def test_plain_card_has_no_alternative(self) -> None:
        card = _card(name="Plain Bear", mana_cost="{1}{G}", cmc=2.0, oracle_text="Trample")
        assert alternative_cast_cost(card) is None

    def test_keyword_absent_from_scryfall_keywords_is_declined(self) -> None:
        # The oracle text alone is not enough: reminder text and ability names
        # both mention keywords they do not grant.
        card = _card(
            name="Talks About Warp",
            mana_cost="{5}",
            cmc=5.0,
            keywords=[],
            oracle_text="Whenever an opponent casts a spell with Warp {2}, draw a card.",
        )
        assert alternative_cast_cost(card) is None

    def test_x_in_alternative_cost_counts_as_zero(self) -> None:
        card = _card(
            name="X Warper",
            mana_cost="{7}",
            cmc=7.0,
            keywords=["Warp"],
            oracle_text="Warp {X}{R}",
        )
        assert alternative_cast_cost(card) == AlternativeCost(
            value=1, keyword="Warp", cost="{X}{R}"
        )


class TestKeywordLengthBound:
    """A keyword arrives from a tool argument a client controls freely.

    Origin (2026-07-29): the pattern cache was bounded in entry COUNT but not in
    entry SIZE. Compiling the regex is linear in the keyword's length and runs on
    the server's single event loop, so one call was enough to freeze it: measured
    819 ms at 100 KB, 3.3 s at 400 KB, minutes at a few MB. The 256-entry cache
    then held every one of them. The longest real Magic keyword is 23 characters
    ("More Than Meets the Eye").
    """

    def test_absurd_keyword_is_refused_before_it_costs_anything(self) -> None:
        card = _card(name="Ninja", oracle_text="Ninjutsu {1}{U}")
        with pytest.raises(ValueError, match="keyword too long"):
            keyword_activation_cost(card, "A" * 100_000)

    def test_refusal_is_fast(self) -> None:
        """The point of the bound is that it costs nothing to enforce."""
        card = _card(name="Ninja", oracle_text="Ninjutsu {1}{U}")
        start = time.perf_counter()
        for _ in range(200):
            with contextlib.suppress(ValueError):
                keyword_activation_cost(card, "A" * 500_000)
        # 200 unbounded calls would have taken minutes.
        assert time.perf_counter() - start < 1.0

    def test_longest_real_magic_keyword_still_works(self) -> None:
        # "More Than Meets the Eye" is the longest keyword Scryfall lists, at 23
        # characters. The bound must not be tight enough to reach it.
        card = _card(
            name="Transformer",
            oracle_text="More Than Meets the Eye {1}{W} (You may cast this card converted.)",
        )
        assert keyword_activation_cost(card, "More Than Meets the Eye") == "{1}{W}"

    def test_carries_keyword_refuses_too(self) -> None:
        """declared.py reaches the same cache through a client-supplied filter."""
        card = _card(name="Ninja", oracle_text="Ninjutsu {1}{U}", keywords=["Ninjutsu"])
        with pytest.raises(ValueError, match="keyword too long"):
            carries_keyword(card, "A" * 100_000)

    def test_a_keyword_at_the_bound_is_accepted(self) -> None:
        card = _card(name="Edge", oracle_text=f"{'K' * MAX_KEYWORD_LENGTH} {{2}}")
        assert keyword_activation_cost(card, "K" * MAX_KEYWORD_LENGTH) == "{2}"

    def test_one_character_past_the_bound_is_refused(self) -> None:
        """The boundary itself, so a future off-by-one cannot pass unnoticed."""
        card = _card(name="Edge", oracle_text="whatever {2}")
        with pytest.raises(ValueError, match="keyword too long"):
            keyword_activation_cost(card, "K" * (MAX_KEYWORD_LENGTH + 1))

    def test_an_accented_keyword_within_the_bound_works(self) -> None:
        """The bound counts characters, not bytes: a non-ASCII keyword of legal
        length must not be refused for being wider once encoded."""
        card = _card(name="Accented", oracle_text="Rôdeur nocturne {1}{B}")
        assert keyword_activation_cost(card, "Rôdeur nocturne") == "{1}{B}"
