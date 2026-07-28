"""Tests for defensive normalisation of HTML-escaped search queries.

Regression origin (2026-07-27): a client sent ``mv&lt;=3`` instead of ``mv<=3``. Scryfall
answered 404 and the tool surfaced "No cards found", which reads exactly like a valid
query with no matches. The agent consuming it narrowed its search instead of fixing the
query, violating the exhaustive-search protocol it was following.
"""

from __future__ import annotations

from mtg_mcp_server.utils.query_sanitize import (
    escaping_warning,
    looks_like_scryfall_syntax,
    normalize_query,
)

# ---------------------------------------------------------------------------
# The reported bug
# ---------------------------------------------------------------------------


class TestReportedBug:
    """The exact query that failed in production."""

    def test_escaped_lte_is_decoded(self) -> None:
        normalized, was_escaped = normalize_query("t:ninja id:ub mv&lt;=3")
        assert normalized == "t:ninja id:ub mv<=3"
        assert was_escaped is True

    def test_escaped_gte_is_decoded(self) -> None:
        normalized, was_escaped = normalize_query("pow&gt;=4")
        assert normalized == "pow>=4"
        assert was_escaped is True

    def test_numeric_entities_are_decoded(self) -> None:
        assert normalize_query("mv&#60;3")[0] == "mv<3"
        assert normalize_query("mv&#62;3")[0] == "mv>3"

    def test_hex_entities_are_decoded(self) -> None:
        assert normalize_query("mv&#x3c;3")[0] == "mv<3"
        assert normalize_query("mv&#x3e;3")[0] == "mv>3"

    def test_le_and_ge_shorthand(self) -> None:
        assert normalize_query("mv&le;3")[0] == "mv<=3"
        assert normalize_query("mv&ge;3")[0] == "mv>=3"


# ---------------------------------------------------------------------------
# Queries that must pass through untouched
# ---------------------------------------------------------------------------


class TestPassThrough:
    """A correct query must never be rewritten, and must not raise the warning flag."""

    def test_raw_operators_untouched(self) -> None:
        normalized, was_escaped = normalize_query("t:ninja id:ub mv<=3")
        assert normalized == "t:ninja id:ub mv<=3"
        assert was_escaped is False

    def test_plain_query_untouched(self) -> None:
        normalized, was_escaped = normalize_query("f:commander is:gamechanger")
        assert normalized == "f:commander is:gamechanger"
        assert was_escaped is False

    def test_empty_query(self) -> None:
        assert normalize_query("") == ("", False)

    def test_bare_ampersand_in_card_name_survives(self) -> None:
        # A lone "&" is not an entity — a card name containing it must not be mangled.
        normalized, was_escaped = normalize_query('!"Tom Bombadil & friends"')
        assert normalized == '!"Tom Bombadil & friends"'
        assert was_escaped is False

    def test_unrelated_entity_is_left_alone(self) -> None:
        # Deliberately NOT html.unescape: only operator-ish entities are decoded, so a
        # name that happens to contain "&copy;" is not silently rewritten.
        normalized, _ = normalize_query("o:&copy;")
        assert normalized == "o:&copy;"


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestDoubleEscaping:
    """``&amp;`` is decoded last so double-escaped input degrades one level at a time."""

    def test_double_escaped_lt_resolves_one_level(self) -> None:
        assert normalize_query("mv&amp;lt;3")[0] == "mv&lt;3"

    def test_escaped_ampersand_alone(self) -> None:
        assert normalize_query("a &amp; b")[0] == "a & b"


# ---------------------------------------------------------------------------
# The warning message
# ---------------------------------------------------------------------------


class TestEscapingWarning:
    """The message must show BOTH forms — that is what makes the bug diagnosable."""

    def test_warning_shows_both_forms(self) -> None:
        message = escaping_warning("mv&lt;=3", "mv<=3")
        assert "mv&lt;=3" in message
        assert "mv<=3" in message


# ---------------------------------------------------------------------------
# Wrong-tool detection for the substring searches
# ---------------------------------------------------------------------------


class TestScryfallSyntaxDetection:
    """Bulk search matches substrings; a filter expression there finds nothing."""

    def test_filter_expression_is_detected(self) -> None:
        assert looks_like_scryfall_syntax("t:creature id<=ub mv<=2") is True

    def test_single_filter_is_detected(self) -> None:
        assert looks_like_scryfall_syntax("t:ninja") is True

    def test_comparison_alone_is_detected(self) -> None:
        assert looks_like_scryfall_syntax("mv<=2") is True

    def test_plain_card_name_is_not_flagged(self) -> None:
        assert looks_like_scryfall_syntax("Yuriko, the Tiger's Shadow") is False

    def test_plain_type_word_is_not_flagged(self) -> None:
        assert looks_like_scryfall_syntax("Legendary Creature") is False

    def test_oracle_phrase_is_not_flagged(self) -> None:
        assert looks_like_scryfall_syntax("deals combat damage to a player") is False
