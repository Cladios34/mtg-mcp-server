"""Tests for defensive normalisation of HTML-escaped search queries.

Regression origin (2026-07-27): a client sent ``mv&lt;=3`` instead of ``mv<=3``. Scryfall
answered 404 and the tool surfaced "No cards found", which reads exactly like a valid
query with no matches. The agent consuming it narrowed its search instead of fixing the
query, violating the exhaustive-search protocol it was following.
"""

from __future__ import annotations

from mtg_mcp_server.utils.query_sanitize import (
    escaping_warning,
    has_legality_filter,
    looks_like_scryfall_syntax,
    normalize_query,
    strip_legality_filter,
    unreleased_probe_query,
    unreleased_warning,
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


# ---------------------------------------------------------------------------
# Legality filters hide unreleased cards
# ---------------------------------------------------------------------------


class TestLegalityFilterDetection:
    """Regression origin (2026-07-30): `f:commander` silently dropped Darksteel Angel.

    The card is real, spoiled, and an Angel, but its set (Reality Fracture Commander)
    releases 2026-10-02, so Scryfall marks it not_legal everywhere and the legality
    filter removed it with nothing in the response saying so.
    """

    def test_f_is_detected(self) -> None:
        assert has_legality_filter("id<=rwb t:angel f:commander") is True

    def test_long_form_format_is_detected(self) -> None:
        assert has_legality_filter("t:angel format:commander") is True

    def test_legal_banned_restricted_are_detected(self) -> None:
        assert has_legality_filter("legal:modern") is True
        assert has_legality_filter("banned:commander") is True
        assert has_legality_filter("restricted:vintage") is True

    def test_negated_filter_is_NOT_detected(self) -> None:
        # Revised 2026-07-30 (adversarial review): a negated legality term hides nothing.
        # `-f:commander` asks for "not legal in commander", which unreleased printings
        # satisfy by definition (their status IS not_legal). Warning on it was a false
        # positive on a query that excludes no upcoming card.
        assert has_legality_filter("t:angel -f:commander") is False

    def test_query_without_legality_is_not_flagged(self) -> None:
        assert has_legality_filter("id<=rwb t:angel year>=2025 -is:reprint") is False

    def test_oracle_text_mentioning_format_is_not_flagged(self) -> None:
        assert has_legality_filter('o:"legendary creature"') is False


class TestStripLegalityFilter:
    def test_filter_is_removed_and_whitespace_normalised(self) -> None:
        assert (
            strip_legality_filter("id<=rwb t:angel f:commander year>=2025")
            == "id<=rwb t:angel year>=2025"
        )

    def test_negated_filter_is_left_alone(self) -> None:
        # Not detected (see above), so not stripped: the probe keeps the user's own term.
        assert strip_legality_filter("t:angel -f:commander") == "t:angel -f:commander"

    def test_legality_only_query_becomes_empty(self) -> None:
        assert strip_legality_filter("f:commander") == ""


class TestUnreleasedProbeQuery:
    def test_probe_swaps_legality_for_a_future_date(self) -> None:
        probe = unreleased_probe_query("id<=rwb t:angel f:commander", "2026-07-30")
        assert probe == "id<=rwb t:angel date>2026-07-30 -is:funny -is:digital"

    def test_probe_excludes_never_legal_printings(self) -> None:
        # Joke sets and Arena-only cards are absent for reasons that have nothing to do
        # with a release date; flagging them as "coming soon" would be a false promise.
        probe = unreleased_probe_query("t:dragon f:commander", "2026-07-30")
        assert probe is not None
        assert "-is:funny" in probe
        assert "-is:digital" in probe

    def test_no_probe_when_legality_was_the_whole_query(self) -> None:
        assert unreleased_probe_query("f:commander", "2026-07-30") is None


class TestUnreleasedWarning:
    def test_warning_names_the_hidden_cards(self) -> None:
        message = unreleased_warning(["Darksteel Angel"], "t:angel date>2026-07-30")
        assert "Darksteel Angel" in message
        assert "not_legal" in message

    def test_warning_reports_the_probe_so_the_claim_is_checkable(self) -> None:
        message = unreleased_warning(["Smaug, Wicked Worm"], "t:dragon date>2026-07-30")
        assert "t:dragon date>2026-07-30" in message

    def test_long_lists_are_truncated_but_counted(self) -> None:
        names = [f"Card {i}" for i in range(15)]
        message = unreleased_warning(names, "probe", shown=10)
        assert "Card 0" in message
        assert "+5 more" in message
        assert "15 such card(s)" in message


# ---------------------------------------------------------------------------
# Adversarial review 2026-07-30: the probe could lie in four distinct ways
# ---------------------------------------------------------------------------


class TestProbeCannotSilentlyLie:
    """Each case below made the probe report "nothing hidden" while cards WERE hidden.

    That false negative is worse than no probe at all: it is the exact failure the whole
    feature exists to prevent, wearing the costume of a successful check.
    """

    def test_existing_date_term_is_removed_before_the_probe_adds_its_own(self) -> None:
        # `date<2020 date>2026-07-30` is satisfiable by nothing, so the probe would return
        # zero and the caller would read "nothing is hidden".
        probe = unreleased_probe_query("t:angel f:commander date<2020", "2026-07-30")
        assert probe is not None
        assert "date<2020" not in probe
        assert "date>2026-07-30" in probe

    def test_year_term_is_removed_too(self) -> None:
        probe = unreleased_probe_query("t:angel f:commander year<=2015", "2026-07-30")
        assert probe is not None
        assert "year" not in probe.lower()

    def test_unbalanced_quote_yields_no_probe_rather_than_a_broken_one(self) -> None:
        # With an open quote, the appended filters become part of the string literal and
        # stop filtering. No probe is the honest answer.
        assert unreleased_probe_query('o:"draw a card f:commander', "2026-07-30") is None

    def test_unbalanced_parenthesis_yields_no_probe(self) -> None:
        assert unreleased_probe_query("(t:angel or t:demon f:commander", "2026-07-30") is None

    def test_balanced_quotes_still_probe(self) -> None:
        probe = unreleased_probe_query('o:"draw a card" f:commander', "2026-07-30")
        assert probe is not None
        assert 'o:"draw a card"' in probe


class TestNegatedLegalityIsNotAFilterToWarnAbout:
    """`-banned:commander` excludes nothing unreleased: warning on it is a false positive.

    Unreleased printings carry status `not_legal`, never `banned`, so "not banned" is
    already true of them.
    """

    def test_negated_forms_are_not_flagged(self) -> None:
        assert has_legality_filter("t:angel -banned:commander") is False
        assert has_legality_filter("t:angel -f:commander") is False

    def test_positive_forms_are_still_flagged(self) -> None:
        assert has_legality_filter("t:angel f:commander") is True
        assert has_legality_filter("t:angel banned:commander") is True


class TestWarningReportsTheCountItMeasured:
    """The probe reads one page; Scryfall knows the real total. Never conflate the two."""

    def test_total_beats_page_size_when_they_differ(self) -> None:
        message = unreleased_warning(["A", "B"], "probe", total=57)
        assert "57 such card(s)" in message
        assert "showing 2 of them" in message

    def test_no_partial_note_when_the_page_holds_everything(self) -> None:
        message = unreleased_warning(["A", "B"], "probe", total=2)
        assert "2 such card(s)" in message
        assert "showing" not in message

    def test_total_defaults_to_the_names_we_hold(self) -> None:
        message = unreleased_warning(["A"], "probe")
        assert "1 such card(s)" in message
