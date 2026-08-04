"""Tests for the shared unreleased-card guard (utils/unreleased.py).

Origin (2026-08-04): Scryfall marks a card not_legal in EVERY format until its
set's release day, so every tool with a format filter silently dropped upcoming
cards while its response still read as exhaustive. Darksteel Angel (set 'frc',
releases 2026-10-02) is the real card that exposed it.
"""

from __future__ import annotations

from mtg_mcp_server.types import Card
from mtg_mcp_server.utils.unreleased import (
    UnreleasedCollector,
    is_unreleased,
    unreleased_param_note,
    utc_today,
)

TODAY = "2026-08-04"


def _card(
    name: str = "Darksteel Angel",
    released_at: str | None = "2026-10-02",
    digital: bool = False,
) -> Card:
    return Card(id="t", name=name, released_at=released_at, digital=digital)


class TestIsUnreleased:
    def test_future_release_is_unreleased(self):
        assert is_unreleased(_card(released_at="2026-10-02"), TODAY) is True

    def test_past_release_is_released(self):
        assert is_unreleased(_card(released_at="2020-01-01"), TODAY) is False

    def test_release_day_counts_as_released(self):
        """On release day the card becomes legal; only strictly-future dates hide."""
        assert is_unreleased(_card(released_at=TODAY), TODAY) is False

    def test_missing_date_is_not_unreleased(self):
        """No claim without data: absence of released_at must not flag anything."""
        assert is_unreleased(_card(released_at=None), TODAY) is False

    def test_digital_printing_never_flagged(self):
        """A digital-only card will never be paper-legal: its not_legal status is
        not a release-date artefact, so calling it 'coming soon' would be a lie."""
        assert is_unreleased(_card(released_at="2026-10-02", digital=True), TODAY) is False

    def test_utc_today_is_iso(self):
        today = utc_today()
        assert len(today) == 10
        assert today[4] == "-"
        assert today[7] == "-"


class TestUnreleasedCollector:
    def test_inactive_collector_offers_nothing_and_fields_none(self):
        """No format filter means nothing was checked: field() must be None, not []."""
        collector = UnreleasedCollector(active=False, today=TODAY)
        assert collector.offer(_card()) is False
        collector.consider(_card())
        assert collector.names == []
        assert collector.field() is None

    def test_active_collector_collects_and_fields_list(self):
        collector = UnreleasedCollector(active=True, today=TODAY)
        assert collector.offer(_card()) is True
        collector.collect(_card())
        assert collector.field() == ["Darksteel Angel"]

    def test_active_but_empty_fields_empty_list(self):
        """Checked and found nothing is [] — a different claim from None."""
        collector = UnreleasedCollector(active=True, today=TODAY)
        assert collector.field() == []

    def test_released_card_not_considered(self):
        collector = UnreleasedCollector(active=True, today=TODAY)
        collector.consider(_card(released_at="2020-01-01"))
        assert collector.names == []

    def test_deduplicates_by_name(self):
        collector = UnreleasedCollector(active=True, today=TODAY)
        collector.collect(_card())
        collector.collect(_card())
        assert collector.names == ["Darksteel Angel"]

    def test_cap_bounds_collection(self):
        collector = UnreleasedCollector(active=True, today=TODAY, cap=2)
        for i in range(5):
            collector.collect(_card(name=f"Upcoming {i}"))
        assert len(collector.names) == 2
        assert collector.full is True

    def test_note_names_the_cards(self):
        collector = UnreleasedCollector(active=True, today=TODAY)
        collector.collect(_card())
        note = collector.note("commander")
        assert note is not None
        assert "Darksteel Angel" in note
        assert "commander" in note
        assert "not_legal" in note

    def test_note_is_none_when_nothing_collected(self):
        collector = UnreleasedCollector(active=True, today=TODAY)
        assert collector.note("commander") is None


class TestUnreleasedParamNote:
    def test_total_beyond_names_is_stated_separately(self):
        """Scryfall's total can exceed the one page of names the probe read;
        announcing len(names) as the total would be a number never measured."""
        note = unreleased_param_note(["Darksteel Angel"], "commander", total=57)
        assert note is not None
        assert "57" in note
        assert "showing 1 of them" in note

    def test_empty_names_returns_none(self):
        assert unreleased_param_note([], "commander") is None
