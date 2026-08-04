"""Unreleased-card guard shared by every discovery tool carrying a format filter.

GOTCHA(2026-07-30): Scryfall marks a printing ``not_legal`` in EVERY format until its
set's release day, including fully spoiled cards from sets months away. Any legality
filter therefore silently drops upcoming cards while the response still reads as
exhaustive. ``scryfall_search_cards`` grew a probe for this (utils/query_sanitize.py);
this module is the same guard for the bulk-data path, where the excluded cards are in
memory and can simply be collected at the point the legality check rejects them.

The contract mirrors ``scryfall_search_cards`` exactly: ``unreleased_excluded`` is
``None`` when no legality filter was active (nothing was checked), and a list —
possibly empty — when one was. ``None`` and ``[]`` are NOT the same claim: only ``[]``
means "checked, found nothing".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mtg_mcp_server.types import Card

# Appended to the `format` parameter description of every discovery tool. This is the
# only part of the guard that protects a caller who never reads the response body.
FORMAT_FILTER_CAVEAT = (
    " WARNING: this is a legality filter — it also excludes cards from sets not yet "
    "released, which Scryfall marks not_legal in every format until release day. Omit "
    "it for discovery searches; cards it removed are listed in `unreleased_excluded`."
)

# One page of names is plenty for a warning; the point is naming cards, not paging them.
UNRELEASED_CAP = 20


def utc_today() -> str:
    """Today's date as an ISO string, in UTC.

    UTC, not local time: the server's timezone must not shift the release-date
    boundary by a day relative to the dates Scryfall publishes.
    """
    return datetime.now(UTC).date().isoformat()


def is_unreleased(card: Card, today: str) -> bool:
    """True when the card's set has not been released yet.

    ISO dates compare lexicographically, so plain string comparison is exact.
    Digital-only printings are excluded: they will never become paper-legal, so
    their ``not_legal`` status is not a release-date artefact and flagging them
    as "coming soon" would be a lie (same reason the Scryfall probe carries
    ``-is:digital``).
    """
    return not card.digital and card.released_at is not None and card.released_at > today


class UnreleasedCollector:
    """Collects cards a format filter rejected ONLY because their set is unreleased.

    Wire it at the exact point a card fails the legality check. Two shapes exist:

    * legality checked FIRST, other criteria after (the filter loops)::

          legal = card.legalities.get(fmt) == "legal"
          if not legal and not collector.offer(card):
              continue          # released and not legal: genuinely out
          ...other criteria, continue on fail...
          if not legal:
              collector.collect(card)   # would have matched without the filter
              continue
          results.append(card)

    * legality checked LAST (everything else already passed)::

          if card.legalities.get(fmt) != "legal":
              collector.consider(card)
              continue

    ``field()`` is the ``unreleased_excluded`` value for structured content and
    ``note()`` the matching markdown warning; both follow the contract stated in
    the module docstring.
    """

    def __init__(self, active: bool, *, today: str | None = None, cap: int = UNRELEASED_CAP):
        self.active = active
        self.today = today if today is not None else utc_today()
        self.cap = cap
        self.cards: list[Card] = []
        self._seen: set[str] = set()

    def offer(self, card: Card) -> bool:
        """True when this card is worth evaluating further despite failing legality."""
        return self.active and is_unreleased(card, self.today)

    def collect(self, card: Card) -> None:
        """Record a card that passed every criterion except the legality filter."""
        key = card.name.lower()
        if key in self._seen or len(self.cards) >= self.cap:
            return
        self._seen.add(key)
        self.cards.append(card)

    def consider(self, card: Card) -> None:
        """``offer`` + ``collect`` in one step, for call sites where legality is checked last."""
        if self.offer(card):
            self.collect(card)

    @property
    def full(self) -> bool:
        """True when the cap is reached — scanning further can collect nothing new."""
        return len(self.cards) >= self.cap

    @property
    def names(self) -> list[str]:
        return [card.name for card in self.cards]

    def field(self) -> list[str] | None:
        """The ``unreleased_excluded`` value: None when no legality filter was active."""
        return self.names if self.active else None

    def note(self, format_name: str) -> str | None:
        """Markdown warning naming the excluded cards, or None when there are none."""
        return unreleased_param_note(self.names, format_name)


def unreleased_param_note(
    names: list[str], format_name: str, total: int | None = None
) -> str | None:
    """Markdown warning naming the cards a ``format`` parameter excluded.

    The names matter more than the count: a generic "beware of legality filters"
    reads once and then goes invisible; a named card the caller expected does not.
    ``total`` is for probes that only read one page (Scryfall API): reporting
    ``len(names)`` as the total would announce a number never measured.
    """
    if not names:
        return None
    listed = ", ".join(names)
    count = total if total is not None else len(names)
    partial = f" (showing {len(names)} of them)" if count > len(names) else ""
    return (
        f"NOTE: the format '{format_name}' legality filter EXCLUDED "
        f"{count} unreleased card(s) matching this search{partial}: {listed}. "
        f"Scryfall marks cards not_legal in every format until their set's release "
        f"day. Omit the format filter to see them."
    )
