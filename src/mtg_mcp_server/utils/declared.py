"""Turn a number the deck's owner stated into a measured constraint.

Regression origin (2026-07-27). The owner said "I have 13 cheap creatures". The list
did have exactly 13. Work then took it down to 11 and nothing said so — while the
probability of opening one, the deck's single most determining statistic, fell from
63.91% to 57.36%. The figure was recomputed at every revision. It was connected to
nothing.

A stated count is a constraint. Measuring it against the actual list is arithmetic,
so it should not depend on anyone remembering to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mtg_mcp_server.utils.mechanics import carries_keyword

if TYPE_CHECKING:
    from mtg_mcp_server.types import Card

__all__ = ["CategoryFilter", "CategoryResult", "evaluate_categories", "parse_filter"]

# Deliberately a small subset of Scryfall syntax: the shapes a deck owner actually
# uses to describe a category out loud. Anything richer belongs in a real search.
_TERM_RE = re.compile(
    r"""
    (?P<key>mv|cmc|t|type|o|oracle|name|kw|keyword)
    \s*(?P<op><=|>=|=|:|<|>)\s*
    (?P<value>"[^"]*"|\S+)
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class CategoryFilter:
    """A parsed category filter. Empty fields mean "no constraint on this axis"."""

    mv_lte: float | None = None
    mv_gte: float | None = None
    mv_eq: float | None = None
    types: tuple[str, ...] = ()
    text: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    unparsed: tuple[str, ...] = ()

    def matches(self, card: Card) -> bool:
        cmc = card.cmc
        if self.mv_eq is not None and cmc != self.mv_eq:
            return False
        if self.mv_lte is not None and cmc > self.mv_lte:
            return False
        if self.mv_gte is not None and cmc < self.mv_gte:
            return False

        type_line = (card.type_line or "").lower()
        if any(t not in type_line for t in self.types):
            return False

        oracle = (card.oracle_text or "").lower()
        if any(t not in oracle for t in self.text):
            return False

        name = card.name.lower()
        if self.names and not any(n in name for n in self.names):
            return False

        # Same keyword normalisation as the mechanic tools. Without it, a card whose
        # only Scryfall keyword is "Commander ninjutsu" answers "no" to kw:ninjutsu
        # here while answering "yes" in deck_mechanic_map — two tools of the same
        # server contradicting each other about the same card.
        return all(carries_keyword(card, k) for k in self.keywords)


@dataclass(frozen=True)
class CategoryResult:
    """A declared category measured against the actual list."""

    name: str
    filter: str
    expected: int | None
    actual: int
    cards: list[str]
    drift: int | None
    unparsed: tuple[str, ...] = ()

    @property
    def drifted(self) -> bool:
        return self.drift is not None and self.drift != 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filter": self.filter,
            "expected": self.expected,
            "actual": self.actual,
            "drift": self.drift,
            "drifted": self.drifted,
            "cards": self.cards,
            "unparsed_terms": list(self.unparsed),
        }


def _strip(value: str) -> str:
    return value.strip('"').lower()


def parse_filter(expression: str) -> CategoryFilter:
    """Parse a small filter expression such as ``mv<=1 t:creature``.

    Terms it does not understand are collected in ``unparsed`` rather than dropped:
    a filter that silently ignores half its input produces a count that looks
    authoritative and is not.
    """
    mv_lte = mv_gte = mv_eq = None
    types: list[str] = []
    text: list[str] = []
    names: list[str] = []
    keywords: list[str] = []

    consumed: list[tuple[int, int]] = []
    for match in _TERM_RE.finditer(expression):
        consumed.append(match.span())
        key = match.group("key").lower()
        op = match.group("op")
        value = _strip(match.group("value"))

        if key in ("mv", "cmc"):
            try:
                number = float(value)
            except ValueError:
                continue
            if op in ("<=", "<"):
                mv_lte = number - 1 if op == "<" else number
            elif op in (">=", ">"):
                mv_gte = number + 1 if op == ">" else number
            else:
                mv_eq = number
        elif key in ("t", "type"):
            types.append(value)
        elif key in ("o", "oracle"):
            text.append(value)
        elif key == "name":
            names.append(value)
        else:
            keywords.append(value)

    # Whatever the term regex did not consume is reported, never assumed harmless.
    leftover = expression
    for start, end in reversed(consumed):
        leftover = leftover[:start] + " " + leftover[end:]
    unparsed = tuple(token for token in leftover.split() if token)

    return CategoryFilter(
        mv_lte=mv_lte,
        mv_gte=mv_gte,
        mv_eq=mv_eq,
        types=tuple(types),
        text=tuple(text),
        names=tuple(names),
        keywords=tuple(keywords),
        unparsed=unparsed,
    )


def evaluate_categories(
    declared: list[dict[str, Any]],
    cards: list[Card],
) -> list[CategoryResult]:
    """Measure each declared category against the cards actually in the list.

    Args:
        declared: Entries with ``name``, ``filter`` and optionally ``expected``.
            An entry may instead carry ``cards``: an explicit list of names.
        cards: The resolved deck.

    Returns:
        One result per declared category, carrying ``actual`` and ``drift``.
    """
    results: list[CategoryResult] = []

    for entry in declared:
        name = str(entry.get("name") or "unnamed category")
        expected_raw = entry.get("expected")
        expected = int(expected_raw) if expected_raw is not None else None

        explicit = entry.get("cards")
        if explicit:
            wanted = {str(n).lower() for n in explicit}
            matched = [c.name for c in cards if c.name.lower() in wanted]
            results.append(
                CategoryResult(
                    name=name,
                    filter=f"explicit list of {len(wanted)} card(s)",
                    expected=expected if expected is not None else len(wanted),
                    actual=len(matched),
                    cards=matched,
                    drift=len(matched) - (expected if expected is not None else len(wanted)),
                )
            )
            continue

        expression = str(entry.get("filter") or "")
        parsed = parse_filter(expression)
        matched = [c.name for c in cards if parsed.matches(c)]
        results.append(
            CategoryResult(
                name=name,
                filter=expression,
                expected=expected,
                actual=len(matched),
                cards=matched,
                drift=(len(matched) - expected) if expected is not None else None,
                unparsed=parsed.unparsed,
            )
        )

    return results
