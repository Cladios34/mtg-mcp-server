"""Defensive normalisation of search queries before they reach Scryfall.

Some MCP clients HTML-escape the comparison operators of a Scryfall query on their way
out, turning ``mv<=3`` into ``mv&lt;=3``. Scryfall then answers 404, which the server
used to surface as a bare "No cards found" — indistinguishable from a query that is
syntactically valid but genuinely matches nothing.

That ambiguity is expensive: an agent applying an exhaustive-search protocol reads the
empty result as proof of absence and narrows its search instead of fixing the query.

So we do two things, and the second matters as much as the first:

1. Decode the escaped operators so the query works anyway.
2. Report that we did it, so the broken client is visible rather than papered over.

Only the entities that can plausibly appear in a Scryfall query are decoded — this is
deliberately not :func:`html.unescape`, which would also rewrite things like ``&copy;``
and could corrupt a legitimate card name.
"""

from __future__ import annotations

import re

# Ordered: ``&amp;`` is decoded LAST so that a double-escaped ``&amp;lt;`` resolves to
# ``&lt;`` rather than collapsing straight to ``<`` in a single pass.
_ENTITIES: tuple[tuple[str, str], ...] = (
    ("&lt;", "<"),
    ("&LT;", "<"),
    ("&#60;", "<"),
    ("&#x3c;", "<"),
    ("&gt;", ">"),
    ("&GT;", ">"),
    ("&#62;", ">"),
    ("&#x3e;", ">"),
    ("&le;", "<="),
    ("&ge;", ">="),
    ("&quot;", '"'),
    ("&#34;", '"'),
    ("&#39;", "'"),
    ("&apos;", "'"),
    ("&amp;", "&"),
)


def normalize_query(query: str) -> tuple[str, bool]:
    """Decode HTML-escaped operators in a search query.

    Args:
        query: The query exactly as the client sent it.

    Returns:
        A ``(normalized, was_escaped)`` tuple. ``was_escaped`` is True when the input
        differed from the output, meaning the caller escaped its query and should be
        told so.
    """
    normalized = query
    for entity, char in _ENTITIES:
        normalized = normalized.replace(entity, char)
    return normalized, normalized != query


def escaping_warning(original: str, normalized: str) -> str:
    """Build the one-line warning shown when a query arrived HTML-escaped."""
    return (
        f"NOTE: your query arrived HTML-escaped and was decoded before sending. "
        f"Received {original!r}, sent {normalized!r}. "
        f"Send raw operators (<, <=, >, >=) — the escaped form matches nothing."
    )


# Scryfall filter syntax, recognised so it can be REFUSED by the substring searches.
# Bulk search matches plain substrings, so `t:creature mv<=2` finds nothing — and a
# bare "no cards found" would read as proof the cards do not exist.
_SCRYFALL_SYNTAX = re.compile(
    r"""
    (?:^|\s)
    (?:
        # A filter key bound TIGHTLY to its value: `t:creature`, `mv<=2`.
        # The colon must not be followed by a space — that rules out ordinary oracle
        # text like "T: Add {C}", which is a legitimate substring to search for and
        # was being refused as if it were filter syntax.
        (?:t|type|id|c|color|f|format|o|oracle|mv|cmc|pow|tou|r|rarity|is|set|e|s)
        (?::(?=\S)|\s*(?:<=|>=|<|>|=)\s*\S)
      | [a-z]{1,4}\s*(?:<=|>=)\s*\S    # any short key with a comparison operator
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def looks_like_scryfall_syntax(query: str) -> bool:
    """True when a substring query is really a Scryfall filter expression.

    The bulk searches match plain substrings. Handed ``t:creature id<=ub mv<=2``
    they match nothing, and an unqualified "No cards found" is then read as
    evidence of absence rather than as a wrong-tool error.
    """
    return bool(_SCRYFALL_SYNTAX.search(query))


# A legality filter, in every spelling Scryfall accepts.
# GOTCHA(2026-07-30): Scryfall marks a card `not_legal` in EVERY format until its set's
# release day, so `f:commander` silently drops spoiled cards from sets that are already
# previewed. A discovery search that carries this filter therefore cannot see anything
# upcoming, and nothing in the response says so.
# Negated forms are deliberately NOT matched: `-banned:commander` asks for "not banned",
# which unreleased cards already satisfy (their status is `not_legal`, never `banned`).
# Warning on it would flag a query that hides nothing.
_LEGALITY_FILTER = re.compile(
    r"(?:^|\s)(?:f|format|legal|banned|restricted):\S+",
    re.IGNORECASE,
)

# Date and release-window terms. The probe appends its own `date>today`, so any date term
# already in the query has to go first: `date<2020` plus `date>today` is satisfiable by
# nothing at all, and the empty result would be reported as "nothing is hidden": the very
# false-negative this whole feature exists to prevent.
_DATE_FILTER = re.compile(
    r"(?:^|\s)-?(?:date|year):?\s*(?:<=|>=|<|>|=|:)\s*\S+",
    re.IGNORECASE,
)

# Never legal in Commander at any point, so their absence is not a release-date artefact
# and flagging them as "coming soon" would be a lie: joke sets and Arena-only printings.
_NEVER_PAPER_LEGAL = "-is:funny -is:digital"


def has_legality_filter(query: str) -> bool:
    """True when the query constrains format legality (``f:``, ``legal:``, ``banned:``...)."""
    return bool(_LEGALITY_FILTER.search(query))


def strip_legality_filter(query: str) -> str:
    """Return the query with every legality term removed, whitespace normalised.

    Used to build the probe that counts what the legality filter is hiding. Returns an
    empty string when legality was the only thing the query asked for.
    """
    return " ".join(_LEGALITY_FILTER.sub(" ", query).split())


def _is_balanced(query: str) -> bool:
    """True when quotes and parentheses pair up.

    An unbalanced query cannot be safely extended: appending `date>...` to a query with an
    open quote makes the appended filters part of a string literal, so they stop being
    filters. The probe would then match almost nothing and wrongly report "nothing hidden".
    """
    if query.count('"') % 2:
        return False
    depth = 0
    for char in query:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def unreleased_probe_query(query: str, today: str) -> str | None:
    """Build the query that finds what a legality filter excludes, or None if moot.

    ``today`` is an ISO date; ``date>today`` is what actually isolates unreleased
    printings. ``is:spoiler`` does NOT: on Scryfall it means "this printing was
    previewed before its release", which is true of nearly every modern card
    (measured 2026-07-30: `id<=rwb t:angel is:spoiler` returns 241 cards, including
    Angel of Despair and Akroma).

    Returns None when no meaningful probe can be built: legality was the whole query, or
    what remains is unbalanced. Returning None is the honest answer, because a probe that
    silently matches nothing is indistinguishable from "nothing is hidden".
    """
    rest = strip_legality_filter(query)
    rest = " ".join(_DATE_FILTER.sub(" ", rest).split())
    if not rest or not _is_balanced(rest):
        return None
    return f"{rest} date>{today} {_NEVER_PAPER_LEGAL}"


def bare_unreleased_query(today: str) -> str:
    """Probe matching ALL unreleased paper cards.

    For tools whose only constraint besides legality is a recency window that
    unreleased cards trivially satisfy (``scryfall_whats_new``): there,
    "everything upcoming" is exactly what the legality filter hid, so an
    unconstrained probe is meaningful where :func:`unreleased_probe_query`
    would return None.
    """
    return f"date>{today} {_NEVER_PAPER_LEGAL}"


def unreleased_warning(
    names: list[str], probe: str, total: int | None = None, shown: int = 10
) -> str:
    """Build the warning naming the cards a legality filter is hiding.

    The names matter more than the count. A generic "beware of legality filters" reads
    once and then goes invisible; a named card the caller expected to see does not.

    ``total`` is Scryfall's own match count, which can exceed the names we hold: the probe
    reads one page. Reporting ``len(names)`` as the total would announce a number the query
    never measured, so the two are stated separately when they differ.
    """
    listed = ", ".join(names[:shown])
    overflow = f" (+{len(names) - shown} more listed)" if len(names) > shown else ""
    count = total if total is not None else len(names)
    partial = f" (showing {len(names)} of them)" if count > len(names) else ""
    return (
        f"NOTE: your query filters on legality, which EXCLUDES cards from sets that "
        f"have not been released yet: Scryfall marks those not_legal in every format "
        f"until release day. {count} such card(s) match the rest of your query"
        f"{partial}: {listed}{overflow}. "
        f"If you are exploring what EXISTS rather than what is playable today, drop the "
        f"legality term and use `{_NEVER_PAPER_LEGAL}` instead. "
        f"Probe used: {probe!r}"
    )
