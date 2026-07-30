"""Connect the rules engine to a decklist.

The rules tools took a question; the deck tools took a list of cards. Nothing
joined them, so a deck could be analysed card by card without a single official
rule ever being read -- and the mechanics a deck is built on are exactly where
the rules bite.

This maps each mechanic the deck actually carries to the rule that governs it,
including the subrules where the operative detail lives: 702.2 only says
deathtouch is a static ability, 702.2b is the one that destroys the creature.

What it deliberately does NOT do: pair mechanics up and claim an interaction.
Measured against the corpus on 2026-07-30, the Comprehensive Rules contain
essentially no rule describing how two keywords interact -- searching for rules
that mention both returns 122.1b and 702.1c, which merely enumerate every
keyword in the game. Interactions emerge from applying the rules; they are not
written down. An "interactions" section here would have been noise wearing the
authority of a citation.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any

import structlog

from mtg_mcp_server.utils.mechanics import card_keywords, carries_keyword
from mtg_mcp_server.workflows import WorkflowResult
from mtg_mcp_server.workflows.card_resolver import resolve_cards

if TYPE_CHECKING:
    from mtg_mcp_server.services.rules import RulesService
    from mtg_mcp_server.services.scryfall import ScryfallClient
    from mtg_mcp_server.services.scryfall_bulk import ScryfallBulkClient
    from mtg_mcp_server.types import Card
    from mtg_mcp_server.utils.formatters import ResponseFormat

log = structlog.get_logger(service="workflow.rules_deck")

# "See rule 702.2, "Deathtouch."" -- how the glossary points at the real rule.
_RULE_REFERENCE_RE = re.compile(r"\brules?\s+(\d{3}\.\d+[a-z]?)", re.IGNORECASE)

# Corpus filename: MagicCompRules%2020250404.txt
#
# GOTCHA(2026-07-30): anchor on the extension, not on "the first 8 digits". The
# URL-encoded space in "%2020250404" puts a 20 in front of the date, so a bare
# \d{8} matched "20202504" and reported a corpus version that never existed --
# wrong, and plausible enough to be believed.
_CORPUS_VERSION_RE = re.compile(r"(\d{8})\.txt", re.IGNORECASE)

# A deck rarely leans on more than a handful of mechanics, and every one costs
# its rule plus subrules. Measured at ~1 KB per mechanic on the real corpus, so
# 15 stays far under the response ceiling. Capped explicitly, total reported.
_MAX_MECHANICS = 15

# How many carriers to name per mechanic. The point is to show the reader the
# mechanic is real in their deck, not to reprint the decklist.
_MAX_CARRIERS = 6


def _corpus_version(rules_url: str) -> str:
    match = _CORPUS_VERSION_RE.search(rules_url)
    return f"MagicCompRules {match.group(1)}" if match else "unknown"


async def deck_rules_map(
    decklist: list[str],
    commander: str,
    *,
    rules: RulesService,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
    response_format: ResponseFormat = "detailed",
) -> WorkflowResult:
    """Map every mechanic a decklist carries to the rule that governs it.

    Args:
        decklist: Card names in the deck (the commander is added if absent).
        commander: The commander's name.
        rules: Initialized RulesService.
        bulk: Bulk data client, or None when the feature flag is off.
        scryfall: Scryfall client, for whatever bulk data misses.
        response_format: ``detailed`` (default) or ``concise``.

    Returns:
        WorkflowResult whose data carries ``mechanics`` (each with its rule and
        subrules), ``uncovered`` (mechanics the corpus does not define),
        ``unresolved``, and ``corpus``.
    """
    names = list(dict.fromkeys([commander, *decklist]))
    log.info("deck_rules_map.start", commander=commander, cards=len(names))

    resolved, unresolved = await resolve_cards(names, bulk=bulk, scryfall=scryfall)

    pool: list[Card] = []
    seen: set[str] = set()
    for name in names:
        card = resolved.get(name.lower())
        if card is not None and card.name not in seen:
            seen.add(card.name)
            pool.append(card)

    counts: Counter[str] = Counter()
    for card in pool:
        counts.update(card_keywords(card))

    glossary_terms = {entry["term"].lower(): entry for entry in await rules.list_keywords()}

    mechanics: list[dict[str, Any]] = []
    uncovered: list[dict[str, Any]] = []

    for keyword, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        carriers = [c.name for c in pool if carries_keyword(c, keyword)]
        entry = glossary_terms.get(keyword.lower())
        reference = _RULE_REFERENCE_RE.findall(entry["definition"]) if entry else []
        rule = await rules.lookup_by_number(reference[0]) if reference else None

        if rule is None:
            # Either the glossary does not know this keyword, or it names no
            # rule. Both mean the same thing to a reader: this corpus cannot
            # tell you how it works.
            uncovered.append(
                {"keyword": keyword, "count": count, "carriers": carriers[:_MAX_CARRIERS]}
            )
            continue

        subrules = await rules.subrules(rule.number)
        mechanics.append(
            {
                "keyword": keyword,
                "count": count,
                "carriers": carriers[:_MAX_CARRIERS],
                "rule": rule.number,
                "rule_text": rule.text,
                "subrules": [{"number": s.number, "text": s.text} for s in subrules],
            }
        )

    mechanics_total = len(mechanics)
    mechanics = mechanics[:_MAX_MECHANICS]

    corpus = _corpus_version(rules.rules_url)
    markdown = _format(
        commander=commander,
        mechanics=mechanics,
        mechanics_total=mechanics_total,
        uncovered=uncovered,
        unresolved=unresolved,
        corpus=corpus,
        response_format=response_format,
    )

    log.info(
        "deck_rules_map.complete",
        mechanics=len(mechanics),
        uncovered=len(uncovered),
        unresolved=len(unresolved),
    )
    return WorkflowResult(
        markdown=markdown,
        data={
            "commander": commander,
            "corpus": corpus,
            "mechanics": mechanics,
            "mechanics_total": mechanics_total,
            "uncovered": uncovered,
            "unresolved": unresolved,
        },
    )


def _join_names(names: list[str]) -> str:
    """Card names contain commas ("Axavar, Fate Thief"), so a bare comma join
    reads as one more card than the list holds. Backticks bound each name."""
    return ", ".join(f"`{n}`" for n in names)


def _format(
    *,
    commander: str,
    mechanics: list[dict[str, Any]],
    mechanics_total: int,
    uncovered: list[dict[str, Any]],
    unresolved: list[str],
    corpus: str,
    response_format: ResponseFormat,
) -> str:
    lines = [f"# Rules Behind {commander}'s Deck", ""]
    lines.append(f"*Rules corpus: {corpus}.*")
    lines.append("")

    if not mechanics and not uncovered:
        lines.append("No keyword mechanics found on the resolved cards.")
    if mechanics_total > len(mechanics):
        lines.append(
            f"Showing {len(mechanics)} of {mechanics_total} mechanics, most-carried first."
        )
        lines.append("")

    for mechanic in mechanics:
        carriers = _join_names(mechanic["carriers"])
        lines.append(f"## {mechanic['keyword']} — {mechanic['count']} card(s)")
        lines.append(f"*Carried by: {carriers}*")
        lines.append("")
        lines.append(f"**{mechanic['rule']}** {mechanic['rule_text']}")
        if response_format == "detailed":
            for sub in mechanic["subrules"]:
                lines.append(f"- **{sub['number']}** {sub['text']}")
        elif mechanic["subrules"]:
            numbers = ", ".join(s["number"] for s in mechanic["subrules"])
            lines.append(f"- Subrules: {numbers}")
        lines.append("")

    if uncovered:
        lines.append("## Not covered by this rules corpus")
        lines.append("")
        lines.append(
            f"These mechanics are in the deck but absent from {corpus}. The corpus is a "
            "dated file that updates a few times a year, so a mechanic newer than it "
            "has no entry. Absence here says nothing about how the mechanic works."
        )
        lines.append("")
        for item in uncovered:
            carriers = _join_names(item["carriers"])
            lines.append(f"- **{item['keyword']}** — {item['count']} card(s): {carriers}")
        lines.append("")

    if unresolved:
        lines.append("## Unresolved cards")
        lines.append("")
        lines.append(
            "These names could not be looked up, so their mechanics are not counted above: "
            + ", ".join(unresolved)
        )

    return "\n".join(lines)
