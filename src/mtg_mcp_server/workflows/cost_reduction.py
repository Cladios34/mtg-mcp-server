"""Apply a cost reducer to a set of costs, mechanically.

Regression origin (2026-07-27). A report asserted "Silver-Fur Master + Skullsnatcher
-> ninjutsu at {0}". False: the reduction is generic, and Skullsnatcher's ninjutsu cost
is {U}{B} — all coloured pips, untouchable (rule 601.2f). Recounting then produced
"reduces 4 ninjutsu" when the answer was 9 of 15. Two errors on one card, in opposite
directions, both from doing the arithmetic by hand.

So the arithmetic stops being a judgement call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import structlog

from mtg_mcp_server.utils.mechanics import (
    apply_generic_reduction,
    keyword_activation_cost,
    reduction_amount,
)
from mtg_mcp_server.workflows import WorkflowResult
from mtg_mcp_server.workflows.card_resolver import resolve_card, resolve_cards

if TYPE_CHECKING:
    from mtg_mcp_server.services.scryfall import ScryfallClient
    from mtg_mcp_server.services.scryfall_bulk import ScryfallBulkClient

log = structlog.get_logger(service="workflow.cost_reduction")

_RULE_NOTE = (
    "Rule 601.2f: a generic cost reduction reduces only the generic component of a "
    "cost. Coloured pips, colorless {C}, hybrid symbols and {X} are never reduced. "
    "Two reducers do not take {U}{B} to {0} — they leave it at {U}{B}."
)


async def cost_reduction_check(
    reducer_card: str,
    *,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
    target_costs: list[str] | None = None,
    target_cards: list[str] | None = None,
    keyword: str | None = None,
    response_format: Literal["detailed", "concise"] = "detailed",
) -> WorkflowResult:
    """Report exactly which costs a reducer reduces, and which it leaves alone.

    Args:
        reducer_card: The card doing the reducing. Its oracle text sets the amount.
        bulk: Bulk data client, or None when the feature flag is off.
        scryfall: Scryfall client, used for whatever bulk data misses.
        target_costs: Raw mana costs to test, e.g. ``["{U}{B}", "{2}{U}{U}"]``.
        target_cards: Card names to test instead of raw costs. Their mana cost is used,
            or their ``keyword`` activation cost when ``keyword`` is given.
        keyword: When testing cards, the keyword whose activation cost is the target
            (e.g. ``"Ninjutsu"``) rather than the card's own mana cost.
        response_format: ``detailed`` (default) or ``concise``.

    Returns:
        WorkflowResult listing every target with ``reduced``, ``result`` and the
        reason — a "no" always says why.
    """
    log.info("cost_reduction_check.start", reducer=reducer_card, keyword=keyword)

    reducer = await resolve_card(reducer_card, bulk=bulk, scryfall=scryfall)
    amount = reduction_amount(reducer)

    if amount is None:
        return WorkflowResult(
            markdown=(
                f"# Cost reduction — {reducer.name}\n\n"
                f"**{reducer.name} does not reduce costs.** Its oracle text carries no "
                f"cost-reduction clause, so nothing below would change.\n\n"
                f"Oracle: {reducer.oracle_text or '(none)'}"
            ),
            data={
                "reducer": reducer.name,
                "reduces": False,
                "amount": 0,
                "targets": [],
                "note": "no cost-reduction clause found in the oracle text",
            },
        )

    targets: list[tuple[str, str]] = [(cost, cost) for cost in (target_costs or [])]
    unresolved: list[str] = []

    if target_cards:
        resolved, unresolved = await resolve_cards(target_cards, bulk=bulk, scryfall=scryfall)
        for name in target_cards:
            card = resolved.get(name.lower())
            if card is None:
                continue
            cost = keyword_activation_cost(card, keyword) if keyword else None
            cost = cost or card.mana_cost or ""
            targets.append((card.name, cost))

    results: list[dict[str, Any]] = []
    for label, cost in targets:
        outcome = apply_generic_reduction(cost, amount)
        results.append(
            {
                "target": label,
                "cost": outcome.cost,
                "reduced": outcome.reduced,
                "result": outcome.result,
                "reason": outcome.reason,
            }
        )

    reduced_count = sum(1 for r in results if r["reduced"])
    data: dict[str, Any] = {
        "reducer": reducer.name,
        "reduces": True,
        "amount": amount,
        "keyword": keyword,
        "targets": results,
        "reduced_count": reduced_count,
        "total_targets": len(results),
        "unresolved": unresolved,
        "note": _RULE_NOTE,
    }

    log.info(
        "cost_reduction_check.complete",
        reducer=reducer.name,
        reduced=reduced_count,
        total=len(results),
    )
    return WorkflowResult(markdown=_format(reducer.name, amount, data, response_format), data=data)


def _format(
    reducer: str,
    amount: int,
    data: dict[str, Any],
    response_format: Literal["detailed", "concise"],
) -> str:
    lines = [
        f"# Cost reduction — {reducer} (-{{{amount}}} generic)",
        "",
        f"**{data['reduced_count']} of {data['total_targets']} targets are actually reduced.**",
        "",
    ]

    if data["targets"]:
        lines.append("| Target | Cost | Reduced | Result |")
        lines.append("|--------|------|---------|--------|")
        for row in data["targets"]:
            mark = "yes" if row["reduced"] else "**no**"
            lines.append(f"| {row['target']} | {row['cost']} | {mark} | {row['result']} |")
        lines.append("")

    if response_format != "concise":
        untouched = [r for r in data["targets"] if not r["reduced"]]
        if untouched:
            lines.append("Not reduced, and why:")
            for row in untouched:
                lines.append(f"- `{row['cost']}` — {row['reason']}")
            lines.append("")

    lines.append(_RULE_NOTE)

    if data["unresolved"]:
        lines.append("")
        lines.append(
            f"**Unresolved ({len(data['unresolved'])})**: {', '.join(data['unresolved'])} "
            "— not tested, and not counted above."
        )

    return "\n".join(lines)
