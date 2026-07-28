"""Map the mechanics a deck shares, rather than the cards it contains.

Every other tool here is indexed BY CARD: one entry per card, one rank per card, one
category per card. Nothing describes a shared mechanic, a trigger's reach, or a deck
resource — so card-by-card analysis is the path of least resistance, because it is the
only path the tools pave.

The cost of that showed up in a real audit (2026-07-27): a deck was described in every
deliverable as "cheap creature -> commander -> reveal", a single destination, when it
actually ran fifteen cards sharing the same keyword. The cheap creature was not there
to deploy the commander; it was there to deploy whichever of the fifteen answered the
turn. That is a different deck from the one that was reviewed.

This tool answers three questions the card-indexed tools cannot:

- Which mechanic do several cards share, and how many carry it?
- What actually modifies its cost — and, precisely, on how many of the carriers?
- Who really has the tribal type, counting changelings (702.73a) and conditional
  creatures, both of which a naive type search misses?
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

import structlog

from mtg_mcp_server.utils.mechanics import (
    EVERGREEN_KEYWORDS,
    apply_generic_reduction,
    card_keywords,
    carries_keyword,
    creature_types,
    has_creature_type,
    keyword_activation_cost,
    parse_mana_cost,
    reduction_amount,
    reduction_clause,
    reduction_depends_on_choice,
)
from mtg_mcp_server.utils.triggers import derive_trigger
from mtg_mcp_server.workflows import WorkflowResult
from mtg_mcp_server.workflows.card_resolver import resolve_cards

if TYPE_CHECKING:
    from mtg_mcp_server.services.scryfall import ScryfallClient
    from mtg_mcp_server.services.scryfall_bulk import ScryfallBulkClient
    from mtg_mcp_server.types import Card

log = structlog.get_logger(service="workflow.mechanic_map")

# A keyword carried by this many cards is a deck mechanic, not an incidental line of
# text. Three is the point where "some cards happen to have it" becomes "the deck does
# this" — below it, a shared keyword says nothing about how the deck operates.
_FREQUENCY_THRESHOLD = 3

# Evergreen keywords (canonical list in utils.mechanics) are excluded from frequency
# detection; still reported when the commander carries one, because then it IS the plan.
_TYPE_THRESHOLD = 3

_GENERIC_ONLY_NOTE = (
    "generic reduction — coloured, colorless, hybrid and {X} symbols are untouched "
    "(rule 601.2f). A cost with no generic component is NOT reduced."
)


def _mechanic_entry(
    keyword: str,
    source: str,
    carriers: list[Card],
    pool: list[Card],
) -> dict[str, Any]:
    """Build one mechanic's entry: its carriers, their cost tiers, and what modifies them."""
    cards: list[dict[str, Any]] = []
    tiers: dict[str, list[str]] = {}

    for card in carriers:
        activation = keyword_activation_cost(card, keyword)
        cost = activation or card.mana_cost
        parsed = parse_mana_cost(cost)
        value = parsed.generic + len(parsed.symbols)
        cards.append(
            {
                "name": card.name,
                "activation_cost": activation,
                "base_cost": card.mana_cost,
                "payoff_text": card.oracle_text,
            }
        )
        tiers.setdefault(str(value), []).append(card.name)

    modifiers: list[dict[str, Any]] = []
    for card in pool:
        amount = reduction_amount(card)
        if amount is None:
            continue
        affected: list[str] = []
        unaffected: list[str] = []
        for carrier in carriers:
            if carrier.name == card.name:
                # "OTHER Ninja spells you cast" — a reducer rarely reduces itself, and
                # counting it inflates the number that gets quoted back as fact.
                continue
            cost = keyword_activation_cost(carrier, keyword) or carrier.mana_cost
            if apply_generic_reduction(cost, amount).reduced:
                affected.append(carrier.name)
            else:
                unaffected.append(carrier.name)
        modifiers.append(
            {
                "name": card.name,
                "amount": amount,
                "reduces": len(affected),
                # The denominator counts what was ACTUALLY tested. A reducer that
                # carries the keyword itself is skipped above, so quoting the full
                # carrier count here would state "8 of 15" when 14 were examined —
                # the same species of wrong ratio this tool exists to prevent.
                "of": len(affected) + len(unaffected),
                "carriers_total": len(carriers),
                "affected": affected,
                "unaffected": unaffected,
                # The counts above answer "would this amount fit these costs", NOT
                # "does this card's clause target these cards". Herald's Horn reduces
                # only the chosen type; asserting a scope we cannot resolve would be
                # the same confident-and-wrong number this tool exists to replace.
                "clause": reduction_clause(card),
                "depends_on_choice": reduction_depends_on_choice(card),
                "note": _GENERIC_ONLY_NOTE,
            }
        )

    return {
        "keyword": keyword,
        "source": source,
        "count": len(carriers),
        "cards": cards,
        "cost_tiers": dict(sorted(tiers.items(), key=lambda kv: int(kv[0]))),
        "cost_modifiers": modifiers,
    }


def _type_entry(creature_type: str, pool: list[Card]) -> dict[str, Any]:
    """Count everything that has ``creature_type``, by which route it has it."""
    typed: list[str] = []
    changelings: list[str] = []
    conditional: list[dict[str, str]] = []

    for card in pool:
        match = has_creature_type(card, creature_type)
        if not match.matches:
            continue
        if match.via == "printed":
            typed.append(card.name)
        elif match.via == "changeling":
            changelings.append(card.name)
        else:
            conditional.append({"name": card.name, "condition": match.condition or ""})

    return {
        "type": creature_type,
        "cards_typed": typed,
        "cards_by_changeling": changelings,
        "cards_conditional": conditional,
        "total": len(typed) + len(changelings) + len(conditional),
        "note": (
            "Changelings are every creature type in every zone (702.73a) and a type "
            "search will not return them. Conditional creatures count only while their "
            "condition holds — the condition travels with the count, never separately."
        ),
    }


async def deck_mechanic_map(
    decklist: list[str],
    commander: str,
    *,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
    response_format: Literal["detailed", "concise"] = "detailed",
) -> WorkflowResult:
    """Map the mechanics, tribal types, and trigger scopes a decklist shares.

    Args:
        decklist: Card names in the deck (the commander is added if absent).
        commander: The commander's name — its keywords seed the mechanic detection.
        bulk: Bulk data client, or None when the feature flag is off.
        scryfall: Scryfall client, used for whatever bulk data misses.
        response_format: ``detailed`` (default) or ``concise``.

    Returns:
        WorkflowResult whose data carries ``mechanics``, ``type_synergy``,
        ``triggers``, and ``unresolved``.
    """
    names = list(dict.fromkeys([commander, *decklist]))
    log.info("deck_mechanic_map.start", commander=commander, cards=len(names))

    resolved, unresolved = await resolve_cards(names, bulk=bulk, scryfall=scryfall)

    pool: list[Card] = []
    seen: set[str] = set()
    for name in names:
        card = resolved.get(name.lower())
        if card is not None and card.name not in seen:
            seen.add(card.name)
            pool.append(card)

    commander_card = resolved.get(commander.lower())

    # --- Mechanics -------------------------------------------------------------
    # The commander's keywords come first and unconditionally: they define the deck's
    # intent even when only a couple of other cards share them.
    commander_keywords = card_keywords(commander_card) if commander_card else set()

    counts: Counter[str] = Counter()
    for card in pool:
        counts.update(card_keywords(card))

    frequent = {
        keyword
        for keyword, n in counts.items()
        if n >= _FREQUENCY_THRESHOLD and keyword.lower() not in EVERGREEN_KEYWORDS
    }

    mechanics: list[dict[str, Any]] = []
    for keyword in sorted(commander_keywords | frequent, key=lambda k: (-counts[k], k)):
        carriers = [c for c in pool if carries_keyword(c, keyword)]
        if not carriers:
            continue
        source = (
            "commander_keyword"
            if keyword in commander_keywords
            else f"frequency>={_FREQUENCY_THRESHOLD}"
        )
        mechanics.append(_mechanic_entry(keyword, source, carriers, pool))

    # --- Tribal types ----------------------------------------------------------
    type_counts: Counter[str] = Counter()
    for card in pool:
        if "creature" in (card.type_line or "").lower():
            type_counts.update(creature_types(card))

    commander_types = creature_types(commander_card) if commander_card else set()
    wanted_types = {t for t, n in type_counts.items() if n >= _TYPE_THRESHOLD} | commander_types
    # Shapeshifter is what a changeling is PRINTED as; it is never the tribe being played.
    wanted_types.discard("Shapeshifter")

    type_synergy = [_type_entry(t, pool) for t in sorted(wanted_types)]
    type_synergy = [t for t in type_synergy if t["total"] >= _TYPE_THRESHOLD]
    type_synergy.sort(key=lambda t: -t["total"])

    # --- Trigger scopes --------------------------------------------------------
    triggers: list[dict[str, Any]] = []
    for card in pool:
        trigger = derive_trigger(card)
        if trigger.scope == "static":
            continue
        triggers.append(
            {
                "name": card.name,
                "scope": trigger.scope,
                "condition": trigger.condition,
                "sources": trigger.sources,
                # A card with two abilities is reported by its widest-reaching one.
                # Without this field the second one is computed and then dropped on
                # the floor — the reader sees a per_source card and never learns it
                # also has an upkeep trigger.
                "other_scopes": list(trigger.other_scopes),
                "notes": trigger.notes,
            }
        )

    data: dict[str, Any] = {
        "commander": commander_card.name if commander_card else commander,
        "cards_analyzed": len(pool),
        "mechanics": mechanics,
        "type_synergy": type_synergy,
        "triggers": triggers,
        "unresolved": unresolved,
    }

    markdown = _format(commander, data, response_format=response_format)
    log.info(
        "deck_mechanic_map.complete",
        mechanics=len(mechanics),
        types=len(type_synergy),
        unresolved=len(unresolved),
    )
    return WorkflowResult(markdown=markdown, data=data)


def _format(
    commander: str,
    data: dict[str, Any],
    *,
    response_format: Literal["detailed", "concise"],
) -> str:
    lines: list[str] = [f"# Mechanic Map — {data['commander']}", ""]

    mechanics = data["mechanics"]
    if not mechanics:
        lines.append(
            "No shared mechanic found. That is itself a finding: this deck has no "
            "keyword its cards route through, so it cannot be described as one loop."
        )
    for mechanic in mechanics:
        count = mechanic["count"]
        lines.append(f"## {mechanic['keyword']} — {count} card(s) ({mechanic['source']})")
        lines.append("")
        lines.append(
            f"**{count} destinations, not one.** A card that enables this mechanic "
            f"enables any of the {count}, not just the commander. Describing the deck "
            f"as a single loop misreads it."
        )
        lines.append("")

        if response_format != "concise":
            lines.append("| Card | Activation | Base cost |")
            lines.append("|------|-----------|-----------|")
            for card in mechanic["cards"]:
                activation = card["activation_cost"] or "-"
                lines.append(f"| {card['name']} | {activation} | {card['base_cost'] or '-'} |")
            lines.append("")

        tiers = mechanic["cost_tiers"]
        if tiers:
            spread = ", ".join(f"{value}: {len(names)}" for value, names in tiers.items())
            lines.append(f"Cost tiers (mana value -> cards): {spread}")
            lines.append("")

        for modifier in mechanic["cost_modifiers"]:
            lines.append(
                f"**{modifier['name']}**: the amount fits {modifier['reduces']} of "
                f"{modifier['of']} costs here, not all of them. {_GENERIC_ONLY_NOTE}"
            )
            if modifier["clause"]:
                lines.append(f"Clause: _{modifier['clause']}_")
            if modifier["depends_on_choice"]:
                lines.append(
                    "Its scope is set by a type chosen on resolution, so whether it "
                    "touches these cards at all depends on that choice, not on the "
                    "count above."
                )
            if modifier["unaffected"]:
                lines.append(f"Unaffected: {', '.join(modifier['unaffected'])}")
            lines.append("")

    for entry in data["type_synergy"]:
        lines.append(f"## Type — {entry['type']}: {entry['total']} total")
        lines.append("")
        lines.append(f"- printed type: {len(entry['cards_typed'])}")
        if entry["cards_by_changeling"]:
            lines.append(
                f"- via changeling (702.73a): {len(entry['cards_by_changeling'])} — "
                f"{', '.join(entry['cards_by_changeling'])}. A `t:` search does NOT "
                f"return these; a tribal count that omits them is wrong."
            )
        else:
            lines.append("- via changeling (702.73a): 0")
        for conditional in entry["cards_conditional"]:
            lines.append(f"- conditional: {conditional['name']} — {conditional['condition']}")
        lines.append("")

    per_source = [t for t in data["triggers"] if t["scope"] == "per_source"]
    per_combat = [t for t in data["triggers"] if t["scope"] == "per_combat"]
    if per_source or per_combat:
        lines.append("## Trigger reach")
        lines.append("")
        if per_source:
            lines.append(
                "**Multiplies per qualifying source** (each creature that connects "
                "triggers it separately): " + ", ".join(t["name"] for t in per_source)
            )
        if per_combat:
            lines.append(
                "**Fires once per combat** however many creatures qualify: "
                + ", ".join(t["name"] for t in per_combat)
            )
        if any(t["notes"] for t in data["triggers"]):
            lines.append("")
            lines.append(
                "A creature with 0 power deals no combat damage and triggers none of "
                "the damage-based abilities above, whatever its types (510.1a)."
            )
        lines.append("")

    if data["unresolved"]:
        lines.append(
            f"**Unresolved ({len(data['unresolved'])})**: {', '.join(data['unresolved'])}. "
            "These were NOT analysed — the counts above exclude them."
        )

    return "\n".join(lines)
