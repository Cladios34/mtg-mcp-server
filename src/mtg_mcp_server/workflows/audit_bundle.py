"""Composite Commander deck audit — the whole mechanical battery in one call.

Runs deck validation, deck analysis, Spellbook decklist combos, bracket
estimation, and the v3 opening-hand simulation CONCURRENTLY (transport
concurrency verified by ``scripts/parallel_probe.mjs``), and returns a single
report where EVERY section carries an explicit ``ok``/``error`` status and
echoes the parameters it actually used.

Design requirements:

- Silent-failure ban (schema-drift bug family, 2026-07-22): a section that
  fails must say so loudly in the report — never a silent zero or missing key
  next to rich data from other sections.
- Forced v3 simulation: ``commander_colors`` is a required argument and
  ``tutor_aware`` is always on, so a color-blind simulation cannot happen by
  omission (the exact incident this bundle exists to prevent).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from mtg_mcp_server.workflows import WorkflowResult

if TYPE_CHECKING:
    from mtg_mcp_server.services.edhrec import EDHRECClient
    from mtg_mcp_server.services.scryfall import ScryfallClient
    from mtg_mcp_server.services.scryfall_bulk import ScryfallBulkClient
    from mtg_mcp_server.services.spellbook import SpellbookClient
from mtg_mcp_server.workflows.analysis import deck_analysis
from mtg_mcp_server.workflows.simulation import simulate_opening_hands
from mtg_mcp_server.workflows.validation import deck_validate

Section = dict[str, Any]


async def _guard(name: str, params: Mapping[str, Any], coro: Awaitable[Any]) -> Section:
    """Run one section, converting ANY failure into an explicit error entry."""
    try:
        data = await coro
    except Exception as exc:  # every failure must surface in the report, none may escape
        return {
            "section": name,
            "ok": False,
            "params_used": dict(params),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"section": name, "ok": True, "params_used": dict(params), "data": data}


async def _fail(message: str) -> Any:
    """Coroutine that raises — used to report unavailable backends per section."""
    raise RuntimeError(message)


async def deck_audit_bundle(
    decklist: list[str],
    commander: str,
    commander_colors: str,
    *,
    iterations: int = 10000,
    seed: int | None = None,
    extra_mana_sources: list[str] | None = None,
    exclude_cards: list[str] | None = None,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
    spellbook: SpellbookClient,
    edhrec: EDHRECClient | None = None,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> WorkflowResult:
    """Run the full mechanical audit battery on a 99-card Commander list.

    ``decklist`` is bare card names, one entry per physical card (repeat basic
    lands), commander EXCLUDED — the same list works for every section.
    """

    async def _noop(step: int, total: int) -> None:
        return None

    progress = on_progress or _noop

    sim_params = {
        "iterations": iterations,
        "seed": seed,
        "keep_rule": "playability",
        "free_mulligan": True,
        "commander_colors": commander_colors,
        "tutor_aware": True,
        "extra_mana_sources": extra_mana_sources,
        "exclude_cards": exclude_cards,
    }

    async def _run_validate(bulk_client: ScryfallBulkClient) -> Any:
        result = await deck_validate(
            decklist,
            "commander",
            commander=commander,
            bulk=bulk_client,
            response_format="concise",
        )
        return result.data

    async def _run_analysis() -> Any:
        result = await deck_analysis(
            decklist,
            commander,
            bulk=bulk,
            scryfall=scryfall,
            spellbook=spellbook,
            edhrec=edhrec,
            on_progress=progress,
            response_format="concise",
        )
        return result.data

    async def _run_combos() -> Any:
        combos = await spellbook.find_decklist_combos([commander], decklist)
        return combos.model_dump()

    async def _run_bracket() -> Any:
        estimate = await spellbook.estimate_bracket([commander], decklist)
        return estimate.model_dump()

    async def _run_simulation() -> Any:
        result = await simulate_opening_hands(
            decklist,
            iterations=iterations,
            seed=seed,
            keep_rule="playability",
            free_mulligan=True,
            commander_colors=commander_colors,
            tutor_aware=True,
            extra_mana_sources=extra_mana_sources,
            exclude_cards=exclude_cards,
            bulk=bulk,
            scryfall=scryfall,
        )
        return result.data

    if bulk is not None:
        validate_coro = _run_validate(bulk)
    else:
        validate_coro = _fail("bulk data backend disabled (enable_bulk_data)")

    sections = await asyncio.gather(
        _guard("validate", {"format": "commander", "commander": commander}, validate_coro),
        _guard("analysis", {"commander": commander, "response_format": "concise"}, _run_analysis()),
        _guard("combos", {"commanders": [commander]}, _run_combos()),
        _guard("bracket", {"commanders": [commander]}, _run_bracket()),
        _guard("simulation", sim_params, _run_simulation()),
    )

    by_name = {s["section"]: s for s in sections}
    failed = [s["section"] for s in sections if not s["ok"]]

    def peek(section: str, *keys: str) -> Any:
        entry = by_name[section]
        if not entry["ok"]:
            return None
        node: Any = entry.get("data")
        for key in keys:
            if not isinstance(node, Mapping):
                return None
            node = node.get(key)
        return node

    summary = f"Sections: {len(sections) - len(failed)}/{len(sections)} ok"
    if failed:
        summary += f" — FAILED: {', '.join(failed)}"
    lines = [f"# Deck audit bundle — {commander}", "", summary]
    for s in sections:
        status = "OK" if s["ok"] else f"FAILED — {s['error']}"
        lines.append(f"- **{s['section']}**: {status}")
    valid = peek("validate", "valid")
    if valid is not None:
        lines.append(f"- validate.valid: {valid}")
    tag = peek("bracket", "bracket_tag")
    if tag is not None:
        lines.append(f"- bracket tag: {tag} (read the FIELDS, never the tag alone)")
    lines.append(
        "\nSimulation forced to v3 (commander_colors + tutor_aware). Audit the "
        "'Detected Card Classes' of the simulation section before reading its numbers."
    )

    data = {
        "commander": commander,
        "deck_size": len(decklist),
        "sections": sections,
        "failed_sections": failed,
    }
    return WorkflowResult(markdown="\n".join(lines), data=data)
