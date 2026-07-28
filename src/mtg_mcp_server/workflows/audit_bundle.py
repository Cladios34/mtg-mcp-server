"""Composite Commander deck audit — the whole mechanical battery in one call.

Runs deck validation, deck analysis, Spellbook decklist combos, bracket
estimation, commander rulings, and the v3 opening-hand simulation CONCURRENTLY
(transport concurrency verified by ``scripts/parallel_probe.mjs``), and returns
a single report where EVERY section carries an explicit ``ok``/``error`` status
and echoes the parameters it actually used.

Design requirements:

- Silent-failure ban (schema-drift bug family, 2026-07-22): a section that
  fails must say so loudly in the report — never a silent zero or missing key
  next to rich data from other sections.
- Forced v3 simulation: ``commander_colors`` is a required argument and
  ``tutor_aware`` is always on, so a color-blind simulation cannot happen by
  omission (the exact incident this bundle exists to prevent).
- Rulings shipped with the audit (2026-07-27): they used to sit behind a
  separate, skippable call. Three rules errors in a single audit all came
  through that gap. A section that is always present cannot be forgotten.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from mtg_mcp_server.utils.decklist import parse_decklist
from mtg_mcp_server.utils.mechanics import EVERGREEN_KEYWORDS, card_keywords, carries_keyword
from mtg_mcp_server.workflows import WorkflowResult

if TYPE_CHECKING:
    from mtg_mcp_server.services.edhrec import EDHRECClient
    from mtg_mcp_server.services.scryfall import ScryfallClient
    from mtg_mcp_server.services.scryfall_bulk import ScryfallBulkClient
    from mtg_mcp_server.services.spellbook import SpellbookClient
    from mtg_mcp_server.types import Card
from mtg_mcp_server.workflows.analysis import deck_analysis
from mtg_mcp_server.workflows.card_resolver import resolve_cards
from mtg_mcp_server.workflows.simulation import simulate_opening_hands
from mtg_mcp_server.workflows.validation import deck_validate

Section = dict[str, Any]

# Rulings cost one serialised Scryfall request each. Beyond this many carriers the
# bundle names what it skipped instead of paying for it.
_SIGNATURE_RULINGS_CAP = 5


async def _guard(name: str, params: Mapping[str, Any], coro: Awaitable[Any]) -> Section:
    """Run one section, converting ANY failure into an explicit error entry.

    Every section reports ``elapsed_ms`` so a slow backend is attributable from
    the response alone, without server-side profiling.
    """
    start = time.perf_counter()
    try:
        data = await coro
    except Exception as exc:  # every failure must surface in the report, none may escape
        return {
            "section": name,
            "ok": False,
            "params_used": dict(params),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": round((time.perf_counter() - start) * 1000),
        }
    return {
        "section": name,
        "ok": True,
        "params_used": dict(params),
        "data": data,
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
    }


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

    # The bundle needs each Spellbook payload three times over (combos section,
    # bracket section, and deck_analysis internally). The Spellbook client is
    # capped at 3 req/s behind a single-slot semaphore, so duplicate calls
    # SERIALIZE: measured 2026-07-27, the redundant pair pushed analysis to
    # 8.7s waiting its turn. Fetch each once and share the awaitable.
    bracket_task = asyncio.ensure_future(spellbook.estimate_bracket([commander], decklist))
    combos_task = asyncio.ensure_future(spellbook.find_decklist_combos([commander], decklist))

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
            bracket_coro=asyncio.shield(bracket_task),
            combo_coro=asyncio.shield(combos_task),
        )
        return result.data

    def _slim_combo(combo: Any) -> dict[str, Any]:
        # Full model_dump carries step-by-step description, prerequisites, legalities
        # and prices PER combo — >100KB total on a real 99-card deck, tripping the
        # ResponseLimitingMiddleware (seen live 2026-07-24 on the Duplicata deck).
        # Step-by-step details stay one call away: spellbook_combo_details(id).
        return {
            "id": combo.id,
            "cards": [c.name for c in combo.cards],
            "results": [r.feature_name for r in combo.produces],
            "identity": combo.identity,
            "mana_needed": combo.mana_needed,
            "bracket_tag": combo.bracket_tag,
            "popularity": combo.popularity,
        }

    async def _run_combos() -> Any:
        combos = await asyncio.shield(combos_task)
        return {
            "identity": combos.identity,
            "included": [_slim_combo(c) for c in combos.included],
            "almost_included": [_slim_combo(c) for c in combos.almost_included],
            "note": "step-by-step d'un combo : spellbook_combo_details(id)",
        }

    async def _run_bracket() -> Any:
        estimate = await asyncio.shield(bracket_task)
        # Keep the fields the bracket gate actually reads (deck-lab step-06);
        # drop the raw per-card/per-combo classified payloads (size).
        return {
            "bracket_tag": estimate.bracket_tag,
            "bracket_tag_name": estimate.bracket_tag_name,
            "banned_cards": estimate.banned_cards,
            "game_changer_cards": estimate.game_changer_cards,
            "mass_land_denial_cards": estimate.mass_land_denial_cards,
            "extra_turn_cards": estimate.extra_turn_cards,
            "two_card_combos": estimate.two_card_combos,
            "lock_combos": estimate.lock_combos,
        }

    def _signature_keyword(commander_card: Card) -> str | None:
        """The commander's defining keyword, if it has one.

        Evergreen combat keywords are skipped: "Flying" says nothing about how a deck
        operates, so pulling rulings for every flier would be noise crowding out the
        rulings that matter.
        """
        for keyword in sorted(card_keywords(commander_card)):
            if keyword.lower() not in EVERGREEN_KEYWORDS:
                return keyword
        return None

    async def _signature_mechanic_rulings(
        commander_card: Card,
    ) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
        """Rulings for the deck's cards that share the commander's signature keyword.

        Bounded to ``_SIGNATURE_RULINGS_CAP`` cards: rulings are one Scryfall request
        each and the client serialises them. Whatever is left out is NAMED in the
        second return value — a silent cap reads as "there was nothing else".
        """
        keyword = _signature_keyword(commander_card)
        if keyword is None:
            return {}, []

        cards_by_name, _ = await resolve_cards(decklist, bulk=bulk, scryfall=scryfall)
        carriers: list[Card] = []
        seen: set[str] = set()
        for card in cards_by_name.values():
            if card.name in seen or card.name == commander_card.name:
                continue
            if carries_keyword(card, keyword):
                seen.add(card.name)
                carriers.append(card)

        carriers.sort(key=lambda c: c.name)
        fetched = carriers[:_SIGNATURE_RULINGS_CAP]
        omitted = [c.name for c in carriers[_SIGNATURE_RULINGS_CAP:]]

        out: dict[str, list[dict[str, str]]] = {}
        for card in fetched:
            try:
                rulings = await scryfall.get_rulings(card.id)
            except Exception as exc:  # one dead lookup must not sink the section
                out[card.name] = [{"published_at": "", "comment": f"lookup failed: {exc}"}]
                continue
            out[card.name] = [
                {"published_at": str(r.published_at), "comment": r.comment} for r in rulings
            ]
        return out, omitted

    async def _run_rulings() -> Any:
        """Official rulings for the commander, plus its oracle text.

        Added 2026-07-27. Rulings used to be reachable only through a separate,
        easily-skipped call, and three rules errors in one audit all came through
        that gap: a commander's signature mechanic was reasoned about from memory
        while the ruling that contradicted it sat unread in an earlier tool
        response. Shipping them WITH the audit removes the option of not looking.

        The oracle text rides along because a ruling is unusable without the text
        it annotates.
        """
        card = await scryfall.get_card_by_name(commander)
        rulings = await scryfall.get_rulings(card.id)

        signature, omitted = await _signature_mechanic_rulings(card)

        return {
            "card": card.name,
            "oracle_text": card.oracle_text,
            "total_rulings": len(rulings),
            "rulings": [
                {"published_at": str(r.published_at), "comment": r.comment} for r in rulings
            ],
            "signature_mechanic": _signature_keyword(card),
            "signature_mechanic_cards": signature,
            "signature_mechanic_not_fetched": omitted,
            "note": (
                "CONFRONT each ruling with your own claims about this commander — "
                "retrieving them is not reading them. Rulings for other cards: "
                "scryfall_card_rulings(name)."
            ),
        }

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

    async def _check_commander() -> bool | None:
        """True/False if the commander name resolves, None if the check itself broke.

        A typo'd commander otherwise yields a confident-looking report whose
        commander-keyed data (EDHREC synergy, combos) is quietly empty.
        """
        from mtg_mcp_server.workflows import card_resolver

        try:
            _, unresolved = await card_resolver.resolve_cards(
                [commander], bulk=bulk, scryfall=scryfall
            )
        except Exception:  # never let a name check take the audit down
            return None
        return not unresolved

    if bulk is not None:
        validate_coro = _run_validate(bulk)
    else:
        validate_coro = _fail("bulk data backend disabled (enable_bulk_data)")

    # The commander name check runs as its own task rather than inside the gather:
    # it returns a bool, and mixing it into the section gather makes the result a
    # union that no longer types as the uniform list the report iterates over. As a
    # task it still starts immediately, so nothing is serialised.
    commander_task = asyncio.create_task(_check_commander())
    section_coros: list[Awaitable[Section]] = [
        _guard("validate", {"format": "commander", "commander": commander}, validate_coro),
        _guard("analysis", {"commander": commander, "response_format": "concise"}, _run_analysis()),
        _guard("combos", {"commanders": [commander]}, _run_combos()),
        _guard("bracket", {"commanders": [commander]}, _run_bracket()),
        _guard("rulings", {"card": commander}, _run_rulings()),
        _guard("simulation", sim_params, _run_simulation()),
    ]
    try:
        sections: list[Section] = list(await asyncio.gather(*section_coros))
    except BaseException:
        # A client disconnect cancels the gather. Without this, commander_task keeps
        # running detached, issuing HTTP the caller will never read and surfacing as
        # "Task exception was never retrieved".
        commander_task.cancel()
        raise
    commander_resolved: bool | None = await commander_task

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
    if commander_resolved is False:
        lines.append(
            f"- **WARNING**: commander '{commander}' was not found in any card source. "
            "Check the spelling: commander-keyed data (EDHREC synergy, combos, bracket) "
            "is meaningless for a name that does not exist."
        )
    for s in sections:
        status = "OK" if s["ok"] else f"FAILED — {s['error']}"
        lines.append(f"- **{s['section']}**: {status}")
    valid = peek("validate", "valid")
    if valid is not None:
        lines.append(f"- validate.valid: {valid}")
    tag = peek("bracket", "bracket_tag")
    if tag is not None:
        lines.append(f"- bracket tag: {tag} (read the FIELDS, never the tag alone)")
    n_rulings = peek("rulings", "total_rulings")
    if n_rulings:
        # Surfaced in the markdown, not only in structured_content: the failure mode
        # this section exists to stop is rulings that get FETCHED and never READ.
        lines.append(
            f"- rulings: {n_rulings} official ruling(s) on {commander} — read them before "
            "asserting anything about its signature mechanic, and put each one FACE TO FACE "
            "with your own claims."
        )
    elif n_rulings == 0:
        lines.append(f"- rulings: none published for {commander}")
    signature = peek("rulings", "signature_mechanic")
    if signature:
        carriers = peek("rulings", "signature_mechanic_cards") or {}
        skipped = peek("rulings", "signature_mechanic_not_fetched") or []
        lines.append(
            f"- signature mechanic '{signature}': rulings included for "
            f"{len(carriers)} carrier(s) in the list"
        )
        if skipped:
            lines.append(
                f"  - NOT fetched ({len(skipped)}, capped to keep the call cheap): "
                f"{', '.join(skipped)} — use scryfall_card_rulings on these before "
                f"asserting anything about them"
            )
    lines.append(
        "\nSimulation forced to v3 (commander_colors + tutor_aware). Audit the "
        "'Detected Card Classes' of the simulation section before reading its numbers."
    )

    data = {
        "commander": commander,
        "commander_resolved": commander_resolved,
        # Physical cards, not entries: "30 Plains" is 30 cards. len(decklist)
        # reported 3 for a quantity-style list while every section counted 35.
        "deck_size": sum(qty for qty, _ in parse_decklist(decklist)),
        "sections": sections,
        "failed_sections": failed,
    }
    return WorkflowResult(markdown="\n".join(lines), data=data)
