"""Scryfall MCP provider — card search, lookup, pricing, rulings, and set info."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

import structlog
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.lifespan import lifespan
from fastmcp.tools import ToolResult
from pydantic import Field

from mtg_mcp_server.config import Settings
from mtg_mcp_server.providers import (
    ATTRIBUTION_SCRYFALL,
    TAGS_LOOKUP,
    TAGS_PRICING,
    TAGS_SEARCH,
    TOOL_ANNOTATIONS,
    format_legalities,
)
from mtg_mcp_server.services.scryfall import CardNotFoundError, ScryfallClient, ScryfallError
from mtg_mcp_server.utils.formatters import ResponseFormat, format_card_detail, format_card_line
from mtg_mcp_server.utils.query_sanitize import (
    bare_unreleased_query,
    escaping_warning,
    has_legality_filter,
    normalize_query,
    unreleased_probe_query,
    unreleased_warning,
)
from mtg_mcp_server.utils.slim import slim_card
from mtg_mcp_server.utils.triggers import derive_trigger
from mtg_mcp_server.utils.unreleased import FORMAT_FILTER_CAVEAT, unreleased_param_note

# Module-level client set by the lifespan. This pattern is required because
# FastMCP's Depends()/lifespan_context DI doesn't propagate through mount().
_client: ScryfallClient | None = None


@lifespan
async def scryfall_lifespan(server: FastMCP):
    """Manage the ScryfallClient lifecycle.

    Create the client once at startup using settings (honoring env var overrides),
    keep the httpx connection pool open for the server's lifetime, then tear down.
    """
    global _client
    settings = Settings()
    client = ScryfallClient(
        base_url=settings.scryfall_base_url,
        rate_limit_rps=1000 / settings.scryfall_rate_limit_ms,
    )
    async with client:
        _client = client
        yield {}
    _client = None


scryfall_mcp = FastMCP("Scryfall", lifespan=scryfall_lifespan, mask_error_details=True)

log = structlog.get_logger(provider="scryfall")

# Deadline for the unreleased-cards probe, well under the shared client's 30s x 3 retries.
# The probe only enriches a warning: losing it costs a hint, waiting on it costs the answer.
_PROBE_TIMEOUT_S = 8.0


def _get_client() -> ScryfallClient:
    """Return the initialized client or raise if the lifespan hasn't started."""
    if _client is None:
        raise RuntimeError("ScryfallClient not initialized — server lifespan not running")
    return _client


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_SEARCH)
async def search_cards(
    query: Annotated[
        str,
        Field(
            description="Scryfall search query (e.g. 'f:commander id:sultai t:creature cmc<=3'). See scryfall.com/docs/syntax"
        ),
    ],
    page: Annotated[int, Field(description="Page number for paginated results, 1-indexed")] = 1,
    limit: Annotated[
        int,
        Field(description="Max cards to return (default 30, 0 for all)"),
    ] = 30,
    response_format: Annotated[
        ResponseFormat,
        Field(description="Output verbosity: 'detailed' (default) or 'concise'"),
    ] = "detailed",
) -> ToolResult:
    """Search for Magic cards using Scryfall syntax.

    Examples: "f:commander id:sultai t:creature", "o:destroy t:instant cmc<=3"
    See https://scryfall.com/docs/syntax for full syntax reference.
    """
    if limit < 0:
        raise ToolError(f"limit must be >= 0 (0 for all), got {limit}")

    # Some clients HTML-escape the comparison operators on the way out, turning `mv<=3`
    # into `mv&lt;=3`. Scryfall 404s on that, and a bare "No cards found" is indis-
    # tinguable from a valid query with no matches — so we decode it AND say so.
    sent_query, was_escaped = normalize_query(query)

    client = _get_client()
    try:
        result = await client.search_cards(sent_query, page=page)
    except CardNotFoundError as exc:
        raise ToolError(
            f"No cards found for query: '{sent_query}'. "
            f"This means zero matches, not necessarily bad syntax — "
            f"widen the query rather than assuming the cards do not exist. "
            f"(Sent to Scryfall verbatim: {sent_query!r})"
        ) from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    cards = result.data if limit == 0 else result.data[:limit]
    showing = len(cards)
    total = len(result.data)

    # A legality filter hides everything from unreleased sets. Say WHICH cards, by name:
    # the caller cannot act on a count, and a generic caveat stops being read.
    # UTC, not local time: the server's timezone must not shift the release-date boundary
    # by a day relative to the dates Scryfall publishes.
    legality_filtered = has_legality_filter(sent_query)
    unreleased: list[str] | None = None
    unreleased_total: int | None = None
    probe = unreleased_probe_query(sent_query, datetime.now(UTC).date().isoformat())
    if legality_filtered and probe is not None:
        unreleased, unreleased_total = await _probe_unreleased(client, probe)

    lines = [f"Found {result.total_cards} cards (showing {showing} of {total}, page {page}):"]
    if unreleased:
        lines.insert(0, unreleased_warning(unreleased, probe or "", total=unreleased_total))
    if was_escaped:
        lines.insert(0, escaping_warning(query, sent_query))
    for card in cards:
        lines.append(format_card_line(card, response_format=response_format))
    if showing < total:
        lines.append(f"\n{total - showing} more on this page — increase limit to see them.")
    if result.has_more:
        lines.append(f"More results available — use page={page + 1}")
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content={
            # `query_sent` is the string Scryfall actually received, after
            # normalisation. It is present on EVERY response, not only when
            # normalisation changed something: a zero-result is only interpretable
            # next to the query that produced it.
            "query": sent_query,
            "query_sent": sent_query,
            "query_received": query,
            "query_was_escaped": was_escaped,
            # `unreleased_excluded` is None when no probe ran, and a list (possibly empty)
            # when one did. None and [] are NOT the same claim: only [] means "checked,
            # found nothing". No probe runs when there is no legality filter, but ALSO
            # when the filter was the whole query or what remained was unbalanced, so
            # `legality_filter_detected: true` with `unreleased_excluded: null` is a real
            # and meaningful combination: a filter was seen, nothing could be checked.
            "legality_filter_detected": legality_filtered,
            "unreleased_excluded": unreleased,
            "total_cards": result.total_cards,
            "page": page,
            "has_more": result.has_more,
            "showing": showing,
            "card_detail_uri_template": "mtg://card/{name}",
            "cards": [slim_card(card) for card in cards],
        },
    )


async def _probe_unreleased(client: ScryfallClient, probe: str) -> tuple[list[str], int | None]:
    """Names (first page) and Scryfall's total for `probe`, or ([], None) on any failure.

    Best-effort by design: this is a warning path, so a probe that fails must never
    take down a search that succeeded.

    GOTCHA(2026-07-30): the catch is deliberately broad. Catching only the two Scryfall
    exceptions was not enough: anything the client does not wrap (a transport error, a
    validation error on an odd payload) escaped and killed a search that had ALREADY
    succeeded, turning a nice-to-have warning into an outage. Caught by the existing
    provider tests on the very first run.

    GOTCHA(2026-07-30): the probe gets its OWN deadline. The shared client retries three
    times with a 30s timeout each, so an unlucky probe could delay a result already in hand
    by ~90s and blow the caller's own timeout. A warning is never worth losing the answer,
    hence `_PROBE_TIMEOUT_S`.

    The name list is one page; `total_cards` is the real match count. Both are returned so
    the message can report the count it actually measured instead of the page size.
    """
    try:
        found = await asyncio.wait_for(client.search_cards(probe, page=1), timeout=_PROBE_TIMEOUT_S)
    except TimeoutError:
        # WARNING, not debug: a probe timing out on every call means the safety net is
        # silently gone. Debug is off in production, so a debug-only log would hide it.
        log.warning("unreleased_probe_timeout", probe=probe, timeout_s=_PROBE_TIMEOUT_S)
        return [], None
    except Exception as exc:
        log.warning("unreleased_probe_failed", probe=probe, error=str(exc))
        return [], None
    return [card.name for card in found.data], found.total_cards


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_LOOKUP)
async def card_details(
    name: Annotated[
        str,
        Field(description="Card name — exact match by default (e.g. 'Muldrotha, the Gravetide')"),
    ],
    fuzzy: Annotated[
        bool,
        Field(
            description="Use fuzzy matching for approximate names (e.g. 'muldrotha' finds 'Muldrotha, the Gravetide')"
        ),
    ] = False,
    response_format: Annotated[
        ResponseFormat,
        Field(description="Output verbosity: 'detailed' (default) or 'concise'"),
    ] = "detailed",
) -> ToolResult:
    """Get full details for a Magic card by exact or fuzzy name."""
    client = _get_client()
    try:
        card = await client.get_card_by_name(name, fuzzy=fuzzy)
    except CardNotFoundError as exc:
        raise ToolError(f"Card not found: '{name}'. Check spelling or try fuzzy=true.") from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    lines = format_card_detail(card, response_format=response_format)

    # How often the ability fires, derived from the oracle text. "Whenever a Ninja you
    # control deals combat damage" multiplies per attacker; "whenever one or more..."
    # does not. Nothing in the card data distinguishes them, and reading them as the
    # same thing changes a card's value outright.
    trigger = derive_trigger(card)
    if response_format == "detailed":
        lines.append(f"Legalities: {format_legalities(card.legalities)}")
        if trigger.scope != "static":
            lines.append(f"Trigger scope: {trigger.scope} (condition: {trigger.condition})")
            if trigger.notes:
                lines.append(trigger.notes)
        lines.append(f"Scryfall: {card.scryfall_uri}")

    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content={
            **card.model_dump(mode="json"),
            "trigger_scope": trigger.scope,
            "trigger_condition": trigger.condition,
            "trigger_sources": trigger.sources,
            "trigger_notes": trigger.notes,
        },
    )


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_PRICING)
async def card_price(
    name: Annotated[str, Field(description="Card name for price lookup (exact match)")],
) -> ToolResult:
    """Get current prices for a Magic card. Prices update once per day."""
    client = _get_client()
    try:
        card = await client.get_card_by_name(name)
    except CardNotFoundError as exc:
        raise ToolError(f"Card not found: '{name}'. Check spelling.") from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    lines = [f"**{card.name}** — Prices"]
    if card.prices.usd:
        lines.append(f"  USD: ${card.prices.usd}")
    if card.prices.usd_foil:
        lines.append(f"  USD (foil): ${card.prices.usd_foil}")
    if card.prices.eur:
        lines.append(f"  EUR: \u20ac{card.prices.eur}")
    if not any([card.prices.usd, card.prices.usd_foil, card.prices.eur]):
        lines.append("  No price data available.")
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content={
            "name": card.name,
            "prices": card.prices.model_dump(mode="json"),
        },
    )


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_LOOKUP)
async def card_rulings(
    name: Annotated[str, Field(description="Card name to get official rulings for (exact match)")],
) -> ToolResult:
    """Get official rulings and clarifications for a Magic card."""
    client = _get_client()
    try:
        card = await client.get_card_by_name(name)
        rulings = await client.get_rulings(card.id)
    except CardNotFoundError as exc:
        raise ToolError(f"Card not found: '{name}'. Check spelling.") from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    rulings_data = [r.model_dump(mode="json") for r in rulings]
    if not rulings:
        return ToolResult(
            content=f"**{card.name}** — No rulings available." + ATTRIBUTION_SCRYFALL,
            structured_content={"name": card.name, "total_rulings": 0, "rulings": []},
        )

    lines = [f"**{card.name}** — {len(rulings)} ruling(s):"]
    for ruling in rulings:
        lines.append(f"  [{ruling.published_at}] {ruling.comment}")
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content={
            "name": card.name,
            "total_rulings": len(rulings),
            "rulings": rulings_data,
        },
    )


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_LOOKUP)
async def set_info(
    set_code: Annotated[
        str,
        Field(description="Set code (e.g. 'dom', 'mh2', 'lci')"),
    ],
) -> ToolResult:
    """Get metadata for a Magic set by its code."""
    client = _get_client()
    try:
        info = await client.get_set(set_code)
    except CardNotFoundError as exc:
        raise ToolError(f"Set not found: '{set_code}'. Check the set code.") from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    lines = [
        f"**{info.name}** ({info.code.upper()})",
        f"Type: {info.set_type}",
    ]
    if info.released_at:
        lines.append(f"Released: {info.released_at}")
    lines.append(f"Card count: {info.card_count}")
    if info.digital:
        lines.append("Digital-only set")
    if info.scryfall_uri:
        lines.append(f"Scryfall: {info.scryfall_uri}")
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content=info.model_dump(mode="json"),
    )


@scryfall_mcp.tool(annotations=TOOL_ANNOTATIONS, tags=TAGS_SEARCH)
async def whats_new(
    days: Annotated[
        int,
        Field(description="Look back this many days for recent cards (minimum 1)"),
    ] = 30,
    set_code: Annotated[
        str | None,
        Field(description="Filter to a specific set code (e.g. 'mh3', 'lci')"),
    ] = None,
    format: Annotated[
        str | None,
        Field(
            description="Filter to cards legal in a format (e.g. 'standard', 'commander', 'modern')."
            + FORMAT_FILTER_CAVEAT
        ),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Max cards to return (default 30, 0 for all)"),
    ] = 30,
    response_format: Annotated[
        ResponseFormat,
        Field(description="Output verbosity: 'detailed' (default) or 'concise'"),
    ] = "detailed",
) -> ToolResult:
    """Find recently printed or released Magic cards.

    Searches Scryfall for cards released within the given number of days.
    Optionally filter by set or format legality.
    """
    if days < 1:
        raise ToolError("days must be at least 1.")
    if limit < 0:
        raise ToolError(f"limit must be >= 0 (0 for all), got {limit}")

    date_str = (date.today() - timedelta(days=days)).isoformat()
    query_parts = [f"date>={date_str}"]
    if set_code:
        query_parts.append(f"s:{set_code}")
    if format:
        query_parts.append(f"f:{format}")
    query = " ".join(query_parts)

    # A format filter hides everything upcoming: unreleased cards satisfy the recency
    # window but are not_legal everywhere until release day. The probe finds them by
    # name. With no set_code the query reduces to legality alone, where the generic
    # probe builder declines — but here "everything upcoming" IS the hidden answer.
    probe: str | None = None
    if format:
        today = datetime.now(UTC).date().isoformat()
        probe = unreleased_probe_query(query, today) or bare_unreleased_query(today)

    client = _get_client()
    unreleased: list[str] | None = None
    unreleased_total: int | None = None
    try:
        result = await client.search_cards(query)
    except CardNotFoundError as exc:
        if probe is not None:
            unreleased, unreleased_total = await _probe_unreleased(client, probe)
        if unreleased:
            # The most dangerous case: a bare "no cards found" error reads as "these
            # cards do not exist", when the truth is the format filter removed them.
            warning = unreleased_param_note(unreleased, format or "", total=unreleased_total)
            return ToolResult(
                content=(
                    f"Found 0 card(s) released in the last {days} day(s)"
                    + (f" for set '{set_code}'" if set_code else "")
                    + f" legal in '{format}'.\n\n{warning}"
                    + ATTRIBUTION_SCRYFALL
                ),
                structured_content={
                    "days": days,
                    "set_code": set_code,
                    "format": format,
                    "total_cards": 0,
                    "showing": 0,
                    "has_more": False,
                    "unreleased_excluded": unreleased,
                    "cards": [],
                },
            )
        raise ToolError(
            f"No new cards found in the last {days} day(s)"
            + (f" for set '{set_code}'" if set_code else "")
            + (f" in format '{format}'" if format else "")
            + "."
        ) from exc
    except ScryfallError as exc:
        raise ToolError(f"Scryfall API error: {exc}") from exc

    if probe is not None:
        unreleased, unreleased_total = await _probe_unreleased(client, probe)

    cards = result.data if limit == 0 else result.data[:limit]
    showing = len(cards)
    total = len(result.data)

    lines = [f"Found {result.total_cards} card(s) released in the last {days} day(s):"]
    warning = unreleased_param_note(unreleased or [], format or "", total=unreleased_total)
    if warning:
        lines.insert(0, warning)
    for card in cards:
        if response_format == "concise":
            set_label = card.set_code.upper() if card.set_code else ""
            lines.append(f"  {card.name} {card.mana_cost or ''} [{set_label}]")
        else:
            set_label = card.set_code.upper() if card.set_code else ""
            lines.append(f"  {card.name} {card.mana_cost or ''} — {card.type_line} [{set_label}]")
    if showing < total:
        lines.append(f"\n{total - showing} more on this page — increase limit to see them.")
    if result.has_more:
        lines.append(
            "\nMore results available — refine your search with set_code or format filters."
        )
    return ToolResult(
        content="\n".join(lines) + ATTRIBUTION_SCRYFALL,
        structured_content={
            "days": days,
            "set_code": set_code,
            "format": format,
            "total_cards": result.total_cards,
            "showing": showing,
            "has_more": result.has_more,
            # Same contract as scryfall_search_cards: None when no probe ran (no
            # format filter), a list — possibly empty — when one did.
            "unreleased_excluded": unreleased,
            "cards": [slim_card(card) for card in cards],
        },
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@scryfall_mcp.resource("mtg://set/{code}")
async def set_resource(code: str) -> str:
    """Set metadata as JSON."""
    client = _get_client()
    try:
        info = await client.get_set(code)
        return info.model_dump_json()
    except CardNotFoundError:
        log.debug("resource.set_not_found", code=code)
        return json.dumps({"error": f"Set not found: {code}"})
    except ScryfallError as exc:
        log.warning("resource.set_error", code=code, error=str(exc))
        return json.dumps({"error": f"Scryfall error: {exc}"})


@scryfall_mcp.resource("mtg://card/{name}")
async def card_resource(name: str) -> str:
    """Get card data as JSON by exact name."""
    client = _get_client()
    try:
        card = await client.get_card_by_name(name)
        return card.model_dump_json()
    except CardNotFoundError:
        log.debug("resource.card_not_found", name=name)
        return json.dumps({"error": f"Card not found: {name}"})
    except ScryfallError as exc:
        log.warning("resource.card_error", name=name, error=str(exc))
        return json.dumps({"error": f"Scryfall error: {exc}"})


@scryfall_mcp.resource("mtg://card/{name}/rulings")
async def card_rulings_resource(name: str) -> str:
    """Get card rulings as JSON by card name."""
    client = _get_client()
    try:
        card = await client.get_card_by_name(name)
        rulings = await client.get_rulings(card.id)
        return json.dumps([r.model_dump() for r in rulings])
    except CardNotFoundError:
        log.debug("resource.rulings_not_found", name=name)
        return json.dumps({"error": f"Card not found: {name}"})
    except ScryfallError as exc:
        log.warning("resource.rulings_error", name=name, error=str(exc))
        return json.dumps({"error": f"Scryfall error: {exc}"})
