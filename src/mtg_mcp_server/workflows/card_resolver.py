"""Card resolver — bulk-data-first card resolution with Scryfall fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from mtg_mcp_server.services.scryfall import ScryfallClient
    from mtg_mcp_server.services.scryfall_bulk import ScryfallBulkClient
    from mtg_mcp_server.types import Card

log = structlog.get_logger(service="workflow.card_resolver")


async def resolve_card(
    name: str,
    *,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
) -> Card:
    """Resolve a card using bulk data first, falling back to Scryfall.

    When bulk data is available, tries it first for rate-limit-free lookup.
    Falls back to Scryfall if the card is not found in bulk data or if
    bulk data is disabled.

    Args:
        name: Card name to look up.
        bulk: Initialized ScryfallBulkClient, or None if disabled.
        scryfall: Initialized ScryfallClient (always available).

    Returns:
        A Card object (from bulk data or Scryfall — same type either way).

    Raises:
        CardNotFoundError: If the card is not found in any source.
    """
    if bulk is not None:
        try:
            card = await bulk.get_card(name)
        except Exception:
            log.warning("resolve_card.bulk_error", name=name, exc_info=True)
            card = None

        if card is not None:
            log.debug("resolve_card.bulk_hit", name=name)
            return card
        log.debug("resolve_card.bulk_miss", name=name)

    log.debug("resolve_card.scryfall_lookup", name=name)
    return await scryfall.get_card_by_name(name)


async def resolve_cards(
    names: list[str],
    *,
    bulk: ScryfallBulkClient | None,
    scryfall: ScryfallClient,
) -> tuple[dict[str, Card], list[str]]:
    """Resolve many cards at once: bulk data first, then one batched Scryfall call.

    Always prefer this over looping :func:`resolve_card`. Per-card lookups are
    serialized by the client's rate limiter, and a deck's worth of them trips
    Scryfall's "less than 10 requests per second" limit — HTTP 429 with a
    60-second cooldown and a network-block warning. A 99-card deck of bulk
    misses took 217s that way; batched it costs 2 requests.

    Args:
        names: Card names to resolve. Duplicates are collapsed.
        bulk: Initialized ScryfallBulkClient, or None if disabled.
        scryfall: Initialized ScryfallClient (always available).

    Returns:
        A ``(cards_by_name, unresolved)`` tuple. ``cards_by_name`` is keyed by
        lowercase name, indexed under both the requested spelling and the
        canonical one Scryfall returned. ``unresolved`` lists names no source
        could resolve.
    """
    unique_names = list(dict.fromkeys(names))
    cards_by_name: dict[str, Card] = {}
    misses: list[str] = []

    for name in unique_names:
        card = None
        if bulk is not None:
            try:
                card = await bulk.get_card(name)
            except Exception:
                log.warning("resolve_cards.bulk_error", name=name, exc_info=True)
        if card is None:
            misses.append(name)
        else:
            cards_by_name[name.lower()] = card

    if not misses:
        return cards_by_name, []

    try:
        found, unresolved = await scryfall.get_cards_collection(misses)
    except Exception as exc:  # a dead backend degrades to "unresolved", never a hard fail
        log.warning(
            "resolve_cards.batch_failed",
            count=len(misses),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return cards_by_name, list(misses)

    # Scryfall resolves loose spellings, so Card.name may differ from the name
    # we asked for. Index both so lookup by requested name still hits.
    not_found = {name.lower() for name in unresolved}
    requested = [name for name in misses if name.lower() not in not_found]
    for name, card in zip(requested, found, strict=False):
        cards_by_name[name.lower()] = card
        cards_by_name[card.name.lower()] = card

    if unresolved:
        log.warning("resolve_cards.unresolved", cards=unresolved, count=len(unresolved))

    return cards_by_name, unresolved
