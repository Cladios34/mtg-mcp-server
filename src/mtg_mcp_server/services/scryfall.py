"""Scryfall API client for card search, lookup, and rulings.

Wraps Scryfall's REST API (https://scryfall.com/docs/api) with rate limiting,
retries, and Pydantic model parsing. All responses are cached via TTLCache.
"""

from __future__ import annotations

from cachetools import TTLCache

from mtg_mcp_server.services.base import DEFAULT_USER_AGENT, BaseClient, ServiceError
from mtg_mcp_server.services.cache import _method_key, async_cached
from mtg_mcp_server.types import Card, CardSearchResult, Ruling, SetInfo

# Scryfall's documented ceiling for POST /cards/collection identifiers.
_COLLECTION_CHUNK = 75


class ScryfallError(ServiceError):
    """Scryfall API error."""


class CardNotFoundError(ScryfallError):
    """Card was not found on Scryfall."""


class ScryfallClient(BaseClient):
    """Async client for the Scryfall REST API.

    Requires ``User-Agent`` and ``Accept`` headers on every request (set by
    BaseClient). Rate limited just under Scryfall's 10 req/sec ceiling.

    To look up many cards, use :meth:`get_cards_collection` rather than looping
    over :meth:`get_card_by_name`: the base client serializes requests, so a
    deck's worth of individual lookups is both slow and rate-limit bait.

    Args:
        base_url: Scryfall API base URL.
        rate_limit_rps: Max requests per second.
        user_agent: User-Agent header value.
    """

    # Class-level caches shared across all instances.
    # Card data rarely changes (24h TTL); searches rotate faster (1h TTL).
    _card_name_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)  # 24h
    _card_id_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)  # 24h
    _search_cache: TTLCache = TTLCache(maxsize=100, ttl=3600)  # 1h
    _rulings_cache: TTLCache = TTLCache(maxsize=200, ttl=86400)  # 24h
    _sets_cache: TTLCache = TTLCache(maxsize=50, ttl=86400)  # 24h

    def __init__(
        self,
        base_url: str = "https://api.scryfall.com",
        # 8.3 req/s. Scryfall requires "less than 10 requests per second"; sitting
        # exactly on 10.0 drew HTTP 429s carrying a 60s cooldown and a
        # network-block warning (incident 2026-07-27).
        rate_limit_rps: float = 8.3,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        super().__init__(
            base_url=base_url,
            rate_limit_rps=rate_limit_rps,
            user_agent=user_agent,
        )

    @async_cached(_card_name_cache, key=_method_key)
    async def get_card_by_name(self, name: str, *, fuzzy: bool = False) -> Card:
        """Look up a card by exact or fuzzy name.

        Args:
            name: Card name to search for.
            fuzzy: If True, use Scryfall's fuzzy matching instead of exact.

        Returns:
            Parsed Card model.

        Raises:
            CardNotFoundError: If no card matches the name.
            ScryfallError: On other API errors.
        """
        param_key = "fuzzy" if fuzzy else "exact"
        try:
            response = await self._get("/cards/named", params={param_key: name})
        except ServiceError as exc:
            if exc.status_code == 404:
                raise CardNotFoundError(f"Card not found: '{name}'", status_code=404) from exc
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return Card.model_validate(response.json())

    async def get_cards_collection(self, names: list[str]) -> tuple[list[Card], list[str]]:
        """Look up many cards by name in as few requests as possible.

        Scryfall accepts up to ``_COLLECTION_CHUNK`` identifiers per request, so
        a 99-card deck costs 2 requests instead of 99. Use this instead of looping
        over :meth:`get_card_by_name`: serialized per-card lookups trip Scryfall's
        "less than 10 requests per second" limit, which answers HTTP 429 with a
        60-second cooldown and a network-block warning.

        Args:
            names: Card names to resolve. Order is preserved in the results.

        Returns:
            A ``(found, missing)`` tuple: parsed Cards, and the names Scryfall
            reported under ``not_found``.

        Raises:
            ScryfallError: On API errors.
        """
        found: list[Card] = []
        missing: list[str] = []
        for start in range(0, len(names), _COLLECTION_CHUNK):
            chunk = names[start : start + _COLLECTION_CHUNK]
            try:
                response = await self._post(
                    "/cards/collection",
                    json={"identifiers": [{"name": name} for name in chunk]},
                )
            except ServiceError as exc:
                raise ScryfallError(exc.message, status_code=exc.status_code) from exc
            payload = response.json()
            found.extend(Card.model_validate(c) for c in payload.get("data", []))
            # not_found echoes the identifiers we sent, so read back the "name" key.
            missing.extend(
                item["name"] for item in payload.get("not_found", []) if item.get("name")
            )
        return found, missing

    @async_cached(_search_cache, key=_method_key)
    async def search_cards(self, query: str, page: int = 1) -> CardSearchResult:
        """Search for cards using Scryfall syntax.

        Args:
            query: Scryfall search query (e.g. ``"f:commander id:sultai"``).
            page: Results page number (1-indexed).

        Returns:
            Paginated search results with card list and ``has_more`` flag.

        Raises:
            CardNotFoundError: If no cards match the query (404).
            ScryfallError: On other API errors.
        """
        try:
            response = await self._get("/cards/search", params={"q": query, "page": str(page)})
        except ServiceError as exc:
            if exc.status_code == 404:
                raise CardNotFoundError(
                    f"No cards found for query: '{query}'", status_code=404
                ) from exc
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return CardSearchResult.model_validate(response.json())

    @async_cached(_card_id_cache, key=_method_key)
    async def get_card_by_id(self, scryfall_id: str) -> Card:
        """Look up a card by Scryfall UUID.

        Args:
            scryfall_id: Scryfall card UUID.

        Returns:
            Parsed Card model.

        Raises:
            CardNotFoundError: If the UUID does not exist.
            ScryfallError: On other API errors.
        """
        try:
            response = await self._get(f"/cards/{scryfall_id}")
        except ServiceError as exc:
            if exc.status_code == 404:
                raise CardNotFoundError(
                    f"Card not found: '{scryfall_id}'", status_code=404
                ) from exc
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return Card.model_validate(response.json())

    @async_cached(_rulings_cache, key=_method_key)
    async def get_rulings(self, scryfall_id: str) -> list[Ruling]:
        """Get official rulings for a card by Scryfall UUID.

        Args:
            scryfall_id: Scryfall card UUID.

        Returns:
            List of rulings with source, date, and comment text.

        Raises:
            ScryfallError: On API errors.
        """
        try:
            response = await self._get(f"/cards/{scryfall_id}/rulings")
        except ServiceError as exc:
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        data = response.json()
        return [Ruling.model_validate(r) for r in data["data"]]

    @async_cached(_sets_cache, key=_method_key)
    async def get_sets(self) -> list[SetInfo]:
        """List all Magic sets.

        Returns:
            List of SetInfo models.

        Raises:
            ScryfallError: On API errors.
        """
        try:
            response = await self._get("/sets")
        except ServiceError as exc:
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return [SetInfo.model_validate(s) for s in response.json()["data"]]

    @async_cached(_sets_cache, key=_method_key)
    async def get_set(self, set_code: str) -> SetInfo:
        """Get set metadata by code.

        Args:
            set_code: Set code (e.g. "dom", "mh2").

        Returns:
            Parsed SetInfo model.

        Raises:
            CardNotFoundError: If the set code doesn't exist.
            ScryfallError: On other API errors.
        """
        try:
            response = await self._get(f"/sets/{set_code}")
        except ServiceError as exc:
            if exc.status_code == 404:
                raise CardNotFoundError(f"Set not found: '{set_code}'", status_code=404) from exc
            raise ScryfallError(exc.message, status_code=exc.status_code) from exc
        return SetInfo.model_validate(response.json())
