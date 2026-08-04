"""Scryfall bulk card data service — lazy download and in-memory card cache.

Download Scryfall's Oracle Cards bulk data once and keep it in memory for O(1)
card lookups and fast substring searches. Refresh automatically when the data
becomes stale (default 12h). On refresh failure with existing data, serve stale
data rather than propagating the error.

Unlike :class:`BaseClient`, this is a standalone service managing its own HTTP
downloads. It does not use rate limiting or retries — the bulk-data endpoint is
a single lightweight request and the bulk download is a large file fetch, neither
benefiting from the per-request rate-limit pattern that ``BaseClient`` provides.

Returns :class:`~mtg_mcp_server.types.Card` objects with prices, legalities,
EDHREC rank, and image URIs. Checks the ``/bulk-data/oracle_cards`` metadata
endpoint before download, supports ETag-based conditional downloads, uses
``asyncio.Lock`` to prevent duplicate concurrent downloads, and runs a
background refresh loop via ``asyncio.create_task()``.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import gzip
import io
import json
import random
import time
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import httpx
import structlog
from pydantic import ValidationError

from mtg_mcp_server.services.base import DEFAULT_USER_AGENT, ServiceError
from mtg_mcp_server.types import Card
from mtg_mcp_server.utils.mechanics import has_creature_type

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mtg_mcp_server.utils.unreleased import UnreleasedCollector

# Card types, as opposed to creature subtypes. A changeling is every creature type
# (702.73a) but it is not an Artifact, so these never pick up changelings.
_CARD_TYPES = frozenset(
    {
        "artifact",
        "battle",
        "creature",
        "enchantment",
        "instant",
        "kindred",
        "land",
        "planeswalker",
        "sorcery",
        "tribal",
        "legendary",
        "basic",
        "snow",
        "token",
    }
)

__all__ = ["ScryfallBulkClient", "ScryfallBulkDownloadError", "ScryfallBulkError"]

# How far a refresh may shrink the card pool before it is treated as corruption
# rather than as Scryfall having removed cards. Real drops are a handful of
# cards; a truncated or re-encoded payload loses most of them.
_MIN_REFRESH_RATIO = 0.5

log = structlog.get_logger(service="ScryfallBulkClient")

# Scryfall Oracle Cards includes non-playable entries (minigames, art series,
# tokens, emblems) that can share names with real cards. Filter these out during
# parsing. Uses a deny-list so new playable layouts are included by default.
_EXCLUDED_LAYOUTS: frozenset[str] = frozenset(
    {
        "art_series",
        "double_faced_token",
        "emblem",
        "minigame",
        "placeholder",
        "token",
    }
)


@dataclass
class _ParseStats:
    """Line counts a generator cannot return alongside the entries it yields."""

    read: int = 0
    malformed: int = 0


class ScryfallBulkError(ServiceError):
    """Base exception for Scryfall bulk data service errors.

    Always passes ``status_code=None`` since bulk data operations don't map
    cleanly to a single HTTP status.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None)


class ScryfallBulkDownloadError(ScryfallBulkError):
    """Error downloading Scryfall bulk data (network or HTTP failure)."""


class ScryfallBulkClient:
    """Manages Scryfall Oracle Cards bulk data with lazy download and refresh.

    Downloads the Oracle Cards JSON on first access and caches parsed
    :class:`Card` models in memory. Refreshes automatically when
    ``refresh_hours`` has elapsed, using ETag-based conditional downloads
    to avoid re-parsing unchanged data.

    Use as an async context manager::

        async with ScryfallBulkClient(base_url=...) as client:
            card = await client.get_card("Sol Ring")
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.scryfall.com",
        refresh_hours: int = 12,
    ) -> None:
        """Initialize the Scryfall bulk client.

        Args:
            base_url: Scryfall API base URL (for metadata endpoint).
            refresh_hours: Hours before re-downloading data.
        """
        self._base_url = base_url
        self._refresh_seconds = refresh_hours * 3600

        # _cards: lowercase-name -> Card for O(1) exact lookup.
        # _unique_cards: deduplicated list for linear substring search.
        # Separation avoids double-counting DFCs which have two keys
        # in _cards (front-face name + full "//" name) but one entry
        # in _unique_cards.
        self._cards: dict[str, Card] = {}
        self._unique_cards: list[Card] = []
        self._loaded_at: float = 0.0  # monotonic timestamp; 0 = never loaded

        # ETag for conditional download (HTTP 304 Not Modified)
        self._etag: str | None = None
        self._etag_url: str | None = None  # URL the ETag applies to

        # asyncio.Lock prevents duplicate concurrent downloads
        self._load_lock = asyncio.Lock()

        # Background refresh task
        self._refresh_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        """Enter async context. Data is loaded lazily on first access, not here."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Cancel background refresh and release in-memory data."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task
            self._refresh_task = None

        self._cards.clear()
        self._unique_cards.clear()
        self._loaded_at = 0.0

    def start_background_refresh(self, *, preload: bool = True) -> None:
        """Start the background refresh loop.

        Creates an ``asyncio.Task`` that periodically calls
        :meth:`ensure_loaded` at the refresh interval. The task is
        cancelled in :meth:`__aexit__`.

        Args:
            preload: Load immediately rather than after one refresh interval.
                Leaving this off means the first request that needs card data
                pays the download itself (~6.5s). Tests pass False so the
                suite never reaches the live API.
        """
        if self._refresh_task is not None:
            return  # Already running
        self._refresh_task = asyncio.create_task(self._refresh_loop(preload=preload))

    async def ensure_loaded(self) -> None:
        """Download and parse Oracle Cards if not loaded or stale.

        On first load failure, the error propagates (server cannot start
        without data). On **refresh** failure (data was previously loaded),
        logs a warning and serves stale data.

        Uses ``asyncio.Lock`` to prevent duplicate concurrent downloads.

        Raises:
            ScryfallBulkDownloadError: On first-load network/HTTP failure.
            ScryfallBulkError: On first-load parse failure.
        """
        if not self._is_stale():
            return

        async with self._load_lock:
            # Double-check after acquiring lock — another coroutine may have
            # already completed the load while we were waiting.
            if not self._is_stale():
                return

            # Whether this is a refresh is "do we already hold data", not the
            # sign of a timestamp. GOTCHA(2026-07-29): this read
            # ``self._loaded_at > 0``, and time.monotonic() counts from boot, so
            # on a host up for less than the refresh interval the timestamp is
            # negative and a refresh was misread as a first load — propagating
            # the error instead of serving the pool we already had. It surfaced
            # as a CI failure on one Python version only, because that runner
            # happened to have a shorter uptime.
            is_refresh = bool(self._cards)
            log.info("scryfall_bulk.loading", base_url=self._base_url, is_refresh=is_refresh)

            try:
                # Step 1: Fetch metadata to get the current download URL.
                # Scryfall renamed 'download_uri' to 'jsonl_download_uri' in
                # July 2026 when the payload moved to gzipped JSONL. The old key
                # is still accepted so a rollback upstream is not an outage here.
                metadata = await self._fetch_metadata()
                download_url = next(
                    (
                        value
                        for key in ("jsonl_download_uri", "download_uri")
                        if isinstance(value := metadata.get(key), str) and value
                    ),
                    None,
                )
                if download_url is None:
                    raise ScryfallBulkError(
                        "Bulk metadata has neither 'jsonl_download_uri' nor 'download_uri'. "
                        f"Keys: {list(metadata.keys())}"
                    )

                # Step 2: Download the bulk data (with ETag if URL matches)
                result = await self._download(download_url)

                if result is not None:
                    # Got new data — parse it
                    self._parse(result)

                # Either we parsed new data or got 304 — update timestamp
                self._loaded_at = time.monotonic()
                log.info(
                    "scryfall_bulk.loaded",
                    card_count=len(self._unique_cards),
                )
            except ScryfallBulkError:
                if is_refresh:
                    # Stale data is better than no data — retry in 5 min, not full interval
                    log.warning(
                        "scryfall_bulk.refresh_failed",
                        base_url=self._base_url,
                        exc_info=True,
                    )
                    self._loaded_at = time.monotonic() - self._refresh_seconds + 300
                    return
                raise

    async def get_card(self, name: str) -> Card | None:
        """Look up a card by exact name (case-insensitive).

        Args:
            name: Card name (front-face or full ``//`` name for DFCs).

        Returns:
            Card data, or None if not found.
        """
        await self.ensure_loaded()
        return self._cards.get(name.lower())

    async def search_cards(self, query: str, limit: int = 20) -> list[Card]:
        """Search cards by name substring (case-insensitive).

        Args:
            query: Substring to match against card names.
            limit: Maximum results to return.

        Returns:
            Matching cards, up to ``limit``.
        """
        await self.ensure_loaded()
        query_lower = query.lower()
        results: list[Card] = []
        for card in self._unique_cards:
            if query_lower in card.name.lower():
                results.append(card)
                if len(results) >= limit:
                    break
        return results

    async def search_by_type(
        self, type_query: str, limit: int = 20, *, include_changelings: bool = True
    ) -> list[Card]:
        """Search cards by type line substring (case-insensitive).

        Changelings are included by default. A changeling's type line says
        "Shapeshifter", but rule 702.73a makes it every creature type in every zone,
        so a plain type-line match silently drops it — which is exactly how a tribal
        count comes out short (observed 2026-07-27: a Ninja search returned 62 cards
        with both of the deck's changelings missing).

        Args:
            type_query: Substring to match against type lines.
            limit: Maximum results to return.
            include_changelings: Also return changelings when ``type_query`` is a
                creature subtype. Turned off, this reverts to a literal type-line
                match — correct only when you want printed types specifically.

        Returns:
            Matching cards, up to ``limit``.
        """
        await self.ensure_loaded()
        query_lower = type_query.lower()
        # Only subtypes get the changeling treatment: a changeling is every CREATURE
        # type, not an Artifact or a Land.
        changelings_apply = include_changelings and query_lower not in _CARD_TYPES

        results: list[Card] = []
        for card in self._unique_cards:
            type_line = card.type_line.lower()
            matched = query_lower in type_line
            # The changeling fallback runs the full rules check, including a regex
            # split of the oracle text. Gated on "creature" in the type line so a
            # miss costs one substring test, not a parse: over ~30k cards the
            # ungated version measured 138ms of blocking work against 5ms.
            if not matched and changelings_apply and "creature" in type_line:
                matched = has_creature_type(card, type_query).matches
            if matched:
                results.append(card)
                if len(results) >= limit:
                    break
        return results

    async def search_by_text(self, text_query: str, limit: int = 20) -> list[Card]:
        """Search cards by oracle text substring (case-insensitive).

        Args:
            text_query: Substring to match against oracle text.
            limit: Maximum results to return.

        Returns:
            Matching cards, up to ``limit``.
        """
        await self.ensure_loaded()
        query_lower = text_query.lower()
        results: list[Card] = []
        for card in self._unique_cards:
            oracle = card.oracle_text or ""
            if query_lower in oracle.lower():
                results.append(card)
                if len(results) >= limit:
                    break
        return results

    async def filter_cards(
        self,
        *,
        format: str | None = None,
        color_identity: frozenset[str] | None = None,
        type_contains: list[str] | None = None,
        text_contains: list[str] | None = None,
        text_any: list[str] | None = None,
        keywords: list[str] | None = None,
        cmc_eq: float | None = None,
        cmc_lte: float | None = None,
        max_price: float | None = None,
        rarity: str | None = None,
        name_contains: str | None = None,
        limit: int = 100,
        unreleased: UnreleasedCollector | None = None,
        include_unreleased: bool = False,
    ) -> list[Card]:
        """Multi-criteria filter over all cards. Returns up to ``limit`` matches.

        All filter parameters are optional. Only cards matching **all** specified
        criteria are returned (AND logic). ``text_any`` is the exception: it uses
        OR logic among its terms.

        Args:
            format: Only cards where ``legalities[format] == "legal"``.
            color_identity: Cards whose color identity is a subset of this set.
            type_contains: ALL strings must appear in the type line (case-insensitive).
            text_contains: ALL strings must appear in oracle text (case-insensitive).
            text_any: ANY string must appear in oracle text (case-insensitive).
            keywords: ALL must be in the card's keywords (case-insensitive).
            cmc_eq: Exact CMC match.
            cmc_lte: CMC at or below this value.
            max_price: Maximum USD price (cards without a USD price are excluded).
            rarity: Exact rarity match (e.g. ``"mythic"``).
            name_contains: Substring match on card name (case-insensitive).
            limit: Maximum results to return.
            unreleased: When given alongside ``format``, receives every card that
                matched all OTHER criteria but failed only the legality check because
                its set has not been released yet. Collecting here rather than at each
                call site keeps the probe's criteria identical to the search's by
                construction.
            include_unreleased: When True (and ``unreleased`` is given), those cards
                are ALSO kept in the results instead of being filtered out — the
                owner's default for discovery tools. Without a collector this flag
                has no effect: legality cannot be waived without knowing why it failed.

        Returns:
            Matching cards, up to ``limit``.
        """
        await self.ensure_loaded()

        # Pre-compute lowercase versions of string filters
        type_lower = [t.lower() for t in type_contains] if type_contains else None
        text_lower = [t.lower() for t in text_contains] if text_contains else None
        text_any_lower = [t.lower() for t in text_any] if text_any else None
        kw_lower = [k.lower() for k in keywords] if keywords else None
        name_lower = name_contains.lower() if name_contains else None

        results: list[Card] = []
        for card in self._unique_cards:
            # Results full: keep scanning only while unreleased hunting remains,
            # otherwise later unreleased matches would be silently missed.
            if len(results) >= limit and (unreleased is None or unreleased.full):
                break
            legal = format is None or card.legalities.get(format) == "legal"
            if not legal:
                # Only unreleased rejects are worth running the other criteria on.
                if unreleased is None or not unreleased.offer(card):
                    continue
            elif len(results) >= limit:
                continue
            if color_identity is not None and not frozenset(card.color_identity).issubset(
                color_identity
            ):
                continue
            if type_lower is not None:
                tl = card.type_line.lower()
                if not all(t in tl for t in type_lower):
                    continue
            if text_lower is not None:
                oracle = (card.oracle_text or "").lower()
                if not all(t in oracle for t in text_lower):
                    continue
            if text_any_lower is not None:
                oracle = (card.oracle_text or "").lower()
                if not any(t in oracle for t in text_any_lower):
                    continue
            if kw_lower is not None:
                card_kw = {k.lower() for k in card.keywords}
                if not all(k in card_kw for k in kw_lower):
                    continue
            if cmc_eq is not None and card.cmc != cmc_eq:
                continue
            if cmc_lte is not None and card.cmc > cmc_lte:
                continue
            if max_price is not None:
                if card.prices.usd is None:
                    continue
                try:
                    if float(card.prices.usd) > max_price:
                        continue
                except ValueError:
                    continue
            if rarity is not None and card.rarity != rarity:
                continue
            if name_lower is not None and name_lower not in card.name.lower():
                continue

            if not legal:
                if unreleased is not None:
                    unreleased.collect(card)
                # Included cards respect `limit` too; past it they are still collected
                # (and therefore named in the note) but no longer listed.
                if not include_unreleased or len(results) >= limit:
                    continue
            results.append(card)

        return results

    async def cards_by_legality(self, format: str, status: str) -> list[Card]:
        """All cards with a specific legality status in a format.

        Args:
            format: Format name (e.g. ``"commander"``, ``"modern"``).
            status: Legality status (e.g. ``"banned"``, ``"restricted"``, ``"legal"``).

        Returns:
            All matching cards (no limit).
        """
        await self.ensure_loaded()
        return [card for card in self._unique_cards if card.legalities.get(format) == status]

    async def all_cards(self) -> list[Card]:
        """All unique cards in the bulk data set.

        Ensures data is loaded before returning. This is the public
        interface for iterating over the full card pool — providers
        should use this instead of accessing ``_unique_cards`` directly.

        Returns:
            All unique Card objects.
        """
        await self.ensure_loaded()
        return self._unique_cards

    async def get_cards(self, names: list[str]) -> dict[str, Card | None]:
        """Batch exact-name lookup. Returns dict of name -> Card|None.

        Keys in the returned dict preserve the original casing from ``names``.
        Lookups are case-insensitive.

        Args:
            names: Card names to look up.

        Returns:
            Dict mapping each input name to its Card or None if not found.
        """
        await self.ensure_loaded()
        return {name: self._cards.get(name.lower()) for name in names}

    async def random_card(
        self,
        *,
        format: str | None = None,
        color_identity: frozenset[str] | None = None,
        type_contains: str | None = None,
        rarity: str | None = None,
        unreleased: UnreleasedCollector | None = None,
        include_unreleased: bool = False,
    ) -> Card | None:
        """Random card from a filtered pool. None if pool is empty.

        Args:
            format: Only cards where ``legalities[format] == "legal"``.
            color_identity: Cards whose color identity is a subset of this set.
            type_contains: String that must appear in the type line (case-insensitive).
            rarity: Exact rarity match.
            unreleased: When given alongside ``format``, receives every card that
                matched all OTHER criteria but failed only the legality check because
                its set has not been released yet.
            include_unreleased: When True (and ``unreleased`` is given), those cards
                stay in the pool and can be drawn.

        Returns:
            A random matching card, or None if no cards match.
        """
        await self.ensure_loaded()

        pool: list[Card] = []
        type_lower = type_contains.lower() if type_contains else None

        for card in self._unique_cards:
            legal = format is None or card.legalities.get(format) == "legal"
            if not legal and (unreleased is None or not unreleased.offer(card)):
                continue
            if color_identity is not None and not frozenset(card.color_identity).issubset(
                color_identity
            ):
                continue
            if type_lower is not None and type_lower not in card.type_line.lower():
                continue
            if rarity is not None and card.rarity != rarity:
                continue
            if not legal:
                if unreleased is not None:
                    unreleased.collect(card)
                if not include_unreleased:
                    continue
            pool.append(card)

        if not pool:
            return None
        return random.choice(pool)

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _is_stale(self) -> bool:
        """Check if the loaded data has exceeded the refresh interval."""
        if self._loaded_at == 0.0:
            return True
        return (time.monotonic() - self._loaded_at) >= self._refresh_seconds

    async def _fetch_metadata(self) -> dict:
        """Fetch the bulk-data metadata to get the current download URI.

        Raises:
            ScryfallBulkDownloadError: On HTTP errors or network failures.
        """
        url = f"{self._base_url}/bulk-data/oracle_cards"
        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": DEFAULT_USER_AGENT,
                    "Accept": "application/json",
                },
            ) as http:
                response = await http.get(url)
                if response.status_code != 200:
                    raise ScryfallBulkDownloadError(
                        f"HTTP {response.status_code} fetching bulk metadata from {url}"
                    )
                try:
                    return response.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ScryfallBulkDownloadError(
                        f"Metadata response is not valid JSON from {url}: {exc}"
                    ) from exc
        except httpx.RequestError as exc:
            raise ScryfallBulkDownloadError(f"Network error fetching bulk metadata: {exc}") from exc

    async def _download(self, url: str) -> bytes | None:
        """Download the Oracle Cards bulk data file.

        Sends ``If-None-Match`` header when the URL matches the previous
        download's URL and we have a saved ETag. Returns ``None`` on HTTP
        304 (data unchanged).

        Raises:
            ScryfallBulkDownloadError: On HTTP errors or network failures.
        """
        headers: dict[str, str] = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        }

        # Only send If-None-Match if the URL matches the one we got the ETag from
        if self._etag is not None and self._etag_url == url:
            headers["If-None-Match"] = self._etag

        try:
            async with httpx.AsyncClient(timeout=120.0, headers=headers) as http:
                response = await http.get(url)

                if response.status_code == 304:
                    log.info("scryfall_bulk.not_modified", url=url)
                    return None

                if response.status_code != 200:
                    raise ScryfallBulkDownloadError(
                        f"HTTP {response.status_code} downloading bulk data from {url}"
                    )

                # Save ETag for future conditional requests
                etag = response.headers.get("ETag")
                if etag:
                    self._etag = etag
                    self._etag_url = url

                return response.content
        except httpx.RequestError as exc:
            raise ScryfallBulkDownloadError(f"Network error downloading bulk data: {exc}") from exc

    @staticmethod
    def _open_payload(raw_bytes: bytes) -> gzip.GzipFile | io.BytesIO:
        """Return a line-readable stream over the payload, gunzipping if needed.

        Scryfall serves ``Content-Type: application/gzip`` with no
        ``Content-Encoding`` header, so httpx does NOT decompress this for us.
        Detection is on the gzip magic number rather than the URL suffix or the
        content type, both of which have already changed once.

        Streamed rather than decompressed in one call on purpose: the live file
        is ~24 MB gzipped and several hundred MB expanded, and this process
        serves production while a background refresh runs.
        """
        if raw_bytes.startswith(b"\x1f\x8b"):
            return gzip.GzipFile(fileobj=io.BytesIO(raw_bytes))
        return io.BytesIO(raw_bytes)

    @classmethod
    def _iter_entries(cls, raw_bytes: bytes, stats: _ParseStats) -> Iterator[object]:
        """Yield card entries from the payload, JSONL first, JSON array as fallback.

        Scryfall moved from a single JSON array to JSONL (one card per line) in
        July 2026. Both are read here: the array form costs a few lines to keep
        and means an upstream rollback does not take the server down.

        A generator rather than a list on purpose. Materialising all ~35k dicts
        before building the first Card put the peak at 803 MB against 628 MB for
        the array path it replaced, on a process that serves production while a
        background refresh runs. Counts land in ``stats`` because a generator
        cannot return them alongside its values, and the caller needs them: it
        reports "parsed N of M", and a denominator that never counted the
        unreadable lines sends whoever reads it looking in the wrong place.

        Raises:
            ScryfallBulkError: If the payload cannot be read as either shape.
        """
        try:
            with cls._open_payload(raw_bytes) as stream:
                for raw_line in stream:
                    line = raw_line.strip().removeprefix(codecs.BOM_UTF8)
                    if not line:
                        continue
                    # A legacy JSON array: hand the whole payload to json.loads.
                    if not stats.read and not stats.malformed and line.startswith(b"["):
                        for entry in cls._parse_legacy_array(raw_bytes):
                            stats.read += 1
                            yield entry
                        return
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # One unreadable line in ~35k must not lose the rest.
                        stats.malformed += 1
                        continue
                    stats.read += 1
                    yield entry
        except (OSError, EOFError, zlib.error) as exc:
            raise ScryfallBulkError(f"Failed to decompress bulk data: {exc}") from exc

        if stats.malformed:
            log.warning("scryfall_bulk.malformed_lines", count=stats.malformed)
        if not stats.read:
            raise ScryfallBulkError(
                "Failed to parse bulk data: payload is neither a JSON array nor JSONL "
                f"({stats.malformed} unreadable line(s), {len(raw_bytes)} bytes)"
            )

    @classmethod
    def _parse_legacy_array(cls, raw_bytes: bytes) -> list[object]:
        """Read the pre-July-2026 shape: one JSON array holding every card."""
        with cls._open_payload(raw_bytes) as stream:
            payload = stream.read()
        try:
            raw = json.loads(payload.removeprefix(codecs.BOM_UTF8))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ScryfallBulkError(f"Failed to parse bulk data: {exc}") from exc
        if not isinstance(raw, list):
            raise ScryfallBulkError("Bulk data is not a JSON array")
        return raw

    def _parse(self, raw_bytes: bytes) -> None:
        """Parse the Oracle Cards payload into the in-memory card dicts.

        Accepts gzipped or plain bytes, holding either JSONL (current Scryfall
        format) or a JSON array (legacy). Each entry is validated through
        ``Card.model_validate()``.

        For DFCs, ``card.name`` contains the full ``"Front // Back"`` name.
        We key lookups by both the full name and the front face name
        (``name.split(" // ")[0]``).

        Raises:
            ScryfallBulkError: If the payload cannot be decompressed or parsed,
                or if too little of it survived to be a usable card pool.
        """
        stats = _ParseStats()

        cards: dict[str, Card] = {}
        unique: list[Card] = []
        skipped = 0

        for entry in self._iter_entries(raw_bytes, stats):
            if not isinstance(entry, dict):
                skipped += 1
                continue

            layout = entry.get("layout", "")
            if layout in _EXCLUDED_LAYOUTS:
                skipped += 1
                continue

            try:
                card = Card.model_validate(entry)
            except (ValidationError, ValueError) as exc:
                log.warning(
                    "scryfall_bulk.card_parse_error",
                    card_name=entry.get("name", "unknown"),
                    error=str(exc),
                )
                skipped += 1
                continue

            unique.append(card)

            # Key by full lowercase name for O(1) case-insensitive lookup
            full_name_lower = card.name.lower()
            cards[full_name_lower] = card

            # For DFCs, card.name is "Front // Back". Also key by front-face
            # name so lookup by either form works.
            if " // " in card.name:
                front_face = card.name.split(" // ")[0].lower()
                if front_face != full_name_lower:
                    cards[front_face] = card

        if skipped or stats.malformed:
            log.info(
                "scryfall_bulk.parse_summary",
                skipped=skipped,
                malformed=stats.malformed,
                loaded=len(unique),
            )

        # The denominator counts every line the payload offered, including the
        # ones dropped before they were ever entries. Quoting only the readable
        # ones would say "0 of 1000" about a file whose other 29000 lines were
        # unreadable, and send whoever reads it looking in the wrong place.
        offered = stats.read + stats.malformed
        if not unique:
            raise ScryfallBulkError(
                f"Parsed 0 cards from {offered} lines "
                f"({stats.malformed} unreadable, {skipped} skipped). "
                "Scryfall bulk data schema may have changed."
            )

        # A payload that parses but is mostly rubbish is the dangerous case: the
        # pool silently shrinks and every downstream tool answers "card not
        # found" with total confidence. Measured against the pool we already
        # had rather than an absolute floor, because only the running server
        # knows what a normal size looks like. Raising here keeps the last good
        # data (ensure_loaded serves stale on refresh failure).
        previous = len(self._unique_cards)
        if previous and len(unique) < previous * _MIN_REFRESH_RATIO:
            raise ScryfallBulkError(
                f"Refresh parsed {len(unique)} cards, down from {previous} "
                f"({stats.malformed} unreadable, {skipped} skipped of {offered} lines). "
                "Refusing to replace a healthy card pool with a truncated one."
            )

        self._cards = cards
        self._unique_cards = unique

    async def _refresh_loop(self, *, preload: bool = True) -> None:
        """Background loop that periodically calls ensure_loaded().

        Runs until cancelled (via __aexit__). Errors are logged but do
        not stop the loop.

        With ``preload``, loads before the first sleep. Sleeping first meant
        nothing was ever preloaded and the first user request paid the ~30MB
        download itself (6.5s on deck_audit_bundle's first call, 2026-07-27).
        ensure_loaded() holds a lock, so a request arriving mid-download waits
        on it instead of starting a second one.
        """
        while True:
            if not preload:
                preload = True  # only the first iteration can be skipped
                await asyncio.sleep(self._refresh_seconds)
                continue
            try:
                await self.ensure_loaded()
            except ScryfallBulkError:
                log.warning("scryfall_bulk.background_refresh_error", exc_info=True)
            except Exception:
                log.error("scryfall_bulk.background_refresh_unexpected_error", exc_info=True)
            await asyncio.sleep(self._refresh_seconds)
