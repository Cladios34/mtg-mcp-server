"""Tests for the Scryfall bulk card data service."""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

from mtg_mcp_server.services.scryfall_bulk import (
    ScryfallBulkClient,
    ScryfallBulkDownloadError,
    ScryfallBulkError,
)
from mtg_mcp_server.utils.unreleased import UnreleasedCollector

FIXTURES = Path(__file__).parent.parent / "fixtures" / "scryfall_bulk"

_BASE_URL = "https://api.scryfall.com"
_JSONL_DOWNLOAD_URL = "https://data.scryfall.io/oracle-cards/oracle-cards-20260728235006.jsonl.gz"
# Kept for the legacy-format tests; the live metadata no longer serves this key.
_LEGACY_DOWNLOAD_URL = "https://data.scryfall.io/oracle-cards/oracle-cards-20260326090226.json"


def _load_metadata() -> dict:
    """Load the bulk-data metadata fixture."""
    return json.loads((FIXTURES / "bulk_metadata.json").read_text())


def _load_oracle_cards() -> str:
    """Load the oracle cards sample fixture as a JSON string."""
    return (FIXTURES / "oracle_cards_sample.json").read_text()


def _load_oracle_cards_bytes() -> bytes:
    """Load the oracle cards sample fixture as a JSON array (legacy format)."""
    return (FIXTURES / "oracle_cards_sample.json").read_bytes()


def _oracle_cards_jsonl() -> bytes:
    """The sample fixture re-serialised as JSONL, the format Scryfall now serves."""
    entries = json.loads((FIXTURES / "oracle_cards_sample.json").read_text())
    return b"\n".join(json.dumps(entry).encode() for entry in entries)


def _oracle_cards_jsonl_gz() -> bytes:
    """The sample fixture as gzipped JSONL -- byte-for-byte what the live URL returns."""
    return gzip.compress(_oracle_cards_jsonl())


# ---------------------------------------------------------------------------
# Helpers for respx-based mocking
# ---------------------------------------------------------------------------


def _mock_metadata_route(
    router: respx.MockRouter,
    metadata: dict | None = None,
    status_code: int = 200,
) -> respx.Route:
    """Register a metadata endpoint route."""
    meta = metadata or _load_metadata()
    return router.get(f"{_BASE_URL}/bulk-data/oracle_cards").mock(
        return_value=httpx.Response(status_code, json=meta)
    )


def _mock_download_route(
    router: respx.MockRouter,
    url: str = _JSONL_DOWNLOAD_URL,
    content: bytes | None = None,
    status_code: int = 200,
    headers: dict | None = None,
) -> respx.Route:
    """Register a bulk data download route."""
    body = content if content is not None else _oracle_cards_jsonl_gz()
    resp_headers = headers or {}
    return router.get(url).mock(
        return_value=httpx.Response(status_code, content=body, headers=resp_headers)
    )


@pytest.fixture
async def loaded_client():
    """A client with pre-loaded fixture data (respx mocked HTTP)."""
    with respx.mock:
        _mock_metadata_route(respx)
        _mock_download_route(respx, headers={"ETag": '"abc123"'})

        client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
        async with client:
            await client.ensure_loaded()
            yield client


# ===========================================================================
# Test Classes
# ===========================================================================


class TestLazyLoading:
    """Test that data is not downloaded until first access."""

    async def test_no_download_on_aenter(self):
        """Creating and entering the client should NOT trigger a download."""
        client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
        async with client:
            assert client._loaded_at == 0.0
            assert len(client._cards) == 0

    async def test_first_get_card_triggers_download(self):
        """First card lookup triggers the bulk data download."""
        with respx.mock:
            meta_route = _mock_metadata_route(respx)
            dl_route = _mock_download_route(respx, headers={"ETag": '"abc123"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                result = await client.get_card("Sol Ring")
                assert result is not None
                assert result.name == "Sol Ring"
                assert meta_route.called
                assert dl_route.called


class TestStaleness:
    """Test that fresh data skips download and stale triggers re-fetch."""

    async def test_fresh_data_skips_download(self):
        """Data within refresh_hours is not re-downloaded."""
        with respx.mock:
            meta_route = _mock_metadata_route(respx)
            _mock_download_route(respx, headers={"ETag": '"abc123"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert meta_route.call_count == 1

                # Second call should skip
                await client.ensure_loaded()
                assert meta_route.call_count == 1

    async def test_stale_data_triggers_refetch(self):
        """Data older than refresh_hours triggers a re-download."""
        with respx.mock:
            meta_route = _mock_metadata_route(respx)
            _mock_download_route(respx, headers={"ETag": '"abc123"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=1)
            async with client:
                await client.ensure_loaded()
                assert meta_route.call_count == 1

                # Simulate stale timestamp (2 hours ago)
                client._loaded_at = time.monotonic() - 7200

                await client.ensure_loaded()
                assert meta_route.call_count == 2


class TestRefreshFailure:
    """Test failure behavior: first-load propagates, refresh-failure serves stale."""

    async def test_first_load_download_error_propagates(self):
        """If the very first download fails, the error propagates."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, status_code=500, content=b"")

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkDownloadError):
                    await client.ensure_loaded()

    async def test_first_load_metadata_error_propagates(self):
        """If the metadata fetch fails on first load, the error propagates."""
        with respx.mock:
            _mock_metadata_route(respx, status_code=500)

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkDownloadError):
                    await client.ensure_loaded()

    async def test_refresh_failure_serves_stale_data(self):
        """If data was loaded but refresh fails, serve stale data."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, headers={"ETag": '"abc123"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=1)
            async with client:
                # First: successful load
                await client.ensure_loaded()
                assert len(client._cards) > 0

                # Simulate stale + failed refresh by clearing routes and
                # re-registering with 503
                respx.reset()
                _mock_metadata_route(respx, status_code=503)

                loaded_at = client._loaded_at
                stale_time = loaded_at + 7200

                with patch(
                    "mtg_mcp_server.services.scryfall_bulk.time.monotonic",
                    return_value=stale_time,
                ):
                    # Should NOT raise — serves stale data
                    await client.ensure_loaded()

                # Stale data should still be available
                card = await client.get_card("Sol Ring")
                assert card is not None
                assert card.name == "Sol Ring"

    async def test_refresh_failure_schedules_earlier_retry(self):
        """After refresh failure, next refresh triggers in ~5 min, not full interval."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, headers={"ETag": '"abc123"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=1)
            async with client:
                await client.ensure_loaded()

                # Simulate stale + failed refresh
                respx.reset()
                _mock_metadata_route(respx, status_code=503)

                loaded_at = client._loaded_at
                stale_time = loaded_at + 7200

                with patch(
                    "mtg_mcp_server.services.scryfall_bulk.time.monotonic",
                    return_value=stale_time,
                ):
                    await client.ensure_loaded()

                # After failure, _loaded_at should be set so data appears stale
                # again in ~300s (not the full 3600s interval)
                expected_retry_at = stale_time - client._refresh_seconds + 300
                assert client._loaded_at == pytest.approx(expected_retry_at, abs=1.0)


class TestETag:
    """Test ETag-based conditional download (304 = skip re-parse)."""

    async def test_etag_saved_from_response(self):
        """ETag from the download response is saved for subsequent requests."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, headers={"ETag": '"my-etag-value"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert client._etag == '"my-etag-value"'

    async def test_304_response_skips_reparse(self):
        """304 Not Modified response skips re-parsing data."""
        with respx.mock:
            metadata = _load_metadata()
            _mock_metadata_route(respx, metadata=metadata)
            _mock_download_route(respx, headers={"ETag": '"etag-v1"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=1)
            async with client:
                await client.ensure_loaded()
                original_count = len(client._unique_cards)
                original_loaded_at = client._loaded_at

                # Simulate stale
                client._loaded_at = time.monotonic() - 7200

                # Swap routes: return 304 on re-fetch
                respx.reset()
                _mock_metadata_route(respx, metadata=metadata)
                _mock_download_route(
                    respx,
                    status_code=304,
                    content=b"",
                    headers={},
                )

                await client.ensure_loaded()
                # Data should NOT have been cleared
                assert len(client._unique_cards) == original_count
                # loaded_at should have been refreshed
                assert client._loaded_at > original_loaded_at

    async def test_etag_only_sent_when_url_matches(self):
        """ETag is only sent when the download URL matches the previous one."""
        metadata_v1 = _load_metadata()
        metadata_v2 = _load_metadata()
        metadata_v2["jsonl_download_uri"] = (
            "https://data.scryfall.io/oracle-cards/oracle-cards-v2.json"
        )

        with respx.mock:
            # First load with URL v1
            _mock_metadata_route(respx, metadata=metadata_v1)
            _mock_download_route(
                respx, url=metadata_v1["jsonl_download_uri"], headers={"ETag": '"etag-v1"'}
            )

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=1)
            async with client:
                await client.ensure_loaded()
                assert client._etag == '"etag-v1"'

        # Simulate stale, metadata now points to a different URL
        client._loaded_at = time.monotonic() - 7200

        with respx.mock:
            _mock_metadata_route(respx, metadata=metadata_v2)
            dl_route = _mock_download_route(
                respx,
                url=metadata_v2["jsonl_download_uri"],
                headers={"ETag": '"etag-v2"'},
            )

            async with client:
                await client.ensure_loaded()
                # Should have fetched the new URL without If-None-Match
                assert dl_route.called
                req = dl_route.calls[0].request
                assert "If-None-Match" not in req.headers
                # ETag should now be v2
                assert client._etag == '"etag-v2"'


class TestConcurrency:
    """Test that multiple concurrent ensure_loaded() calls only download once."""

    async def test_concurrent_ensure_loaded_only_downloads_once(self):
        """Multiple concurrent ensure_loaded() calls only download once (lock)."""
        with respx.mock:
            meta_route = _mock_metadata_route(respx)
            dl_route = _mock_download_route(respx, headers={"ETag": '"abc"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                # Launch 5 concurrent ensure_loaded calls
                await asyncio.gather(
                    client.ensure_loaded(),
                    client.ensure_loaded(),
                    client.ensure_loaded(),
                    client.ensure_loaded(),
                    client.ensure_loaded(),
                )
                # Only one download should have happened
                assert meta_route.call_count == 1
                assert dl_route.call_count == 1


class TestParsing:
    """Test that the bulk data is parsed correctly into Card models."""

    async def test_correct_card_count(self, loaded_client: ScryfallBulkClient):
        """30 playable cards (4 non-playable layouts filtered), 31 dict entries for DFC."""
        assert len(loaded_client._unique_cards) == 30
        # 30 normal keys + 1 extra for DFC front-face-only key
        assert len(loaded_client._cards) == 31

    async def test_card_has_prices(self, loaded_client: ScryfallBulkClient):
        """Parsed cards have price data."""
        card = await loaded_client.get_card("Sol Ring")
        assert card is not None
        assert card.prices.usd == "1.50"
        assert card.prices.usd_foil == "3.00"

    async def test_card_has_legalities(self, loaded_client: ScryfallBulkClient):
        """Parsed cards have legality data."""
        card = await loaded_client.get_card("Sol Ring")
        assert card is not None
        assert card.legalities["commander"] == "legal"
        assert card.legalities["legacy"] == "banned"

    async def test_card_has_edhrec_rank(self, loaded_client: ScryfallBulkClient):
        """Parsed cards have EDHREC rank."""
        card = await loaded_client.get_card("Sol Ring")
        assert card is not None
        assert card.edhrec_rank == 1

    async def test_card_without_edhrec_rank(self, loaded_client: ScryfallBulkClient):
        """Cards without edhrec_rank (like Forest) have None."""
        card = await loaded_client.get_card("Forest")
        assert card is not None
        assert card.edhrec_rank is None


class TestLayoutFiltering:
    """Test that non-playable card layouts are excluded from parsed data."""

    async def test_minigame_excluded(self, loaded_client: ScryfallBulkClient):
        """Minigame Sol Ring (acmm) is filtered; real Sol Ring returned."""
        card = await loaded_client.get_card("Sol Ring")
        assert card is not None
        assert card.set_code != "acmm"
        assert card.layout == "normal"
        assert card.type_line == "Artifact"

    async def test_art_series_excluded(self, loaded_client: ScryfallBulkClient):
        """Art series Lightning Bolt is filtered; real Lightning Bolt returned."""
        card = await loaded_client.get_card("Lightning Bolt")
        assert card is not None
        assert card.set_code != "sld"
        assert card.layout == "normal"
        assert card.type_line == "Instant"

    async def test_token_excluded(self, loaded_client: ScryfallBulkClient):
        """Token Forest is filtered; real Basic Land returned."""
        card = await loaded_client.get_card("Forest")
        assert card is not None
        assert "Basic Land" in card.type_line

    async def test_emblem_excluded(self, loaded_client: ScryfallBulkClient):
        """Emblem cards are filtered and not searchable."""
        results = await loaded_client.search_cards("Emblem")
        assert len(results) == 0

    async def test_real_card_preserved_despite_collision(self, loaded_client: ScryfallBulkClient):
        """Sol Ring retains correct data despite minigame entry later in fixture."""
        card = await loaded_client.get_card("Sol Ring")
        assert card is not None
        assert card.prices.usd == "1.50"
        assert card.legalities["commander"] == "legal"
        assert card.edhrec_rank == 1

    async def test_excluded_layouts_constant(self):
        """All 6 non-playable layouts are in the exclusion set."""
        from mtg_mcp_server.services.scryfall_bulk import _EXCLUDED_LAYOUTS

        expected = {
            "art_series",
            "double_faced_token",
            "emblem",
            "minigame",
            "placeholder",
            "token",
        }
        assert expected == _EXCLUDED_LAYOUTS

    async def test_layout_field_populated(self, loaded_client: ScryfallBulkClient):
        """Parsed cards have the layout field populated from Scryfall data."""
        card = await loaded_client.get_card("Sol Ring")
        assert card is not None
        assert card.layout == "normal"

        dfc = await loaded_client.get_card("Delver of Secrets")
        assert dfc is not None
        assert dfc.layout == "transform"


class TestDFC:
    """Test double-faced card handling."""

    async def test_dfc_accessible_by_full_name(self, loaded_client: ScryfallBulkClient):
        """DFC is accessible by full '// ' name."""
        card = await loaded_client.get_card("Delver of Secrets // Insectile Aberration")
        assert card is not None
        assert card.name == "Delver of Secrets // Insectile Aberration"

    async def test_dfc_accessible_by_front_face(self, loaded_client: ScryfallBulkClient):
        """DFC is accessible by front-face name only."""
        card = await loaded_client.get_card("Delver of Secrets")
        assert card is not None
        assert card.name == "Delver of Secrets // Insectile Aberration"

    async def test_dfc_not_duplicated_in_unique_cards(self, loaded_client: ScryfallBulkClient):
        """DFC should appear only once in _unique_cards."""
        delver_count = sum(1 for c in loaded_client._unique_cards if "Delver" in c.name)
        assert delver_count == 1

    async def test_dfc_has_front_face_data(self, loaded_client: ScryfallBulkClient):
        """DFC has oracle_text from front face (via _fill_from_card_faces)."""
        card = await loaded_client.get_card("Delver of Secrets")
        assert card is not None
        assert card.oracle_text is not None
        assert "transform" in card.oracle_text.lower()


class TestGetCard:
    """Test exact card lookup."""

    async def test_case_insensitive(self, loaded_client: ScryfallBulkClient):
        """Lookup is case-insensitive."""
        result = await loaded_client.get_card("sol ring")
        assert result is not None
        assert result.name == "Sol Ring"

    async def test_mixed_case(self, loaded_client: ScryfallBulkClient):
        """All-uppercase input resolves correctly."""
        result = await loaded_client.get_card("SOL RING")
        assert result is not None
        assert result.name == "Sol Ring"

    async def test_not_found_returns_none(self, loaded_client: ScryfallBulkClient):
        """Nonexistent card name returns None."""
        result = await loaded_client.get_card("Nonexistent Card")
        assert result is None

    async def test_legendary_creature(self, loaded_client: ScryfallBulkClient):
        """Legendary creature lookup includes power and toughness."""
        result = await loaded_client.get_card("Muldrotha, the Gravetide")
        assert result is not None
        assert result.power == "6"
        assert result.toughness == "6"
        assert result.cmc == 6.0

    async def test_special_characters(self, loaded_client: ScryfallBulkClient):
        """Card with non-ASCII characters is found correctly."""
        result = await loaded_client.get_card("Jötun Grunt")
        assert result is not None
        assert result.name == "Jötun Grunt"

    async def test_basic_land(self, loaded_client: ScryfallBulkClient):
        """Basic land has no colors and no power/toughness."""
        result = await loaded_client.get_card("Forest")
        assert result is not None
        assert result.type_line == "Basic Land — Forest"
        assert result.colors == []
        assert result.power is None
        assert result.toughness is None


class TestSearchCards:
    """Test name substring search."""

    async def test_search_by_name(self, loaded_client: ScryfallBulkClient):
        """Substring match on card name returns matching cards."""
        results = await loaded_client.search_cards("ring")
        assert len(results) == 1
        assert results[0].name == "Sol Ring"

    async def test_search_case_insensitive(self, loaded_client: ScryfallBulkClient):
        """Name search is case-insensitive."""
        results = await loaded_client.search_cards("BOLT")
        assert len(results) == 1
        assert results[0].name == "Lightning Bolt"

    async def test_search_no_results(self, loaded_client: ScryfallBulkClient):
        """Search with no matches returns an empty list."""
        results = await loaded_client.search_cards("xyzzynonexistent")
        assert results == []

    async def test_search_limit(self, loaded_client: ScryfallBulkClient):
        """Limit parameter caps the number of returned results."""
        results = await loaded_client.search_cards("", limit=3)
        assert len(results) == 3

    async def test_search_partial_match(self, loaded_client: ScryfallBulkClient):
        """Partial name substring matches cards containing that text."""
        results = await loaded_client.search_cards("counter")
        assert len(results) >= 1
        names = [r.name for r in results]
        assert "Counterspell" in names


class TestSearchByType:
    """Test type line substring search."""

    async def test_search_creature(self, loaded_client: ScryfallBulkClient):
        """Type search for 'Creature' includes creatures and excludes non-creatures."""
        results = await loaded_client.search_by_type("Creature")
        names = [r.name for r in results]
        assert "Spore Frog" in names
        assert "Sol Ring" not in names

    async def test_search_instant(self, loaded_client: ScryfallBulkClient):
        """Type search for 'Instant' returns instants."""
        results = await loaded_client.search_by_type("Instant")
        names = [r.name for r in results]
        assert "Lightning Bolt" in names
        assert "Counterspell" in names

    async def test_search_no_match(self, loaded_client: ScryfallBulkClient):
        """Type search with no matching cards returns empty list."""
        results = await loaded_client.search_by_type("Planeswalker")
        assert results == []


class TestChangelingsInTypeSearch:
    """Rule 702.73a — a changeling is every creature type, in every zone.

    Regression origin (2026-07-27): a Ninja type search returned 62 cards with both of
    the deck's changelings missing. The cause was here, not upstream — the search
    compared the type line, and a changeling's type line says "Shapeshifter".
    A tribal deck is judged on that count.
    """

    async def test_changeling_is_returned_for_a_creature_subtype(
        self, loaded_client: ScryfallBulkClient
    ):
        results = await loaded_client.search_by_type("Ninja")
        names = [r.name for r in results]
        assert "Ingenious Infiltrator" in names  # printed Ninja
        assert "Changeling Outcast" in names  # Ninja by 702.73a

    async def test_opting_out_gives_the_literal_type_line_match(
        self, loaded_client: ScryfallBulkClient
    ):
        results = await loaded_client.search_by_type("Ninja", include_changelings=False)
        names = [r.name for r in results]
        assert "Ingenious Infiltrator" in names
        assert "Changeling Outcast" not in names

    async def test_changeling_is_not_returned_for_a_card_type(
        self, loaded_client: ScryfallBulkClient
    ):
        # A changeling is every CREATURE type — it is not an Artifact.
        results = await loaded_client.search_by_type("Artifact", limit=100)
        assert "Changeling Outcast" not in [r.name for r in results]

    async def test_changeling_still_matches_its_own_printed_type(
        self, loaded_client: ScryfallBulkClient
    ):
        results = await loaded_client.search_by_type("Shapeshifter")
        assert "Changeling Outcast" in [r.name for r in results]


class TestSearchByText:
    """Test oracle text substring search."""

    async def test_search_damage(self, loaded_client: ScryfallBulkClient):
        """Oracle text search for 'damage' returns cards mentioning damage."""
        results = await loaded_client.search_by_text("damage")
        names = [r.name for r in results]
        assert "Lightning Bolt" in names
        assert "Spore Frog" in names

    async def test_search_counter(self, loaded_client: ScryfallBulkClient):
        """Oracle text search for 'Counter target spell' finds counterspells."""
        results = await loaded_client.search_by_text("Counter target spell")
        names = [r.name for r in results]
        assert "Counterspell" in names

    async def test_search_no_match(self, loaded_client: ScryfallBulkClient):
        """Text search with no matches returns an empty list."""
        results = await loaded_client.search_by_text("xyzzynonexistent")
        assert results == []


class TestBackgroundRefresh:
    """Test background refresh task lifecycle."""

    async def test_start_preloads_immediately(self):
        """The refresh task loads on start, it does not wait a full interval first.

        Regression guard: the loop used to sleep `refresh_hours` BEFORE its first
        load, so nothing was ever preloaded and the first user request paid the
        ~30MB download. Measured 2026-07-27: it put 6.5s on deck_audit_bundle's
        first call, with three sections queued behind the same download.
        """
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx)
            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                client.start_background_refresh()
                # Yield to the loop until it has loaded, without waiting an interval.
                for _ in range(100):
                    if client._loaded_at > 0:
                        break
                    await asyncio.sleep(0.01)
                assert client._loaded_at > 0, "background task did not preload on start"
                assert len(client._unique_cards) > 0

    async def test_preload_false_does_not_load_on_start(self):
        """preload=False waits a full interval, so tests never hit the live API."""
        with respx.mock:
            meta = _mock_metadata_route(respx)
            _mock_download_route(respx)
            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                client.start_background_refresh(preload=False)
                for _ in range(20):
                    await asyncio.sleep(0.01)
                assert client._loaded_at == 0.0
                assert meta.call_count == 0

    async def test_start_creates_task(self):
        """start_background_refresh() creates a background task."""
        client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
        async with client:
            client.start_background_refresh()
            assert client._refresh_task is not None
            assert not client._refresh_task.done()
            # Clean up
            client._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await client._refresh_task

    async def test_aexit_cancels_task(self):
        """__aexit__ cancels the background refresh task."""
        client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
        async with client:
            client.start_background_refresh()
            task = client._refresh_task
            assert task is not None
        # After __aexit__, the task should be cancelled
        assert task.cancelled() or task.done()

    async def test_aexit_clears_data(self):
        """__aexit__ clears in-memory card data."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, headers={"ETag": '"abc"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert len(client._cards) > 0

            # After __aexit__, data should be cleared
            assert len(client._cards) == 0
            assert len(client._unique_cards) == 0
            assert client._loaded_at == 0.0


class TestExceptionTypes:
    """Test that the correct exception types are raised."""

    async def test_bulk_error_is_service_error(self):
        """ScryfallBulkError inherits from ServiceError."""
        from mtg_mcp_server.services.base import ServiceError

        err = ScryfallBulkError("test")
        assert isinstance(err, ServiceError)
        assert err.status_code is None

    async def test_download_error_is_bulk_error(self):
        """ScryfallBulkDownloadError inherits from ScryfallBulkError."""
        err = ScryfallBulkDownloadError("test")
        assert isinstance(err, ScryfallBulkError)

    async def test_network_error_raises_download_error(self):
        """Network connection failure on metadata raises ScryfallBulkDownloadError."""
        with respx.mock:
            respx.get(f"{_BASE_URL}/bulk-data/oracle_cards").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkDownloadError, match="Network error"):
                    await client.ensure_loaded()

    async def test_download_network_error_raises(self):
        """Network failure during bulk data download raises ScryfallBulkDownloadError."""
        with respx.mock:
            _mock_metadata_route(respx)
            respx.get(_JSONL_DOWNLOAD_URL).mock(side_effect=httpx.ReadTimeout("Read timed out"))

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkDownloadError, match="Network error"):
                    await client.ensure_loaded()


class TestMetadataValidation:
    """Test metadata response validation and edge cases."""

    async def test_missing_download_uri_raises(self):
        """Missing download_uri key in metadata raises ScryfallBulkError."""
        with respx.mock:
            _mock_metadata_route(respx, metadata={"object": "bulk_data", "type": "oracle_cards"})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkError, match="jsonl_download_uri"):
                    await client.ensure_loaded()

    async def test_non_json_metadata_raises(self):
        """Non-JSON metadata response raises ScryfallBulkDownloadError."""
        with respx.mock:
            respx.get(f"{_BASE_URL}/bulk-data/oracle_cards").mock(
                return_value=httpx.Response(
                    200, content=b"<html>error</html>", headers={"Content-Type": "text/html"}
                )
            )

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkDownloadError, match="not valid JSON"):
                    await client.ensure_loaded()


class TestParseFailures:
    """Test _parse error handling for malformed data."""

    async def test_non_json_bulk_data_raises(self):
        """Non-JSON bulk data raises ScryfallBulkError on first load."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=b"<html>CDN error</html>")

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkError, match="Failed to parse"):
                    await client.ensure_loaded()

    async def test_error_object_instead_of_card_data_raises(self):
        """An API error object where card data was expected raises, loudly.

        Under JSONL this is a syntactically valid single line, so it is rejected
        at card validation rather than at parse. What matters to the caller is
        unchanged: a ScryfallBulkError naming a schema problem, never a silently
        empty card pool.
        """
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=b'{"error": "not found"}')

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkError, match="schema may have changed"):
                    await client.ensure_loaded()

    async def test_all_invalid_cards_raises_zero_parsed(self):
        """Array of all-invalid entries raises ScryfallBulkError (zero cards)."""
        bad_data = json.dumps([{"not_a": "card"}, {"also": "bad"}]).encode()
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=bad_data)

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkError, match="Parsed 0 cards"):
                    await client.ensure_loaded()

    async def test_empty_json_array_raises_zero_parsed(self):
        """Empty JSON array [] raises ScryfallBulkError (zero cards)."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=b"[]")

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkError, match="Parsed 0 cards"):
                    await client.ensure_loaded()

    async def test_bad_card_skipped_good_cards_loaded(self):
        """A malformed card in the middle is skipped; valid cards still load."""
        sample = json.loads(_load_oracle_cards())
        sample.insert(1, {"not_a": "valid_card"})  # inject bad entry
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=json.dumps(sample).encode())

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                # Fixture has 30 playable cards (non-playable layouts filtered)
                assert len(client._unique_cards) == 30
                assert await client.get_card("Sol Ring") is not None

    async def test_corrupt_data_on_refresh_serves_stale(self):
        """Corrupt data during refresh serves stale data instead of failing."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, headers={"ETag": '"abc"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=1)
            async with client:
                await client.ensure_loaded()
                original_count = len(client._unique_cards)
                assert original_count > 0

                # Simulate stale + corrupt download
                respx.reset()
                _mock_metadata_route(respx)
                _mock_download_route(respx, content=b"not json at all")

                loaded_at = client._loaded_at
                stale_time = loaded_at + 7200

                with patch(
                    "mtg_mcp_server.services.scryfall_bulk.time.monotonic",
                    return_value=stale_time,
                ):
                    await client.ensure_loaded()  # Should NOT raise

                # Stale data still available
                assert len(client._unique_cards) == original_count
                assert await client.get_card("Sol Ring") is not None


class TestUnreleasedCollection:
    """A format filter must not silently drop cards from unreleased sets.

    The fixture carries the real Darksteel Angel (set 'frc', released_at
    2026-10-02, not_legal in every format until then). Collectors are built with
    a frozen `today` so these tests do not rot when that set actually releases.
    """

    _TODAY = "2026-08-04"

    async def test_filter_cards_collects_unreleased_matches(
        self, loaded_client: ScryfallBulkClient
    ):
        """An unreleased card matching every other criterion is named, not returned."""
        collector = UnreleasedCollector(active=True, today=self._TODAY)
        results = await loaded_client.filter_cards(
            format="commander",
            type_contains=["Angel"],
            unreleased=collector,
        )
        assert all(c.legalities.get("commander") == "legal" for c in results)
        assert "Darksteel Angel" in collector.names

    async def test_filter_cards_ignores_unreleased_failing_other_criteria(
        self, loaded_client: ScryfallBulkClient
    ):
        """A card the OTHER criteria reject was not hidden by the legality filter,
        so reporting it as excluded would be a false claim."""
        collector = UnreleasedCollector(active=True, today=self._TODAY)
        await loaded_client.filter_cards(
            format="commander",
            type_contains=["Sorcery"],  # Darksteel Angel is a creature
            unreleased=collector,
        )
        assert "Darksteel Angel" not in collector.names

    async def test_filter_cards_collects_even_when_limit_reached(
        self, loaded_client: ScryfallBulkClient
    ):
        """A tiny limit must not stop the unreleased hunt: the guard exists
        precisely for searches that look complete."""
        collector = UnreleasedCollector(active=True, today=self._TODAY)
        results = await loaded_client.filter_cards(
            format="commander",
            type_contains=["Creature"],
            limit=1,
            unreleased=collector,
        )
        assert len(results) == 1
        assert "Darksteel Angel" in collector.names

    async def test_filter_cards_without_collector_unchanged(
        self, loaded_client: ScryfallBulkClient
    ):
        """No collector, no behavior change: unreleased cards are simply filtered."""
        results = await loaded_client.filter_cards(format="commander", type_contains=["Angel"])
        assert all(c.name != "Darksteel Angel" for c in results)

    async def test_released_card_never_reported(self, loaded_client: ScryfallBulkClient):
        """After release day the same card is a genuine legality miss, not a hidden one."""
        collector = UnreleasedCollector(active=True, today="2027-01-01")
        await loaded_client.filter_cards(
            format="commander",
            type_contains=["Angel"],
            unreleased=collector,
        )
        assert "Darksteel Angel" not in collector.names

    async def test_random_card_collects_unreleased_matches(self, loaded_client: ScryfallBulkClient):
        collector = UnreleasedCollector(active=True, today=self._TODAY)
        await loaded_client.random_card(
            format="commander",
            type_contains="Angel",
            unreleased=collector,
        )
        assert "Darksteel Angel" in collector.names

    async def test_filter_cards_include_mode_keeps_unreleased_in_results(
        self, loaded_client: ScryfallBulkClient
    ):
        """Owner default (wired by the tools): the upcoming card IS a result,
        and still collected so callers can mark it."""
        collector = UnreleasedCollector(active=True, today=self._TODAY)
        results = await loaded_client.filter_cards(
            format="commander",
            type_contains=["Angel"],
            unreleased=collector,
            include_unreleased=True,
        )
        assert any(c.name == "Darksteel Angel" for c in results)
        assert "Darksteel Angel" in collector.names

    async def test_random_card_include_mode_pool_contains_unreleased(
        self, loaded_client: ScryfallBulkClient
    ):
        """With a pool narrowed to the upcoming card, include mode can draw it."""
        collector = UnreleasedCollector(active=True, today=self._TODAY)
        card = await loaded_client.random_card(
            format="commander",
            type_contains="Artifact Creature",  # Darksteel Angel's type line
            unreleased=collector,
            include_unreleased=True,
        )
        assert "Darksteel Angel" in collector.names
        assert card is not None

    async def test_include_mode_without_collector_changes_nothing(
        self, loaded_client: ScryfallBulkClient
    ):
        """Without a collector, legality cannot be waived: the flag is inert."""
        results = await loaded_client.filter_cards(
            format="commander",
            type_contains=["Angel"],
            include_unreleased=True,
        )
        assert all(c.name != "Darksteel Angel" for c in results)


class TestETagEdgeCases:
    """Test ETag edge cases."""

    async def test_missing_etag_header_leaves_etag_none(self):
        """Response without ETag header does not set _etag."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx)  # No ETag header

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert client._etag is None


class TestBackgroundRefreshEdgeCases:
    """Test background refresh edge cases."""

    async def test_start_background_refresh_idempotent(self):
        """Calling start_background_refresh twice does not create duplicate tasks."""
        client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
        async with client:
            client.start_background_refresh()
            first_task = client._refresh_task
            assert first_task is not None

            client.start_background_refresh()
            assert client._refresh_task is first_task  # Same task, not a new one


# ===========================================================================
# New method test classes
# ===========================================================================


class TestFilterCards:
    """Test multi-criteria card filtering."""

    async def test_filter_by_format(self, loaded_client: ScryfallBulkClient):
        """Filter by format returns only cards legal in that format."""
        results = await loaded_client.filter_cards(format="modern")
        assert len(results) > 0
        for card in results:
            assert card.legalities.get("modern") == "legal"

    async def test_filter_by_color_identity(self, loaded_client: ScryfallBulkClient):
        """Filter by color identity returns cards with subset identity."""
        sultai = frozenset({"B", "G", "U"})
        results = await loaded_client.filter_cards(color_identity=sultai)
        assert len(results) > 0
        for card in results:
            assert frozenset(card.color_identity).issubset(sultai)

    async def test_filter_by_type(self, loaded_client: ScryfallBulkClient):
        """Filter by type_contains returns cards with matching type line."""
        results = await loaded_client.filter_cards(type_contains=["Creature"])
        assert len(results) > 0
        for card in results:
            assert "Creature" in card.type_line

    async def test_filter_by_text_any(self, loaded_client: ScryfallBulkClient):
        """Filter by text_any returns cards matching ANY oracle text term."""
        results = await loaded_client.filter_cards(text_any=["destroy", "exile"])
        assert len(results) > 0
        for card in results:
            text = (card.oracle_text or "").lower()
            assert "destroy" in text or "exile" in text

    async def test_filter_by_cmc(self, loaded_client: ScryfallBulkClient):
        """Filter by cmc_eq returns cards with exact CMC."""
        results = await loaded_client.filter_cards(cmc_eq=2.0)
        assert len(results) > 0
        for card in results:
            assert card.cmc == 2.0

    async def test_filter_by_max_price(self, loaded_client: ScryfallBulkClient):
        """Filter by max_price returns only cards at or below that price."""
        results = await loaded_client.filter_cards(max_price=1.0)
        assert len(results) > 0
        for card in results:
            assert card.prices.usd is not None
            assert float(card.prices.usd) <= 1.0

    async def test_filter_by_rarity(self, loaded_client: ScryfallBulkClient):
        """Filter by rarity returns only cards of that rarity."""
        results = await loaded_client.filter_cards(rarity="mythic")
        assert len(results) > 0
        for card in results:
            assert card.rarity == "mythic"

    async def test_filter_combined(self, loaded_client: ScryfallBulkClient):
        """Combined filters narrow results correctly."""
        results = await loaded_client.filter_cards(
            format="commander", type_contains=["Creature"], cmc_lte=3.0
        )
        assert len(results) > 0
        for card in results:
            assert card.legalities.get("commander") == "legal"
            assert "Creature" in card.type_line
            assert card.cmc <= 3.0

    async def test_filter_limit(self, loaded_client: ScryfallBulkClient):
        """Limit parameter caps the number of returned results."""
        results = await loaded_client.filter_cards(limit=3)
        assert len(results) <= 3

    async def test_filter_empty_result(self, loaded_client: ScryfallBulkClient):
        """Nonexistent rarity returns empty list."""
        results = await loaded_client.filter_cards(rarity="nonexistent")
        assert results == []

    async def test_filter_name_contains(self, loaded_client: ScryfallBulkClient):
        """Filter by name_contains finds cards with matching name substring."""
        results = await loaded_client.filter_cards(name_contains="bolt")
        assert any(c.name == "Lightning Bolt" for c in results)

    async def test_filter_text_contains_all(self, loaded_client: ScryfallBulkClient):
        """Filter by text_contains requires ALL strings to match."""
        results = await loaded_client.filter_cards(text_contains=["destroy", "creature"])
        assert len(results) > 0
        for card in results:
            text = (card.oracle_text or "").lower()
            assert "destroy" in text
            assert "creature" in text

    async def test_filter_keywords(self, loaded_client: ScryfallBulkClient):
        """Filter by keywords returns cards with matching keywords."""
        results = await loaded_client.filter_cards(keywords=["Persist"])
        assert any(c.name == "Kitchen Finks" for c in results)

    async def test_filter_cmc_lte(self, loaded_client: ScryfallBulkClient):
        """Filter by cmc_lte returns cards at or below that CMC."""
        results = await loaded_client.filter_cards(cmc_lte=1.0)
        assert len(results) > 0
        for card in results:
            assert card.cmc <= 1.0


class TestCardsByLegality:
    """Test legality-based card listing."""

    async def test_banned_in_commander(self, loaded_client: ScryfallBulkClient):
        """Returns cards banned in commander."""
        results = await loaded_client.cards_by_legality("commander", "banned")
        names = {c.name for c in results}
        assert "Black Lotus" in names
        assert "Channel" in names

    async def test_banned_in_modern(self, loaded_client: ScryfallBulkClient):
        """Returns cards banned in modern."""
        results = await loaded_client.cards_by_legality("modern", "banned")
        names = {c.name for c in results}
        assert "Birthing Pod" in names

    async def test_legal_returns_cards(self, loaded_client: ScryfallBulkClient):
        """Returns cards legal in commander."""
        results = await loaded_client.cards_by_legality("commander", "legal")
        assert len(results) > 0
        for card in results:
            assert card.legalities.get("commander") == "legal"

    async def test_empty_for_nonexistent_format(self, loaded_client: ScryfallBulkClient):
        """Nonexistent format returns empty list."""
        results = await loaded_client.cards_by_legality("nonexistent_format", "legal")
        assert results == []


class TestGetCards:
    """Test batch exact-name lookup."""

    async def test_batch_lookup(self, loaded_client: ScryfallBulkClient):
        """Batch lookup returns cards for known names."""
        result = await loaded_client.get_cards(["Sol Ring", "Lightning Bolt"])
        assert result["Sol Ring"] is not None
        assert result["Sol Ring"].name == "Sol Ring"
        assert result["Lightning Bolt"] is not None

    async def test_missing_card(self, loaded_client: ScryfallBulkClient):
        """Missing cards return None in the dict."""
        result = await loaded_client.get_cards(["Sol Ring", "Nonexistent Card"])
        assert result["Sol Ring"] is not None
        assert result["Nonexistent Card"] is None

    async def test_case_insensitive(self, loaded_client: ScryfallBulkClient):
        """Batch lookup is case-insensitive, preserving input key casing."""
        result = await loaded_client.get_cards(["sol ring"])
        assert result["sol ring"] is not None
        assert result["sol ring"].name == "Sol Ring"

    async def test_empty_list(self, loaded_client: ScryfallBulkClient):
        """Empty input returns empty dict."""
        result = await loaded_client.get_cards([])
        assert result == {}


class TestRandomCard:
    """Test random card selection."""

    async def test_random_returns_card(self, loaded_client: ScryfallBulkClient):
        """Random card returns a Card instance."""
        from mtg_mcp_server.types import Card

        card = await loaded_client.random_card()
        assert card is not None
        assert isinstance(card, Card)

    async def test_random_with_format(self, loaded_client: ScryfallBulkClient):
        """Random card with format filter returns a legal card."""
        card = await loaded_client.random_card(format="modern")
        if card is not None:
            assert card.legalities.get("modern") == "legal"

    async def test_random_with_type(self, loaded_client: ScryfallBulkClient):
        """Random card with type filter returns a matching card."""
        card = await loaded_client.random_card(type_contains="Creature")
        if card is not None:
            assert "Creature" in card.type_line

    async def test_random_with_rarity(self, loaded_client: ScryfallBulkClient):
        """Random card with rarity filter returns a matching card."""
        card = await loaded_client.random_card(rarity="mythic")
        if card is not None:
            assert card.rarity == "mythic"

    async def test_random_empty_pool(self, loaded_client: ScryfallBulkClient):
        """Random card from empty pool returns None."""
        card = await loaded_client.random_card(rarity="nonexistent")
        assert card is None

    async def test_random_with_color_identity(self, loaded_client: ScryfallBulkClient):
        """Random card with color_identity filter returns a subset match."""
        card = await loaded_client.random_card(color_identity=frozenset({"R"}))
        if card is not None:
            assert frozenset(card.color_identity).issubset({"R"})


# ===========================================================================
# Scryfall bulk-data format (2026-07 schema change)
# ===========================================================================


class TestJsonlGzipFormat:
    """Scryfall serves bulk data as gzipped JSONL under a renamed key.

    Origin (2026-07-29): ``deck_validate`` and every other bulk-backed tool went
    down. Three changes landed together and only the first was obvious:

    1. ``download_uri`` became ``jsonl_download_uri``.
    2. The payload became JSONL (one card per line), not a JSON array.
    3. It is served as ``Content-Type: application/gzip`` with NO
       ``Content-Encoding``, so httpx hands back compressed bytes untouched.

    Renaming the key alone would have swapped one error message for another.
    """

    async def test_loads_from_jsonl_download_uri(self):
        """The live metadata shape: jsonl_download_uri, gzipped JSONL body."""
        metadata = {
            "object": "bulk_data",
            "id": "27bf3214-1271-490b-bdfe-c0be6c23d02e",
            "type": "oracle_cards",
            "updated_at": "2026-07-28T23:50:06.858+00:00",
            "uri": f"{_BASE_URL}/bulk-data/27bf3214-1271-490b-bdfe-c0be6c23d02e",
            "name": "Oracle Cards",
            "description": "A JSON file containing one Scryfall card object for each Oracle ID.",
            "jsonl_download_uri": _JSONL_DOWNLOAD_URL,
            "compressed_size": 24330439,
        }
        with respx.mock:
            _mock_metadata_route(respx, metadata=metadata)
            _mock_download_route(respx, url=_JSONL_DOWNLOAD_URL)

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert len(await client.all_cards()) > 0
                assert await client.get_card("Lightning Bolt") is not None

    async def test_uncompressed_jsonl_still_parses(self):
        """Gzip is detected from the bytes, not assumed from the file extension."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=_oracle_cards_jsonl())

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert len(await client.all_cards()) > 0

    async def test_legacy_json_array_still_parses(self):
        """A JSON array body keeps working, so a Scryfall rollback is not an outage."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=_load_oracle_cards_bytes())

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert len(await client.all_cards()) > 0

    async def test_legacy_download_uri_key_still_accepted(self):
        """A rollback to the old key name must not need a redeploy."""
        metadata = {
            "object": "bulk_data",
            "type": "oracle_cards",
            "download_uri": _JSONL_DOWNLOAD_URL,
        }
        with respx.mock:
            _mock_metadata_route(respx, metadata=metadata)
            _mock_download_route(respx, url=_JSONL_DOWNLOAD_URL)

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert len(await client.all_cards()) > 0

    async def test_error_names_both_accepted_keys(self):
        """When neither key is present, say which ones were looked for."""
        with respx.mock:
            _mock_metadata_route(respx, metadata={"object": "bulk_data", "type": "oracle_cards"})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkError, match="jsonl_download_uri"):
                    await client.ensure_loaded()

    async def test_corrupt_gzip_is_reported_as_a_parse_failure(self):
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=b"\x1f\x8b\x08" + b"garbage")

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkError, match=r"[Ff]ailed to parse|decompress"):
                    await client.ensure_loaded()

    async def test_blank_and_malformed_lines_are_skipped_not_fatal(self):
        """One bad line in 30000 must not take the whole download down."""
        body = _oracle_cards_jsonl().split(b"\n")
        body.insert(1, b"")
        body.insert(2, b"{not json")
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=b"\n".join(body))

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert len(await client.all_cards()) > 0


class TestTruncatedPayloadGuards:
    """A payload that parses but is mostly rubbish is worse than one that fails.

    The pool shrinks silently and every downstream tool then answers "card not
    found" with total confidence, which is the exact failure class the JSONL fix
    was meant to close rather than reopen.
    """

    async def test_refresh_refuses_to_shrink_the_pool(self):
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, headers={"ETag": '"v1"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=1)
            async with client:
                await client.ensure_loaded()
                healthy = len(await client.all_cards())
                assert healthy > 4

                # A refresh that returns almost nothing must not replace it.
                one_card = json.loads((FIXTURES / "oracle_cards_sample.json").read_text())[:1]
                respx.reset()
                _mock_metadata_route(respx)
                _mock_download_route(respx, content=gzip.compress(json.dumps(one_card[0]).encode()))
                client._loaded_at = time.monotonic() - 7200

                await client.ensure_loaded()
                # ensure_loaded swallows refresh failures and serves stale data.
                assert len(await client.all_cards()) == healthy

    async def test_first_load_is_not_blocked_by_the_shrink_guard(self):
        """The guard compares against a pool we already have, so a small first
        load (every test fixture) must still be allowed through."""
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx)

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                await client.ensure_loaded()
                assert len(await client.all_cards()) > 0

    async def test_refresh_failure_is_swallowed_on_a_freshly_booted_host(self):
        """A refresh failure must serve stale data whatever the machine's uptime.

        Origin (2026-07-29): "is this a refresh?" was answered by
        ``self._loaded_at > 0``, a monotonic timestamp. ``time.monotonic()``
        counts from boot, so on a host up for less than the refresh interval the
        stored timestamp is NEGATIVE and a refresh was misread as a first load
        — which propagates instead of serving what we already have. It failed in
        CI on one Python version and passed on the other purely because the two
        runners had different uptimes, and it would hit a production server for
        the first hours after every restart.
        """
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, headers={"ETag": '"v1"'})

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=1)
            async with client:
                await client.ensure_loaded()
                healthy = len(await client.all_cards())
                assert healthy > 4

                respx.reset()
                _mock_metadata_route(respx)
                _mock_download_route(respx, content=b"{not json at all")
                # A host booted 10 minutes ago, asked for a 1-hour refresh.
                client._loaded_at = -3000.0

                await client.ensure_loaded()
                assert len(await client.all_cards()) == healthy

    async def test_unreadable_lines_are_counted_in_the_denominator(self):
        """ "0 of 1000" about a file whose other 29000 lines were unreadable sends
        whoever reads it looking in the wrong place."""
        body = b"\n".join([b"{not json"] * 9 + [json.dumps({"no": "card"}).encode()])
        with respx.mock:
            _mock_metadata_route(respx)
            _mock_download_route(respx, content=body)

            client = ScryfallBulkClient(base_url=_BASE_URL, refresh_hours=24)
            async with client:
                with pytest.raises(ScryfallBulkError) as exc_info:
                    await client.ensure_loaded()
                message = str(exc_info.value)
                assert "from 10 lines" in message
                assert "9 unreadable" in message
