"""Tests for the Scryfall service client."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from mtg_mcp_server.services.scryfall import CardNotFoundError, ScryfallClient, ScryfallError
from mtg_mcp_server.types import Card, CardSearchResult, Ruling, SetInfo

FIXTURES = Path(__file__).parent.parent / "fixtures" / "scryfall"
BASE_URL = "https://api.scryfall.com"


def _load_fixture(name: str) -> dict:
    """Load a Scryfall JSON fixture by filename."""
    return json.loads((FIXTURES / name).read_text())


class TestGetCardByName:
    """Card lookup by exact or fuzzy name."""

    @respx.mock
    async def test_exact_match(self):
        """Exact name lookup returns a fully populated Card model."""
        fixture = _load_fixture("card_muldrotha.json")
        respx.get(f"{BASE_URL}/cards/named", params={"exact": "Muldrotha, the Gravetide"}).mock(
            return_value=httpx.Response(200, json=fixture)
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            card = await client.get_card_by_name("Muldrotha, the Gravetide")

        assert isinstance(card, Card)
        assert card.name == "Muldrotha, the Gravetide"
        assert card.mana_cost == "{3}{B}{G}{U}"
        assert card.cmc == 6.0
        assert card.type_line == "Legendary Creature \u2014 Elemental Avatar"
        assert card.colors == ["B", "G", "U"]
        assert card.power == "6"
        assert card.toughness == "6"
        assert card.rarity == "mythic"
        assert card.prices.usd == "0.57"
        assert card.edhrec_rank == 1140

    @respx.mock
    async def test_fuzzy_match(self):
        """Fuzzy name lookup resolves partial input to the correct card."""
        fixture = _load_fixture("card_muldrotha.json")
        respx.get(f"{BASE_URL}/cards/named", params={"fuzzy": "muldrotha"}).mock(
            return_value=httpx.Response(200, json=fixture)
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            card = await client.get_card_by_name("muldrotha", fuzzy=True)

        assert card.name == "Muldrotha, the Gravetide"

    @respx.mock
    async def test_not_found_raises(self):
        """Nonexistent card name raises CardNotFoundError."""
        fixture = _load_fixture("card_not_found.json")
        respx.get(f"{BASE_URL}/cards/named", params={"exact": "Xyzzy"}).mock(
            return_value=httpx.Response(404, json=fixture)
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            with pytest.raises(CardNotFoundError, match="Xyzzy"):
                await client.get_card_by_name("Xyzzy")

    @respx.mock
    async def test_mdfc_extracts_front_face_mana_cost(self):
        """Modal DFCs have mana_cost in card_faces[0], not at top level."""
        fixture = _load_fixture("card_shatterskull_smashing.json")
        respx.get(
            f"{BASE_URL}/cards/named",
            params={"exact": "Shatterskull Smashing"},
        ).mock(return_value=httpx.Response(200, json=fixture))
        async with ScryfallClient(base_url=BASE_URL) as client:
            card = await client.get_card_by_name("Shatterskull Smashing")

        assert card.mana_cost == "{X}{R}{R}"
        assert card.oracle_text is not None
        assert "damage" in card.oracle_text.lower()
        assert card.colors == ["R"]


class TestSearchCards:
    """Card search using Scryfall query syntax."""

    @respx.mock
    async def test_returns_search_result(self):
        """Search returns a CardSearchResult with pagination and Card models."""
        fixture = _load_fixture("search_sultai_commander.json")
        respx.get(
            f"{BASE_URL}/cards/search",
            params={"q": "f:commander id:sultai t:creature", "page": "1"},
        ).mock(return_value=httpx.Response(200, json=fixture))
        async with ScryfallClient(base_url=BASE_URL) as client:
            result = await client.search_cards("f:commander id:sultai t:creature")

        assert isinstance(result, CardSearchResult)
        assert result.total_cards == 9273
        assert result.has_more is True
        assert len(result.data) == 3
        assert all(isinstance(c, Card) for c in result.data)

    @respx.mock
    async def test_search_not_found_raises(self):
        """Search with no matching cards raises CardNotFoundError."""
        respx.get(f"{BASE_URL}/cards/search", params={"q": "xyzzy_nonexistent", "page": "1"}).mock(
            return_value=httpx.Response(
                404,
                json={
                    "object": "error",
                    "code": "not_found",
                    "status": 404,
                    "details": "No cards found",
                },
            )
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            with pytest.raises(CardNotFoundError):
                await client.search_cards("xyzzy_nonexistent")


class TestGetCardById:
    """Card lookup by Scryfall UUID."""

    @respx.mock
    async def test_returns_card(self):
        """ID lookup returns the correct Card model."""
        fixture = _load_fixture("card_sol_ring.json")
        card_id = fixture["id"]
        respx.get(f"{BASE_URL}/cards/{card_id}").mock(
            return_value=httpx.Response(200, json=fixture)
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            card = await client.get_card_by_id(card_id)

        assert isinstance(card, Card)
        assert card.name == "Sol Ring"


class TestGetRulings:
    """Official card rulings retrieval."""

    @respx.mock
    async def test_returns_rulings(self):
        """Rulings endpoint returns a list of Ruling models with source and date."""
        fixture = _load_fixture("rulings_muldrotha.json")
        card_id = "705b4d97-2f50-47f7-9053-d748f4337553"
        respx.get(f"{BASE_URL}/cards/{card_id}/rulings").mock(
            return_value=httpx.Response(200, json=fixture)
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            rulings = await client.get_rulings(card_id)

        assert len(rulings) == 8
        assert all(isinstance(r, Ruling) for r in rulings)
        assert rulings[0].source == "wotc"
        assert "2020-11-10" in rulings[0].published_at


class TestGetSet:
    """Set metadata lookup by code."""

    @respx.mock
    async def test_get_set(self):
        """Set lookup returns a fully populated SetInfo model."""
        fixture = _load_fixture("set_dominaria.json")
        respx.get(f"{BASE_URL}/sets/dom").mock(return_value=httpx.Response(200, json=fixture))
        async with ScryfallClient(base_url=BASE_URL) as client:
            set_info = await client.get_set("dom")

        assert isinstance(set_info, SetInfo)
        assert set_info.code == "dom"
        assert set_info.name == "Dominaria"
        assert set_info.card_count > 0
        assert set_info.released_at is not None

    @respx.mock
    async def test_get_set_not_found(self):
        """Nonexistent set code raises CardNotFoundError."""
        respx.get(f"{BASE_URL}/sets/zzz").mock(
            return_value=httpx.Response(
                404,
                json={"object": "error", "code": "not_found", "details": "No set found"},
            )
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            with pytest.raises(CardNotFoundError):
                await client.get_set("zzz")


class TestScryfallServerErrors:
    """Scryfall API server error handling."""

    @respx.mock
    async def test_500_raises_scryfall_error(self):
        """500 response from Scryfall raises ScryfallError."""
        respx.get(f"{BASE_URL}/cards/named", params={"exact": "Sol Ring"}).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            with pytest.raises(ScryfallError):
                await client.get_card_by_name("Sol Ring")


class TestGetCardsCollection:
    """Batch card lookup via POST /cards/collection (75 identifiers per request)."""

    @respx.mock
    async def test_single_batch_returns_found_and_missing(self):
        """One request; unknown names come back through Scryfall's not_found list."""
        card = _load_fixture("card_muldrotha.json")
        route = respx.post(f"{BASE_URL}/cards/collection").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [card],
                    "not_found": [{"name": "Zzzz Not A Card"}],
                },
            )
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            found, missing = await client.get_cards_collection(
                ["Muldrotha, the Gravetide", "Zzzz Not A Card"]
            )

        assert route.call_count == 1
        assert [c.name for c in found] == ["Muldrotha, the Gravetide"]
        assert missing == ["Zzzz Not A Card"]
        assert json.loads(route.calls[0].request.content) == {
            "identifiers": [{"name": "Muldrotha, the Gravetide"}, {"name": "Zzzz Not A Card"}]
        }

    @respx.mock
    async def test_chunks_at_scryfall_75_identifier_ceiling(self):
        """160 names split into 3 requests of at most 75 identifiers each."""
        route = respx.post(f"{BASE_URL}/cards/collection").mock(
            return_value=httpx.Response(200, json={"object": "list", "data": [], "not_found": []})
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            await client.get_cards_collection([f"Card {i}" for i in range(160)])

        assert route.call_count == 3
        sizes = [len(json.loads(c.request.content)["identifiers"]) for c in route.calls]
        assert sizes == [75, 75, 10]

    @respx.mock
    async def test_empty_input_makes_no_request(self):
        """An empty name list short-circuits without touching the network."""
        route = respx.post(f"{BASE_URL}/cards/collection")
        async with ScryfallClient(base_url=BASE_URL) as client:
            found, missing = await client.get_cards_collection([])

        assert route.call_count == 0
        assert found == []
        assert missing == []

    @respx.mock
    async def test_api_error_raises_scryfall_error(self):
        """A non-200 from the batch endpoint surfaces as ScryfallError."""
        respx.post(f"{BASE_URL}/cards/collection").mock(
            return_value=httpx.Response(422, json={"object": "error", "details": "bad body"})
        )
        async with ScryfallClient(base_url=BASE_URL) as client:
            with pytest.raises(ScryfallError):
                await client.get_cards_collection(["Sol Ring"])
