"""Tests for the Commander Spellbook MCP provider."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from fastmcp import Client

from mtg_mcp_server.providers.spellbook import spellbook_mcp

FIXTURES = Path(__file__).parent.parent / "fixtures" / "spellbook"
BASE_URL = "https://backend.commanderspellbook.com"


def _load_fixture(name: str) -> dict:
    """Load a Commander Spellbook JSON fixture file by name."""
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
async def client():
    """Provide an in-memory MCP client connected to the Spellbook provider."""
    async with Client(transport=spellbook_mcp) as c:
        yield c


class TestFindCombos:
    """Spellbook find_combos tool behavior."""

    @respx.mock
    async def test_returns_formatted_results(self, client: Client):
        """find_combos returns formatted combo list with IDs and attribution."""
        fixture = _load_fixture("combos_muldrotha.json")
        respx.get(
            f"{BASE_URL}/variants/",
            params={"q": 'card:"Muldrotha, the Gravetide"', "limit": "10"},
        ).mock(return_value=httpx.Response(200, json=fixture))

        result = await client.call_tool("find_combos", {"card_name": "Muldrotha, the Gravetide"})
        text = result.content[0].text
        assert "5 combo(s)" in text
        assert "Muldrotha, the Gravetide" in text
        assert "1414-2730-5131-5256" in text
        assert "Data provided by [Commander Spellbook]" in text

        # Structured output
        sc = result.structured_content
        assert isinstance(sc, dict)
        assert sc["card_name"] == "Muldrotha, the Gravetide"
        assert sc["total_combos"] == 5
        assert isinstance(sc["combos"], list)
        assert len(sc["combos"]) == 5
        assert sc["combos"][0]["id"] == "1414-2730-5131-5256"
        # Slim combo fields only
        combo = sc["combos"][0]
        assert "cards" in combo
        assert "results" in combo
        assert "color_identity" in combo
        assert isinstance(combo["cards"], list)
        assert isinstance(combo["results"], list)
        # Bloat fields excluded
        assert "description" not in combo
        assert "popularity" not in combo
        assert "legalities" not in combo
        assert "prices" not in combo
        assert "bracket_tag" not in combo

    @respx.mock
    async def test_no_combos_returns_message(self, client: Client):
        """find_combos returns a 'no combos found' message when none match."""
        fixture = _load_fixture("combos_not_found.json")
        respx.get(
            f"{BASE_URL}/variants/",
            params={"q": 'card:"Xyzzy Nonexistent"', "limit": "10"},
        ).mock(return_value=httpx.Response(200, json=fixture))

        result = await client.call_tool("find_combos", {"card_name": "Xyzzy Nonexistent"})
        text = result.content[0].text
        assert "No combos found" in text

        # Structured output — empty result
        sc = result.structured_content
        assert isinstance(sc, dict)
        assert sc["total_combos"] == 0
        assert sc["combos"] == []


class TestComboDetails:
    """Spellbook combo_details tool behavior."""

    @respx.mock
    async def test_returns_combo_data(self, client: Client):
        """combo_details returns full combo data including cards, identity, and results."""
        fixture = _load_fixture("combo_detail.json")
        respx.get(f"{BASE_URL}/variants/1414-2730-5131-5256/").mock(
            return_value=httpx.Response(200, json=fixture)
        )

        result = await client.call_tool("combo_details", {"combo_id": "1414-2730-5131-5256"})
        text = result.content[0].text
        assert "1414-2730-5131-5256" in text
        assert "Muldrotha, the Gravetide" in text
        assert "BGU" in text
        assert "Infinite death triggers" in text

        # Structured output
        sc = result.structured_content
        assert isinstance(sc, dict)
        assert sc["id"] == "1414-2730-5131-5256"
        assert sc["identity"] == "BGU"
        assert isinstance(sc["cards"], list)
        assert any(c["name"] == "Muldrotha, the Gravetide" for c in sc["cards"])

    @respx.mock
    async def test_not_found_returns_error(self, client: Client):
        """combo_details returns an error response for nonexistent combo IDs."""
        respx.get(f"{BASE_URL}/variants/9999-9999/").mock(
            return_value=httpx.Response(404, json={"detail": "Not found."})
        )

        result = await client.call_tool(
            "combo_details", {"combo_id": "9999-9999"}, raise_on_error=False
        )
        assert result.is_error


class TestFindDecklistCombos:
    """Spellbook find_decklist_combos tool behavior."""

    @respx.mock
    async def test_returns_analysis(self, client: Client):
        """find_decklist_combos returns combo analysis for a decklist."""
        fixture = _load_fixture("find_my_combos_response.json")
        respx.post(f"{BASE_URL}/find-my-combos").mock(
            return_value=httpx.Response(200, json=fixture)
        )

        result = await client.call_tool(
            "find_decklist_combos",
            {
                "commanders": ["Muldrotha, the Gravetide"],
                "decklist": ["Sol Ring", "Spore Frog"],
            },
        )
        text = result.content[0].text
        assert "BGU" in text
        assert "combo" in text.lower()

        # Structured output
        sc = result.structured_content
        assert isinstance(sc, dict)
        assert sc["identity"] == "BGU"
        assert "included" in sc
        assert "almost_included" in sc

    @respx.mock
    async def test_structured_content_is_slim(self, client: Client):
        """Structured combos are slim dicts, not full model dumps.

        The full Combo dump measured 72KB for a 5-card decklist against the
        live API; past the 100KB ResponseLimitingMiddleware ceiling the whole
        structured_content was silently dropped, leaving the tool response
        empty (seen live 2026-07-29).
        """
        fixture = _load_fixture("find_my_combos_response.json")
        respx.post(f"{BASE_URL}/find-my-combos").mock(
            return_value=httpx.Response(200, json=fixture)
        )

        result = await client.call_tool(
            "find_decklist_combos",
            {
                "commanders": ["Muldrotha, the Gravetide"],
                "decklist": ["Sol Ring", "Spore Frog"],
            },
        )
        sc = result.structured_content
        combos = sc["included"] + sc["almost_included"]
        assert combos, "fixture should yield at least one combo"
        for combo in combos:
            assert set(combo) == {"id", "cards", "results", "color_identity"}
            assert all(isinstance(name, str) for name in combo["cards"])

    @respx.mock
    async def test_almost_included_capped_by_popularity(self, client: Client):
        """Almost-included combos are capped to the most popular, total stated.

        A combo-dense 99-card deck returned 920 almost-included combos against
        the live API — the uncapped markdown alone exceeded the 100KB
        middleware ceiling and the structured_content was dropped (2026-07-29).
        """

        def _combo(i: int) -> dict:
            return {
                "id": str(i),
                "uses": [{"card": {"name": f"Card {i}"}, "zoneLocations": ["H"]}],
                "produces": [{"feature": {"name": "Win the game"}, "quantity": 1}],
                "popularity": i,
            }

        # One combo has popularity: null (nullable upstream) — the sort must
        # rank it lowest instead of crashing.
        null_combo = _combo(0)
        null_combo["popularity"] = None
        fixture = {
            "results": {
                "identity": "BGU",
                "included": [],
                "almostIncluded": [null_combo, *(_combo(i) for i in range(1, 61))],
            }
        }
        respx.post(f"{BASE_URL}/find-my-combos").mock(
            return_value=httpx.Response(200, json=fixture)
        )

        result = await client.call_tool(
            "find_decklist_combos",
            {
                "commanders": ["Muldrotha, the Gravetide"],
                "decklist": ["Sol Ring", "Spore Frog"],
            },
        )
        sc = result.structured_content
        assert len(sc["almost_included"]) == 50
        assert sc["almost_included_total"] == 61
        # The cap keeps the MOST popular combos (popularity == id here);
        # the null-popularity combo ranks last and is cut.
        kept_ids = {c["id"] for c in sc["almost_included"]}
        assert "60" in kept_ids
        assert "1" not in kept_ids
        assert "0" not in kept_ids

        text = result.content[0].text
        assert "top 50 of 61" in text
        assert "11 more" in text


class TestEstimateBracket:
    """Spellbook estimate_bracket tool behavior."""

    @respx.mock
    async def test_returns_bracket(self, client: Client):
        """estimate_bracket returns bracket tag for a decklist."""
        fixture = _load_fixture("estimate_bracket_response.json")
        respx.post(f"{BASE_URL}/estimate-bracket").mock(
            return_value=httpx.Response(200, json=fixture)
        )

        result = await client.call_tool(
            "estimate_bracket",
            {
                "commanders": ["Muldrotha, the Gravetide"],
                "decklist": ["Sol Ring", "Spore Frog"],
            },
        )
        text = result.content[0].text
        assert "R (Ruthless)" in text
        assert "Ancient Tomb" in text  # game changer surfaced in the summary

        # Structured output
        sc = result.structured_content
        assert isinstance(sc, dict)
        assert "bracket_tag" in sc
        assert sc["game_changer_cards"] == ["Ancient Tomb", "Mana Vault"]
        assert sc["two_card_combos"] == ["Satya, Aetherflux Genius + Lightning Runner"]
        assert "banned_cards" in sc
        assert "two_card_combos" in sc


class TestToolRegistration:
    """Spellbook provider tool registration."""

    async def test_all_tools_registered(self, client: Client):
        """All four Spellbook tools are registered on the provider."""
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        assert tool_names == {
            "find_combos",
            "combo_details",
            "find_decklist_combos",
            "estimate_bracket",
        }
