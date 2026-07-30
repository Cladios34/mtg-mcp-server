"""Tests for deck_rules_map — the deck's mechanics, and the rules that govern them.

The rules engine and the deck tools had no connection at all: a deck could be
analysed card by card without a single rule ever being consulted.
"""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from mtg_mcp_server.types import Card, CardPrices
from mtg_mcp_server.workflows import WorkflowResult
from mtg_mcp_server.workflows.rules_deck import deck_rules_map

_RULES_URL = "https://media.wizards.com/2025/downloads/MagicCompRules%2020250404.txt"


@pytest.fixture
async def rules_service():
    """The real fixture corpus: which rules come back is the whole question."""
    from mtg_mcp_server.services.rules import RulesService

    fixture = (
        pathlib.Path(__file__).parent.parent / "fixtures" / "rules" / "comprehensive_rules.txt"
    )
    with respx.mock:
        respx.get(_RULES_URL).mock(return_value=httpx.Response(200, content=fixture.read_bytes()))
        service = RulesService(rules_url=_RULES_URL, refresh_hours=168)
        await service.ensure_loaded()
        yield service


def _card(name: str, keywords: list[str] | None = None, type_line: str = "Creature") -> Card:
    return Card(
        id=f"id-{name}",
        name=name,
        type_line=type_line,
        oracle_text="",
        keywords=keywords or [],
        prices=CardPrices(),
    )


def _scryfall(cards: dict[str, Card]) -> AsyncMock:
    """resolve_cards batches through get_cards_collection, never per-card."""
    mock = AsyncMock()

    async def get_cards_collection(names: list[str]) -> tuple[list[Card], list[str]]:
        found = [cards[n.lower()] for n in names if n.lower() in cards]
        missing = [n for n in names if n.lower() not in cards]
        return found, missing

    mock.get_cards_collection = AsyncMock(side_effect=get_cards_collection)
    return mock


class TestDeckRulesMap:
    async def test_maps_a_mechanic_to_its_governing_rule(self, rules_service):
        """A deck full of deathtouch must come back with the deathtouch rule."""
        cards = {
            "biting snake": _card("Biting Snake", ["Deathtouch"]),
            "venom drake": _card("Venom Drake", ["Deathtouch"]),
        }
        result = await deck_rules_map(
            decklist=["Venom Drake"],
            commander="Biting Snake",
            rules=rules_service,
            bulk=None,
            scryfall=_scryfall(cards),
        )

        assert isinstance(result, WorkflowResult)
        mechanics = {m["keyword"].lower(): m for m in result.data["mechanics"]}
        assert "deathtouch" in mechanics
        assert mechanics["deathtouch"]["rule"] == "702.2"

    async def test_ships_the_subrules_that_carry_the_detail(self, rules_service):
        """702.2 says "deathtouch is a static ability"; 702.2b is the one that
        makes it kill. Returning only the parent answers nothing."""
        cards = {"biting snake": _card("Biting Snake", ["Deathtouch"])}
        result = await deck_rules_map(
            decklist=[],
            commander="Biting Snake",
            rules=rules_service,
            bulk=None,
            scryfall=_scryfall(cards),
        )
        deathtouch = next(
            m for m in result.data["mechanics"] if m["keyword"].lower() == "deathtouch"
        )
        numbers = [s["number"] for s in deathtouch["subrules"]]
        assert "702.2b" in numbers, f"expected the operative subrule, got {numbers}"

    async def test_a_sibling_rule_is_not_mistaken_for_a_subrule(self, rules_service):
        """702.2 must not drag in 702.20 through 702.29."""
        cards = {"biting snake": _card("Biting Snake", ["Deathtouch"])}
        result = await deck_rules_map(
            decklist=[],
            commander="Biting Snake",
            rules=rules_service,
            bulk=None,
            scryfall=_scryfall(cards),
        )
        deathtouch = next(
            m for m in result.data["mechanics"] if m["keyword"].lower() == "deathtouch"
        )
        for sub in deathtouch["subrules"]:
            assert sub["number"][len("702.2") :].isalpha(), sub["number"]

    async def test_a_mechanic_the_corpus_does_not_know_is_reported(self, rules_service):
        """Silence would read as "nothing to know about it".

        The corpus is a dated file. Warp is absent from the April 2025 rules
        entirely, and it is the central mechanic of a real deck this server is
        used on. An uncovered mechanic has to be named, with the corpus date,
        rather than quietly dropped from the output.
        """
        cards = {"rift walker": _card("Rift Walker", ["Warp"])}
        result = await deck_rules_map(
            decklist=[],
            commander="Rift Walker",
            rules=rules_service,
            bulk=None,
            scryfall=_scryfall(cards),
        )
        uncovered = [u["keyword"].lower() for u in result.data["uncovered"]]
        assert "warp" in uncovered
        assert "warp" in result.markdown.lower()
        # The exact version, not merely "something non-empty": the first attempt
        # read the first 8 digits of the URL and reported 20202504, a corpus that
        # never existed, and an assertion on truthiness passed it happily.
        assert result.data["corpus"] == "MagicCompRules 20250404", result.data["corpus"]

    async def test_unresolved_cards_are_reported(self, rules_service):
        """A card the lookup missed is not a card without mechanics."""
        result = await deck_rules_map(
            decklist=["Nonexistent Card"],
            commander="Also Missing",
            rules=rules_service,
            bulk=None,
            scryfall=_scryfall({}),
        )
        assert set(result.data["unresolved"]) == {"Nonexistent Card", "Also Missing"}

    async def test_empty_deck_does_not_crash(self, rules_service):
        result = await deck_rules_map(
            decklist=[],
            commander="Also Missing",
            rules=rules_service,
            bulk=None,
            scryfall=_scryfall({}),
        )
        assert isinstance(result, WorkflowResult)
        assert result.data["mechanics"] == []
