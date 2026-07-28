"""Tests for cost_reduction_check.

Regression origin (2026-07-27): "Silver-Fur Master + Skullsnatcher -> ninjutsu at {0}"
(false — the cost is all coloured pips), then "reduces 4 ninjutsu" (the answer was 9 of
15). Same card, two wrong numbers, opposite directions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mtg_mcp_server.types import Card
from mtg_mcp_server.workflows.cost_reduction import cost_reduction_check


def _card(name: str, *, oracle: str | None = None, mana_cost: str = "{1}{U}") -> Card:
    return Card(
        id=name.lower(),
        name=name,
        type_line="Creature — Ninja",
        oracle_text=oracle,
        mana_cost=mana_cost,
    )


SILVER_FUR = _card(
    "Silver-Fur Master",
    oracle=(
        "Ninjutsu {1}{U}\nOther Ninja spells you cast cost {1} less to cast. "
        "Ninjutsu abilities you activate cost {1} less to activate."
    ),
    mana_cost="{2}{U}",
)
SKULLSNATCHER = _card("Skullsnatcher", oracle="Ninjutsu {U}{B}", mana_cost="{1}{B}")
MISTBLADE = _card("Mistblade Shinobi", oracle="Ninjutsu {1}{U}", mana_cost="{2}{U}")
PLAIN = _card("Ornithopter", oracle="Flying", mana_cost="{0}")

_ALL = [SILVER_FUR, SKULLSNATCHER, MISTBLADE, PLAIN]


@pytest.fixture
def bulk() -> AsyncMock:
    client = AsyncMock()
    by_name = {c.name.lower(): c for c in _ALL}

    async def get_card(name: str) -> Card | None:
        return by_name.get(name.lower())

    client.get_card.side_effect = get_card
    return client


@pytest.fixture
def scryfall() -> AsyncMock:
    client = AsyncMock()

    async def collection(names: list[str]) -> tuple[list[Card], list[str]]:
        return [], list(names)

    client.get_cards_collection.side_effect = collection
    return client


class TestRawCosts:
    async def test_coloured_only_cost_is_not_reduced(self, bulk, scryfall) -> None:
        result = await cost_reduction_check(
            "Silver-Fur Master", target_costs=["{U}{B}"], bulk=bulk, scryfall=scryfall
        )
        row = result.data["targets"][0]
        assert row["reduced"] is False
        assert row["result"] == "{U}{B}"

    async def test_generic_component_is_reduced(self, bulk, scryfall) -> None:
        result = await cost_reduction_check(
            "Silver-Fur Master", target_costs=["{2}{U}{U}"], bulk=bulk, scryfall=scryfall
        )
        row = result.data["targets"][0]
        assert row["reduced"] is True
        assert row["result"] == "{1}{U}{U}"

    async def test_count_is_reported_not_assumed(self, bulk, scryfall) -> None:
        result = await cost_reduction_check(
            "Silver-Fur Master",
            target_costs=["{U}{B}", "{1}{U}", "{2}{U}{U}"],
            bulk=bulk,
            scryfall=scryfall,
        )
        assert result.data["reduced_count"] == 2
        assert result.data["total_targets"] == 3

    async def test_markdown_cites_the_rule(self, bulk, scryfall) -> None:
        result = await cost_reduction_check(
            "Silver-Fur Master", target_costs=["{U}{B}"], bulk=bulk, scryfall=scryfall
        )
        assert "601.2f" in result.markdown


class TestTargetCards:
    async def test_keyword_activation_cost_is_the_target(self, bulk, scryfall) -> None:
        result = await cost_reduction_check(
            "Silver-Fur Master",
            target_cards=["Skullsnatcher", "Mistblade Shinobi"],
            keyword="Ninjutsu",
            bulk=bulk,
            scryfall=scryfall,
        )
        rows = {r["target"]: r for r in result.data["targets"]}
        # Skullsnatcher's NINJUTSU cost is {U}{B} even though its mana cost is {1}{B}:
        # testing the wrong cost is how "at {0}" got asserted in the first place.
        assert rows["Skullsnatcher"]["cost"] == "{U}{B}"
        assert rows["Skullsnatcher"]["reduced"] is False
        assert rows["Mistblade Shinobi"]["reduced"] is True

    async def test_mana_cost_is_used_without_a_keyword(self, bulk, scryfall) -> None:
        result = await cost_reduction_check(
            "Silver-Fur Master",
            target_cards=["Skullsnatcher"],
            bulk=bulk,
            scryfall=scryfall,
        )
        assert result.data["targets"][0]["cost"] == "{1}{B}"

    async def test_unresolved_targets_are_reported(self, bulk, scryfall) -> None:
        result = await cost_reduction_check(
            "Silver-Fur Master",
            target_cards=["Not A Card"],
            bulk=bulk,
            scryfall=scryfall,
        )
        assert "Not A Card" in result.data["unresolved"]


class TestNonReducer:
    async def test_card_without_a_reduction_clause_says_so(self, bulk, scryfall) -> None:
        result = await cost_reduction_check(
            "Ornithopter", target_costs=["{2}{U}"], bulk=bulk, scryfall=scryfall
        )
        assert result.data["reduces"] is False
        assert result.data["targets"] == []
        assert "does not reduce costs" in result.markdown
