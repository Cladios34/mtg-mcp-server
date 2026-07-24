"""Tests for the deck_audit_bundle composite workflow."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from mtg_mcp_server.services.spellbook import SpellbookError
from mtg_mcp_server.types import BracketEstimate, DecklistCombos
from mtg_mcp_server.workflows import WorkflowResult
from mtg_mcp_server.workflows.audit_bundle import deck_audit_bundle

COMMANDER = "Kaalia of the Vast"
DECKLIST = ["Sol Ring", "Swords to Plowshares", "Mountain", "Plains", "Swamp"]


def _make_bracket() -> BracketEstimate:
    return BracketEstimate.model_validate({"bracketTag": "R", "cards": [], "combos": []})


def _make_decklist_combos() -> DecklistCombos:
    return DecklistCombos(identity="WBR", included=[], almost_included=[])


def _make_spellbook(
    *,
    bracket_error: Exception | None = None,
) -> AsyncMock:
    spellbook = AsyncMock()
    spellbook.find_decklist_combos = AsyncMock(return_value=_make_decklist_combos())
    if bracket_error is not None:
        spellbook.estimate_bracket = AsyncMock(side_effect=bracket_error)
    else:
        spellbook.estimate_bracket = AsyncMock(return_value=_make_bracket())
    return spellbook


@pytest.fixture
def patched_impls(monkeypatch: pytest.MonkeyPatch) -> dict[str, AsyncMock]:
    """Patch the three heavy section impls with observable mocks."""
    validate = AsyncMock(return_value=WorkflowResult("md", {"valid": True}))
    analysis = AsyncMock(return_value=WorkflowResult("md", {"curve": {"2": 3}}))
    simulate = AsyncMock(return_value=WorkflowResult("md", {"keep_first_deal_pct": 79.3}))
    monkeypatch.setattr("mtg_mcp_server.workflows.audit_bundle.deck_validate", validate)
    monkeypatch.setattr("mtg_mcp_server.workflows.audit_bundle.deck_analysis", analysis)
    monkeypatch.setattr("mtg_mcp_server.workflows.audit_bundle.simulate_opening_hands", simulate)
    return {"validate": validate, "analysis": analysis, "simulate": simulate}


def _sections_by_name(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["section"]: s for s in data["sections"]}


class TestHappyPath:
    async def test_all_sections_ok(self, patched_impls: dict[str, AsyncMock]) -> None:
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            seed=42,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        sections = _sections_by_name(result.data)
        assert set(sections) == {"validate", "analysis", "combos", "bracket", "simulation"}
        assert all(s["ok"] for s in sections.values())
        assert result.data["failed_sections"] == []
        assert "5/5 ok" in result.markdown

    async def test_simulation_v3_forced(self, patched_impls: dict[str, AsyncMock]) -> None:
        await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "WBR",
            seed=7,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        kwargs = patched_impls["simulate"].call_args.kwargs
        assert kwargs["commander_colors"] == "WBR"
        assert kwargs["tutor_aware"] is True
        assert kwargs["keep_rule"] == "playability"
        assert kwargs["free_mulligan"] is True
        assert kwargs["seed"] == 7

    async def test_params_echoed_in_report(self, patched_impls: dict[str, AsyncMock]) -> None:
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            seed=42,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        sim = _sections_by_name(result.data)["simulation"]
        assert sim["params_used"]["commander_colors"] == "mardu"
        assert sim["params_used"]["tutor_aware"] is True
        assert sim["params_used"]["seed"] == 42


class TestFailureIsolation:
    async def test_failed_section_reported_not_raised(
        self, patched_impls: dict[str, AsyncMock]
    ) -> None:
        spellbook = _make_spellbook(bracket_error=SpellbookError("upstream 500"))
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=spellbook,
        )
        sections = _sections_by_name(result.data)
        assert sections["bracket"]["ok"] is False
        assert "upstream 500" in sections["bracket"]["error"]
        assert sections["combos"]["ok"] is True
        assert result.data["failed_sections"] == ["bracket"]
        assert "FAILED: bracket" in result.markdown

    async def test_bulk_disabled_fails_validate_only(
        self, patched_impls: dict[str, AsyncMock]
    ) -> None:
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            bulk=None,
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        sections = _sections_by_name(result.data)
        assert sections["validate"]["ok"] is False
        assert "bulk" in sections["validate"]["error"]
        assert sections["simulation"]["ok"] is True

    async def test_no_silent_zero_every_section_has_status(
        self, patched_impls: dict[str, AsyncMock]
    ) -> None:
        spellbook = _make_spellbook(bracket_error=SpellbookError("boom"))
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=spellbook,
        )
        for section in result.data["sections"]:
            assert "ok" in section
            assert section["ok"] or "error" in section
            assert "params_used" in section
