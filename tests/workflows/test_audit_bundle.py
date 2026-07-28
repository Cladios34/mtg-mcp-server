"""Tests for the deck_audit_bundle composite workflow."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from mtg_mcp_server.services.spellbook import SpellbookError
from mtg_mcp_server.types import (
    BracketEstimate,
    Card,
    Combo,
    ComboCard,
    ComboResult,
    DecklistCombos,
    Ruling,
)
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
        assert set(sections) == {
            "validate",
            "analysis",
            "combos",
            "bracket",
            "rulings",
            "simulation",
        }
        assert all(s["ok"] for s in sections.values())
        assert result.data["failed_sections"] == []
        assert "6/6 ok" in result.markdown

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


class TestSlimReport:
    async def test_combos_are_slimmed_not_dumped(self, patched_impls: dict[str, AsyncMock]) -> None:
        """Full Combo dumps (description, prices, legalities) tripped the 100KB
        response limit on a real 99-card deck (2026-07-24): the report must keep
        names only and point to spellbook_combo_details for the steps."""
        fat_combo = Combo(
            id="1-2-3",
            cards=[ComboCard(name="Ashnod's Altar"), ComboCard(name="Nim Deathmantle")],
            produces=[ComboResult(feature_name="Infinite colorless mana")],
            identity="C",
            description="Step 1: very long step-by-step text. " * 50,
            prices={"tcgplayer": "12.34"},
            legalities={"commander": True},
        )
        spellbook = _make_spellbook()
        spellbook.find_decklist_combos = AsyncMock(
            return_value=DecklistCombos(identity="WBR", included=[fat_combo], almost_included=[])
        )
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=spellbook,
        )
        combos = _sections_by_name(result.data)["combos"]["data"]
        entry = combos["included"][0]
        assert entry["cards"] == ["Ashnod's Altar", "Nim Deathmantle"]
        assert entry["results"] == ["Infinite colorless mana"]
        assert "description" not in entry
        assert "prices" not in entry
        assert "legalities" not in entry

    async def test_bracket_keeps_gate_fields_only(
        self, patched_impls: dict[str, AsyncMock]
    ) -> None:
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        bracket = _sections_by_name(result.data)["bracket"]["data"]
        assert bracket["bracket_tag"] == "R"
        assert set(bracket) == {
            "bracket_tag",
            "bracket_tag_name",
            "banned_cards",
            "game_changer_cards",
            "mass_land_denial_cards",
            "extra_turn_cards",
            "two_card_combos",
            "lock_combos",
        }


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


class TestReportedDeckSize:
    """`deck_size` counts physical cards, matching what the sections report."""

    async def test_quantity_entries_count_as_cards(self, patched_impls):
        """An entry of "30 Plains" is 30 cards, not 1 entry.

        Regression guard: deck_size used len(decklist), so a quantity-style list
        reported 3 while every section counted 35 (observed 2026-07-27).
        """
        result = await deck_audit_bundle(
            ["1 Sol Ring", "4 Lightning Bolt", "30 Plains"],
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        assert result.data["deck_size"] == 35

    async def test_plain_entries_still_count_one_each(self, patched_impls):
        """A bare one-name-per-card list is unaffected."""
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        assert result.data["deck_size"] == len(DECKLIST)


class TestCommanderVerification:
    """An unknown commander is called out instead of silently audited."""

    @staticmethod
    def _resolver(monkeypatch, *, unresolved: list[str]):
        async def fake(names, *, bulk, scryfall):
            return {}, unresolved

        monkeypatch.setattr("mtg_mcp_server.workflows.card_resolver.resolve_cards", fake)

    async def test_unknown_commander_is_flagged(self, patched_impls, monkeypatch):
        """A commander no source can resolve is reported, loudly.

        Regression guard: the bundle happily returned a full report for
        "Zzzz Not A Card" (observed 2026-07-27), so a typo produced an audit
        that looked authoritative while its commander-keyed data was empty.
        """
        self._resolver(monkeypatch, unresolved=["Zzzz Not A Card"])
        result = await deck_audit_bundle(
            DECKLIST,
            "Zzzz Not A Card",
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        assert result.data["commander_resolved"] is False
        assert "Zzzz Not A Card" in result.markdown
        assert "not found" in result.markdown.lower()

    async def test_known_commander_reports_resolved(self, patched_impls, monkeypatch):
        """A real commander sets the flag and adds no warning."""
        self._resolver(monkeypatch, unresolved=[])
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        assert result.data["commander_resolved"] is True
        assert "not found" not in result.markdown.lower()

    async def test_resolver_failure_does_not_fail_the_bundle(self, patched_impls, monkeypatch):
        """If the lookup itself dies, the audit still runs and says nothing false."""

        async def boom(names, *, bulk, scryfall):
            raise RuntimeError("scryfall down")

        monkeypatch.setattr("mtg_mcp_server.workflows.card_resolver.resolve_cards", boom)
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        assert result.data["commander_resolved"] is None
        assert result.data["failed_sections"] == []


class TestSpellbookCallDeduplication:
    """Each Spellbook payload is fetched once and shared across sections."""

    async def test_each_endpoint_called_once(self, patched_impls):
        """combos, bracket and analysis share two calls, not four.

        The Spellbook client is capped at 3 req/s behind a single-slot
        semaphore, so duplicate calls serialize. Measured 2026-07-27: the
        redundant pair pushed the analysis section to 8.7s waiting its turn.
        """
        spellbook = _make_spellbook()
        await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=spellbook,
        )
        assert spellbook.estimate_bracket.await_count == 1
        assert spellbook.find_decklist_combos.await_count == 1

    async def test_analysis_receives_the_shared_fetches(self, patched_impls):
        """deck_analysis is handed the in-flight awaitables instead of re-fetching."""
        await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=AsyncMock(),
            spellbook=_make_spellbook(),
        )
        kwargs = patched_impls["analysis"].await_args.kwargs
        assert kwargs["bracket_coro"] is not None
        assert kwargs["combo_coro"] is not None


class TestRulingsSection:
    """The rulings section (added 2026-07-27).

    Origin: three rules errors in one audit all came from rulings that were
    fetched through a separate, skippable call and then never confronted with
    the report's own claims. Shipping them inside the bundle removes the option
    of not looking — so the section must exist, must carry the oracle text, and
    must be VISIBLE in the markdown, not only in structured_content.
    """

    @staticmethod
    def _scryfall_with_rulings(comments: list[str]) -> AsyncMock:
        scryfall = AsyncMock()
        card = AsyncMock()
        card.id = "abc-123"
        card.name = "Yuriko, the Tiger's Shadow"
        card.oracle_text = "Commander ninjutsu {U}{B}"
        scryfall.get_card_by_name = AsyncMock(return_value=card)
        scryfall.get_rulings = AsyncMock(
            return_value=[
                Ruling(source="wotc", published_at="2020-11-10", comment=c) for c in comments
            ]
        )
        return scryfall

    async def test_rulings_are_returned_with_oracle_text(self, patched_impls) -> None:
        # The real ruling that was sitting unread during the Yuriko audit.
        comment = "Activating commander ninjutsu won't increase the commander tax."
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=self._scryfall_with_rulings([comment]),
            spellbook=_make_spellbook(),
        )
        section = _sections_by_name(result.data)["rulings"]
        assert section["ok"] is True
        assert section["data"]["total_rulings"] == 1
        assert section["data"]["rulings"][0]["comment"] == comment
        # A ruling is unusable without the text it annotates.
        assert section["data"]["oracle_text"] == "Commander ninjutsu {U}{B}"

    async def test_rulings_count_is_surfaced_in_markdown(self, patched_impls) -> None:
        """Buried in structured_content is exactly how they got ignored before."""
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=self._scryfall_with_rulings(["a", "b"]),
            spellbook=_make_spellbook(),
        )
        assert "2 official ruling(s)" in result.markdown

    async def test_no_rulings_says_so_explicitly(self, patched_impls) -> None:
        """Zero rulings is a fact to state, never a silently missing line."""
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=self._scryfall_with_rulings([]),
            spellbook=_make_spellbook(),
        )
        assert "rulings: none published" in result.markdown

    async def test_signature_mechanic_carriers_ship_their_rulings(self, patched_impls) -> None:
        """The commander's mechanic is carried by other cards, and they have rulings too.

        The Yuriko audit reasoned about a signature mechanic from memory while the
        rulings for the cards carrying it were one skippable call away.
        """
        yuriko = Card(
            id="yuriko",
            name="Yuriko, the Tiger's Shadow",
            type_line="Legendary Creature — Human Ninja",
            oracle_text="Commander ninjutsu {1}{U}{B}",
            keywords=["Commander ninjutsu"],
        )
        thief = Card(
            id="thief",
            name="Prosperous Thief",
            type_line="Creature — Ninja",
            oracle_text="Ninjutsu {1}{U}",
            keywords=["Ninjutsu"],
        )

        scryfall = AsyncMock()
        scryfall.get_card_by_name = AsyncMock(return_value=yuriko)
        scryfall.get_rulings = AsyncMock(
            return_value=[Ruling(source="wotc", published_at="2020-11-10", comment="a ruling")]
        )
        bulk = AsyncMock()
        bulk.get_card = AsyncMock(
            side_effect=lambda name: thief if name.lower() == "prosperous thief" else None
        )
        scryfall.get_cards_collection = AsyncMock(return_value=([], []))

        result = await deck_audit_bundle(
            ["Prosperous Thief"],
            "Yuriko, the Tiger's Shadow",
            "ub",
            iterations=100,
            bulk=bulk,
            scryfall=scryfall,
            spellbook=_make_spellbook(),
        )
        data = _sections_by_name(result.data)["rulings"]["data"]
        assert data["signature_mechanic"] == "Ninjutsu"
        assert "Prosperous Thief" in data["signature_mechanic_cards"]
        assert "signature mechanic 'Ninjutsu'" in result.markdown

    async def test_commander_without_a_signature_mechanic_is_not_forced(
        self, patched_impls
    ) -> None:
        """No signature keyword means no section noise, and no invented one."""
        plain = Card(
            id="plain",
            name="Kaalia of the Vast",
            type_line="Legendary Creature — Human Cleric",
            oracle_text="Flying",
            keywords=["Flying"],  # evergreen — never a signature mechanic
        )
        scryfall = AsyncMock()
        scryfall.get_card_by_name = AsyncMock(return_value=plain)
        scryfall.get_rulings = AsyncMock(return_value=[])

        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=scryfall,
            spellbook=_make_spellbook(),
        )
        data = _sections_by_name(result.data)["rulings"]["data"]
        assert data["signature_mechanic"] is None
        assert data["signature_mechanic_cards"] == {}

    async def test_rulings_failure_never_takes_the_audit_down(self, patched_impls) -> None:
        """Silent-failure ban: the section reports its own error, the rest survives."""
        scryfall = AsyncMock()
        scryfall.get_card_by_name = AsyncMock(side_effect=RuntimeError("scryfall down"))
        result = await deck_audit_bundle(
            DECKLIST,
            COMMANDER,
            "mardu",
            iterations=100,
            bulk=AsyncMock(),
            scryfall=scryfall,
            spellbook=_make_spellbook(),
        )
        section = _sections_by_name(result.data)["rulings"]
        assert section["ok"] is False
        assert "scryfall down" in section["error"]
        assert result.data["failed_sections"] == ["rulings"]
