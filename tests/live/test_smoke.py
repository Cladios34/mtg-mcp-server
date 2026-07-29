"""Live smoke tests — starts a real server and hits real external APIs.

These tests are SLOW (30+ seconds for bulk data download) and require network
access. They are excluded from the default test run and must be invoked
explicitly via ``mise run test:live`` or ``pytest -m live``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


class TestServerHealth:
    """Basic server health and tool registration checks."""

    async def test_ping(self, live_client):
        result = await live_client.call_tool("ping", {})
        assert result.content[0].text == "pong"

    async def test_all_tools_registered(self, live_client):
        """The running server exposes exactly what this build registers.

        Derived from the server itself rather than hardcoded: the count lived in
        three files, and this copy sat at 69 while the other two had moved on —
        the assertion meant to catch a missing tool was itself stale.
        """
        from fastmcp import Client

        from mtg_mcp_server.server import mcp

        async with Client(transport=mcp) as local:
            expected = {t.name for t in await local.list_tools()}

        tools = await live_client.list_tools()
        tool_names = {t.name for t in tools}
        assert tool_names == expected, (
            f"Live server differs from this build. "
            f"Missing: {sorted(expected - tool_names)}. "
            f"Unexpected: {sorted(tool_names - expected)}"
        )

    async def test_no_mtgjson_tools(self, live_client):
        tools = await live_client.list_tools()
        mtgjson_tools = [t.name for t in tools if "mtgjson" in t.name]
        assert mtgjson_tools == [], f"Unexpected MTGJSON tools: {mtgjson_tools}"


class TestSpicerackLive:
    """Hit the real Spicerack API."""

    async def test_recent_tournaments(self, live_client):
        """Spicerack answers, or it is unreachable — those are different verdicts.

        The upstream service has gone fully dark before (site unreachable from two
        networks, 2026-07-22). A red suite every time a third party is down trains
        everyone to ignore it, and the one failure that matters gets ignored with it.

        So an unreachable host SKIPS, while any other error still FAILS: a schema
        drift or a broken wrapper does not produce a network error, and that is the
        regression this test exists to catch.
        """
        result = await live_client.call_tool(
            "spicerack_recent_tournaments",
            {"format": "Legacy", "num_days": 30},
            raise_on_error=False,
        )
        text = result.content[0].text

        if result.is_error:
            if "Network error" in text or "timed out" in text.lower():
                pytest.skip(f"Spicerack unreachable, not our regression: {text}")
            pytest.fail(f"Spicerack returned an error that is not a network fault: {text}")

        assert "Legacy" in text


class TestBulkDataLive:
    """Hit the real Scryfall bulk data. First call triggers a ~30MB download."""

    async def test_sol_ring_is_artifact(self, live_client):
        result = await live_client.call_tool("bulk_card_lookup", {"name": "Sol Ring"})
        assert "Artifact" in result.content[0].text

    async def test_lightning_bolt_has_prices(self, live_client):
        result = await live_client.call_tool("bulk_card_lookup", {"name": "Lightning Bolt"})
        text = result.content[0].text
        assert "$" in text

    async def test_sol_ring_has_legalities(self, live_client):
        result = await live_client.call_tool("bulk_card_lookup", {"name": "Sol Ring"})
        assert "commander" in result.content[0].text.lower()

    async def test_sol_ring_has_edhrec_rank(self, live_client):
        result = await live_client.call_tool("bulk_card_lookup", {"name": "Sol Ring"})
        assert "EDHREC Rank" in result.content[0].text

    async def test_search_returns_results(self, live_client):
        result = await live_client.call_tool("bulk_card_search", {"query": "Lightning Bolt"})
        assert "Found" in result.content[0].text

    async def test_dfc_lookup(self, live_client):
        result = await live_client.call_tool("bulk_card_lookup", {"name": "Delver of Secrets"})
        text = result.content[0].text
        assert "Delver of Secrets" in text


class TestScryfallLive:
    """Hit the real Scryfall API."""

    async def test_card_details(self, live_client):
        result = await live_client.call_tool("scryfall_card_details", {"name": "Sol Ring"})
        assert "Artifact" in result.content[0].text

    async def test_search_cards(self, live_client):
        result = await live_client.call_tool(
            "scryfall_search_cards", {"query": "t:creature c:green cmc=1"}
        )
        assert "Found" in result.content[0].text

    async def test_set_info(self, live_client):
        result = await live_client.call_tool("scryfall_set_info", {"set_code": "dom"})
        text = result.content[0].text
        assert "Dominaria" in text


class TestDeckBuildingLive:
    """Hit real bulk data with deck building workflows."""

    async def test_theme_search(self, live_client):
        result = await live_client.call_tool(
            "theme_search", {"theme": "sacrifice", "format": "commander", "limit": 5}
        )
        text = result.content[0].text
        assert "sacrifice" in text.lower() or "Sacrifice" in text

    async def test_tribal_staples(self, live_client):
        result = await live_client.call_tool(
            "tribal_staples", {"tribe": "Elf", "format": "commander"}
        )
        text = result.content[0].text
        assert "Elf" in text

    async def test_color_identity_staples(self, live_client):
        result = await live_client.call_tool("color_identity_staples", {"color_identity": "simic"})
        text = result.content[0].text
        assert len(text) > 100  # Should have real card data

    async def test_rotation_check(self, live_client):
        result = await live_client.call_tool("rotation_check", {})
        text = result.content[0].text
        assert "Standard" in text


class TestCommanderDepthLive:
    """Hit real APIs with commander depth workflows."""

    async def test_commander_comparison(self, live_client):
        result = await live_client.call_tool(
            "commander_comparison",
            {"commanders": ["Muldrotha, the Gravetide", "Meren of Clan Nel Toth"]},
        )
        text = result.content[0].text
        assert "Muldrotha" in text
        assert "Meren" in text


class TestValidationLive:
    """Hit real bulk data with validation and utility workflows."""

    async def test_deck_validate_catches_illegal(self, live_client):
        result = await live_client.call_tool(
            "deck_validate",
            {
                "decklist": ["4 Lightning Bolt", "4 Sol Ring", "52 Island"],
                "format": "modern",
            },
        )
        text = result.content[0].text
        assert "INVALID" in text or "not legal" in text.lower()

    async def test_price_comparison(self, live_client):
        result = await live_client.call_tool(
            "price_comparison", {"cards": ["Sol Ring", "Lightning Bolt"]}
        )
        text = result.content[0].text
        assert "$" in text


class TestRulesEngineLive:
    """Hit the real rules engine (downloads Comprehensive Rules on first access)."""

    async def test_rules_lookup_by_number(self, live_client):
        result = await live_client.call_tool("rules_lookup", {"query": "704.5k"})
        text = result.content[0].text
        assert "704.5k" in text
        assert "world" in text.lower()

    async def test_keyword_explain(self, live_client):
        result = await live_client.call_tool("keyword_explain", {"keyword": "deathtouch"})
        text = result.content[0].text
        assert "deathtouch" in text.lower()
        assert "702" in text  # deathtouch rules section

    async def test_rules_interaction(self, live_client):
        result = await live_client.call_tool(
            "rules_interaction", {"mechanic_a": "deathtouch", "mechanic_b": "trample"}
        )
        text = result.content[0].text
        assert "deathtouch" in text.lower()
        assert "trample" in text.lower()

    async def test_rules_scenario(self, live_client):
        result = await live_client.call_tool(
            "rules_scenario",
            {"scenario": "A 1/1 with deathtouch blocks a 5/5 creature"},
        )
        text = result.content[0].text
        assert "deathtouch" in text.lower()

    async def test_combat_calculator(self, live_client):
        result = await live_client.call_tool(
            "combat_calculator",
            {"attackers": ["Typhoid Rats"], "blockers": ["Grizzly Bears"]},
        )
        text = result.content[0].text
        assert "combat" in text.lower() or "damage" in text.lower()


class TestDeckBuildingDepthLive:
    """Branch B deck building tools against real data."""

    async def test_build_around(self, live_client):
        result = await live_client.call_tool(
            "build_around",
            {"cards": ["Muldrotha, the Gravetide"], "format": "commander"},
        )
        text = result.content[0].text
        assert "Muldrotha" in text or len(text) > 100

    async def test_complete_deck(self, live_client):
        result = await live_client.call_tool(
            "complete_deck",
            {
                "decklist": ["Sol Ring", "Spore Frog", "Sakura-Tribe Elder"],
                "format": "commander",
                "commander": "Muldrotha, the Gravetide",
            },
        )
        text = result.content[0].text
        assert len(text) > 100  # Should have gap analysis

    async def test_precon_upgrade(self, live_client):
        result = await live_client.call_tool(
            "precon_upgrade",
            {
                "decklist": [
                    "Sol Ring",
                    "Spore Frog",
                    "Sakura-Tribe Elder",
                    "Mulldrifter",
                    "Coiling Oracle",
                    "Ravenous Chupacabra",
                ],
                "commander": "Muldrotha, the Gravetide",
                "budget": 5.0,
                "num_upgrades": 3,
            },
        )
        text = result.content[0].text
        assert len(text) > 50


class TestLimitedLive:
    """Branch B limited tools against real data."""

    async def test_sealed_pool_build(self, live_client):
        # Minimal pool — just enough to test the tool runs
        pool = [
            "Plains",
            "Island",
            "Swamp",
            "Mountain",
            "Forest",
            "Serra Angel",
            "Air Elemental",
            "Doom Blade",
            "Giant Growth",
            "Lightning Bolt",
            "Cancel",
            "Grizzly Bears",
            "Wind Drake",
            "Glory Seeker",
        ]
        result = await live_client.call_tool("sealed_pool_build", {"pool": pool, "set_code": "FDN"})
        text = result.content[0].text
        assert len(text) > 50

    async def test_draft_signal_read(self, live_client):
        result = await live_client.call_tool(
            "draft_signal_read",
            {
                "picks": ["Serra Angel", "Doom Blade", "Wind Drake"],
                "set_code": "FDN",
            },
        )
        text = result.content[0].text
        assert "signal" in text.lower() or "color" in text.lower() or len(text) > 50

    async def test_draft_log_review(self, live_client):
        result = await live_client.call_tool(
            "draft_log_review",
            {
                "picks": [
                    "Serra Angel",
                    "Doom Blade",
                    "Wind Drake",
                    "Lightning Bolt",
                    "Grizzly Bears",
                    "Cancel",
                ],
                "set_code": "FDN",
            },
        )
        text = result.content[0].text
        assert len(text) > 50


class TestCrossFormatLive:
    """New cross-format tools against real data."""

    async def test_ban_list_modern(self, live_client):
        result = await live_client.call_tool("bulk_ban_list", {"format": "modern"})
        text = result.content[0].text
        # Modern has banned cards
        assert "Banned" in text or "banned" in text

    async def test_format_staples_commander(self, live_client):
        result = await live_client.call_tool(
            "bulk_format_staples", {"format": "commander", "limit": 5}
        )
        text = result.content[0].text
        assert "Sol Ring" in text or "Commander" in text.lower()

    async def test_format_staples_modern_nonsingleton(self, live_client):
        """Non-singleton format should use tournament or competitive mode, not EDHREC."""
        result = await live_client.call_tool(
            "bulk_format_staples", {"format": "modern", "limit": 5}
        )
        text = result.content[0].text
        assert "Modern" in text
        # Should use tournament (% Decks) or competitive (Score), NOT edhrec (Rank #)
        assert "% Decks" in text or "Score" in text

    async def test_card_in_formats(self, live_client):
        result = await live_client.call_tool(
            "bulk_card_in_formats", {"card_name": "Lightning Bolt"}
        )
        text = result.content[0].text
        assert "Lightning Bolt" in text
        assert "modern" in text.lower()


@pytest.mark.live
async def test_scenario_ranking_on_the_real_corpus():
    """Scenario ranking measured against all 3047 rules, not a 26-rule fixture.

    The fixture cannot show this defect: ranking is about which rules come back
    FIRST out of many, and 26 rules produce no competition. Measured here on the
    real corpus, where the same scenario once put the two governing rules at
    ranks 105 and 114 of 368 inside a 215 KB response.

    This is a floor, not a target. Cases whose wording names a mechanic
    ("deathtouch", "trample") rank well; cases phrased in plain language without
    a glossary term still fail, which is the known limit of lexical retrieval
    and the reason a larger annotated question set comes before any further
    tuning.
    """
    from mtg_mcp_server.config import Settings
    from mtg_mcp_server.services.rules import RulesService
    from mtg_mcp_server.workflows.rules import rules_scenario

    service = RulesService(rules_url=Settings().rules_url, refresh_hours=168)
    await service.ensure_loaded()

    result = await rules_scenario(
        "My creature has deathtouch and trample. It is blocked by a 5/5 "
        "creature. How much damage must I assign to the blocker?",
        rules=service,
    )
    ranked = [r["number"] for r in result.data["rules"]]

    # 702.19b is the rule that answers the question.
    assert "702.19b" in ranked[:5], f"702.19b should rank top-5, got {ranked[:8]}"
    # A verification aid has to be readable; 215 KB is a haystack.
    assert len(result.markdown) < 20_000, f"{len(result.markdown)} chars"


def _annotated_questions() -> dict:
    import json
    import pathlib

    path = pathlib.Path(__file__).parent.parent / "fixtures" / "rules" / "annotated_questions.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.live
async def test_every_annotated_rule_exists_in_the_corpus():
    """The question set is only worth what its annotations are worth.

    An expected rule that does not exist scores as a permanent miss, and the
    tuning it motivates chases a rule that was never there. Three numbers
    recalled from memory were wrong while this set was built (702.82a is Devour,
    702.22b is Bands With Other, 702.111a answers nothing), so the annotations
    are checked against the corpus rather than trusted.

    This also catches a Comprehensive Rules update renumbering a rule: the set
    is pinned to a corpus version, and a silent renumber would quietly turn
    good annotations into misses.
    """
    from mtg_mcp_server.config import Settings
    from mtg_mcp_server.services.rules import RulesService

    service = RulesService(rules_url=Settings().rules_url, refresh_hours=168)
    await service.ensure_loaded()

    missing = [
        (q["id"], number)
        for q in _annotated_questions()["questions"]
        for number in q["expected_rules"]
        if await service.lookup_by_number(number) is None
    ]
    assert not missing, f"annotated rules absent from the corpus: {missing}"


@pytest.mark.live
async def test_rules_recall_does_not_regress():
    """Floor on retrieval recall, measured 2026-07-29 — not a target.

    First baseline, word-by-word search: 9/30 in the top 5, 14/30 never returned
    at all, 8/17 named against 1/13 plain.

    After scoring the scenario's terms jointly and weighting each by inverse
    document frequency: 17/30 in the top 5, 8/30 absent, 15/17 named against
    2/13 plain. The named/plain gap is the real result. Retrieval was never the
    limit for questions that name a mechanic; it is still the limit for
    questions phrased the way a player speaks, and no amount of lexical work
    reaches those. "My 3/3 gets -3/-3, what happens to it?" is answered by rule
    704.5f, which says "toughness 0 or less" — a word the question does not
    contain. That gap is what an embedding would have to close.

    The floors are separate on purpose: a change that lifts the average while
    quietly losing the plain-language cases would still pass a single number.
    """
    from mtg_mcp_server.config import Settings
    from mtg_mcp_server.services.rules import RulesService
    from mtg_mcp_server.workflows.rules import rules_scenario

    service = RulesService(rules_url=Settings().rules_url, refresh_hours=168)
    await service.ensure_loaded()

    hits_at_5 = {"named": 0, "plain": 0}
    for question in _annotated_questions()["questions"]:
        result = await rules_scenario(question["question_en"], rules=service)
        ranked = [r["number"] for r in result.data["rules"]]
        if any(number in ranked[:5] for number in question["expected_rules"]):
            hits_at_5[question["phrasing"]] += 1

    total = hits_at_5["named"] + hits_at_5["plain"]
    assert total >= 17, f"recall@5 regressed: {total}/30, floor is 17/30"
    assert hits_at_5["named"] >= 15, f"named recall@5 regressed: {hits_at_5['named']}/17"
    assert hits_at_5["plain"] >= 2, f"plain recall@5 regressed: {hits_at_5['plain']}/13"
