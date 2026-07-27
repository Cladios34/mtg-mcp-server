"""Settings that are advertised in .env.example must actually reach the clients.

A setting documented but wired to nothing is worse than no setting: operators
tune it and nothing changes. `scryfall_rate_limit_ms` was exactly that until
2026-07-27 -- published in .env.example, read by no one.
"""

from __future__ import annotations

from mtg_mcp_server.config import Settings
from mtg_mcp_server.services.scryfall import ScryfallClient


class TestScryfallRateLimitWiring:
    """The advertised rate-limit setting drives the real client."""

    def test_default_stays_under_scryfall_ceiling(self):
        """Scryfall requires "less than 10 requests per second" -- strictly under.

        Regression guard: the default was 100ms, i.e. exactly 10 req/s, which
        drew HTTP 429s carrying a 60s cooldown and a network-block warning.
        Wiring the setting without raising this default would reinstate them.
        """
        rps = 1000 / Settings().scryfall_rate_limit_ms
        assert rps < 10.0, f"default rate {rps:.2f} req/s sits on Scryfall's ceiling"

    def test_setting_reaches_the_client(self, monkeypatch):
        """An operator override changes the client's effective rate."""
        monkeypatch.setenv("MTG_MCP_SCRYFALL_RATE_LIMIT_MS", "250")
        settings = Settings()
        client = ScryfallClient(
            base_url=settings.scryfall_base_url,
            rate_limit_rps=1000 / settings.scryfall_rate_limit_ms,
        )
        assert client._rate_limit_rps == 4.0

    def test_client_default_also_stays_under_ceiling(self):
        """Constructing without an explicit rate must not land on 10 req/s either."""
        assert ScryfallClient()._rate_limit_rps < 10.0
