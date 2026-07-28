"""Tests for the parameter echo middleware.

Regression origin (2026-07-27): a client HTML-escaped the comparison operators of a
Scryfall query. The tool answered "No cards found" — which reads exactly like a valid
query with zero matches. Nothing in the response showed what the server had actually
received, so the bug survived roughly forty calls before anyone noticed.

``deck_audit_bundle`` already echoed its parameters per section and was the one tool
where this class of bug was visible immediately. This middleware generalises that.
"""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP
from fastmcp.tools.base import ToolResult

from mtg_mcp_server.middleware import ParamsEchoMiddleware


@pytest.fixture
def echo_server() -> FastMCP:
    """A minimal server carrying the middleware, one tool per response shape."""
    server = FastMCP("EchoTest")
    server.add_middleware(ParamsEchoMiddleware())

    @server.tool
    async def structured(query: str, limit: int = 10) -> ToolResult:
        """A tool with a structured_content dict — the common shape here."""
        return ToolResult(content="ok", structured_content={"query": query, "hits": 0})

    @server.tool
    async def text_only(name: str) -> ToolResult:
        """A tool that returns markdown and no structured content."""
        return ToolResult(content=f"hello {name}")

    @server.tool
    async def plain_string(name: str) -> str:
        """A tool whose return type makes FastMCP wrap the payload."""
        return f"hello {name}"

    return server


@pytest.fixture
async def client(echo_server: FastMCP):
    async with Client(transport=echo_server) as c:
        yield c


class TestEchoIsPresent:
    """Every tool call reports the arguments the server actually received."""

    async def test_structured_tool_carries_params_received(self, client) -> None:
        result = await client.call_tool("structured", {"query": "mv<=2", "limit": 5})
        assert result.structured_content is not None
        assert result.structured_content["params_received"] == {"query": "mv<=2", "limit": 5}

    async def test_original_keys_are_preserved(self, client) -> None:
        result = await client.call_tool("structured", {"query": "t:ninja"})
        assert result.structured_content["query"] == "t:ninja"
        assert result.structured_content["hits"] == 0

    async def test_escaped_operator_is_visible_in_the_echo(self, client) -> None:
        # The whole point: what arrived is inspectable without a second call.
        result = await client.call_tool("structured", {"query": "mv&lt;=2"})
        assert result.structured_content["params_received"]["query"] == "mv&lt;=2"

    async def test_text_only_tool_gains_a_structured_echo(self, client) -> None:
        result = await client.call_tool("text_only", {"name": "yuriko"})
        assert result.structured_content == {"params_received": {"name": "yuriko"}}

    async def test_defaults_are_not_invented(self, client) -> None:
        # The echo reports what the CLIENT sent, not what the signature defaults to.
        # Reporting a default the client never sent would hide a missing argument.
        result = await client.call_tool("structured", {"query": "x"})
        assert result.structured_content["params_received"] == {"query": "x"}


class TestWrappedResultsAreLeftAlone:
    """A tool whose output FastMCP wraps must not have its payload rewritten."""

    async def test_plain_string_result_is_untouched(self, client) -> None:
        result = await client.call_tool("plain_string", {"name": "yuriko"})
        assert result.structured_content == {"result": "hello yuriko"}


class TestEchoSurvivesTruncation:
    """The echo must outlive response limiting.

    Truncation replaces the result with a plain text block and drops
    structured_content. An echo applied underneath the limiter would disappear on
    exactly the large responses that are hardest to debug, so the echo middleware is
    registered as the outermost one.
    """

    async def test_echo_present_on_a_truncated_response(self) -> None:
        from fastmcp.server.middleware.response_limiting import ResponseLimitingMiddleware

        server = FastMCP("Truncating")
        server.add_middleware(ParamsEchoMiddleware())  # outermost, as in server.py
        server.add_middleware(ResponseLimitingMiddleware(max_size=500))

        @server.tool
        async def huge(query: str) -> ToolResult:
            """Returns far more than the limiter allows."""
            return ToolResult(content="x" * 5000, structured_content={"query": query})

        async with Client(transport=server) as c:
            result = await c.call_tool("huge", {"query": "mv<=2"})

        assert result.structured_content is not None
        assert result.structured_content["params_received"] == {"query": "mv<=2"}


class TestEchoDoesNotClobber:
    """A tool that already reports its own params keeps its version."""

    async def test_existing_key_wins(self) -> None:
        server = FastMCP("NoClobber")
        server.add_middleware(ParamsEchoMiddleware())

        @server.tool
        async def bundle(commander: str) -> ToolResult:
            """Mimics deck_audit_bundle, which echoes params per section."""
            return ToolResult(
                content="ok",
                structured_content={"params_received": {"per": "section"}},
            )

        async with Client(transport=server) as c:
            result = await c.call_tool("bundle", {"commander": "Yuriko"})
        assert result.structured_content["params_received"] == {"per": "section"}


class TestEchoIsBounded:
    """The echo sits OUTSIDE the response size limiters, so it bounds itself.

    A decklist argument is ~99 names. Repeating it verbatim on every response adds
    kilobytes the size ceiling never sees. The point of the echo is to make a mangled
    argument visible — the first few entries and a count do that just as well.
    """

    @pytest.fixture
    def big_arg_client(self):
        server = FastMCP("BigArgs")
        server.add_middleware(ParamsEchoMiddleware())

        @server.tool
        async def analyse(decklist: list[str]) -> ToolResult:
            """Takes a decklist, like the real deck tools."""
            return ToolResult(content="ok", structured_content={"count": len(decklist)})

        return server

    async def test_long_list_is_summarised_not_repeated(self, big_arg_client) -> None:
        deck = [f"Card {i}" for i in range(99)]
        async with Client(transport=big_arg_client) as c:
            result = await c.call_tool("analyse", {"decklist": deck})

        echoed = result.structured_content["params_received"]["decklist"]
        assert len(echoed) < len(deck)
        # The count survives, so a truncated argument is still diagnosable.
        assert "99" in echoed[-1]

    async def test_short_list_is_echoed_verbatim(self, big_arg_client) -> None:
        async with Client(transport=big_arg_client) as c:
            result = await c.call_tool("analyse", {"decklist": ["Sol Ring", "Island"]})
        assert result.structured_content["params_received"]["decklist"] == ["Sol Ring", "Island"]

    async def test_long_string_is_truncated_with_its_length(self, big_arg_client) -> None:
        async with Client(transport=big_arg_client) as c:
            result = await c.call_tool("analyse", {"decklist": ["x" * 2000]})
        echoed = result.structured_content["params_received"]["decklist"][0]
        assert "truncated" in echoed
        assert "2000" in echoed
