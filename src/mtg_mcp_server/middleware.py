"""Server-wide middleware.

Currently one concern: every tool response echoes the arguments the server actually
received. See :class:`ParamsEchoMiddleware` for why that is not cosmetic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp.server.middleware.middleware import Middleware

if TYPE_CHECKING:
    import mcp.types as mt
    from fastmcp.server.middleware.middleware import CallNext, MiddlewareContext
    from fastmcp.tools.base import ToolResult

__all__ = ["ParamsEchoMiddleware"]

_ECHO_KEY = "params_received"

# A decklist argument is ~99 names, and this echo is applied AFTER the response size
# limiters, so it escapes their ceiling. Long list values are summarised rather than
# repeated: the point of the echo is to make a mangled argument visible, and the first
# few entries plus a count do that just as well as 99 verbatim strings.
_MAX_LIST_ITEMS = 5
_MAX_STRING_CHARS = 500

# FastMCP wraps a non-object tool return in a single ``result`` key, driven by the
# generated output schema. Injecting into that dict would produce a payload the
# client validates against a schema it no longer matches.
_WRAPPED_KEYS = frozenset({"result"})


def _summarise(value: Any) -> Any:
    """Shrink a value that is too large to echo verbatim, saying so explicitly."""
    if isinstance(value, str) and len(value) > _MAX_STRING_CHARS:
        return f"{value[:_MAX_STRING_CHARS]}... [truncated, {len(value)} chars total]"
    if isinstance(value, list):
        # Items are summarised even when the list is short: one 2000-character entry
        # is as expensive as a hundred short ones.
        head = [_summarise(item) for item in value[:_MAX_LIST_ITEMS]]
        if len(value) <= _MAX_LIST_ITEMS:
            return head
        return [*head, f"... [{len(value) - _MAX_LIST_ITEMS} more, {len(value)} total]"]
    return value


class ParamsEchoMiddleware(Middleware):
    """Attach the received arguments to every tool response.

    A tool that returns nothing but "No cards found" is indistinguishable from a tool
    whose input was mangled in transit. That ambiguity is what let an HTML-escaping
    client survive about forty calls before it was noticed: each empty result read as
    evidence that the cards did not exist.

    Echoing the arguments makes the mangling visible on the FIRST call. It is the
    cheapest possible instrumentation, so it applies to every tool rather than to the
    handful we currently suspect.

    The echo reports what the client sent, never what the signature defaults to —
    a fabricated default would hide the very thing this exists to reveal.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        result = await call_next(context)

        raw = context.message.arguments or {}
        if not isinstance(raw, dict):
            return result
        arguments: dict[str, Any] = {key: _summarise(value) for key, value in raw.items()}
        structured = result.structured_content

        if structured is None:
            result.structured_content = {_ECHO_KEY: arguments}
            return result

        if not isinstance(structured, dict):
            return result

        # A tool that already reports its own parameters (deck_audit_bundle does it
        # per section) knows better than we do — don't overwrite it.
        if _ECHO_KEY in structured or set(structured) <= _WRAPPED_KEYS:
            return result

        structured[_ECHO_KEY] = arguments
        return result
