"""Tool Protocol — generic interface for any agent-invokable tool.

Source: tech-spec.py §2.6
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# ── Supporting value type ──────────────────────────────────────────────


class ToolResult(BaseModel):
    """Result returned by any tool invocation."""

    success: bool
    output: Any = None
    error: str | None = None
    tokens_used: int = 0


# ── Protocol ───────────────────────────────────────────────────────────


@runtime_checkable
class Tool(Protocol):
    """
    A capability an agent can invoke. Maps to the MCP tool interface.
    MCP servers, LSP servers, test runners, linters — all are Tools.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given parameters."""
        ...
