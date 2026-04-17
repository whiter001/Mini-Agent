"""Tools for durable memory storage and search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .base import Tool, ToolResult
from ..memory_store import MemoryStore


class RememberTool(Tool):
    """Store durable agent memory notes."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return "Store a durable agent memory note in ~/.mini-agent/MEMORY.md and the SQLite memory store."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short note title"},
                "content": {"type": "string", "description": "The durable fact or workflow to remember"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for recall",
                },
            },
            "required": ["content"],
        }

    async def execute(self, content: str, title: str = "", tags: Optional[List[str]] = None) -> ToolResult:
        entry = self.memory_store.store_memory(content=content, title=title or None, tags=tags)
        return ToolResult(
            success=True,
            content=f"Stored memory #{entry.id} in {self.memory_store.memory_path}",
        )


class RememberUserTool(Tool):
    """Store durable user profile facts."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "remember_user"

    @property
    def description(self) -> str:
        return "Store a durable user profile fact in ~/.mini-agent/USER.md and the SQLite memory store."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short profile title"},
                "content": {"type": "string", "description": "The user preference or profile fact to remember"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for recall",
                },
            },
            "required": ["content"],
        }

    async def execute(self, content: str, title: str = "", tags: Optional[List[str]] = None) -> ToolResult:
        entry = self.memory_store.store_user_profile(content=content, title=title or None, tags=tags)
        return ToolResult(
            success=True,
            content=f"Stored user fact #{entry.id} in {self.memory_store.user_path}",
        )


class SearchMemoryTool(Tool):
    """Search durable memory notes and user facts."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return "Search durable memory stored in ~/.mini-agent/MEMORY.md and ~/.mini-agent/USER.md."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                "kind": {
                    "type": "string",
                    "enum": ["memory", "user", "all"],
                    "default": "all",
                    "description": "Which memory bucket to search",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, limit: int = 5, kind: str = "all") -> ToolResult:
        kinds: Sequence[str] | None
        if kind == "all":
            kinds = None
        else:
            kinds = [kind]

        results = self.memory_store.search(query=query, limit=limit, kinds=kinds)
        if not results:
            return ToolResult(success=True, content="No memory matches found.")

        lines = ["Memory search results:"]
        for index, entry in enumerate(results, 1):
            lines.append(f"{index}. [{entry.kind}] {entry.title}: {entry.content}")
        return ToolResult(success=True, content="\n".join(lines))


def create_memory_tools(memory_store: MemoryStore | None = None) -> tuple[list[Tool], MemoryStore]:
    """Create memory tools and their backing store."""
    store = memory_store or MemoryStore()
    tools: list[Tool] = [
        RememberTool(store),
        RememberUserTool(store),
        SearchMemoryTool(store),
    ]
    return tools, store
