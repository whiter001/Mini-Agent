"""Tests for the persistent memory system."""

from pathlib import Path
import tempfile

import pytest

from mini_agent.memory_store import MemoryStore
from mini_agent.tools.memory_tools import create_memory_tools


def test_memory_store_persists_snapshots_and_search():
    """Memory notes should persist to disk and be searchable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(root_dir=tmpdir)

        store.store_memory("Prefer concise responses", title="response-style", tags=["preference"])
        store.store_user_profile("User works on Windows", title="platform", tags=["environment"])

        assert (Path(tmpdir) / "MEMORY.md").exists()
        assert (Path(tmpdir) / "USER.md").exists()
        assert (Path(tmpdir) / "memory.sqlite3").exists()

        results = store.search("concise", limit=5)
        assert results
        assert any("concise responses" in entry.content for entry in results)

        prompt = store.build_system_prompt()
        assert "MEMORY.md" in prompt
        assert "USER.md" in prompt
        assert "Windows" in prompt


def test_memory_store_builds_turn_context():
    """Relevant search hits should be injected as a temporary system message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(root_dir=tmpdir)
        store.store_memory("Project uses uv for dependency management", title="tooling", tags=["uv"])

        context = store.build_turn_context("How do I manage dependencies with uv?")

        assert len(context) == 1
        assert context[0].role == "system"
        assert "Relevant Persistent Memory" in context[0].content
        assert "uv" in context[0].content.lower()


def test_memory_store_search_falls_back_for_chinese_query():
    """Chinese queries should still find relevant memory even when FTS tokenization cannot express them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(root_dir=tmpdir)
        store.store_memory("用户偏好：使用中文回复，少用英文术语", title="中文偏好", tags=["偏好"])

        results = store.search("中文回复", limit=5)

        assert results
        assert any("使用中文回复" in entry.content for entry in results)


def test_memory_store_builds_turn_context_for_chinese_query():
    """Chinese queries should populate turn context from matching durable memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(root_dir=tmpdir)
        store.store_memory("用户偏好：使用中文回复，少用英文术语", title="中文偏好", tags=["偏好"])

        context = store.build_turn_context("请继续用中文回复")

        assert len(context) == 1
        assert "使用中文回复" in context[0].content


@pytest.mark.asyncio
async def test_memory_tools_store_and_search():
    """Memory tools should write durable entries and recall them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(root_dir=tmpdir)
        tools, _ = create_memory_tools(store)
        remember_tool = next(tool for tool in tools if tool.name == "remember")
        search_tool = next(tool for tool in tools if tool.name == "search_memory")

        result = await remember_tool.execute(
            title="workflow",
            content="Use uv sync before running tests",
            tags=["uv", "tests"],
        )
        assert result.success

        search_result = await search_tool.execute(query="uv sync", limit=5)
        assert search_result.success
        assert "uv sync" in search_result.content


@pytest.mark.asyncio
async def test_memory_tools_search_chinese_query():
    """search_memory should return Chinese memory hits instead of reporting no matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(root_dir=tmpdir)
        tools, _ = create_memory_tools(store)
        remember_tool = next(tool for tool in tools if tool.name == "remember")
        search_tool = next(tool for tool in tools if tool.name == "search_memory")

        result = await remember_tool.execute(
            title="中文偏好",
            content="用户偏好：使用中文回复，少用英文术语",
            tags=["偏好"],
        )
        assert result.success

        search_result = await search_tool.execute(query="中文回复", limit=5)
        assert search_result.success
        assert "使用中文回复" in search_result.content
