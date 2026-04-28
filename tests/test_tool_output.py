"""Tests for large tool output truncation helpers."""

from pathlib import Path

from mini_agent.tool_output import ToolOutputStore


def test_tool_output_store_keeps_small_output(tmp_path):
    store = ToolOutputStore(tmp_path, max_lines=10, max_bytes=1024)

    prepared = store.prepare("short output", label="test output")

    assert prepared.truncated is False
    assert prepared.preview == "short output"
    assert prepared.full_output_path is None


def test_tool_output_store_truncates_and_persists_large_output(tmp_path):
    store = ToolOutputStore(tmp_path, max_lines=2, max_bytes=24)
    original = "line 1\nline 2\nline 3\nline 4"

    prepared = store.prepare(original, label="test output")

    assert prepared.truncated is True
    assert prepared.full_output_path is not None
    assert prepared.full_output_path.exists()
    assert prepared.full_output_path.read_text(encoding="utf-8") == original
    assert "output truncated" in prepared.preview
    assert "Full test output saved to:" in prepared.preview
    assert str(Path(".mini-agent") / "truncation") in prepared.preview
