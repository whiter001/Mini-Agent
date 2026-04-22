"""Tests for CLI skill initialization."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from mini_agent.cli import initialize_base_tools


@pytest.mark.asyncio
async def test_initialize_base_tools_strips_skill_paths_and_passes_extra_dirs(monkeypatch):
    """Skill directory settings should tolerate accidental surrounding whitespace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundled_dir = Path(tmpdir) / "bundled"
        extra_dir = Path(tmpdir) / "extra"
        bundled_dir.mkdir()
        extra_dir.mkdir()

        captured = {}

        def fake_create_skill_tools(skills_dir, extra_skills_dirs=None):
            captured["skills_dir"] = skills_dir
            captured["extra_skills_dirs"] = extra_skills_dirs
            return [], object()

        monkeypatch.setattr("mini_agent.cli.create_skill_tools", fake_create_skill_tools)

        config = SimpleNamespace(
            tools=SimpleNamespace(
                enable_bash=False,
                enable_memory=False,
                enable_skills=True,
                enable_mcp=False,
                skills_dir=f"  {bundled_dir}  ",
                skills_external_dirs=[f"  {extra_dir}  "],
            )
        )

        tools, skill_loader, memory_store = await initialize_base_tools(config)

        assert captured["skills_dir"] == str(bundled_dir)
        assert captured["extra_skills_dirs"] == [str(extra_dir)]
        assert tools == []
        assert skill_loader is not None
        assert memory_store is None