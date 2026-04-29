"""CLI tests for mini-agent."""

import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import pytest

from mini_agent import cli


@pytest.mark.parametrize("flag", ["-p", "--prompt"])
def test_parse_args_supports_prompt_flag(flag):
    args = cli.parse_args([flag, "列出当前的skills有哪些"])

    assert args.command is None
    assert args.prompt == "列出当前的skills有哪些"


def test_parse_args_supports_help_command():
    args = cli.parse_args(["help"])

    assert args.command == "help"


def test_parse_args_supports_cleanup_auto_skills_command():
    args = cli.parse_args(["cleanup-auto-skills", "--apply", "--skip-candidates"])

    assert args.command == "cleanup-auto-skills"
    assert args.apply is True
    assert args.skip_candidates is True


def test_parse_args_supports_help_topic():
    args = cli.parse_args(["help", "log"])

    assert args.command == "help"
    assert args.topic == "log"


def test_parse_args_keeps_legacy_task_flag():
    args = cli.parse_args(["--task", "列出当前的skills有哪些"])

    assert args.command is None
    assert args.prompt == "列出当前的skills有哪些"


def test_parse_args_supports_version_flag():
    args = cli.parse_args(["--version"])

    assert args.command is None
    assert args.version is True


def test_get_installation_time_prefers_distribution_metadata(monkeypatch, tmp_path):
    dist_info_dir = tmp_path / "mini_agent-0.1.0.dist-info"
    dist_info_dir.mkdir()
    record_path = dist_info_dir / "RECORD"
    record_path.write_text("", encoding="utf-8")
    executable_path = tmp_path / "mini-agent"
    executable_path.write_text("#!/bin/sh\n", encoding="utf-8")

    metadata_mtime = 1_700_000_123
    executable_mtime = metadata_mtime - 3600
    os.utime(dist_info_dir, (metadata_mtime, metadata_mtime))
    os.utime(record_path, (metadata_mtime, metadata_mtime))
    os.utime(executable_path, (executable_mtime, executable_mtime))

    class FakeDistribution:
        files = [Path("mini_agent-0.1.0.dist-info/RECORD")]
        _path = dist_info_dir

        def locate_file(self, package_file):
            return tmp_path / Path(package_file)

    monkeypatch.setattr(cli.metadata, "distribution", lambda name: FakeDistribution())

    installed_at = cli.get_installation_time(executable_path=executable_path)

    assert installed_at == datetime.fromtimestamp(metadata_mtime)


def test_get_version_text_includes_install_time(monkeypatch):
    expected = datetime(2026, 4, 29, 23, 58, 12)
    monkeypatch.setattr(cli, "get_installation_time", lambda dist_name="mini-agent", executable_path=None: expected)

    version_text = cli.get_version_text()

    assert "mini-agent 0.1.0" in version_text
    assert "Installed at: 2026-04-29 23:58:12" in version_text


def test_main_forwards_prompt_to_run_agent(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    async def fake_run_agent(workspace_dir: Path, prompt: str | None = None):
        captured["workspace_dir"] = workspace_dir
        captured["prompt"] = prompt

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda argv=None: Namespace(command=None, workspace=str(tmp_path), prompt="列出当前的skills有哪些"),
    )
    monkeypatch.setattr(cli, "run_agent", fake_run_agent)

    cli.main()

    assert captured["workspace_dir"] == tmp_path.resolve()
    assert captured["prompt"] == "列出当前的skills有哪些"


def test_main_handles_help_command(monkeypatch):
    captured = {"called": False, "topic": None}

    def fake_print_help(topic=None):
        captured["called"] = True
        captured["topic"] = topic

    monkeypatch.setattr(cli, "parse_args", lambda argv=None: Namespace(command="help", topic=None))
    monkeypatch.setattr(cli, "print_help", fake_print_help)

    cli.main()

    assert captured["called"] is True
    assert captured["topic"] is None


def test_main_handles_version_flag(monkeypatch):
    captured = {"called": False}

    def fake_print_version_info():
        captured["called"] = True

    async def fail_run_agent(*args, **kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("run_agent should not be called when --version is used")

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda argv=None: Namespace(command=None, workspace=None, prompt=None, version=True),
    )
    monkeypatch.setattr(cli, "print_version_info", fake_print_version_info)
    monkeypatch.setattr(cli, "run_agent", fail_run_agent)

    cli.main()

    assert captured["called"] is True


def test_main_handles_help_topic(monkeypatch):
    captured = {"called": False, "topic": None}

    def fake_print_help(topic=None):
        captured["called"] = True
        captured["topic"] = topic

    monkeypatch.setattr(cli, "parse_args", lambda argv=None: Namespace(command="help", topic="log"))
    monkeypatch.setattr(cli, "print_help", fake_print_help)

    cli.main()

    assert captured["called"] is True
    assert captured["topic"] == "log"


def test_main_handles_cleanup_auto_skills_command(monkeypatch):
    captured: dict[str, object] = {}

    def fake_cleanup_auto_skills(skills_dir, *, apply, include_candidates, archive_dir):
        captured["skills_dir"] = skills_dir
        captured["apply"] = apply
        captured["include_candidates"] = include_candidates
        captured["archive_dir"] = archive_dir
        return object()

    def fake_print_auto_skill_cleanup_report(report):
        captured["report"] = report

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda argv=None: Namespace(
            command="cleanup-auto-skills",
            apply=True,
            skills_dir="~/custom-skills",
            archive_dir="~/archive-skills",
            skip_candidates=False,
        ),
    )
    monkeypatch.setattr(cli, "cleanup_auto_skills", fake_cleanup_auto_skills)
    monkeypatch.setattr(cli, "print_auto_skill_cleanup_report", fake_print_auto_skill_cleanup_report)

    cli.main()

    assert captured["skills_dir"] == "~/custom-skills"
    assert captured["apply"] is True
    assert captured["include_candidates"] is True
    assert captured["archive_dir"] == "~/archive-skills"
    assert captured["report"] is not None


def test_print_help_includes_cli_and_interactive_commands(capsys):
    cli.print_help()

    out = capsys.readouterr().out

    assert "Mini-Agent CLI" in out
    assert "mini-agent -p" in out
    assert "mini-agent help log" in out
    assert "/help" in out
    assert "Ctrl+J" in out


def test_print_help_log_topic(capsys):
    cli.print_help("log")

    out = capsys.readouterr().out

    assert "Log Command" in out
    assert "mini-agent log" in out
    assert "mini-agent help log" in out


def test_print_help_cleanup_topic(capsys):
    cli.print_help("cleanup-auto-skills")

    out = capsys.readouterr().out

    assert "cleanup-auto-skills" in out
    assert "--apply" in out
    assert "_archived" in out
