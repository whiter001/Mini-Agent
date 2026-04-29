"""CLI tests for mini-agent."""

from argparse import Namespace
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
