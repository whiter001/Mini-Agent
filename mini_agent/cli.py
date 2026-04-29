"""Mini Agent CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import List

from mini_agent import __version__
from mini_agent.config import Config
from mini_agent.memory_store import MemoryStore
from mini_agent.runtime import (
    build_runtime_context,
    build_turn_context,
    execute_prompt,
    persist_auto_skill_if_needed,
    run_interactive_session,
)
from mini_agent.tools.auto_skill import AutoSkillCleanupReport, cleanup_auto_skills
from mini_agent.terminal_ui import (
    Colors,
    get_log_directory,
    print_banner,
    print_help,
    print_session_info,
    print_stats,
    read_log_file,
    show_log_directory,
)
from mini_agent.tools.base import Tool
from mini_agent.tools.bash_tool import BashKillTool, BashOutputTool, BashTool
from mini_agent.tools.file_tools import EditTool, ReadTool, WriteTool
from mini_agent.tools.mcp_loader import load_mcp_tools_async, set_mcp_timeout_config
from mini_agent.tools.memory_tools import create_memory_tools
from mini_agent.tools.note_tool import SessionNoteTool
from mini_agent.tools.skill_tool import create_skill_tools


_INSTALL_MARKER_FILENAMES = {
    "direct_url.json",
    "entry_points.txt",
    "installer",
    "metadata",
    "pkg-info",
    "record",
    "sources.txt",
}


def _collect_installation_marker_paths(
    distribution: metadata.Distribution,
    *,
    executable_path: str | Path | None = None,
) -> list[Path]:
    """Collect metadata paths whose mtimes approximate the local install time."""
    candidates: dict[Path, None] = {}

    def add_candidate(path_like: str | Path | None) -> None:
        if path_like is None:
            return
        path = Path(path_like)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved.exists():
            candidates[resolved] = None

    for package_file in distribution.files or []:
        try:
            located = Path(distribution.locate_file(package_file)).resolve()
        except OSError:
            continue

        metadata_parent = next(
            (
                parent
                for parent in [located, *located.parents]
                if parent.name.lower().endswith((".dist-info", ".egg-info"))
            ),
            None,
        )
        if metadata_parent is not None:
            add_candidate(metadata_parent)

        if located.name.lower() in _INSTALL_MARKER_FILENAMES:
            add_candidate(located)

    dist_path = getattr(distribution, "_path", None)
    if dist_path is not None:
        add_candidate(dist_path)
        try:
            add_candidate(distribution.locate_file(dist_path))
        except OSError:
            pass

    executable = Path(executable_path) if executable_path is not None else Path(sys.argv[0])
    add_candidate(executable)

    return list(candidates)


def get_installation_time(
    dist_name: str = "mini-agent",
    *,
    executable_path: str | Path | None = None,
) -> datetime | None:
    """Return the best-effort local install/update time for the current CLI."""
    try:
        distribution = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        distribution = None

    candidate_paths: list[Path] = []
    if distribution is not None:
        candidate_paths.extend(_collect_installation_marker_paths(distribution, executable_path=executable_path))
    elif executable_path is not None:
        executable = Path(executable_path).resolve()
        if executable.exists():
            candidate_paths.append(executable)

    if not candidate_paths:
        return None

    latest_mtime = max(path.stat().st_mtime for path in candidate_paths)
    return datetime.fromtimestamp(latest_mtime)


def get_version_text() -> str:
    """Build version output including best-effort local install time."""
    lines = [f"mini-agent {__version__}"]
    installed_at = get_installation_time()
    if installed_at is not None:
        lines.append(f"Installed at: {installed_at.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        lines.append("Installed at: unavailable")
    return "\n".join(lines)


def print_version_info() -> None:
    """Print version information for the installed CLI."""
    print(get_version_text())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="Mini Agent - AI assistant with file tools and MCP support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mini-agent                              # Use current directory as workspace
  mini-agent --workspace /path/to/dir     # Use specific workspace directory
  mini-agent -p "列出当前的skills有哪些"     # Execute a prompt non-interactively
    mini-agent cleanup-auto-skills          # Preview cleanup of historical auto skills
  mini-agent help                         # Show command help
  mini-agent help log                     # Show log command help
    mini-agent help cleanup-auto-skills     # Show cleanup command help
  mini-agent log                          # Show log directory and recent files
  mini-agent log agent_run_xxx.log        # Read a specific log file
        """,
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=None,
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        metavar="TEXT",
        help="Execute a prompt in non-interactive mode (exits after completion)",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="prompt",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version information, including the local install time",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    log_parser = subparsers.add_parser("log", help="Show log directory or read log files")
    log_parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Log filename to read (optional, shows directory if omitted)",
    )

    help_parser = subparsers.add_parser("help", help="Show command help")
    help_parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Optional help topic (for example: log)",
    )

    cleanup_parser = subparsers.add_parser(
        "cleanup-auto-skills",
        help="Review or clean up historical auto-generated skills",
    )
    cleanup_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the cleanup plan. Without this flag, only a dry-run preview is shown.",
    )
    cleanup_parser.add_argument(
        "--skills-dir",
        type=str,
        default=None,
        help="Auto-skill root directory (default: ~/.mini-agent/skills)",
    )
    cleanup_parser.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="Archive directory for migrated skill folders (default: <skills-dir>/_archived/auto-skill-cleanup-<timestamp>)",
    )
    cleanup_parser.add_argument(
        "--skip-candidates",
        action="store_true",
        help="Only inspect approved auto-skills in the root directory.",
    )

    return parser.parse_args(argv)


def print_auto_skill_cleanup_report(report: AutoSkillCleanupReport) -> None:
    """Render a human-readable summary for the auto-skill cleanup command."""
    mode_label = "Applied cleanup" if report.apply else "Dry-run cleanup plan"
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_YELLOW}{mode_label}:{Colors.RESET}")
    print(f"  Root: {Colors.BRIGHT_CYAN}{report.root}{Colors.RESET}")
    print(f"  Archive: {Colors.BRIGHT_CYAN}{report.archive_root}{Colors.RESET}")
    print(f"  Scanned generated skills: {Colors.BRIGHT_WHITE}{report.scanned_skills}{Colors.RESET}")
    print(f"  Historical families: {Colors.BRIGHT_WHITE}{report.family_groups}{Colors.RESET}")
    print(f"  To archive: {Colors.BRIGHT_WHITE}{report.archived_count}{Colors.RESET}")
    print(f"  To rename: {Colors.BRIGHT_WHITE}{report.renamed_count}{Colors.RESET}")
    print(f"  Kept active: {Colors.BRIGHT_WHITE}{report.kept_count}{Colors.RESET}")

    if not report.actions:
        print(f"  {Colors.GREEN}No cleanup actions needed.{Colors.RESET}\n")
        return

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_YELLOW}Planned Actions:{Colors.RESET}")
    for index, action in enumerate(report.actions, 1):
        if action.kind == "archive-skill" and action.target_path is not None:
            print(
                f"  {index:2d}. {Colors.YELLOW}archive{Colors.RESET} "
                f"{action.skill_path} -> {action.target_path} {Colors.DIM}({action.detail}){Colors.RESET}"
            )
            continue
        if action.kind == "rename-skill-name":
            print(
                f"  {index:2d}. {Colors.CYAN}rename{Colors.RESET} "
                f"{action.skill_path} -> name={action.replacement_name} {Colors.DIM}({action.detail}){Colors.RESET}"
            )
            continue
        print(f"  {index:2d}. {action.kind} {action.skill_path}")

    if not report.apply:
        print(
            f"\n{Colors.DIM}Re-run with {Colors.BRIGHT_GREEN}mini-agent cleanup-auto-skills --apply{Colors.DIM} to execute these changes.{Colors.RESET}\n"
        )
    else:
        print()


async def initialize_base_tools(config: Config):
    """Initialize tools that do not depend on the workspace directory."""
    tools: list[Tool] = []
    skill_loader = None
    memory_store: MemoryStore | None = None

    if config.tools.enable_bash:
        tools.append(BashOutputTool())
        print(f"{Colors.GREEN}✅ Loaded Bash Output tool{Colors.RESET}")

        tools.append(BashKillTool())
        print(f"{Colors.GREEN}✅ Loaded Bash Kill tool{Colors.RESET}")

    if config.tools.enable_memory:
        print(f"{Colors.BRIGHT_CYAN}Loading durable memory...{Colors.RESET}")
        try:
            memory_tools, memory_store = create_memory_tools()
            tools.extend(memory_tools)
            print(f"{Colors.GREEN}✅ Loaded durable memory tools{Colors.RESET}")
        except Exception as exc:
            print(f"{Colors.YELLOW}⚠️  Failed to load durable memory: {exc}{Colors.RESET}")

    if config.tools.enable_skills:
        print(f"{Colors.BRIGHT_CYAN}Loading Claude Skills...{Colors.RESET}")
        try:
            skills_path = Path(str(config.tools.skills_dir).strip()).expanduser()
            if skills_path.is_absolute():
                skills_dir = str(skills_path)
            else:
                search_paths = [
                    skills_path,
                    Path("mini_agent") / skills_path,
                    Config.get_package_dir() / skills_path,
                ]
                skills_dir = str(skills_path)
                for path in search_paths:
                    if path.exists():
                        skills_dir = str(path.resolve())
                        break

            extra_skills_dirs = [str(Path(str(path).strip()).expanduser()) for path in config.tools.skills_external_dirs]
            skill_tools, skill_loader = create_skill_tools(skills_dir, extra_skills_dirs=extra_skills_dirs)
            if skill_tools:
                tools.extend(skill_tools)
                loaded_tool_names = ", ".join(tool.name for tool in skill_tools)
                print(f"{Colors.GREEN}✅ Loaded Skill tools ({loaded_tool_names}){Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠️  No available Skills found{Colors.RESET}")
        except Exception as exc:
            print(f"{Colors.YELLOW}⚠️  Failed to load Skills: {exc}{Colors.RESET}")

    if config.tools.enable_mcp:
        print(f"{Colors.BRIGHT_CYAN}Loading MCP tools...{Colors.RESET}")
        try:
            mcp_config = config.tools.mcp
            set_mcp_timeout_config(
                connect_timeout=mcp_config.connect_timeout,
                execute_timeout=mcp_config.execute_timeout,
                sse_read_timeout=mcp_config.sse_read_timeout,
            )
            print(
                f"{Colors.DIM}  MCP timeouts: connect={mcp_config.connect_timeout}s, "
                f"execute={mcp_config.execute_timeout}s, sse_read={mcp_config.sse_read_timeout}s{Colors.RESET}"
            )

            mcp_config_path = Config.find_config_file(config.tools.mcp_config_path)
            if mcp_config_path:
                mcp_tools = await load_mcp_tools_async(str(mcp_config_path))
                if mcp_tools:
                    tools.extend(mcp_tools)
                    print(f"{Colors.GREEN}✅ Loaded {len(mcp_tools)} MCP tools (from: {mcp_config_path}){Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}⚠️  No available MCP tools found{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠️  MCP config file not found: {config.tools.mcp_config_path}{Colors.RESET}")
        except Exception as exc:
            print(f"{Colors.YELLOW}⚠️  Failed to load MCP tools: {exc}{Colors.RESET}")

    print()
    return tools, skill_loader, memory_store


def add_workspace_tools(tools: List[Tool], config: Config, workspace_dir: Path):
    """Add tools that depend on the workspace directory."""
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if config.tools.enable_bash:
        bash_tool = BashTool(workspace_dir=str(workspace_dir))
        tools.append(bash_tool)
        print(f"{Colors.GREEN}✅ Loaded Bash tool (cwd: {workspace_dir}){Colors.RESET}")

    if config.tools.enable_file_tools:
        tools.extend(
            [
                ReadTool(workspace_dir=str(workspace_dir)),
                WriteTool(workspace_dir=str(workspace_dir)),
                EditTool(workspace_dir=str(workspace_dir)),
            ]
        )
        print(f"{Colors.GREEN}✅ Loaded file operation tools (workspace: {workspace_dir}){Colors.RESET}")

    if config.tools.enable_note:
        tools.append(SessionNoteTool(memory_file=str(workspace_dir / ".agent_memory.json")))
        print(f"{Colors.GREEN}✅ Loaded session note tool{Colors.RESET}")


def _maybe_persist_auto_skill(config: Config, skill_loader, agent, turn_start_index: int, final_result: str) -> None:
    """Backward-compatible wrapper for callers that still import this helper from cli.py."""
    runtime_context = type(
        "_CompatRuntimeContext",
        (),
        {
            "config": config,
            "skill_loader": skill_loader,
            "agent": agent,
        },
    )()
    persist_auto_skill_if_needed(runtime_context, turn_start_index, final_result)


async def run_agent(workspace_dir: Path, prompt: str | None = None):
    """Run Agent in interactive or non-interactive mode."""
    context = await build_runtime_context(
        workspace_dir,
        initialize_base_tools=initialize_base_tools,
        add_workspace_tools=add_workspace_tools,
    )
    if context is None:
        return

    if prompt:
        await execute_prompt(context, prompt)
        return

    print_banner()
    print_session_info(context.agent, context.workspace_dir, context.config.llm.model)
    await run_interactive_session(context)


def main():
    """Main entry point for CLI."""
    args = parse_args()

    if getattr(args, "version", False):
        print_version_info()
        return

    if args.command == "help":
        print_help(args.topic)
        return

    if args.command == "log":
        if args.filename:
            read_log_file(args.filename)
        else:
            show_log_directory(open_file_manager=True)
        return

    if args.command == "cleanup-auto-skills":
        skills_dir = args.skills_dir or "~/.mini-agent/skills"
        report = cleanup_auto_skills(
            skills_dir,
            apply=args.apply,
            include_candidates=not args.skip_candidates,
            archive_dir=args.archive_dir,
        )
        print_auto_skill_cleanup_report(report)
        return

    if args.workspace:
        workspace_dir = Path(args.workspace).expanduser().absolute()
    else:
        workspace_dir = Path.cwd()

    workspace_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_agent(workspace_dir, prompt=args.prompt))


if __name__ == "__main__":
    main()
