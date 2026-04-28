"""Mini Agent CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List

from mini_agent.config import Config
from mini_agent.memory_store import MemoryStore
from mini_agent.runtime import (
    build_runtime_context,
    build_turn_context,
    execute_prompt,
    persist_auto_skill_if_needed,
    run_interactive_session,
)
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
  mini-agent help                         # Show command help
  mini-agent help log                     # Show log command help
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
        action="version",
        version="mini-agent 0.1.0",
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

    return parser.parse_args(argv)


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

    if args.command == "help":
        print_help(args.topic)
        return

    if args.command == "log":
        if args.filename:
            read_log_file(args.filename)
        else:
            show_log_directory(open_file_manager=True)
        return

    if args.workspace:
        workspace_dir = Path(args.workspace).expanduser().absolute()
    else:
        workspace_dir = Path.cwd()

    workspace_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_agent(workspace_dir, prompt=args.prompt))


if __name__ == "__main__":
    main()
