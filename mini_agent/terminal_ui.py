"""Terminal UI helpers for the Mini-Agent CLI."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import calculate_display_width

if TYPE_CHECKING:
    from .agent import Agent


class Colors:
    """Terminal color definitions."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def get_log_directory() -> Path:
    """Get the log directory path."""
    return Path.home() / ".mini-agent" / "log"


def show_log_directory(open_file_manager: bool = True) -> None:
    """Show log directory contents and optionally open file manager."""
    log_dir = get_log_directory()

    print(f"\n{Colors.BRIGHT_CYAN}📁 Log Directory: {log_dir}{Colors.RESET}")

    if not log_dir.exists() or not log_dir.is_dir():
        print(f"{Colors.RED}Log directory does not exist: {log_dir}{Colors.RESET}\n")
        return

    log_files = list(log_dir.glob("*.log"))

    if not log_files:
        print(f"{Colors.YELLOW}No log files found in directory.{Colors.RESET}\n")
        return

    log_files.sort(key=lambda item: item.stat().st_mtime, reverse=True)

    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}Available Log Files (newest first):{Colors.RESET}")

    for index, log_file in enumerate(log_files[:10], 1):
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        size = log_file.stat().st_size
        size_str = f"{size:,}" if size < 1024 else f"{size / 1024:.1f}K"
        print(f"  {Colors.GREEN}{index:2d}.{Colors.RESET} {Colors.BRIGHT_WHITE}{log_file.name}{Colors.RESET}")
        print(f"      {Colors.DIM}Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}, Size: {size_str}{Colors.RESET}")

    if len(log_files) > 10:
        print(f"  {Colors.DIM}... and {len(log_files) - 10} more files{Colors.RESET}")

    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")

    if open_file_manager:
        _open_directory_in_file_manager(log_dir)

    print()


def _open_directory_in_file_manager(directory: Path) -> None:
    """Open directory in the system file manager."""
    system = platform.system()

    try:
        if system == "Darwin":
            subprocess.run(["open", str(directory)], check=False)
        elif system == "Windows":
            subprocess.run(["explorer", str(directory)], check=False)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(directory)], check=False)
    except FileNotFoundError:
        print(f"{Colors.YELLOW}Could not open file manager. Please navigate manually.{Colors.RESET}")
    except Exception as exc:  # pragma: no cover - best effort UI
        print(f"{Colors.YELLOW}Error opening file manager: {exc}{Colors.RESET}")


def read_log_file(filename: str) -> None:
    """Read and display a specific log file."""
    log_dir = get_log_directory()
    log_file = log_dir / filename

    if not log_file.exists() or not log_file.is_file():
        print(f"\n{Colors.RED}❌ Log file not found: {log_file}{Colors.RESET}\n")
        return

    print(f"\n{Colors.BRIGHT_CYAN}📄 Reading: {log_file}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")

    try:
        with open(log_file, "r", encoding="utf-8") as file:
            content = file.read()
        print(content)
        print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")
        print(f"\n{Colors.GREEN}✅ End of file{Colors.RESET}\n")
    except Exception as exc:  # pragma: no cover - best effort UI
        print(f"\n{Colors.RED}❌ Error reading file: {exc}{Colors.RESET}\n")


def print_banner() -> None:
    """Print welcome banner with proper alignment."""
    box_width = 58
    banner_text = f"{Colors.BOLD}🤖 Mini Agent - Multi-turn Interactive Session{Colors.RESET}"
    banner_width = calculate_display_width(banner_text)

    total_padding = box_width - banner_width
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╔{'═' * box_width}╗{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET}{' ' * left_padding}{banner_text}{' ' * right_padding}{Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╚{'═' * box_width}╝{Colors.RESET}")
    print()


def print_help(topic: str | None = None) -> None:
    """Print CLI or topic-specific help information."""
    topic_normalized = topic.lower().strip() if isinstance(topic, str) and topic.strip() else None

    if topic_normalized == "log":
        help_text = f"""
{Colors.BOLD}{Colors.BRIGHT_YELLOW}Log Command:{Colors.RESET}
    {Colors.BRIGHT_GREEN}mini-agent log{Colors.RESET}           Show log directory and recent files
    {Colors.BRIGHT_GREEN}mini-agent log <file>{Colors.RESET}     Read a specific log file
    {Colors.BRIGHT_GREEN}mini-agent help log{Colors.RESET}      Show log command help

{Colors.BOLD}{Colors.BRIGHT_YELLOW}Log File Tips:{Colors.RESET}
    - Use {Colors.BRIGHT_GREEN}mini-agent log{Colors.RESET} to browse recent logs
    - Use {Colors.BRIGHT_GREEN}mini-agent log <file>{Colors.RESET} to inspect a specific log file
"""
        print(help_text)
        return

    if topic_normalized == "cleanup-auto-skills":
        help_text = f"""
{Colors.BOLD}{Colors.BRIGHT_YELLOW}Cleanup Auto Skills Command:{Colors.RESET}
    {Colors.BRIGHT_GREEN}mini-agent cleanup-auto-skills{Colors.RESET}                 Preview cleanup actions for generated auto skills
    {Colors.BRIGHT_GREEN}mini-agent cleanup-auto-skills --apply{Colors.RESET}         Apply the cleanup and archive stale variants
    {Colors.BRIGHT_GREEN}mini-agent cleanup-auto-skills --skills-dir DIR{Colors.RESET} Use a custom generated-skills directory
    {Colors.BRIGHT_GREEN}mini-agent cleanup-auto-skills --skip-candidates{Colors.RESET} Ignore the _candidates/ pool
    {Colors.BRIGHT_GREEN}mini-agent help cleanup-auto-skills{Colors.RESET}            Show this help topic

{Colors.BOLD}{Colors.BRIGHT_YELLOW}What it does:{Colors.RESET}
    - Finds auto-generated skills whose metadata source is {Colors.BRIGHT_GREEN}mini-agent{Colors.RESET}
    - Archives stale historical variants under {Colors.BRIGHT_GREEN}_archived/{Colors.RESET}
    - Syncs mismatched frontmatter names with their directory names
    - Defaults to a dry run so you can review the plan before changing files
"""
        print(help_text)
        return

    if topic_normalized and topic_normalized not in {"general", "cli", "commands", "interactive"}:
        print(f"{Colors.YELLOW}Unknown help topic: {topic}{Colors.RESET}")
        print(f"{Colors.DIM}Showing the main help instead. Available topics: log, cleanup-auto-skills{Colors.RESET}\n")

    help_text = f"""
{Colors.BOLD}{Colors.BRIGHT_YELLOW}Mini-Agent CLI:{Colors.RESET}
    {Colors.BRIGHT_GREEN}mini-agent{Colors.RESET}                   Start interactive mode
    {Colors.BRIGHT_GREEN}mini-agent -p \"<text>\"{Colors.RESET}     Run a prompt non-interactively and exit
    {Colors.BRIGHT_GREEN}mini-agent --workspace DIR{Colors.RESET} Use a specific workspace directory
    {Colors.BRIGHT_GREEN}mini-agent log [file]{Colors.RESET}      Show logs or read a specific log file
    {Colors.BRIGHT_GREEN}mini-agent cleanup-auto-skills{Colors.RESET} Review or clean historical generated skills
    {Colors.BRIGHT_GREEN}mini-agent help [topic]{Colors.RESET}    Show this help or a topic-specific help
    {Colors.BRIGHT_GREEN}mini-agent help log{Colors.RESET}      Show log command help
    {Colors.BRIGHT_GREEN}mini-agent help cleanup-auto-skills{Colors.RESET} Show cleanup command help
    {Colors.BRIGHT_GREEN}mini-agent --version{Colors.RESET}       Show version information

{Colors.BOLD}{Colors.BRIGHT_YELLOW}Interactive Commands:{Colors.RESET}
    {Colors.BRIGHT_GREEN}/help{Colors.RESET}      - Show this help message
    {Colors.BRIGHT_GREEN}/clear{Colors.RESET}     - Clear session history (keep system prompt)
    {Colors.BRIGHT_GREEN}/history{Colors.RESET}   - Show current session message count
    {Colors.BRIGHT_GREEN}/stats{Colors.RESET}     - Show session statistics
    {Colors.BRIGHT_GREEN}/log{Colors.RESET}       - Show log directory and recent files
    {Colors.BRIGHT_GREEN}/log <file>{Colors.RESET} - Read a specific log file
    {Colors.BRIGHT_GREEN}/exit{Colors.RESET}      - Exit program (also: exit, quit, q)

{Colors.BOLD}{Colors.BRIGHT_YELLOW}Keyboard Shortcuts:{Colors.RESET}
    {Colors.BRIGHT_CYAN}Esc{Colors.RESET}        - Cancel current agent execution
    {Colors.BRIGHT_CYAN}Ctrl+C{Colors.RESET}     - Exit program
    {Colors.BRIGHT_CYAN}Ctrl+U{Colors.RESET}     - Clear current input line
    {Colors.BRIGHT_CYAN}Ctrl+L{Colors.RESET}     - Clear screen
    {Colors.BRIGHT_CYAN}Ctrl+J{Colors.RESET}     - Insert newline (also Ctrl+Enter)
    {Colors.BRIGHT_CYAN}Tab{Colors.RESET}        - Auto-complete commands
    {Colors.BRIGHT_CYAN}↑/↓{Colors.RESET}        - Browse command history
    {Colors.BRIGHT_CYAN}→{Colors.RESET}          - Accept auto-suggestion

{Colors.BOLD}{Colors.BRIGHT_YELLOW}Usage:{Colors.RESET}
    - Enter your task directly, Agent will help you complete it
    - Agent remembers all conversation content in this session
    - Use {Colors.BRIGHT_GREEN}/clear{Colors.RESET} to start a new session
    - Press {Colors.BRIGHT_CYAN}Enter{Colors.RESET} to submit your message
    - Use {Colors.BRIGHT_CYAN}Ctrl+J{Colors.RESET} to insert line breaks within your message
    - Try {Colors.BRIGHT_GREEN}mini-agent -p \"列出当前的skills有哪些\"{Colors.RESET} for a one-shot prompt
"""
    print(help_text)


def print_session_info(agent: "Agent", workspace_dir: Path, model: str) -> None:
    """Print session information with proper alignment."""
    box_width = 58

    def print_info_line(text: str) -> None:
        text_width = calculate_display_width(text)
        padding = max(0, box_width - 1 - text_width)
        print(f"{Colors.DIM}│{Colors.RESET} {text}{' ' * padding}{Colors.DIM}│{Colors.RESET}")

    print(f"{Colors.DIM}┌{'─' * box_width}┐{Colors.RESET}")

    header_text = f"{Colors.BRIGHT_CYAN}Session Info{Colors.RESET}"
    header_width = calculate_display_width(header_text)
    header_padding_total = box_width - 1 - header_width
    header_padding_left = header_padding_total // 2
    header_padding_right = header_padding_total - header_padding_left
    print(f"{Colors.DIM}│{Colors.RESET} {' ' * header_padding_left}{header_text}{' ' * header_padding_right}{Colors.DIM}│{Colors.RESET}")

    print(f"{Colors.DIM}├{'─' * box_width}┤{Colors.RESET}")
    print_info_line(f"Model: {model}")
    print_info_line(f"Workspace: {workspace_dir}")
    print_info_line(f"Message History: {len(agent.messages)} messages")
    print_info_line(f"Available Tools: {len(agent.tools)} tools")
    print(f"{Colors.DIM}└{'─' * box_width}┘{Colors.RESET}")
    print()
    print(f"{Colors.DIM}Type {Colors.BRIGHT_GREEN}/help{Colors.DIM} for help, {Colors.BRIGHT_GREEN}/exit{Colors.DIM} to quit{Colors.RESET}")
    print()


def print_stats(agent: "Agent", session_start: datetime) -> None:
    """Print session statistics."""
    duration = datetime.now() - session_start
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    user_msgs = sum(1 for message in agent.messages if message.role == "user")
    assistant_msgs = sum(1 for message in agent.messages if message.role == "assistant")
    tool_msgs = sum(1 for message in agent.messages if message.role == "tool")

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}Session Statistics:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
    print(f"  Session Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"  Total Messages: {len(agent.messages)}")
    print(f"    - User Messages: {Colors.BRIGHT_GREEN}{user_msgs}{Colors.RESET}")
    print(f"    - Assistant Replies: {Colors.BRIGHT_BLUE}{assistant_msgs}{Colors.RESET}")
    print(f"    - Tool Calls: {Colors.BRIGHT_YELLOW}{tool_msgs}{Colors.RESET}")
    print(f"  Available Tools: {len(agent.tools)}")
    if agent.api_total_tokens > 0:
        print(f"  API Tokens Used: {Colors.BRIGHT_MAGENTA}{agent.api_total_tokens:,}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}\n")
