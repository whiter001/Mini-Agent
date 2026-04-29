"""Readiness heuristics for shell commands that start long-running services."""

from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bash_runtime import BackgroundShell

BACKGROUND_STARTUP_GRACE_SECONDS = 0.05
READY_WAIT_TIMEOUT_SECONDS = 8.0
READY_POLL_INTERVAL_SECONDS = 0.1
STARTUP_OUTPUT_PREVIEW_LINES = 8
LONG_RUNNING_COMMAND_PATTERNS = (
    re.compile(r"\bnpm\s+run\s+(dev|serve|preview|watch)\b", re.IGNORECASE),
    re.compile(r"\bpnpm\s+(dev|serve|preview|watch)\b", re.IGNORECASE),
    re.compile(r"\byarn\s+(dev|serve|preview|watch)\b", re.IGNORECASE),
    re.compile(r"\bbun\s+(run\s+)?(dev|serve|preview|watch)\b", re.IGNORECASE),
    re.compile(r"\bvite(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bnext\s+dev\b", re.IGNORECASE),
    re.compile(r"\bnuxt\s+(dev|preview)\b", re.IGNORECASE),
    re.compile(r"\buvicorn\b", re.IGNORECASE),
    re.compile(r"\bflask\s+run\b", re.IGNORECASE),
    re.compile(r"\bdjango-admin\s+runserver\b", re.IGNORECASE),
    re.compile(r"\bmanage\.py\s+runserver\b", re.IGNORECASE),
    re.compile(r"\bhttp\.server\b", re.IGNORECASE),
    re.compile(r"--(?:watch|reload)\b", re.IGNORECASE),
)
LONG_RUNNING_COMMAND_BLOCKERS = (
    re.compile(r"\bvite\s+build\b", re.IGNORECASE),
    re.compile(r"\b(?:npm\s+run|pnpm|yarn|bun(?:\s+run)?)\s+(build|test|lint|format|typecheck)\b", re.IGNORECASE),
    re.compile(r"\b--(?:help|version)\b", re.IGNORECASE),
)
READY_OUTPUT_PATTERNS = (
    re.compile(r"\bready in\b", re.IGNORECASE),
    re.compile(r"\blocal:\s*https?://", re.IGNORECASE),
    re.compile(r"\bnetwork:\s*https?://", re.IGNORECASE),
    re.compile(r"\bserving http on\b", re.IGNORECASE),
    re.compile(r"\bapplication startup complete\b", re.IGNORECASE),
    re.compile(r"\buvicorn running on\b", re.IGNORECASE),
    re.compile(r"\brunning on https?://", re.IGNORECASE),
    re.compile(r"\blistening on\b", re.IGNORECASE),
    re.compile(r"\bserver started\b", re.IGNORECASE),
    re.compile(r"\bcompiled successfully\b", re.IGNORECASE),
)


def normalize_command(command: str) -> str:
    """Normalize whitespace to improve command heuristics."""
    return " ".join(command.strip().split())


def looks_like_long_running_command(command: str) -> bool:
    """Return True when the command appears to start a long-running dev/service process."""
    normalized = normalize_command(command)
    if any(pattern.search(normalized) for pattern in LONG_RUNNING_COMMAND_BLOCKERS):
        return False
    return any(pattern.search(normalized) for pattern in LONG_RUNNING_COMMAND_PATTERNS)


def extract_ready_target(command: str) -> tuple[str, int] | None:
    """Best-effort extraction of a local host/port pair for readiness checks."""
    normalized = normalize_command(command)

    host_match = re.search(r"--(?:host|bind)(?:=|\s+)([^\s]+)", normalized, re.IGNORECASE)
    host = host_match.group(1).strip("\"'") if host_match else "127.0.0.1"
    if host in {"0.0.0.0", "::", "[::]", "::1"}:
        host = "127.0.0.1"

    port: int | None = None
    patterns = (
        re.compile(r"\bhttp\.server(?:\s+--bind\s+[^\s]+)?\s+(\d{2,5})\b", re.IGNORECASE),
        re.compile(r"\b--port(?:=|\s+)(\d{2,5})\b", re.IGNORECASE),
        re.compile(r"\brunserver\s+(?:[^\s:]+:)?(\d{2,5})\b", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(normalized)
        if match:
            port = int(match.group(1))
            break

    lowered = normalized.lower()
    if port is None:
        default_ports = (
            ("http.server", 8000),
            ("uvicorn", 8000),
            ("flask run", 5000),
            ("vite", 5173),
            ("next dev", 3000),
        )
        for token, default_port in default_ports:
            if token in lowered:
                port = default_port
                break

    if port is None:
        return None

    return host, port


def extract_startup_output(shell: "BackgroundShell", max_lines: int = STARTUP_OUTPUT_PREVIEW_LINES) -> str:
    """Return a compact startup output preview without consuming unread lines."""
    lines = [line for line in shell.output_lines[-max_lines:] if line.strip()]
    return "\n".join(lines)


def find_ready_line(lines: list[str]) -> str | None:
    """Detect a service readiness indicator from captured output."""
    for line in reversed(lines[-(STARTUP_OUTPUT_PREVIEW_LINES * 2) :]):
        stripped = line.strip()
        if stripped and any(pattern.search(stripped) for pattern in READY_OUTPUT_PATTERNS):
            return stripped
    return None


async def is_port_accepting_connections(host: str, port: int) -> bool:
    """Return True when a local TCP port starts accepting connections."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=0.25)
    except (asyncio.TimeoutError, OSError):
        return False

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass

    return True


async def wait_for_background_ready(shell: "BackgroundShell", command: str, timeout: float) -> str | None:
    """Wait briefly for a readiness signal from a backgrounded service."""
    deadline = time.monotonic() + timeout
    ready_target = extract_ready_target(command)

    while time.monotonic() < deadline:
        ready_line = find_ready_line(shell.output_lines)
        if ready_line:
            return ready_line

        if ready_target and await is_port_accepting_connections(*ready_target):
            host, port = ready_target
            return f"Port {host}:{port} is accepting connections"

        if shell.process.returncode is not None:
            shell.update_status(is_alive=False, exit_code=shell.process.returncode)
            return None

        await asyncio.sleep(READY_POLL_INTERVAL_SECONDS)

    ready_line = find_ready_line(shell.output_lines)
    if ready_line:
        return ready_line

    if ready_target and await is_port_accepting_connections(*ready_target):
        host, port = ready_target
        return f"Port {host}:{port} is accepting connections"

    return None


__all__ = [
    "BACKGROUND_STARTUP_GRACE_SECONDS",
    "READY_WAIT_TIMEOUT_SECONDS",
    "STARTUP_OUTPUT_PREVIEW_LINES",
    "extract_startup_output",
    "looks_like_long_running_command",
    "wait_for_background_ready",
]
