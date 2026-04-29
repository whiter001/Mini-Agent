"""Shell command execution tool with background process management.

Supports both bash (Unix/Linux/macOS) and PowerShell (Windows).
"""

from __future__ import annotations

import asyncio
import platform
import time
import uuid
from typing import Any

from pydantic import Field, model_validator

from .base import Tool, ToolResult
from .bash_readiness import (
    BACKGROUND_STARTUP_GRACE_SECONDS,
    READY_WAIT_TIMEOUT_SECONDS,
    STARTUP_OUTPUT_PREVIEW_LINES,
    extract_startup_output,
    looks_like_long_running_command,
    wait_for_background_ready,
)
from .bash_runtime import BackgroundShell, BackgroundShellManager, close_process_transport


class BashOutputResult(ToolResult):
    """Bash command execution result with separated stdout and stderr."""

    stdout: str = Field(description="The command's standard output")
    stderr: str = Field(description="The command's standard error output")
    exit_code: int = Field(description="The command's exit code")
    bash_id: str | None = Field(default=None, description="Shell process ID when the command runs in background")

    @model_validator(mode="after")
    def format_content(self) -> "BashOutputResult":
        """Auto-format content from stdout and stderr if content is empty."""
        output = ""
        if self.stdout:
            output += self.stdout
        if self.stderr:
            output += f"\n[stderr]:\n{self.stderr}"
        if self.bash_id:
            output += f"\n[bash_id]:\n{self.bash_id}"
        if self.exit_code:
            output += f"\n[exit_code]:\n{self.exit_code}"

        if not output:
            output = "(no output)"

        self.content = output
        return self


class BashTool(Tool):
    """Execute shell commands in foreground or background."""

    def __init__(self, workspace_dir: str | None = None):
        self.is_windows = platform.system() == "Windows"
        self.shell_name = "PowerShell" if self.is_windows else "bash"
        self.workspace_dir = workspace_dir

    def _build_subprocess_command(self, command: str) -> tuple[list[str] | str, bool]:
        """Build the subprocess invocation for the current platform shell."""
        if self.is_windows:
            return ["powershell.exe", "-NoProfile", "-Command", command], False
        return command, True

    async def _create_process(
        self,
        command: str,
        *,
        stdout,
        stderr,
    ) -> "asyncio.subprocess.Process":
        """Create a subprocess using the current platform shell."""
        shell_cmd, use_shell = self._build_subprocess_command(command)
        if use_shell:
            return await asyncio.create_subprocess_shell(
                shell_cmd,
                stdout=stdout,
                stderr=stderr,
                cwd=self.workspace_dir,
            )

        return await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdout=stdout,
            stderr=stderr,
            cwd=self.workspace_dir,
        )

    def _looks_like_long_running_command(self, command: str) -> bool:
        """Return True when the command appears to start a long-running dev/service process."""
        return looks_like_long_running_command(command)

    async def _run_background_command(
        self,
        command: str,
        *,
        auto_backgrounded: bool,
        startup_timeout: float,
    ) -> BashOutputResult:
        """Start a command in the background and optionally wait for a readiness signal."""
        bash_id = str(uuid.uuid4())[:8]

        process = await self._create_process(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        await asyncio.sleep(BACKGROUND_STARTUP_GRACE_SECONDS)
        if process.returncode not in (None, 0):
            stdout, _ = await process.communicate()
            await close_process_transport(process)
            stderr_text = stdout.decode("utf-8", errors="replace")
            return BashOutputResult(
                success=False,
                error=f"Command failed to start in {self.shell_name} (exit code {process.returncode})",
                stdout="",
                stderr=stderr_text,
                exit_code=process.returncode,
            )

        bg_shell = BackgroundShell(bash_id=bash_id, command=command, process=process, start_time=time.time())
        BackgroundShellManager.add(bg_shell)
        await BackgroundShellManager.start_monitor(bash_id)

        wait_for_ready = auto_backgrounded or self._looks_like_long_running_command(command)
        ready_line: str | None = None
        if wait_for_ready:
            ready_line = await wait_for_background_ready(bg_shell, command, startup_timeout)

        if process.returncode not in (None, 0):
            startup_output = extract_startup_output(bg_shell, max_lines=STARTUP_OUTPUT_PREVIEW_LINES * 2)
            await BackgroundShellManager.terminate(bash_id)
            error_msg = f"Command failed shortly after startup in {self.shell_name} (exit code {process.returncode})"
            return BashOutputResult(
                success=False,
                error=error_msg,
                stdout="",
                stderr=startup_output or error_msg,
                exit_code=process.returncode,
            )

        stdout_lines = [f"Background command started with ID: {bash_id}"]
        if auto_backgrounded:
            stdout_lines.append(
                "Auto-switched to background because the command appears to start a long-running service."
            )
        if ready_line:
            stdout_lines.append(f"Detected service readiness: {ready_line}")
        elif wait_for_ready:
            stdout_lines.append(
                "No readiness signal detected yet. The process is still running in background; use bash_output to monitor it."
            )

        startup_output = extract_startup_output(bg_shell)
        if startup_output:
            stdout_lines.append(f"Startup output:\n{startup_output}")

        return BashOutputResult(
            success=True,
            stdout="\n".join(stdout_lines),
            stderr="",
            exit_code=0,
            bash_id=bash_id,
        )

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        shell_examples = {
            "Windows": """Execute shell commands in the current platform shell.

On Windows this tool runs commands via PowerShell. The tool name remains `bash` for backward compatibility.

For terminal operations like git, npm, docker, etc. DO NOT use for file operations - use specialized tools.

Parameters:
  - command (required): PowerShell command to execute
  - timeout (optional): Timeout in seconds (default: 120, max: 600) for foreground commands
  - run_in_background (optional): Set true for long-running commands (servers, etc.)

Tips:
  - Quote file paths with spaces: cd "My Documents"
  - Chain dependent commands with semicolon: git add . ; git commit -m "msg"
  - Use absolute paths instead of cd when possible
    - For Windows CLI tools that accept JavaScript or stdin (for example `autobrowser.cmd eval`), avoid PowerShell redirection like `<` and `>`; prefer `--file`/`--base64` or a single fully quoted one-liner
        - Bootstrap autobrowser with supported commands like `autobrowser.cmd server start` / `autobrowser.cmd connect`; do not assume a `start --headless` subcommand exists
        - Navigate with supported autobrowser commands like `open` / `goto`; do not assume a `navigate` subcommand exists
        - For `autobrowser.cmd find`, use one strategy at a time (for example `find text "我来答"`); do not combine strategies like `find role=text ...`
        - For `autobrowser.cmd click`, pass a selector or a ref from `find` / `snapshot`; do not use `click --text ...`
        - Prefer `autobrowser.cmd wait ms <milliseconds>` for waits; do not chain Windows shell `timeout` into autobrowser commands
                - For page scrolling, prefer `eval "window.scrollBy(0, 500)"` or pass an explicit selector to `scroll`; do not call `autobrowser.cmd scroll 500`
                - For iframe-based rich-text editors (for example UEditor), select the real iframe with `frame "<iframe-selector>"`, type into `body`, then `frame top` before clicking the page-level submit button
                - Avoid generic submit selectors like `[class*=submit]`; prefer exact answer-submit selectors such as `.new-editor-deliver-btn`, and verify success via durable signals like `newAnswer=1` or a visible `我的回答` block
    - Quote snapshot refs that include page epochs, for example `"@e1#p71"`, so PowerShell does not mangle the selector
  - Common service commands like vite, npm run dev, uvicorn, and python -m http.server are auto-started in background
  - For background commands, monitor with bash_output and terminate with bash_kill

Examples:
  - git status
  - npm test
  - python -m http.server 8080 (with run_in_background=true)""",
            "Unix": """Execute shell commands in the current platform shell.

On Unix-like systems this tool runs commands via bash. The tool name remains `bash` for backward compatibility.

For terminal operations like git, npm, docker, etc. DO NOT use for file operations - use specialized tools.

Parameters:
  - command (required): Bash command to execute
  - timeout (optional): Timeout in seconds (default: 120, max: 600) for foreground commands
  - run_in_background (optional): Set true for long-running commands (servers, etc.)

Tips:
  - Quote file paths with spaces: cd "My Documents"
  - Chain dependent commands with &&: git add . && git commit -m "msg"
  - Use absolute paths instead of cd when possible
  - Common service commands like vite, npm run dev, uvicorn, and python -m http.server are auto-started in background
  - For background commands, monitor with bash_output and terminate with bash_kill

Examples:
  - git status
  - npm test
  - python3 -m http.server 8080 (with run_in_background=true)""",
        }
        return shell_examples["Windows"] if self.is_windows else shell_examples["Unix"]

    @property
    def parameters(self) -> dict[str, Any]:
        cmd_desc = f"The {self.shell_name} command to execute. Quote file paths with spaces using double quotes."
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": cmd_desc,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional: Timeout in seconds (default: 120, max: 600). Only applies to foreground commands.",
                    "default": 120,
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Optional: Set to true to run the command in the background. Use this for long-running commands like servers. Common dev/server commands are auto-backgrounded even when omitted. You can monitor output using bash_output tool.",
                    "default": False,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        timeout: int = 120,
        run_in_background: bool = False,
    ) -> ToolResult:
        """Execute shell command with optional background execution."""
        try:
            if timeout > 600:
                timeout = 600
            elif timeout < 1:
                timeout = 120

            auto_backgrounded = not run_in_background and self._looks_like_long_running_command(command)
            startup_timeout = min(float(timeout), READY_WAIT_TIMEOUT_SECONDS)
            if run_in_background or auto_backgrounded:
                return await self._run_background_command(
                    command,
                    auto_backgrounded=auto_backgrounded,
                    startup_timeout=startup_timeout,
                )

            process = await self._create_process(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await process.communicate()
                except Exception:
                    try:
                        await process.wait()
                    except Exception:
                        pass
                await close_process_transport(process)
                error_msg = f"Command timed out after {timeout} seconds"
                return BashOutputResult(
                    success=False,
                    error=error_msg,
                    stdout="",
                    stderr=error_msg,
                    exit_code=-1,
                )

            await close_process_transport(process)

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            is_success = process.returncode == 0
            error_msg = None
            if not is_success:
                error_msg = f"Command failed with exit code {process.returncode}"
                if stderr_text:
                    error_msg += f"\n{stderr_text.strip()}"

            return BashOutputResult(
                success=is_success,
                error=error_msg,
                stdout=stdout_text,
                stderr=stderr_text,
                exit_code=process.returncode or 0,
            )

        except Exception as exc:
            return BashOutputResult(
                success=False,
                error=str(exc),
                stdout="",
                stderr=str(exc),
                exit_code=-1,
            )


class BashOutputTool(Tool):
    """Retrieve output from background bash shells."""

    @property
    def name(self) -> str:
        return "bash_output"

    @property
    def description(self) -> str:
        return """Retrieves output from a running or completed background bash shell.

        - Takes a bash_id parameter identifying the shell
        - Always returns only new output since the last check
        - Returns stdout and stderr output along with shell status
        - Supports optional regex filtering to show only lines matching a pattern
        - Use this tool when you need to monitor or check the output of a long-running shell
        - Shell IDs can be found from commands started in background, including auto-backgrounded service commands

        Process status values:
          - "running": Still executing
          - "completed": Finished successfully
          - "failed": Finished with error
          - "terminated": Was terminated
          - "error": Error occurred

        Example: bash_output(bash_id="abc12345")"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bash_id": {
                    "type": "string",
                    "description": "The ID of the background shell to retrieve output from. Shell IDs are returned when commands are started in background.",
                },
                "filter_str": {
                    "type": "string",
                    "description": "Optional regular expression to filter the output lines. Only lines matching this regex will be included in the result. Any lines that do not match will no longer be available to read.",
                },
            },
            "required": ["bash_id"],
        }

    async def execute(
        self,
        bash_id: str,
        filter_str: str | None = None,
    ) -> BashOutputResult:
        """Retrieve output from background shell."""
        try:
            bg_shell = BackgroundShellManager.get(bash_id)
            if not bg_shell:
                available_ids = BackgroundShellManager.get_available_ids()
                return BashOutputResult(
                    success=False,
                    error=f"Shell not found: {bash_id}. Available: {available_ids or 'none'}",
                    stdout="",
                    stderr="",
                    exit_code=-1,
                )

            new_lines = bg_shell.get_new_output(filter_pattern=filter_str)
            stdout = "\n".join(new_lines) if new_lines else ""

            return BashOutputResult(
                success=True,
                stdout=stdout,
                stderr="",
                exit_code=bg_shell.exit_code if bg_shell.exit_code is not None else 0,
                bash_id=bash_id,
            )

        except Exception as exc:
            return BashOutputResult(
                success=False,
                error=f"Failed to get bash output: {str(exc)}",
                stdout="",
                stderr=str(exc),
                exit_code=-1,
            )


class BashKillTool(Tool):
    """Terminate a running background bash shell."""

    @property
    def name(self) -> str:
        return "bash_kill"

    @property
    def description(self) -> str:
        return """Kills a running background bash shell by its ID.

        - Takes a bash_id parameter identifying the shell to kill
        - Attempts graceful termination (SIGTERM) first, then forces (SIGKILL) if needed
        - Returns the final status and any remaining output before termination
        - Cleans up all resources associated with the shell
        - Use this tool when you need to terminate a long-running shell
        - Shell IDs can be found from commands started in background, including auto-backgrounded service commands

        Example: bash_kill(bash_id="abc12345")"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bash_id": {
                    "type": "string",
                    "description": "The ID of the background shell to terminate. Shell IDs are returned when commands are started in background.",
                },
            },
            "required": ["bash_id"],
        }

    async def execute(self, bash_id: str) -> BashOutputResult:
        """Terminate a background shell process."""
        try:
            bg_shell = BackgroundShellManager.get(bash_id)
            remaining_lines = bg_shell.get_new_output() if bg_shell else []

            bg_shell = await BackgroundShellManager.terminate(bash_id)
            stdout = "\n".join(remaining_lines) if remaining_lines else ""

            return BashOutputResult(
                success=True,
                stdout=stdout,
                stderr="",
                exit_code=bg_shell.exit_code if bg_shell.exit_code is not None else 0,
                bash_id=bash_id,
            )

        except ValueError as exc:
            available_ids = BackgroundShellManager.get_available_ids()
            return BashOutputResult(
                success=False,
                error=f"{str(exc)}. Available: {available_ids or 'none'}",
                stdout="",
                stderr=str(exc),
                exit_code=-1,
            )
        except Exception as exc:
            return BashOutputResult(
                success=False,
                error=f"Failed to terminate bash shell: {str(exc)}",
                stdout="",
                stderr=str(exc),
                exit_code=-1,
            )


class ShellTool(BashTool):
    """Compatibility alias with a clearer name for platform-native shell execution."""


async def cleanup_background_shells() -> int:
    """Terminate all tracked background shell processes and release their transports."""
    return await BackgroundShellManager.cleanup_all()


__all__ = [
    "BackgroundShell",
    "BackgroundShellManager",
    "BashKillTool",
    "BashOutputResult",
    "BashOutputTool",
    "BashTool",
    "ShellTool",
    "cleanup_background_shells",
]
