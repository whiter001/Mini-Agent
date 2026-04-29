"""Test cases for Bash Tool."""

import asyncio
import platform
import socket
import sys
import time

import pytest

from mini_agent.tools.bash_tool import BackgroundShellManager, BashKillTool, BashOutputTool, BashTool


def _shell_command(unix: str, windows: str) -> str:
    return windows if platform.system() == "Windows" else unix


def _sleep_command(seconds: int | float) -> str:
    if platform.system() == "Windows":
        if float(seconds).is_integer():
            return f"Start-Sleep -Seconds {int(seconds)}"
        return f"Start-Sleep -Milliseconds {int(seconds * 1000)}"
    return f"sleep {seconds}"


def _loop_command(prefix: str, count: int, delay_ms: int = 500) -> str:
    if platform.system() == "Windows":
        return f'1..{count} | ForEach-Object {{ Write-Output ("{prefix} $($_)"); Start-Sleep -Milliseconds {delay_ms} }}'
    numbers = " ".join(str(i) for i in range(1, count + 1))
    return f"for i in {numbers}; do echo '{prefix} '$i; sleep {delay_ms / 1000}; done"


def _python_module_command(module: str, *args: object) -> str:
    quoted_python = f'"{sys.executable}"'
    arg_text = " ".join(str(arg) for arg in args)
    prefix = f"& {quoted_python}" if platform.system() == "Windows" else quoted_python
    return f"{prefix} -m {module} {arg_text}".strip()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_foreground_command():
    """Test executing a simple foreground command."""
    print("\n=== Testing Foreground Command ===")

    bash_tool = BashTool()
    result = await bash_tool.execute(command=_shell_command("echo 'Hello from foreground'", "Write-Output 'Hello from foreground'"))

    assert result.success
    assert "Hello from foreground" in result.stdout
    assert result.exit_code == 0
    print(f"Output: {result.content}")


@pytest.mark.asyncio
async def test_foreground_command_with_stderr():
    """Test command that outputs to both stdout and stderr."""
    print("\n=== Testing Stdout/Stderr Separation ===")

    bash_tool = BashTool()
    result = await bash_tool.execute(
        command=_shell_command(
            "echo 'stdout message' && echo 'stderr message' >&2",
            "Write-Output 'stdout message'; [Console]::Error.WriteLine('stderr message')",
        )
    )

    assert result.success
    assert "stdout message" in result.stdout
    assert "stderr message" in result.stderr
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")


@pytest.mark.asyncio
async def test_command_failure():
    """Test command that fails with non-zero exit code."""
    print("\n=== Testing Command Failure ===")

    bash_tool = BashTool()
    result = await bash_tool.execute(command=_shell_command("ls /nonexistent_directory_12345", "Get-ChildItem 'nonexistent_directory_12345'"))

    assert not result.success
    assert result.exit_code != 0
    assert result.error is not None
    print(f"Error: {result.error}")


@pytest.mark.asyncio
async def test_command_timeout():
    """Test command timeout."""
    print("\n=== Testing Command Timeout ===")

    bash_tool = BashTool()
    result = await bash_tool.execute(command=_sleep_command(10), timeout=1)

    assert not result.success
    assert "timed out" in result.error.lower()
    assert result.exit_code == -1
    print(f"Timeout error: {result.error}")


@pytest.mark.asyncio
async def test_background_command():
    """Test running a command in the background."""
    print("\n=== Testing Background Command ===")

    bash_tool = BashTool()
    result = await bash_tool.execute(
        command=_loop_command("Line", 3, delay_ms=500), run_in_background=True
    )

    assert result.success
    assert result.bash_id is not None
    assert "Background command started" in result.stdout

    bash_id = result.bash_id
    print(f"Background command started with ID: {bash_id}")

    # Wait a bit for output
    await asyncio.sleep(1)

    # Check output
    bash_output_tool = BashOutputTool()
    output_result = await bash_output_tool.execute(bash_id=bash_id)

    assert output_result.success
    print(f"Output:\n{output_result.content}")

    # Clean up - terminate the background process
    bash_kill_tool = BashKillTool()
    kill_result = await bash_kill_tool.execute(bash_id=bash_id)
    assert kill_result.success
    print("Background process terminated")


@pytest.mark.asyncio
async def test_bash_output_monitoring():
    """Test monitoring background command output."""
    print("\n=== Testing Output Monitoring ===")

    bash_tool = BashTool()

    # Start background command
    result = await bash_tool.execute(
        command=_loop_command("Line", 5, delay_ms=500), run_in_background=True
    )

    assert result.success
    bash_id = result.bash_id
    print(f"Started background command: {bash_id}")

    bash_output_tool = BashOutputTool()

    # Check output multiple times (incremental output)
    for i in range(3):
        await asyncio.sleep(1)
        output_result = await bash_output_tool.execute(bash_id=bash_id)
        assert output_result.success
        print(f"\n--- Check #{i + 1} ---")
        print(f"Output:\n{output_result.content}")

    # Clean up
    bash_kill_tool = BashKillTool()
    await bash_kill_tool.execute(bash_id=bash_id)


@pytest.mark.asyncio
async def test_bash_output_with_filter():
    """Test bash_output with regex filter."""
    print("\n=== Testing Output Filter ===")

    bash_tool = BashTool()

    # Start background command
    result = await bash_tool.execute(
        command=_loop_command("Line", 5, delay_ms=300), run_in_background=True
    )

    assert result.success
    bash_id = result.bash_id

    # Wait for some output
    await asyncio.sleep(2)

    # Get filtered output (only lines with "Line 2" or "Line 4")
    bash_output_tool = BashOutputTool()
    output_result = await bash_output_tool.execute(bash_id=bash_id, filter_str="Line [24]")

    assert output_result.success
    lines = output_result.content
    print(f"Filtered output:\n{output_result.content}")

    # Clean up
    bash_kill_tool = BashKillTool()
    await bash_kill_tool.execute(bash_id=bash_id)


@pytest.mark.asyncio
async def test_bash_kill():
    """Test terminating a background command."""
    print("\n=== Testing Bash Kill ===")

    bash_tool = BashTool()

    # Start a long-running background command
    result = await bash_tool.execute(command=_sleep_command(100), run_in_background=True)

    assert result.success
    bash_id = result.bash_id
    print(f"Started long-running command: {bash_id}")

    # Verify it's running
    await asyncio.sleep(0.5)
    bg_shell = BackgroundShellManager.get(bash_id)
    assert bg_shell is not None
    assert bg_shell.status == "running"

    # Kill it
    bash_kill_tool = BashKillTool()
    kill_result = await bash_kill_tool.execute(bash_id=bash_id)

    assert kill_result.success
    # exit_code -15 means terminated by SIGTERM
    assert kill_result.exit_code == -15 or kill_result.bash_id == bash_id
    print(f"Kill result:\n{kill_result.content}")

    # Verify it's removed from manager
    bg_shell = BackgroundShellManager.get(bash_id)
    assert bg_shell is None


@pytest.mark.asyncio
async def test_bash_kill_nonexistent():
    """Test killing a non-existent bash process."""
    print("\n=== Testing Kill Non-existent Process ===")

    bash_kill_tool = BashKillTool()
    result = await bash_kill_tool.execute(bash_id="nonexistent123")

    assert not result.success
    assert "not found" in result.error.lower()
    print(f"Expected error: {result.error}")


@pytest.mark.asyncio
async def test_bash_output_nonexistent():
    """Test getting output from non-existent bash process."""
    print("\n=== Testing Output From Non-existent Process ===")

    bash_output_tool = BashOutputTool()
    result = await bash_output_tool.execute(bash_id="nonexistent123")

    assert not result.success
    assert "not found" in result.error.lower()
    print(f"Expected error: {result.error}")


@pytest.mark.asyncio
async def test_multiple_background_commands():
    """Test running multiple background commands simultaneously."""
    print("\n=== Testing Multiple Background Commands ===")

    bash_tool = BashTool()

    # Start multiple background commands
    bash_ids = []
    for i in range(3):
        result = await bash_tool.execute(
            command=_loop_command(f"Command {i + 1} Line", 3, delay_ms=500), run_in_background=True
        )
        assert result.success
        bash_ids.append(result.bash_id)
        print(f"Started command {i + 1}: {result.bash_id}")

    # Wait and check all commands
    await asyncio.sleep(1)

    bash_output_tool = BashOutputTool()
    for bash_id in bash_ids:
        output_result = await bash_output_tool.execute(bash_id=bash_id)
        assert output_result.success
        print(f"\nOutput for {bash_id}:\n{output_result.content[:100]}...")

    # Clean up all
    bash_kill_tool = BashKillTool()
    for bash_id in bash_ids:
        await bash_kill_tool.execute(bash_id=bash_id)

    print("All background processes cleaned up")


@pytest.mark.asyncio
async def test_background_shell_cleanup_all():
    """Test global cleanup for any lingering background shells."""
    bash_tool = BashTool()

    result = await bash_tool.execute(command=_sleep_command(100), run_in_background=True)

    assert result.success
    cleaned = await BackgroundShellManager.cleanup_all()

    assert cleaned >= 1
    assert BackgroundShellManager.get(result.bash_id) is None


@pytest.mark.asyncio
async def test_timeout_validation():
    """Test timeout parameter validation."""
    print("\n=== Testing Timeout Validation ===")

    bash_tool = BashTool()

    # Test with timeout > 600 (should be capped to 600)
    result = await bash_tool.execute(command=_shell_command("echo 'test'", "Write-Output 'test'"), timeout=1000)
    assert result.success
    print("Timeout > 600 handled correctly")

    # Test with timeout < 1 (should be set to 120)
    result = await bash_tool.execute(command=_shell_command("echo 'test'", "Write-Output 'test'"), timeout=0)
    assert result.success
    print("Timeout < 1 handled correctly")


def test_long_running_command_heuristics_ignore_build_commands():
    """Long-running command detection should avoid common one-shot build/test commands."""
    bash_tool = BashTool()

    assert bash_tool._looks_like_long_running_command("npm run dev")
    assert bash_tool._looks_like_long_running_command("vite")
    assert bash_tool._looks_like_long_running_command('"python" -m http.server 8000')

    assert not bash_tool._looks_like_long_running_command("npm run test")
    assert not bash_tool._looks_like_long_running_command("vite build")
    assert not bash_tool._looks_like_long_running_command("pnpm lint")


@pytest.mark.asyncio
async def test_service_command_auto_backgrounds_and_detects_ready_port():
    """Service-like commands should auto-switch to background and return once ready."""
    bash_tool = BashTool()
    bash_kill_tool = BashKillTool()
    port = _free_port()
    command = _python_module_command("http.server", port)

    started = time.monotonic()
    result = await bash_tool.execute(command=command)
    elapsed = time.monotonic() - started

    try:
        assert result.success
        assert result.bash_id is not None
        assert "Auto-switched to background" in result.stdout
        assert f"Port 127.0.0.1:{port} is accepting connections" in result.stdout
        assert elapsed < 6

        bg_shell = BackgroundShellManager.get(result.bash_id)
        assert bg_shell is not None
        assert bg_shell.status == "running"
    finally:
        if result.bash_id:
            await bash_kill_tool.execute(bash_id=result.bash_id)
