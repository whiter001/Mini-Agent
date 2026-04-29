"""Runtime helpers for shell command execution and background process management."""

from __future__ import annotations

import asyncio
import re
from typing import Any


async def close_process_transport(process: "asyncio.subprocess.Process") -> None:
    """Explicitly close the underlying subprocess transport on Windows-friendly paths."""
    transport = getattr(process, "_transport", None)
    if transport is None:
        return

    try:
        transport.close()
    except Exception:
        pass


class BackgroundShell:
    """Background shell data container."""

    def __init__(self, bash_id: str, command: str, process: "asyncio.subprocess.Process", start_time: float):
        self.bash_id = bash_id
        self.command = command
        self.process = process
        self.start_time = start_time
        self.output_lines: list[str] = []
        self.last_read_index = 0
        self.status = "running"
        self.exit_code: int | None = None

    def add_output(self, line: str) -> None:
        """Add new output line."""
        self.output_lines.append(line)

    def get_new_output(self, filter_pattern: str | None = None) -> list[str]:
        """Get new output since last check, optionally filtered by regex."""
        new_lines = self.output_lines[self.last_read_index :]
        self.last_read_index = len(self.output_lines)

        if filter_pattern:
            try:
                pattern = re.compile(filter_pattern)
                new_lines = [line for line in new_lines if pattern.search(line)]
            except re.error:
                pass

        return new_lines

    def update_status(self, is_alive: bool, exit_code: int | None = None) -> None:
        """Update process status."""
        if not is_alive:
            self.status = "completed" if exit_code == 0 else "failed"
            self.exit_code = exit_code
        else:
            self.status = "running"

    async def terminate(self) -> None:
        """Terminate the background process."""
        was_running = self.process.returncode is None

        try:
            if was_running:
                self.process.terminate()

            try:
                stdout, _ = await asyncio.wait_for(self.process.communicate(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                stdout, _ = await asyncio.wait_for(self.process.communicate(), timeout=5)

            if stdout:
                for line in stdout.decode("utf-8", errors="replace").splitlines():
                    self.add_output(line)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                await self.process.wait()
            except Exception:
                pass
        finally:
            await close_process_transport(self.process)

        if was_running:
            self.status = "terminated"
            self.exit_code = self.process.returncode
        else:
            self.update_status(is_alive=False, exit_code=self.process.returncode)


class BackgroundShellManager:
    """Manager for all background shell processes."""

    _shells: dict[str, BackgroundShell] = {}
    _monitor_tasks: dict[str, asyncio.Task[Any]] = {}

    @classmethod
    def add(cls, shell: BackgroundShell) -> None:
        """Add a background shell to management."""
        cls._shells[shell.bash_id] = shell

    @classmethod
    def get(cls, bash_id: str) -> BackgroundShell | None:
        """Get a background shell by ID."""
        return cls._shells.get(bash_id)

    @classmethod
    def get_available_ids(cls) -> list[str]:
        """Get all available bash IDs."""
        return list(cls._shells.keys())

    @classmethod
    def _remove(cls, bash_id: str) -> None:
        """Remove a background shell from management."""
        cls._shells.pop(bash_id, None)

    @classmethod
    async def start_monitor(cls, bash_id: str) -> None:
        """Start monitoring a background shell's output."""
        shell = cls.get(bash_id)
        if not shell:
            return

        async def monitor() -> None:
            try:
                process = shell.process
                while process.returncode is None:
                    try:
                        if process.stdout:
                            line = await asyncio.wait_for(process.stdout.readline(), timeout=0.1)
                            if line:
                                shell.add_output(line.decode("utf-8", errors="replace").rstrip("\n"))
                            else:
                                break
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.1)
                        continue
                    except Exception:
                        await asyncio.sleep(0.1)
                        continue

                try:
                    returncode = await process.wait()
                except Exception:
                    returncode = -1

                shell.update_status(is_alive=False, exit_code=returncode)
            except Exception as exc:
                if bash_id in cls._shells:
                    cls._shells[bash_id].status = "error"
                    cls._shells[bash_id].add_output(f"Monitor error: {str(exc)}")
            finally:
                cls._monitor_tasks.pop(bash_id, None)

        cls._monitor_tasks[bash_id] = asyncio.create_task(monitor())

    @classmethod
    async def _cancel_monitor(cls, bash_id: str) -> None:
        """Cancel and remove a monitoring task."""
        task = cls._monitor_tasks.pop(bash_id, None)
        if task is None or task.done():
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    @classmethod
    async def terminate(cls, bash_id: str) -> BackgroundShell:
        """Terminate a background shell and clean up all resources."""
        shell = cls.get(bash_id)
        if not shell:
            raise ValueError(f"Shell not found: {bash_id}")

        await cls._cancel_monitor(bash_id)
        await shell.terminate()
        cls._remove(bash_id)
        return shell

    @classmethod
    async def cleanup_all(cls) -> int:
        """Terminate and remove every tracked background shell."""
        bash_ids = list(cls._shells.keys())

        for bash_id in bash_ids:
            shell = cls._shells.get(bash_id)
            if shell is None:
                continue

            await cls._cancel_monitor(bash_id)
            try:
                await shell.terminate()
            except Exception:
                pass
            finally:
                cls._remove(bash_id)

        return len(bash_ids)


__all__ = [
    "BackgroundShell",
    "BackgroundShellManager",
    "close_process_transport",
]
