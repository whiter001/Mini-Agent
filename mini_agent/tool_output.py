"""Utilities for truncating large tool outputs while preserving the full text on disk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

DEFAULT_MAX_LINES = 200
DEFAULT_MAX_BYTES = 16 * 1024


@dataclass(frozen=True)
class PreparedToolOutput:
    """Prepared text for agent context/logging."""

    preview: str
    truncated: bool
    full_output_path: Path | None = None


class ToolOutputStore:
    """Persist full tool output when it is too large for chat context or logs."""

    def __init__(
        self,
        workspace_dir: str | Path,
        *,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.output_dir = self.workspace_dir / ".mini-agent" / "truncation"

    def prepare(self, text: str | None, *, label: str = "tool output") -> PreparedToolOutput:
        """Return preview text and persist the full content when it exceeds limits."""
        normalized = text or ""
        if not normalized:
            return PreparedToolOutput(preview=normalized, truncated=False)

        lines = normalized.splitlines()
        total_bytes = len(normalized.encode("utf-8"))
        if len(lines) <= self.max_lines and total_bytes <= self.max_bytes:
            return PreparedToolOutput(preview=normalized, truncated=False)

        output_path = self._write_full_output(normalized)
        preview = self._build_preview(lines)
        display_path = self._display_path(output_path)
        suffix = (
            f"\n\n... output truncated ({len(lines)} lines, {total_bytes} bytes) ..."
            f"\nFull {label} saved to: `{display_path}`"
        )
        return PreparedToolOutput(preview=f"{preview}{suffix}", truncated=True, full_output_path=output_path)

    def _write_full_output(self, text: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tool-output-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}.txt"
        output_path = self.output_dir / filename
        output_path.write_text(text, encoding="utf-8")
        return output_path

    def _build_preview(self, lines: list[str]) -> str:
        preview_lines: list[str] = []
        used_bytes = 0
        for line in lines[: self.max_lines]:
            line_bytes = len(line.encode("utf-8"))
            separator_bytes = 1 if preview_lines else 0
            if used_bytes + line_bytes + separator_bytes > self.max_bytes:
                break
            preview_lines.append(line)
            used_bytes += line_bytes + separator_bytes
        return "\n".join(preview_lines)

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_dir))
        except ValueError:
            return str(path)
