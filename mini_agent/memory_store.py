"""Persistent memory store for Mini-Agent."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .schema import Message

DEFAULT_MEMORY_ROOT = Path.home() / ".mini-agent"
_STOPWORDS = {
    "a",
    "an",
    "and",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
}


@dataclass(frozen=True)
class MemoryEntry:
    """Stored memory entry."""

    id: int
    kind: str
    title: str
    content: str
    tags: str
    created_at: str
    updated_at: str

    def format_line(self) -> str:
        tag_suffix = f" [{self.tags}]" if self.tags else ""
        return f"- {self.title}{tag_suffix}: {self.content}"


class MemoryStore:
    """Manage durable memory files and SQLite-backed recall."""

    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir).expanduser() if root_dir else DEFAULT_MEMORY_ROOT
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.root_dir / "MEMORY.md"
        self.user_path = self.root_dir / "USER.md"
        self.db_path = self.root_dir / "memory.sqlite3"
        self._memory_snapshot_limit = 16
        self._user_snapshot_limit = 8
        self._search_limit = 5
        self._fts_available = False
        self._initialize()

    def store_memory(self, content: str, title: str | None = None, tags: Sequence[str] | None = None) -> MemoryEntry:
        """Store a durable agent memory note."""
        return self._insert_entry("memory", title or self._guess_title(content), content, tags)

    def store_user_profile(
        self,
        content: str,
        title: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> MemoryEntry:
        """Store a user profile fact."""
        return self._insert_entry("user", title or self._guess_title(content), content, tags)

    def search(self, query: str, limit: int | None = None, kinds: Sequence[str] | None = None) -> list[MemoryEntry]:
        """Search persisted memory using SQLite FTS5 when available."""
        query = query.strip()
        if not query:
            return self._latest_entries(limit=limit or self._search_limit, kinds=kinds)

        limit = limit or self._search_limit
        kinds = list(kinds or [])
        with closing(self._connect()) as conn:
            if self._fts_available:
                fts_query = self._build_fts_query(query)
                if fts_query:
                    clauses = ["memory_fts MATCH ?"]
                    params: list[object] = [fts_query]
                    if kinds:
                        clauses.append(f"e.kind IN ({','.join('?' for _ in kinds)})")
                        params.extend(kinds)

                    rows = conn.execute(
                        f"""
                        SELECT e.id, e.kind, e.title, e.content, e.tags, e.created_at, e.updated_at
                        FROM memory_fts
                        JOIN memory_entries e ON memory_fts.rowid = e.id
                        WHERE {' AND '.join(clauses)}
                        ORDER BY bm25(memory_fts), e.updated_at DESC
                        LIMIT ?
                        """,
                        [*params, limit],
                    ).fetchall()
                    return [self._row_to_entry(row) for row in rows]

                # FTS query construction currently targets Latin tokens. For queries like Chinese phrases,
                # fall back to substring matching instead of returning an empty result set.
                return self._search_like(conn, query=query, limit=limit, kinds=kinds)

            return self._search_like(conn, query=query, limit=limit, kinds=kinds)

    def build_system_prompt(self) -> str:
        """Build compact persistent context for the system prompt."""
        parts: list[str] = ["## Persistent Memory", "The following durable notes and user facts are stored under `~/.mini-agent/`."]

        user_snapshot = self._read_snapshot(self.user_path, 1200)
        if user_snapshot:
            parts.append("### USER.md")
            parts.append(user_snapshot)

        memory_snapshot = self._read_snapshot(self.memory_path, 1800)
        if memory_snapshot:
            parts.append("### MEMORY.md")
            parts.append(memory_snapshot)

        if len(parts) == 2:
            return ""
        return "\n\n".join(parts)

    def build_turn_context(self, query: str, limit: int | None = None) -> list[Message]:
        """Build a temporary system message with query-relevant memory hits."""
        limit = limit or self._search_limit
        hits = self.search(query, limit=limit)
        if not hits:
            hits = self._latest_entries(limit=limit, kinds=["memory", "user"])
        if not hits:
            return []

        lines = [
            "## Relevant Persistent Memory",
            "Use these stored facts and preferences when answering the current request.",
        ]
        for index, hit in enumerate(hits, 1):
            tag_suffix = f" [{hit.tags}]" if hit.tags else ""
            lines.append(f"{index}. [{hit.kind}] {hit.title}{tag_suffix}: {hit.content}")

        return [Message(role="system", content="\n".join(lines))]

    def _initialize(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_entries_kind_updated ON memory_entries(kind, updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_entries_updated ON memory_entries(updated_at)")

            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(title, content, tags, kind, created_at, updated_at)
                    """
                )
                self._fts_available = True
            except sqlite3.OperationalError:
                self._fts_available = False

            if self._fts_available:
                self._rebuild_fts(conn)
            conn.commit()
            self._refresh_snapshots(conn)

    def _insert_entry(self, kind: str, title: str, content: str, tags: Sequence[str] | None) -> MemoryEntry:
        normalized_title = title.strip() or self._guess_title(content)
        normalized_content = content.strip()
        normalized_tags = self._normalize_tags(tags)
        now = self._now()

        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO memory_entries (kind, title, content, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kind, normalized_title, normalized_content, normalized_tags, now, now),
            )
            entry_id = int(cursor.lastrowid)
            entry = MemoryEntry(
                id=entry_id,
                kind=kind,
                title=normalized_title,
                content=normalized_content,
                tags=normalized_tags,
                created_at=now,
                updated_at=now,
            )
            if self._fts_available:
                conn.execute(
                    """
                    INSERT INTO memory_fts(rowid, title, content, tags, kind, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (entry.id, entry.title, entry.content, entry.tags, entry.kind, entry.created_at, entry.updated_at),
                )
            conn.commit()
            self._refresh_snapshots(conn)
            return entry

    def _latest_entries(self, limit: int, kinds: Sequence[str] | None = None) -> list[MemoryEntry]:
        with closing(self._connect()) as conn:
            clauses = []
            params: list[object] = []
            if kinds:
                clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
                params.extend(kinds)
            where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = conn.execute(
                f"""
                SELECT id, kind, title, content, tags, created_at, updated_at
                FROM memory_entries
                {where_clause}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
            return [self._row_to_entry(row) for row in rows]

    def _search_like(
        self,
        conn: sqlite3.Connection,
        *,
        query: str,
        limit: int,
        kinds: Sequence[str] | None = None,
    ) -> list[MemoryEntry]:
        like = f"%{query}%"
        clauses = ["(title LIKE ? OR content LIKE ? OR tags LIKE ?)"]
        params: list[object] = [like, like, like]
        if kinds:
            clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
            params.extend(kinds)
        rows = conn.execute(
            f"""
            SELECT id, kind, title, content, tags, created_at, updated_at
            FROM memory_entries
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def _refresh_snapshots(self, conn: sqlite3.Connection) -> None:
        memory_snapshot = self._render_snapshot(
            "MEMORY.md",
            self._fetch_recent_rows(conn, "memory", self._memory_snapshot_limit),
        )
        self.memory_path.write_text(memory_snapshot, encoding="utf-8")

        user_snapshot = self._render_snapshot(
            "USER.md",
            self._fetch_recent_rows(conn, "user", self._user_snapshot_limit),
        )
        self.user_path.write_text(user_snapshot, encoding="utf-8")

    def _fetch_recent_rows(
        self,
        conn: sqlite3.Connection,
        kind: str,
        limit: int,
    ) -> list[MemoryEntry]:
        rows = conn.execute(
            """
            SELECT id, kind, title, content, tags, created_at, updated_at
            FROM memory_entries
            WHERE kind = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (kind, limit),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def _render_snapshot(self, filename: str, entries: list[MemoryEntry]) -> str:
        timestamp = self._now()
        if not entries:
            title = "Agent memory" if filename == "MEMORY.md" else "User profile"
            return f"# {filename}\n\nUpdated: {timestamp}\n\n## {title}\n- No entries yet.\n"

        header = "# MEMORY.md" if filename == "MEMORY.md" else "# USER.md"
        section = "Recent agent notes" if filename == "MEMORY.md" else "User facts"
        lines = [header, "", f"Updated: {timestamp}", "", f"## {section}"]
        for entry in entries:
            tag_suffix = f" [{entry.tags}]" if entry.tags else ""
            lines.append(f"- [{entry.updated_at}] {entry.title}{tag_suffix}: {entry.content}")
        lines.append("")
        return "\n".join(lines)

    def _rebuild_fts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM memory_fts")
        rows = conn.execute(
            """
            SELECT id, kind, title, content, tags, created_at, updated_at
            FROM memory_entries
            ORDER BY updated_at ASC, id ASC
            """
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                INSERT INTO memory_fts(rowid, title, content, tags, kind, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["title"],
                    row["content"],
                    row["tags"],
                    row["kind"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _read_snapshot(self, path: Path, max_chars: int) -> str:
        if not path.exists():
            return ""
        content = path.read_text(encoding="utf-8").strip()
        return self._truncate(content, max_chars)

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=int(row["id"]),
            kind=str(row["kind"]),
            title=str(row["title"]),
            content=str(row["content"]),
            tags=str(row["tags"] or ""),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _normalize_tags(self, tags: Sequence[str] | None) -> str:
        if not tags:
            return ""
        items: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            normalized = str(tag).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            items.append(normalized)
        return ", ".join(items)

    def _guess_title(self, content: str) -> str:
        text = content.strip().splitlines()[0].strip() if content.strip() else "memory"
        return self._truncate(text, 64)

    def _build_fts_query(self, query: str) -> str:
        tokens = [token.lower() for token in _tokenize(query)]
        if not tokens:
            return ""
        parts = [f"{token}*" if len(token) > 2 else token for token in tokens]
        return " AND ".join(parts)

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> Iterable[str]:
    for token in re.findall(r"[A-Za-z0-9]+", text):
        normalized = token.lower()
        if len(normalized) > 1 and normalized not in _STOPWORDS:
            yield normalized
