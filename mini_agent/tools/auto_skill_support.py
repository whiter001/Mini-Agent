"""Shared text and document helpers for auto-generated skills."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import yaml

from .auto_skill_cleanup import _SKILL_FRONTMATTER_PATTERN

_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "how",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
_WEB_URL_PATTERN = re.compile(r"https?://[^\s)]+|www\.[^\s)]+", re.IGNORECASE)
_WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n\s]*")
_UNIX_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_])/(?:[^/\s]+/)*[^/\s]+")
_TIMESTAMP_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b")
_ID_PATTERN = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE)
_SIMPLE_COMMA_LIST_ITEM_PATTERN = re.compile(r"^[A-Za-z0-9_./:+#<>-]+$")


@dataclass
class _AutoSkillKnowledge:
    observed_requests: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    decision_notes: list[str] = field(default_factory=list)
    workflow_outline: list[str] = field(default_factory=list)
    watchouts: list[str] = field(default_factory=list)
    legacy_family_keys: list[str] = field(default_factory=list)
    latest_outcome: str = ""
    run_count: int = 0


def _normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            items = [part.strip() for part in text.split(",")]
            if all(item and _SIMPLE_COMMA_LIST_ITEM_PATTERN.fullmatch(item) for item in items):
                return [item for item in _dedupe(items) if item]
        return [text]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for entry in value:
            items.extend(_normalize_string_list(entry))
        return [item for item in _dedupe(items) if item]
    return [str(value).strip()] if str(value).strip() else []


def _merge_string_lists(*collections: list[str], limit: int | None = None) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for collection in collections:
        for value in collection:
            text = str(value).strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            items.append(text)
            if limit is not None and len(items) >= limit:
                return items
    return items


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _parse_skill_document(content: str) -> tuple[dict[str, Any], str] | None:
    frontmatter_match = _SKILL_FRONTMATTER_PATTERN.match(content)
    if not frontmatter_match:
        return None
    try:
        frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(frontmatter, dict):
        return None
    return frontmatter, frontmatter_match.group(2)


def _sanitize_summary_text(text: str, *, max_chars: int) -> str:
    cleaned = _sanitize_multiline_text(text, max_chars=max_chars)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _summarize_final_outcome(text: str, *, max_chars: int) -> str:
    cleaned = _sanitize_multiline_text(text, max_chars=max(max_chars * 4, max_chars))
    if not cleaned:
        return ""

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned) if paragraph.strip()]
    summary = paragraphs[0] if paragraphs else cleaned.strip()
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def _sanitize_multiline_text(text: str, *, max_chars: int) -> str:
    cleaned = _stringify_message_content(text)
    cleaned, protected_urls = _protect_web_urls(cleaned)
    cleaned = _WINDOWS_PATH_PATTERN.sub("<path>", cleaned)
    cleaned = _UNIX_PATH_PATTERN.sub("<path>", cleaned)
    cleaned = _TIMESTAMP_PATTERN.sub("<timestamp>", cleaned)
    cleaned = _ID_PATTERN.sub("<id>", cleaned)
    cleaned = _restore_web_urls(cleaned, protected_urls)
    cleaned = cleaned.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned


def _strip_web_urls(text: str) -> str:
    return _WEB_URL_PATTERN.sub(" ", text)


def _protect_web_urls(text: str) -> tuple[str, dict[str, str]]:
    protected_urls: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        placeholder = f"__mini_agent_web_url_{len(protected_urls)}__"
        protected_urls[placeholder] = match.group(0)
        return placeholder

    return _WEB_URL_PATTERN.sub(replace, text), protected_urls


def _restore_web_urls(text: str, protected_urls: dict[str, str]) -> str:
    restored = text
    for placeholder, url in protected_urls.items():
        restored = restored.replace(placeholder, url)
    return restored


def _stringify_message_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    ]


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        items.append(value)
    return items