"""Naming and guidance generation for auto-generated skills."""

from __future__ import annotations

import re
from typing import Sequence
from urllib.parse import urlparse

from .auto_skill_quality import (
    AutoSkillQualityReport,
    _ANSWER_REQUEST_PATTERNS,
    _extract_requested_count,
    _matches_any_pattern,
)
from .auto_skill_support import (
    _STOPWORDS,
    _dedupe,
    _merge_string_lists,
    _slugify,
    _tokenize,
)

_URL_TOKEN_STOPWORDS = {
    "http",
    "https",
    "www",
    "com",
    "cn",
    "net",
    "org",
    "io",
    "co",
    "question",
    "questions",
    "page",
    "pages",
    "html",
    "htm",
}
_URL_PATTERN = re.compile(r"https?://[^\s)]+|www\.[^\s)]+", re.IGNORECASE)
_NAME_TOKEN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9._/-]*")
_NAME_TOKEN_SPLIT_PATTERN = re.compile(r"[._/-]+")
_GUIDANCE_BUCKET_RULES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "decision": (
        ("inspect", ("inspect ",)),
        ("tool-path", ("prefer the proven tool path", "prefer write_file plus autobrowser eval --file", "prefer write_file")),
        ("extract", ("for x.com feed requests, start with the cheapest visible `article` extraction",)),
        ("count", ("for count-based requests",)),
        ("recovery", ("when a step fails",)),
        ("validate", ("keep validation", "add an explicit verification", "finish with a post-action verification", "validate ")),
    ),
    "workflow": (
        ("inspect", ("inspect ", "reuse an existing x.com/home tab")),
        ("extract", ("execute the main task through", "start with a simple visible-post extraction", "use a single extraction path")),
        ("recovery", ("if a step fails", "if the initial visible extract is short", "if the first pass is short")),
        ("validate", ("validate ", "finish with a post-action verification")),
    ),
}
_TASK_LABEL_RULES = (
    ("answer", ("answer", "answering", "reply", "respond", "回答", "答题", "作答", "答复")),
    ("review", ("review", "pull request", "code review", "审查", "评审", "复核")),
    ("fix", ("fix", "repair", "debug", "troubleshoot", "修复", "排查", "故障")),
    ("update", ("update", "edit", "modify", "更新", "修改", "编辑")),
    ("create", ("create", "generate", "build", "生成", "创建", "新建")),
    ("search", ("search", "find", "lookup", "搜索", "查找")),
    ("test", ("test", "verify", "validate", "pytest", "测试", "验证", "校验")),
)
_SITE_LABEL_RULES = (
    ("zhihu", ("zhihu", "zhihu.com", "知乎")),
    ("baidu-zhidao", ("zhidao.baidu.com", "百度知道")),
    ("github", ("github", "github.com")),
    ("x", ("x.com", "twitter.com")),
    ("slack", ("slack",)),
    ("jira", ("jira",)),
    ("notion", ("notion",)),
)
_WARNING_GUIDANCE = {
    "environment-specific-data-detected": "Avoid hard-coding local paths, timestamps, run IDs, or other environment-specific data.",
    "multiple-failed-steps": "When several steps fail, isolate the root cause before treating the whole trace as reusable guidance.",
    "partial-completion-detected": "Do not treat a partially completed run as reusable until the requested scope is actually covered.",
    "requirement-mismatch": "Compare the requested quantity or action with the observed result before reusing this workflow.",
    "incomplete-tool-results": "Every critical tool call should have an observable result before the workflow is trusted.",
    "requested-action-not-observed": "Browsing or reading is not enough when the user asked for a concrete action such as answering or publishing.",
    "final-result-not-successful": "A polite summary can still describe failure; watch for blockers such as login, permissions, or anti-automation limits.",
    "final-outcome-too-generic": "Record a concrete completion signal so the next run can verify the same outcome.",
    "too-few-steps": "One-shot runs are usually too shallow to become durable guidance.",
}


def _contains_han(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text)


def _expand_name_token(token: str) -> list[str]:
    trimmed = token.strip().lower()
    if not trimmed:
        return []

    items = [trimmed]
    if _NAME_TOKEN_SPLIT_PATTERN.search(trimmed):
        items.extend(part for part in _NAME_TOKEN_SPLIT_PATTERN.split(trimmed) if part)

    if _contains_han(trimmed):
        return items

    return [item for item in _dedupe(items) if item and item not in _URL_TOKEN_STOPWORDS and len(item) > 2]


def _extract_distinctive_url_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_url in _URL_PATTERN.findall(text):
        candidate = raw_url if not raw_url.lower().startswith("www.") else f"https://{raw_url}"
        parsed = urlparse(candidate)
        host_parts = [part for part in parsed.netloc.lower().split(".") if part]
        path_parts = [part for part in _NAME_TOKEN_SPLIT_PATTERN.split(parsed.path.lower()) if part]
        for part in [*host_parts, *path_parts]:
            tokens.extend(_expand_name_token(part))
    return [item for item in _dedupe(tokens) if item and item not in _URL_TOKEN_STOPWORDS]


def _extract_ascii_request_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    request_without_urls = _URL_PATTERN.sub(" ", text)
    for match in _NAME_TOKEN_PATTERN.findall(request_without_urls.lower()):
        if _contains_han(match):
            continue
        tokens.extend(_expand_name_token(match))
    return [item for item in _dedupe(tokens) if item not in _STOPWORDS and item not in _URL_TOKEN_STOPWORDS]


def _infer_request_labels(user_request: str) -> list[str]:
    lowered_request = user_request.lower()
    labels: list[str] = []
    for label, markers in _TASK_LABEL_RULES:
        if any(marker in lowered_request for marker in markers):
            labels.append(label)
    if _matches_any_pattern(lowered_request, _ANSWER_REQUEST_PATTERNS) and "answer" not in labels:
        labels.insert(0, "answer")
    return _dedupe(labels)


def _infer_site_labels(user_request: str) -> list[str]:
    lowered_request = user_request.lower()
    labels: list[str] = []
    url_tokens = _extract_distinctive_url_tokens(user_request)
    for label, markers in _SITE_LABEL_RULES:
        if any(marker in lowered_request for marker in markers) or any(marker in url_tokens for marker in markers):
            labels.append(label)
    for token in url_tokens:
        if token in _URL_TOKEN_STOPWORDS or token in labels:
            continue
        labels.append(token)
    return _dedupe(labels)


def _build_legacy_skill_name(user_request: str, trace: Sequence[dict[str, Any]]) -> str:
    tokens = _tokenize(user_request)
    tool_names = _dedupe([str(step.get("name", "")) for step in trace if step.get("name")])
    parts = ["auto"]
    parts.extend(tool_names[:2])
    parts.extend(tokens[:4])
    slug = _slugify("-".join(parts))
    return slug[:80] or "auto-workflow"


def _build_skill_name(user_request: str, trace: Sequence[dict[str, Any]]) -> str:
    semantic_parts = ["auto"]
    semantic_parts.extend(_infer_request_labels(user_request)[:2])
    semantic_parts.extend(_infer_site_labels(user_request)[:2])

    for token in _extract_ascii_request_tokens(user_request):
        if token in semantic_parts:
            continue
        semantic_parts.append(token)
        if len(semantic_parts) >= 6:
            break

    slug = _slugify("-".join(semantic_parts))
    if slug and slug != "auto":
        return slug[:80]
    return _build_legacy_skill_name(user_request, trace)


def _is_autobrowser_x_feed_request(user_request: str) -> bool:
    lowered_request = user_request.lower()
    if "autobrowser" not in lowered_request:
        return False

    site_labels = set(_infer_site_labels(user_request))
    if not ({"x", "twitter"} & site_labels or "x.com" in lowered_request or "twitter.com" in lowered_request):
        return False

    return any(marker in user_request for marker in ("消息", "推文", "帖子", "动态")) or _extract_requested_count(user_request) is not None


def _guidance_request_keywords(user_request: str) -> list[str]:
    keywords: list[str] = []
    keywords.extend(_infer_request_labels(user_request))
    keywords.extend(_infer_site_labels(user_request))
    keywords.extend(_extract_ascii_request_tokens(user_request))

    lowered_request = user_request.lower()
    if "x.com" in lowered_request:
        keywords.append("x.com")
    if "twitter.com" in lowered_request:
        keywords.append("twitter.com")
    if "autobrowser" in lowered_request:
        keywords.append("autobrowser")

    return _dedupe([keyword for keyword in keywords if keyword])


def _guidance_specificity_score(item: str, request_keywords: Sequence[str]) -> int:
    lowered_item = item.lower()
    overlap_score = sum(1 for keyword in request_keywords if keyword and keyword in lowered_item)
    structured_score = sum(
        1
        for marker in (
            "x.com",
            "x.com/home",
            "autobrowser",
            "/status/",
            "article",
            "eval --file",
            "write_file",
            "structured json",
            "requested number",
            "unique posts",
            "查看新帖子",
            "scrolling/dedupe",
        )
        if marker in lowered_item
    )
    token_score = len(re.findall(r"[A-Za-z0-9]+", lowered_item))
    phrase_bonus = 0
    if "start with a simple visible-post extraction" in lowered_item or "cheapest visible `article` extraction" in lowered_item:
        phrase_bonus += 20
    if "validate that at least the requested number of unique posts" in lowered_item:
        phrase_bonus += 20
    return overlap_score * 8 + structured_score * 4 + min(token_score, 24) + phrase_bonus


def _guidance_bucket(item: str, kind: str) -> str | None:
    lowered_item = item.lower()
    for bucket_name, markers in _GUIDANCE_BUCKET_RULES.get(kind, ()):
        if any(lowered_item.startswith(marker) for marker in markers):
            return bucket_name
    return None


def _merge_guidance_lists(
    primary_items: Sequence[str],
    existing_items: Sequence[str],
    user_request: str,
    *,
    kind: str,
    limit: int | None = None,
) -> list[str]:
    merged = _merge_string_lists(primary_items, existing_items)
    bucket_rules = _GUIDANCE_BUCKET_RULES.get(kind)
    if not merged or not bucket_rules:
        return merged[:limit] if limit is not None else merged

    request_keywords = _guidance_request_keywords(user_request)
    bucket_winners: dict[str, tuple[int, int, str]] = {}
    extras: list[tuple[int, str]] = []

    for index, item in enumerate(merged):
        bucket_name = _guidance_bucket(item, kind)
        if bucket_name is None:
            extras.append((index, item))
            continue

        score = _guidance_specificity_score(item, request_keywords)
        current = bucket_winners.get(bucket_name)
        if current is None or score > current[0] or (score == current[0] and index < current[1]):
            bucket_winners[bucket_name] = (score, index, item)

    items: list[str] = []
    seen: set[str] = set()
    for bucket_name, _markers in bucket_rules:
        winner = bucket_winners.get(bucket_name)
        if winner is None:
            continue
        item = winner[2]
        if item in seen:
            continue
        seen.add(item)
        items.append(item)
        if limit is not None and len(items) >= limit:
            return items

    for _index, item in extras:
        if item in seen:
            continue
        seen.add(item)
        items.append(item)
        if limit is not None and len(items) >= limit:
            return items

    return items


def _build_decision_notes(
    user_request: str,
    trace: Sequence[dict[str, Any]],
    quality: AutoSkillQualityReport,
) -> list[str]:
    metrics = quality.metrics
    categories = set(metrics.get("categories", []))
    tool_names = _dedupe([str(step.get("name", "")) for step in trace if step.get("name")])

    notes: list[str] = []
    if "discovery" in categories and "action" in categories:
        notes.append("Inspect the current state first, then act; avoid editing or submitting before you confirm the real target.")
    if tool_names:
        notes.append(f"Prefer the proven tool path ({', '.join(tool_names[:4])}) instead of switching approaches mid-task without evidence.")
    if _is_autobrowser_x_feed_request(user_request):
        notes.append("For x.com feed requests, start with the cheapest visible `article` extraction and only switch to file-backed scrolling/dedupe when the first pass is short.")
        notes.append("Prefer write_file plus autobrowser eval --file for any fallback script that needs scrolling or dedupe state.")
    if metrics.get("recovered"):
        notes.append("When a step fails, read the tool output and repair the current path before inventing a brand-new workflow.")
    if metrics.get("requested_count"):
        notes.append("For count-based requests, track the requested quantity explicitly and report the actual completed count.")
    if metrics.get("task_kind") == "answering":
        notes.append("For answering tasks, completion requires entering the editor, drafting content, and observing a real submission signal.")
    if "validation" in categories:
        notes.append("Keep validation inside the main workflow so success is confirmed by evidence rather than assumption.")
    else:
        notes.append("Add an explicit verification step after the action so the outcome is evidence-based.")
    if not metrics.get("environment_risk"):
        notes.append("Persist reusable patterns only; strip paths, timestamps, IDs, and other environment-specific data.")
    return _merge_string_lists(notes, limit=6)


def _build_workflow_outline(
    user_request: str,
    quality: AutoSkillQualityReport,
) -> list[str]:
    metrics = quality.metrics
    categories = set(metrics.get("categories", []))
    outline: list[str] = []

    if "discovery" in categories:
        outline.append("Inspect the current state, constraints, and true target before committing to any action.")
    if _is_autobrowser_x_feed_request(user_request):
        outline.append("Reuse an existing x.com/home tab or open the feed, then confirm the target timeline is active before extracting.")
        outline.append("Start with a simple visible-post extraction from `article` nodes to see whether the target count is already available.")
        outline.append("If the initial visible extract is short, switch to one file-backed scrolling/dedupe script that accumulates unique `/status/` links across passes instead of spawning new ad-hoc scripts each retry.")
    if metrics.get("task_kind") == "answering":
        outline.append("Open the real answering entry point, draft the content in the verified editor, and keep the interaction focused on that flow.")
        outline.append("Confirm a durable submission signal instead of assuming that opening the editor or clicking a button already finished the task.")
    elif "action" in categories:
        outline.append("Execute the main task through the verified tool path and keep the successful sequence stable.")
    if metrics.get("recovered"):
        outline.append("If a step fails, use the failure signal to repair only the affected part of the flow and retry with evidence.")
    if "validation" in categories:
        outline.append("Validate the final state with a separate check that proves the user goal was actually achieved.")
    else:
        outline.append("Finish with a post-action verification step that checks the resulting state instead of trusting the first success message.")
    if _is_autobrowser_x_feed_request(user_request):
        outline.append("Validate that at least the requested number of unique posts were returned before summarizing the result.")

    if not outline:
        outline = [
            "Inspect the current state before acting.",
            "Execute the main task through the most stable tool path.",
            "Validate the final state with an explicit post-check.",
        ]
    return _merge_string_lists(outline, limit=6)


def _build_watchouts(
    user_request: str,
    quality: AutoSkillQualityReport,
) -> list[str]:
    metrics = quality.metrics
    categories = set(metrics.get("categories", []))
    watchouts = [_WARNING_GUIDANCE[warning] for warning in quality.warnings if warning in _WARNING_GUIDANCE]

    if metrics.get("task_kind") == "answering":
        watchouts.append(
            "Opening a question page or reading existing answers does not count as answering unless authoring and submission signals are observed."
        )
    if _is_autobrowser_x_feed_request(user_request):
        watchouts.append("Do not jump straight to multi-pass scrolling when the visible `article` list already satisfies the requested count.")
        watchouts.append("If fallback scrolling is needed, keep one file-backed script and one dedupe state keyed by `/status/` links; multiple script variants increase retries and tool calls.")
    if "validation" not in categories:
        watchouts.append("Without a post-action verification step, the workflow can drift even when tool calls look successful.")
    if not watchouts:
        watchouts.append("Review the workflow whenever the environment, page structure, or tool behavior changes.")

    return _merge_string_lists(watchouts, limit=6)