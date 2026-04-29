"""Trace collection and quality evaluation for auto-generated skills."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from mini_agent.schema import Message

from .auto_skill_support import (
    _ID_PATTERN,
    _TIMESTAMP_PATTERN,
    _UNIX_PATH_PATTERN,
    _WINDOWS_PATH_PATTERN,
    _dedupe,
    _sanitize_summary_text,
    _stringify_message_content,
    _summarize_final_outcome,
)

_DISCOVERY_TOOL_NAMES = {"read_file", "get_skill", "list_skills", "search_memory", "recall_note"}
_ACTION_TOOL_NAMES = {"bash", "bash_kill", "edit_file", "write_file", "remember", "remember_user"}
_VALIDATION_TOOL_NAMES = {"bash_output"}
_DISCOVERY_MARKERS = ("list ", "find ", "grep", "search", "read", "show", "cat ", "type ", "status")
_ACTION_MARKERS = ("apply", "edit", "write", "create", "generate", "install", "run ", "python ", "node ", "mkdir", "copy ", "move ")
_VALIDATION_MARKERS = ("pytest", "unittest", "test", "lint", "mypy", "ruff", "validate", "check", "verify", "diff", "status")
_NO_ERROR_MARKERS = ("no error", "no errors", "without error", "without errors", "0 failed")
_ERROR_MARKERS = (
    "[error]",
    "traceback",
    "exception",
    "failed",
    "cancelled",
    "timed out",
    "syntaxerror",
    "element not found",
    "not a valid selector",
    "requires login",
    "need to log in",
    "please log in",
    "无法完成",
    "未能完成",
    "无法处理",
    "未能处理",
    "无法回答",
    "未能回答",
    "无法继续",
    "未能继续",
    "请先登录",
    "需要登录",
    "登录验证",
    "publish failed",
    "submit failed",
    "failed to publish",
    "unable to submit",
    "not published",
    "submission blocked",
    "publish blocked",
    "anti-automation",
    "blocked by anti-automation",
    "发布失败",
    "提交失败",
    "无法发布",
    "未能发布",
    "无法提交",
    "未能提交",
    "未提交成功",
    "提交未成功",
    "反自动化",
    "似乎被阻止",
)
_SUCCESS_BLOCKERS = (
    "couldn't be completed",
    "unable to complete",
    "task failed",
    "i'm sorry",
    "sorry, but",
    "requires login",
    "need to log in",
    "please log in",
    "很抱歉",
    "无法完成",
    "未能完成",
    "无法处理",
    "未能处理",
    "无法回答",
    "未能回答",
    "无法继续",
    "未能继续",
    "请先登录",
    "需要登录",
    "登录验证",
    "publish failed",
    "submit failed",
    "failed to publish",
    "unable to submit",
    "not published",
    "submission blocked",
    "publish blocked",
    "anti-automation",
    "blocked by anti-automation",
    "发布失败",
    "提交失败",
    "无法发布",
    "未能发布",
    "无法提交",
    "未能提交",
    "未提交成功",
    "提交未成功",
    "反自动化",
    "被阻止",
    "似乎被阻止",
)
_PARTIAL_COMPLETION_MARKERS = (
    "只显示了",
    "仅显示了",
    "只获取了",
    "仅获取了",
    "只返回了",
    "仅返回了",
    "只加载了",
    "仅加载了",
    "只能获取",
    "只能返回",
    "当前页面只显示",
    "目前只显示",
    "无法获取更多",
    "未能获取更多",
    "数量有限",
    "only showed",
    "only displayed",
    "only found",
    "only returned",
    "only loaded",
    "could only",
    "currently shows",
    "currently only",
    "unable to fetch more",
    "could not fetch more",
    "limited to",
)
_REQUEST_COUNT_PATTERNS = (
    re.compile(r"\b(?:top|first|latest)\s+(\d+)\s*(?:items?|results?|messages?|tweets?|posts?|records?|entries?|questions?|answers?|replies?)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*(?:items?|results?|messages?|tweets?|posts?|records?|entries?|questions?|answers?|replies?)\b", re.IGNORECASE),
    re.compile(r"(\d+)\s*(?:条|个|篇|项|道|题)\s*(?:消息|推文|帖子|结果|记录|内容|问题|题目|题|回答|回复)?"),
)
_LIMITED_RESULT_COUNT_PATTERNS = (
    re.compile(r"(?:只|仅|目前只|当前只|当前页面只|只能|仅能)\s*(?:显示|获取|返回|找到|抓取|加载|看到|提供)?\s*[^\d]{0,12}(\d+)\s*(?:条|个|篇|项)?"),
    re.compile(r"(?:only|just|currently|could only|limited to)\s*(?:show|display|find|fetch|get|return|load)?(?:ed|s)?\s*[^\d]{0,12}(\d+)\s*(?:items?|results?|messages?|tweets?|posts?|records?|entries?)?", re.IGNORECASE),
)
_ANSWER_REQUEST_PATTERNS = (
    re.compile(r"答\s*\d*\s*(?:道|个)?\s*(?:题|问题|题目)"),
    re.compile(r"(?:回答|作答|答复|回复)\s*\d*\s*(?:道|个)?\s*(?:题|问题|题目)"),
    re.compile(r"\b(?:answer|reply to|respond to)\s+\d+\s+(?:questions?|replies?)\b", re.IGNORECASE),
    re.compile(r"答题"),
)
_ANSWER_ENTRY_MARKERS = (
    "我来答",
    "answer box",
    "answer editor",
    "reply box",
    "回答框",
    "回答区域",
)
_ANSWER_AUTHORING_PATTERNS = (
    re.compile(r"\b(?:fill|type)\b", re.IGNORECASE),
    re.compile(r"(?:textarea|contenteditable|editor|reply-box|answer-box)", re.IGNORECASE),
    re.compile(r"(?:输入回答|填写回答|回答内容|撰写回答|编辑回答)"),
)
_ANSWER_SUBMISSION_PATTERNS = (
    re.compile(r"(?:提交回答|提交答案|发布回答|发布答案|发表回答|发送回复|提交回复|发布回复)"),
    re.compile(r"\b(?:submit|post|publish|send)\b", re.IGNORECASE),
    re.compile(r"(?:\.submit\s*\(|submit\s*\()", re.IGNORECASE),
)
_ANSWER_SUCCESS_MARKERS = (
    "回答成功",
    "提交成功",
    "发布成功",
    "发送成功",
    "已回答",
    "已提交",
    "已发布",
    "submitted answer",
    "posted answer",
    "reply sent",
    "answered successfully",
)
_ANSWER_BROWSE_ONLY_MARKERS = (
    "访问了",
    "查看了",
    "浏览了",
    "打开了",
    "获取了",
    "visited",
    "viewed",
    "browsed",
    "opened",
)
_EXISTING_ANSWER_SUMMARY_MARKERS = (
    "已有回答",
    "已有人回答",
    "already answered",
    "existing answer",
)
_MIN_FINAL_OUTCOME_CHARS = 24


@dataclass
class AutoSkillQualityReport:
    should_create: bool
    score: int = 0
    tier: str = ""
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _collect_turn_trace(turn_messages: Sequence[Message]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    pending_indices: list[int] = []
    pending_by_call_id: dict[str, list[int]] = {}
    for message in turn_messages:
        role = getattr(message, "role", "")
        if role == "assistant":
            tool_calls = getattr(message, "tool_calls", None) or []
            for call in tool_calls:
                arguments = _tool_call_args(call)
                trace.append(
                    {
                        "name": _tool_call_name(call),
                        "arguments": _format_arguments(arguments),
                        "arguments_dict": arguments,
                        "result": "",
                    }
                )
                index = len(trace) - 1
                pending_indices.append(index)
                call_id = str(getattr(call, "id", "") or "")
                if call_id:
                    pending_by_call_id.setdefault(call_id, []).append(index)
            continue

        if role == "tool" and trace:
            content = _stringify_message_content(getattr(message, "content", ""))
            if not content:
                continue

            index = None
            tool_call_id = str(getattr(message, "tool_call_id", "") or "")
            if tool_call_id:
                matching_indices = pending_by_call_id.get(tool_call_id) or []
                if matching_indices:
                    index = matching_indices.pop(0)

            if index is None and pending_indices:
                while pending_indices and trace[pending_indices[0]]["result"]:
                    pending_indices.pop(0)
                if pending_indices:
                    index = pending_indices.pop(0)

            if index is not None and not trace[index]["result"]:
                trace[index]["result"] = content
    return trace


def _evaluate_auto_skill_quality(
    user_request: str,
    trace: Sequence[dict[str, Any]],
    final_result: str,
    *,
    min_tool_calls: int,
    candidate_score_threshold: int,
    approved_score_threshold: int,
) -> AutoSkillQualityReport:
    tool_names = _dedupe([str(step.get("name", "")) for step in trace if step.get("name")])
    categories = _dedupe([_categorize_step(step) for step in trace if _categorize_step(step) != "other"])
    completed_steps = [step for step in trace if step.get("result")]
    successful_steps = [step for step in completed_steps if not _looks_like_error(_stringify_message_content(step.get("result", "")))]
    failed_steps = [step for step in completed_steps if _looks_like_error(_stringify_message_content(step.get("result", "")))]
    completion_gap = _analyze_completion_gap(user_request, final_result)
    # “答3道题”这类任务不能只看最终总结像不像成功，还要确认轨迹里真的出现了写回答/提交流程。
    action_gap = _analyze_requested_action_gap(user_request, trace, final_result)
    requirement_mismatch = completion_gap["requirement_mismatch"] or action_gap["requirement_mismatch"]
    successful = _looks_like_success(final_result) and not requirement_mismatch
    recovered = bool(failed_steps) and successful
    stable_completion = _has_stable_completion(trace, successful)
    meets_step_threshold = len(trace) >= min_tool_calls
    has_discovery = "discovery" in categories
    has_action = "action" in categories
    has_validation = "validation" in categories
    multi_tool = len(tool_names) >= 2
    environment_risk = any(_contains_environment_specific_data(_flatten_step_text(step)) for step in trace) or _contains_environment_specific_data(final_result)
    result_specific = len(_sanitize_summary_text(final_result, max_chars=320)) >= _MIN_FINAL_OUTCOME_CHARS
    too_few_steps = len(trace) < 2 and not recovered

    metrics = {
        "total_steps": len(trace),
        "completed_steps": len(completed_steps),
        "successful_steps": len(successful_steps),
        "failed_steps": len(failed_steps),
        "unique_tools": len(tool_names),
        "categories": categories,
        "meets_step_threshold": meets_step_threshold,
        "recovered": recovered,
        "stable_completion": stable_completion,
        "environment_risk": environment_risk,
        "result_specific": result_specific,
        "user_request_chars": len(user_request.strip()),
        "requested_count": completion_gap["requested_count"],
        "reported_count": completion_gap["reported_count"],
        "partial_completion": completion_gap["partial_completion"],
        "task_kind": action_gap["task_kind"],
        "required_action_count": action_gap["required_action_count"],
        "observed_action_count": action_gap["observed_action_count"],
        "observed_authoring_count": action_gap["observed_authoring_count"],
        "action_requirement_mismatch": action_gap["requirement_mismatch"],
        "requirement_mismatch": requirement_mismatch,
    }

    reasons: list[str] = []
    warnings: list[str] = []
    score = 0

    if successful:
        score += 2
        reasons.append("final-result-successful")
    else:
        warnings.append("final-result-not-successful")

    if completion_gap["partial_completion"]:
        warnings.append("partial-completion-detected")
    if requirement_mismatch:
        warnings.append("requirement-mismatch")
    if action_gap["requirement_mismatch"]:
        warnings.append("requested-action-not-observed")

    if meets_step_threshold:
        score += 2
        reasons.append("multi-step-workflow")

    if len(successful_steps) >= max(2, len(trace) - 1):
        score += 1
        reasons.append("successful-tool-trace")
    elif len(successful_steps) >= 2:
        score += 1
        reasons.append("multiple-successful-steps")

    if stable_completion:
        score += 1
        reasons.append("stable-completion")

    if not failed_steps:
        score += 1
        reasons.append("clean-run")
    elif recovered:
        score += 1
        reasons.append("recovered-from-failure")

    if len(failed_steps) >= 2:
        warnings.append("multiple-failed-steps")
        score -= min(2, len(failed_steps) - 1)

    if multi_tool:
        score += 1
        reasons.append("multi-tool-workflow")

    if has_discovery and has_action:
        score += 1
        reasons.append("discovery-action-loop")

    if has_validation:
        score += 1
        reasons.append("has-validation-step")

    if result_specific:
        score += 1
        reasons.append("specific-final-outcome")
    else:
        warnings.append("final-outcome-too-generic")

    if not environment_risk:
        score += 1
        reasons.append("environment-neutral")
    else:
        warnings.append("environment-specific-data-detected")

    if len(completed_steps) < len(trace):
        warnings.append("incomplete-tool-results")
    if too_few_steps:
        warnings.append("too-few-steps")

    score = max(score, 0)

    should_create = successful and not too_few_steps and score >= candidate_score_threshold
    tier = ""
    if should_create:
        tier = "approved" if score >= approved_score_threshold else "candidate"

    return AutoSkillQualityReport(
        should_create=should_create,
        score=score,
        tier=tier,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
    )


def _extract_user_request(turn_messages: Sequence[Message]) -> str:
    for message in turn_messages:
        if getattr(message, "role", "") == "user":
            return _stringify_message_content(getattr(message, "content", ""))
    return ""


def _extract_requested_count(text: str) -> int | None:
    for pattern in _REQUEST_COUNT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            count = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if count > 1:
            return count
    return None


def _analyze_completion_gap(user_request: str, final_result: str) -> dict[str, Any]:
    requested_count = _extract_requested_count(user_request)
    result_summary = _summarize_final_outcome(final_result, max_chars=320)
    reported_count = _extract_limited_result_count(result_summary)
    lowered_result = result_summary.lower()
    partial_completion = bool(requested_count and any(marker in lowered_result for marker in _PARTIAL_COMPLETION_MARKERS))

    requirement_mismatch = False
    if requested_count:
        if reported_count is not None:
            requirement_mismatch = reported_count < requested_count
        elif partial_completion:
            requirement_mismatch = True

    return {
        "requested_count": requested_count,
        "reported_count": reported_count,
        "partial_completion": partial_completion,
        "requirement_mismatch": requirement_mismatch,
    }


def _analyze_requested_action_gap(
    user_request: str,
    trace: Sequence[dict[str, Any]],
    final_result: str,
) -> dict[str, Any]:
    lowered_request = user_request.lower()
    if not _matches_any_pattern(lowered_request, _ANSWER_REQUEST_PATTERNS):
        return {
            "task_kind": "",
            "required_action_count": 0,
            "observed_action_count": 0,
            "observed_authoring_count": 0,
            "requirement_mismatch": False,
        }

    required_action_count = _extract_requested_count(user_request) or 1
    observed_action_count = 0
    observed_authoring_count = 0

    for step in trace:
        arguments_text = _step_argument_text(step).lower()
        result_text = _stringify_message_content(step.get("result", "")).lower()

        authoring_signal = (
            _contains_any(arguments_text, _ANSWER_ENTRY_MARKERS)
            or _matches_any_pattern(arguments_text, _ANSWER_AUTHORING_PATTERNS)
            or _matches_any_pattern(result_text, _ANSWER_AUTHORING_PATTERNS)
        )
        if authoring_signal:
            observed_authoring_count += 1

        submission_signal = (
            _matches_any_pattern(arguments_text, _ANSWER_SUBMISSION_PATTERNS)
            or _matches_any_pattern(result_text, _ANSWER_SUBMISSION_PATTERNS)
            or _contains_any(result_text, _ANSWER_SUCCESS_MARKERS)
        )
        if submission_signal:
            observed_action_count += 1

    lowered_result = final_result.lower()
    browse_only_summary = _contains_any(lowered_result, _ANSWER_BROWSE_ONLY_MARKERS)
    existing_answer_summary = _contains_any(lowered_result, _EXISTING_ANSWER_SUMMARY_MARKERS)
    requirement_mismatch = observed_action_count < required_action_count
    if observed_action_count == 0 and (observed_authoring_count == 0 or browse_only_summary or existing_answer_summary):
        requirement_mismatch = True

    return {
        "task_kind": "answering",
        "required_action_count": required_action_count,
        "observed_action_count": observed_action_count,
        "observed_authoring_count": observed_authoring_count,
        "requirement_mismatch": requirement_mismatch,
    }


def _extract_limited_result_count(text: str) -> int | None:
    for pattern in _LIMITED_RESULT_COUNT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            continue
    return None


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _matches_any_pattern(text: str, patterns: Sequence[re.Pattern]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _step_argument_text(step: dict[str, Any]) -> str:
    parts = [
        _stringify_message_content(step.get("arguments", "")),
        _flatten_search_text(step.get("arguments_dict", {})),
    ]
    return " ".join(part for part in parts if part)


def _looks_like_error(text: str) -> bool:
    lowered = text.lower()
    if not lowered.strip():
        return False
    if any(marker in lowered for marker in _NO_ERROR_MARKERS):
        return False
    return any(marker in lowered for marker in _ERROR_MARKERS) or lowered.startswith("error")


def _looks_like_success(text: str) -> bool:
    lowered = text.lower()
    if not lowered.strip():
        return False
    # 自动技能质量闸门必须识别中文失败总结，否则中文任务会把“需要登录/无法处理”这类失败结论误判成成功，
    # 进而把失败轨迹沉淀成可自动加载的技能。
    if any(token in lowered for token in _SUCCESS_BLOCKERS):
        return False
    return not _looks_like_error(text)


def _tool_call_name(call: object) -> str:
    function = getattr(call, "function", None)
    if function is None:
        return "tool"
    return str(getattr(function, "name", "tool"))


def _tool_call_args(call: object) -> dict[str, object]:
    function = getattr(call, "function", None)
    if function is None:
        return {}
    arguments = getattr(function, "arguments", {})
    return arguments if isinstance(arguments, dict) else {}


def _format_arguments(arguments: dict[str, object]) -> str:
    if not arguments:
        return ""
    items = []
    for key, value in list(arguments.items())[:3]:
        value_text = _stringify_message_content(value)
        if len(value_text) > 80:
            value_text = value_text[:77] + "..."
        items.append(f"{key}={value_text}")
    return ", ".join(items)


def _categorize_step(step: dict[str, Any]) -> str:
    tool_name = str(step.get("name", "") or "").lower()
    context = _flatten_step_text(step).lower()
    if tool_name in _VALIDATION_TOOL_NAMES or any(marker in context for marker in _VALIDATION_MARKERS):
        return "validation"
    if tool_name in _DISCOVERY_TOOL_NAMES or any(marker in context for marker in _DISCOVERY_MARKERS):
        return "discovery"
    if tool_name in _ACTION_TOOL_NAMES or any(marker in context for marker in _ACTION_MARKERS):
        return "action"
    return "other"


def _flatten_step_text(step: dict[str, Any]) -> str:
    parts = [
        _stringify_message_content(step.get("name", "")),
        _stringify_message_content(step.get("arguments", "")),
        _flatten_search_text(step.get("arguments_dict", {})),
        _stringify_message_content(step.get("result", "")),
    ]
    return " ".join(part for part in parts if part)


def _flatten_search_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_search_text(subvalue)}" for key, subvalue in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_search_text(item) for item in value)
    return str(value)


def _has_stable_completion(trace: Sequence[dict[str, Any]], successful: bool) -> bool:
    if not successful:
        return False
    recent_results = [
        _stringify_message_content(step.get("result", ""))
        for step in trace
        if _stringify_message_content(step.get("result", ""))
    ][-2:]
    if not recent_results:
        return False
    return all(not _looks_like_error(result) for result in recent_results)


def _contains_environment_specific_data(text: str) -> bool:
    return bool(
        _WINDOWS_PATH_PATTERN.search(text)
        or _UNIX_PATH_PATTERN.search(text)
        or _TIMESTAMP_PATTERN.search(text)
        or _ID_PATTERN.search(text)
    )