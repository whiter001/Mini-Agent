"""Cleanup and record-loading helpers for auto-generated skills."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

_CANDIDATE_DIRNAME = "_candidates"
_ARCHIVED_DIRNAME = "_archived"
_AUTO_SKILL_BLOCKING_WARNINGS = {
    "environment-specific-data-detected",
    "multiple-failed-steps",
    "partial-completion-detected",
    "requirement-mismatch",
    "incomplete-tool-results",
    "requested-action-not-observed",
}
_SKILL_FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_NUMERIC_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+)-(?P<suffix>\d+)$")


@dataclass
class AutoSkillCleanupAction:
    kind: str
    skill_path: Path
    detail: str = ""
    target_path: Path | None = None
    replacement_name: str = ""


@dataclass
class AutoSkillCleanupReport:
    root: Path
    archive_root: Path
    apply: bool
    scanned_skills: int = 0
    family_groups: int = 0
    renamed_count: int = 0
    archived_count: int = 0
    kept_count: int = 0
    actions: list[AutoSkillCleanupAction] = field(default_factory=list)


@dataclass
class _AutoSkillRecord:
    tier: str
    skill_dir: Path
    skill_file: Path
    dir_name: str
    content: str
    skill_name: str
    normalized_content: str
    family_key: str
    aliases: set[str] = field(default_factory=set)
    score: int = 0
    warnings: set[str] = field(default_factory=set)
    generated_at: datetime | None = None


def cleanup_auto_skills(
    auto_skill_dir: str,
    *,
    apply: bool = False,
    include_candidates: bool = True,
    archive_dir: str | None = None,
) -> AutoSkillCleanupReport:
    """Review and optionally clean up historical auto-generated skills."""
    root = Path(str(auto_skill_dir).strip()).expanduser()
    archive_root = Path(str(archive_dir).strip()).expanduser() if archive_dir else _default_auto_skill_archive_root(root)
    report = AutoSkillCleanupReport(root=root, archive_root=archive_root, apply=apply)

    records = _collect_generated_auto_skill_records(root, include_candidates=include_candidates)
    report.scanned_skills = len(records)
    if not records:
        return report

    dir_names = {record.dir_name for record in records}
    for record in records:
        record.family_key = _resolve_cleanup_family_key(record, dir_names)

    archived_dirs: set[Path] = set()

    duplicate_groups: dict[str, list[_AutoSkillRecord]] = {}
    for record in records:
        duplicate_groups.setdefault(record.normalized_content, []).append(record)
    for group in duplicate_groups.values():
        if len(group) <= 1:
            continue
        primary = _select_preferred_cleanup_record(group)
        for record in group:
            if record is primary:
                continue
            _stage_archive_cleanup_action(report, record, archive_root, reason="duplicate-content", archived_dirs=archived_dirs)

    survivors = [record for record in records if record.skill_dir not in archived_dirs]

    family_groups: dict[str, list[_AutoSkillRecord]] = {}
    for record in survivors:
        family_groups.setdefault(record.family_key, []).append(record)
    for group in family_groups.values():
        if len(group) <= 1:
            continue
        report.family_groups += 1
        primary = _select_preferred_cleanup_record(group)
        for record in group:
            if record is primary:
                continue
            _stage_archive_cleanup_action(
                report,
                record,
                archive_root,
                reason=f"historical-version:{record.family_key}",
                archived_dirs=archived_dirs,
            )

    survivors = [record for record in survivors if record.skill_dir not in archived_dirs]
    report.kept_count = len(survivors)
    for record in survivors:
        if record.skill_name == record.dir_name:
            continue
        report.renamed_count += 1
        report.actions.append(
            AutoSkillCleanupAction(
                kind="rename-skill-name",
                skill_path=record.skill_file,
                replacement_name=record.dir_name,
                detail="sync frontmatter name with directory name",
            )
        )

    if apply:
        _apply_auto_skill_cleanup(report)

    return report


def _find_matching_auto_skill_record(
    auto_skill_dir: str,
    family_keys: Sequence[str],
) -> _AutoSkillRecord | None:
    keys = {str(key).strip() for key in family_keys if str(key).strip()}
    if not keys:
        return None

    root = Path(str(auto_skill_dir).strip()).expanduser()
    matches: list[_AutoSkillRecord] = []
    for record in _collect_generated_auto_skill_records(root, include_candidates=True):
        record_keys = {record.family_key, record.skill_name, record.dir_name, *record.aliases}
        if any(key and key in record_keys for key in keys):
            matches.append(record)

    if not matches:
        return None
    return _select_preferred_cleanup_record(matches)


def _rename_generated_skill_content(skill_content: str, skill_name: str) -> str:
    frontmatter_match = _SKILL_FRONTMATTER_PATTERN.match(skill_content)
    if not frontmatter_match:
        return skill_content

    try:
        frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
    except yaml.YAMLError:
        return skill_content

    if not isinstance(frontmatter, dict):
        return skill_content

    frontmatter["name"] = skill_name
    body = frontmatter_match.group(2).lstrip()
    body = re.sub(r"^#\s+.+$", f"# {skill_name}", body, count=1, flags=re.MULTILINE)
    frontmatter_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{frontmatter_yaml}\n---\n\n{body}"


def _normalize_skill_content(content: str) -> str:
    frontmatter_match = _SKILL_FRONTMATTER_PATTERN.match(content)
    if not frontmatter_match:
        return content.strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
    except yaml.YAMLError:
        return "\n".join(
            line for line in content.splitlines() if not line.startswith("  generated_at: ")
        )

    if not isinstance(frontmatter, dict):
        return content.strip()

    frontmatter["name"] = "<skill-name>"
    metadata = frontmatter.get("metadata")
    if isinstance(metadata, dict):
        if "generated_at" in metadata:
            metadata["generated_at"] = "<timestamp>"
        if "updated_at" in metadata:
            metadata["updated_at"] = "<timestamp>"

    body = frontmatter_match.group(2).lstrip()
    body = re.sub(r"^#\s+.+$", "# <skill-name>", body, count=1, flags=re.MULTILINE)
    frontmatter_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{frontmatter_yaml}\n---\n\n{body.strip()}"


def _default_auto_skill_archive_root(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return root / _ARCHIVED_DIRNAME / f"auto-skill-cleanup-{timestamp}"


def _collect_generated_auto_skill_records(root: Path, *, include_candidates: bool) -> list[_AutoSkillRecord]:
    records: list[_AutoSkillRecord] = []
    tier_roots = [("approved", root)]
    if include_candidates:
        tier_roots.append(("candidate", root / _CANDIDATE_DIRNAME))

    for tier, tier_root in tier_roots:
        if not tier_root.exists() or not tier_root.is_dir():
            continue
        for child in sorted(tier_root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_dir() or child.name in {_CANDIDATE_DIRNAME, _ARCHIVED_DIRNAME}:
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.is_file():
                continue
            record = _load_generated_auto_skill_record(skill_file, tier)
            if record is not None:
                records.append(record)

    return records


def _load_generated_auto_skill_record(skill_file: Path, tier: str) -> _AutoSkillRecord | None:
    try:
        content = skill_file.read_text(encoding="utf-8")
    except OSError:
        return None

    frontmatter_match = _SKILL_FRONTMATTER_PATTERN.match(content)
    if not frontmatter_match:
        return None

    try:
        frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
    except yaml.YAMLError:
        return None

    if not isinstance(frontmatter, dict):
        return None

    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict) or str(metadata.get("source") or "").strip().lower() != "mini-agent":
        return None

    auto_skill_meta = metadata.get("auto_skill") if isinstance(metadata.get("auto_skill"), dict) else {}
    warnings = {
        str(item).strip().lower()
        for item in auto_skill_meta.get("warnings", [])
        if str(item).strip()
    }

    try:
        score = int(auto_skill_meta.get("score") or 0)
    except (TypeError, ValueError):
        score = 0

    family_key = str(auto_skill_meta.get("family_key") or "").strip()
    aliases = {
        str(item).strip()
        for item in auto_skill_meta.get("legacy_family_keys", [])
        if str(item).strip()
    }

    return _AutoSkillRecord(
        tier=tier,
        skill_dir=skill_file.parent,
        skill_file=skill_file,
        dir_name=skill_file.parent.name,
        content=content,
        skill_name=str(frontmatter.get("name") or "").strip(),
        normalized_content=_normalize_skill_content(content),
        family_key=family_key,
        aliases=aliases,
        score=score,
        warnings=warnings,
        generated_at=_parse_generated_at(metadata.get("generated_at")),
    )


def _parse_generated_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _resolve_cleanup_family_key(record: _AutoSkillRecord, dir_names: set[str]) -> str:
    if record.family_key:
        return record.family_key

    if record.skill_name and record.skill_name != record.dir_name:
        match = _NUMERIC_SUFFIX_PATTERN.fullmatch(record.dir_name)
        if match and match.group("base") == record.skill_name:
            return record.skill_name

    if record.dir_name in dir_names:
        return record.dir_name

    return record.skill_name or record.dir_name


def _cleanup_record_sort_key(record: _AutoSkillRecord) -> tuple[int, int, int, int, int, int, int, str]:
    blocking_warning_count = len(record.warnings & _AUTO_SKILL_BLOCKING_WARNINGS)
    generated_at_epoch = int(record.generated_at.timestamp()) if record.generated_at is not None else 0
    match = _NUMERIC_SUFFIX_PATTERN.fullmatch(record.dir_name)
    suffix_value = int(match.group("suffix")) if match else 0
    return (
        1 if record.tier == "approved" else 0,
        record.score,
        -blocking_warning_count,
        -len(record.warnings),
        generated_at_epoch,
        1 if record.dir_name == record.family_key else 0,
        -suffix_value,
        record.dir_name.lower(),
    )


def _select_preferred_cleanup_record(records: Sequence[_AutoSkillRecord]) -> _AutoSkillRecord:
    return max(records, key=_cleanup_record_sort_key)


def _stage_archive_cleanup_action(
    report: AutoSkillCleanupReport,
    record: _AutoSkillRecord,
    archive_root: Path,
    *,
    reason: str,
    archived_dirs: set[Path],
) -> None:
    if record.skill_dir in archived_dirs:
        return

    archived_dirs.add(record.skill_dir)
    report.archived_count += 1
    report.actions.append(
        AutoSkillCleanupAction(
            kind="archive-skill",
            skill_path=record.skill_dir,
            target_path=_build_archive_target_path(record, archive_root),
            detail=reason,
        )
    )


def _build_archive_target_path(record: _AutoSkillRecord, archive_root: Path) -> Path:
    return archive_root / record.tier / record.dir_name


def _apply_auto_skill_cleanup(report: AutoSkillCleanupReport) -> None:
    for action in report.actions:
        if action.kind != "archive-skill" or action.target_path is None:
            continue

        destination = _next_available_directory(action.target_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(action.skill_path), str(destination))
        action.target_path = destination

    for action in report.actions:
        if action.kind != "rename-skill-name" or not action.replacement_name:
            continue

        try:
            content = action.skill_path.read_text(encoding="utf-8")
        except OSError:
            continue
        action.skill_path.write_text(_rename_generated_skill_content(content, action.replacement_name), encoding="utf-8")


def _next_available_directory(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = 2
    candidate = path.parent / f"{path.name}-{suffix}"
    while candidate.exists():
        suffix += 1
        candidate = path.parent / f"{path.name}-{suffix}"
    return candidate