"""
Skill Loader - Load Claude Skills

Supports loading skills from SKILL.md files and providing them to Agent
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SkillSection:
    """Structured markdown section extracted from a skill."""

    heading: str
    level: int
    content: str


@dataclass
class QueryProfile:
    """Normalized query profile used for skill ranking."""

    raw: str
    tokens: List[str]
    compact: str


@dataclass
class Skill:
    """Skill data structure"""

    name: str
    description: str
    content: str
    license: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    skill_path: Optional[Path] = None
    tags: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    platform: str = ""
    sections: List[SkillSection] = field(default_factory=list)

    def to_prompt(self, max_content_chars: int | None = None) -> str:
        """Convert skill to prompt format."""
        # Inject skill root directory path for context
        skill_root = str(self.skill_path.parent) if self.skill_path else "unknown"

        skill_content = self.content
        truncation_note = ""
        if max_content_chars is not None and max_content_chars >= 0 and len(skill_content) > max_content_chars:
            skill_content = skill_content[:max_content_chars].rstrip()
            truncation_note = (
                "\n\n... [Skill content truncated to keep the request within the context window. "
                "Use get_skill for the full version.] ..."
            )

        return f"""
# Skill: {self.name}

{self.description}

**Skill Root Directory:** `{skill_root}`

All files and references in this skill are relative to this directory.

---

{skill_content}{truncation_note}
"""


class SkillLoader:
    """Skill loader"""

    _TOKEN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9._/-]*")
    _TOKEN_SPLIT_PATTERN = re.compile(r"[._/-]+")

    def __init__(
        self,
        skills_dir: str = "./skills",
        extra_skills_dirs: Optional[List[str]] = None,
        ignored_skill_dir_names: Optional[List[str]] = None,
    ):
        """
        Initialize Skill Loader

        Args:
            skills_dir: Skills directory path
            extra_skills_dirs: Additional skill directories to scan
        """
        # Be forgiving about accidental leading/trailing whitespace in config values.
        self.skills_dir = Path(str(skills_dir).strip())
        self.extra_skills_dirs = [Path(str(path).strip()) for path in (extra_skills_dirs or [])]
        self.ignored_skill_dir_names = {
            str(name).strip()
            for name in (ignored_skill_dir_names or ["_candidates"])
            if str(name).strip()
        }
        self.loaded_skills: Dict[str, Skill] = {}

    def _flatten_metadata(self, value: Any) -> str:
        """Convert nested metadata values into searchable text."""
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(
                f"{key} {self._flatten_metadata(subvalue)}"
                for key, subvalue in value.items()
            )
        if isinstance(value, (list, tuple, set)):
            return " ".join(self._flatten_metadata(item) for item in value)
        return str(value)

    def _unique_ordered_strings(self, values: List[str]) -> List[str]:
        """Deduplicate strings while preserving their original order."""
        items: List[str] = []
        seen = set()
        for value in values:
            trimmed = str(value).strip()
            if not trimmed:
                continue
            key = trimmed.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(trimmed)
        return items

    def _normalize_string_list(self, value: Any) -> List[str]:
        """Normalize YAML scalar/list fields into a clean string list."""
        if value is None:
            return []
        if isinstance(value, str):
            items = [part.strip() for part in value.split(",")] if "," in value else [value.strip()]
            return self._unique_ordered_strings(items)
        if isinstance(value, (list, tuple, set)):
            items: List[str] = []
            for item in value:
                items.extend(self._normalize_string_list(item))
            return self._unique_ordered_strings(items)
        return self._unique_ordered_strings([str(value)])

    def _expand_token(self, token: str) -> List[str]:
        """Expand mixed-language tokens into searchable variants."""
        trimmed = token.strip().lower()
        if not trimmed:
            return []

        items = [trimmed]
        if self._TOKEN_SPLIT_PATTERN.search(trimmed):
            items.extend(part for part in self._TOKEN_SPLIT_PATTERN.split(trimmed) if part)

        if self._contains_han(trimmed):
            runes = list(trimmed)
            if len(runes) > 2:
                max_gram = min(4, len(runes))
                for size in range(2, max_gram + 1):
                    for start in range(0, len(runes) - size + 1):
                        items.append("".join(runes[start : start + size]))

        return self._unique_ordered_strings(items)

    def _contains_han(self, text: str) -> bool:
        """Return True when the text contains CJK Han characters."""
        return any("\u3400" <= char <= "\u9fff" for char in text)

    def _tokenize(self, text: str) -> List[str]:
        """Split text into normalized search tokens."""
        tokens: List[str] = []
        seen = set()
        for match in self._TOKEN_PATTERN.findall(text.lower()):
            for token in self._expand_token(match):
                normalized = token.strip(" ,.;:!?()[]{}<>\"'`")
                if len(normalized) <= 1 or normalized in seen:
                    continue
                seen.add(normalized)
                tokens.append(normalized)
        return tokens

    def _compact_text(self, text: str) -> str:
        """Compact text into a separator-free representation for phrase matching."""
        return "".join(self._TOKEN_PATTERN.findall(text.lower()))

    def _build_query_profile(self, query: str) -> QueryProfile:
        """Build a reusable query profile for ranking and context extraction."""
        normalized = query.strip()
        return QueryProfile(
            raw=normalized,
            tokens=self._tokenize(normalized),
            compact=self._compact_text(normalized),
        )

    def _field_token_score(self, text: str, tokens: List[str], weight: int) -> int:
        """Return a weighted score for token matches inside a text field."""
        if not tokens or not text or weight <= 0:
            return 0
        normalized = text.lower()
        return sum(weight for token in tokens if token in normalized)

    def _phrase_score(self, text: str, query_compact: str, weight: int) -> int:
        """Score exact-ish phrase matches after removing separators."""
        if not text or not query_compact or weight <= 0:
            return 0
        return weight if query_compact in self._compact_text(text) else 0

    def _matches_all_tokens(self, text: str, tokens: List[str]) -> bool:
        """Return True when every normalized token appears in the text."""
        if not tokens:
            return False
        normalized = text.lower()
        return all(token in normalized for token in tokens)

    def _build_search_text(self, skill: Skill) -> str:
        """Build the searchable text for a skill."""
        metadata_text = self._flatten_metadata(skill.metadata)
        return " ".join(
            part
            for part in [
                skill.name,
                skill.description,
                " ".join(skill.tags),
                " ".join(skill.tools),
                " ".join(skill.triggers),
                skill.platform,
                metadata_text,
                " ".join(section.heading for section in skill.sections if section.heading),
                skill.content[:2000],
            ]
            if part
        )

    def _score_skill(self, skill: Skill, query: QueryProfile) -> int:
        """Score how relevant a skill is for the query."""
        metadata_text = self._flatten_metadata(skill.metadata).lower()
        score = 0

        score += self._phrase_score(skill.name, query.compact, 28)
        score += self._phrase_score(" ".join(skill.tools), query.compact, 22)
        score += self._phrase_score(" ".join(skill.triggers), query.compact, 18)
        score += self._phrase_score(skill.description, query.compact, 12)

        score += self._field_token_score(skill.name, query.tokens, 12)
        score += self._field_token_score(" ".join(skill.tools), query.tokens, 10)
        score += self._field_token_score(" ".join(skill.triggers), query.tokens, 9)
        score += self._field_token_score(" ".join(skill.tags), query.tokens, 8)
        score += self._field_token_score(skill.platform, query.tokens, 8)
        score += self._field_token_score(skill.description, query.tokens, 6)
        score += self._field_token_score(metadata_text, query.tokens, 4)
        for section in skill.sections:
            score += self._field_token_score(section.heading, query.tokens, 4)
        score += self._field_token_score(skill.content, query.tokens, 2)

        if self._matches_all_tokens(
            " ".join(
                part
                for part in [
                    skill.name,
                    skill.description,
                    " ".join(skill.tools),
                    " ".join(skill.triggers),
                    " ".join(skill.tags),
                    skill.platform,
                    metadata_text,
                ]
                if part
            ),
            query.tokens,
        ):
            score += 10

        return score

    def _parse_heading(self, line: str) -> tuple[str, int] | None:
        """Parse a markdown heading line into heading text and level."""
        trimmed = line.strip()
        if not trimmed or not trimmed.startswith("#"):
            return None

        level = 0
        while level < len(trimmed) and trimmed[level] == "#":
            level += 1

        if level == 0 or level >= len(trimmed) or trimmed[level] != " ":
            return None

        return trimmed[level:].strip(), level

    def _parse_skill_sections(self, content: str) -> List[SkillSection]:
        """Split a skill markdown body into structured sections."""
        sections: List[SkillSection] = []
        current_heading = ""
        current_level = 0
        current_lines: List[str] = []

        def flush() -> None:
            nonlocal current_heading, current_level, current_lines
            text = "\n".join(current_lines).strip()
            if not current_heading and not text:
                current_lines = []
                return
            sections.append(SkillSection(heading=current_heading, level=current_level, content=text))
            current_heading = ""
            current_level = 0
            current_lines = []

        for line in content.splitlines():
            heading = self._parse_heading(line)
            if heading is not None:
                flush()
                current_heading, current_level = heading
                continue
            current_lines.append(line)

        flush()
        return sections

    def _first_populated_sections(self, sections: List[SkillSection], max_sections: int) -> List[SkillSection]:
        """Return the first non-empty sections as a stable fallback."""
        selected: List[SkillSection] = []
        for section in sections:
            if not section.content.strip():
                continue
            selected.append(section)
            if len(selected) >= max_sections:
                break
        return selected

    def _score_section(self, section: SkillSection, query: QueryProfile) -> int:
        """Score how relevant a section is for a query."""
        score = 0
        score += self._phrase_score(section.heading, query.compact, 14)
        score += self._phrase_score(section.content, query.compact, 6)
        score += self._field_token_score(section.heading, query.tokens, 6)
        score += self._field_token_score(section.content, query.tokens, 2)
        return score

    def _select_relevant_sections(
        self,
        skill: Skill,
        query: QueryProfile,
        max_sections: int = 2,
    ) -> List[SkillSection]:
        """Select the most relevant sections for the current request."""
        if not skill.sections:
            return []
        if max_sections <= 0:
            max_sections = 2
        if not query.tokens:
            return self._first_populated_sections(skill.sections, max_sections)

        scored_sections: List[tuple[int, int, SkillSection]] = []
        for index, section in enumerate(skill.sections):
            if not section.content.strip():
                continue
            score = self._score_section(section, query)
            if index == 0 and score == 0:
                score = 1
            if score > 0:
                scored_sections.append((score, index, section))

        if not scored_sections:
            return self._first_populated_sections(skill.sections, max_sections)

        scored_sections.sort(key=lambda item: (-item[0], item[1]))
        return [section for _, _, section in scored_sections[:max_sections]]

    def _truncate_middle(self, text: str, max_chars: int) -> tuple[str, bool]:
        """Truncate text from the middle while keeping both ends visible."""
        cleaned = text.strip()
        if max_chars < 0 or len(cleaned) <= max_chars:
            return cleaned, False
        if max_chars <= 3:
            return cleaned[:max_chars], True

        remaining = max_chars - 3
        left = remaining // 2
        right = remaining - left
        return f"{cleaned[:left].rstrip()}...{cleaned[-right:].lstrip()}", True

    def _build_relevant_excerpt(
        self,
        skill: Skill,
        query: QueryProfile,
        max_content_chars: int,
    ) -> tuple[str, bool]:
        """Render a focused excerpt that favors the most relevant sections."""
        selected_sections = self._select_relevant_sections(skill, query, max_sections=2)
        if not selected_sections:
            return self._truncate_middle(skill.content, max_content_chars)

        per_section_limit = max(240, max_content_chars // max(1, len(selected_sections)))
        parts: List[str] = []
        section_truncated = False
        for section in selected_sections:
            if not section.content.strip():
                continue
            heading = section.heading or "Overview"
            snippet, was_truncated = self._truncate_middle(section.content, per_section_limit)
            section_truncated = section_truncated or was_truncated
            parts.append(f"### {heading}\n{snippet}")

        if not parts:
            return self._truncate_middle(skill.content, max_content_chars)

        excerpt, excerpt_truncated = self._truncate_middle("\n\n".join(parts), max_content_chars)
        return excerpt, section_truncated or excerpt_truncated

    def _render_skill_context(
        self,
        index: int,
        skill: Skill,
        query: QueryProfile,
        max_content_chars: int,
    ) -> str:
        """Render a compact, section-aware prompt block for an auto-selected skill."""
        skill_root = str(skill.skill_path.parent) if skill.skill_path else "unknown"
        lines = [
            f"{index}. {skill.name}: {skill.description}",
            f"Skill Root Directory: `{skill_root}`",
            "All files and references in this skill are relative to this directory.",
        ]

        metadata_lines: List[str] = []
        if skill.tools:
            metadata_lines.append("Tools: " + ", ".join(skill.tools))
        if skill.tags:
            metadata_lines.append("Tags: " + ", ".join(skill.tags))
        if skill.triggers:
            metadata_lines.append("Triggers: " + ", ".join(skill.triggers))
        if skill.platform:
            metadata_lines.append("Platform: " + skill.platform)
        if metadata_lines:
            lines.append("\n".join(metadata_lines))

        excerpt, truncated = self._build_relevant_excerpt(skill, query, max_content_chars=max_content_chars)
        if excerpt:
            lines.append(excerpt)
        if truncated:
            lines.append(
                "... [Skill content truncated to keep the request within the context window. Use get_skill for the full version.] ..."
            )

        return "\n".join(lines)

    def load_skill(self, skill_path: Path) -> Optional[Skill]:
        """
        Load single skill from SKILL.md file

        Args:
            skill_path: SKILL.md file path

        Returns:
            Skill object, or None if loading fails
        """
        try:
            content = skill_path.read_text(encoding="utf-8")

            # Parse YAML frontmatter
            frontmatter_match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)

            if not frontmatter_match:
                print(f"⚠️  {skill_path} missing YAML frontmatter")
                return None

            frontmatter_text = frontmatter_match.group(1)
            skill_content = frontmatter_match.group(2).strip()

            # Parse YAML
            try:
                frontmatter = yaml.safe_load(frontmatter_text)
            except yaml.YAMLError as e:
                print(f"❌ Failed to parse YAML frontmatter: {e}")
                return None

            if not isinstance(frontmatter, dict):
                print(f"⚠️  {skill_path} frontmatter must be a mapping")
                return None

            # Required fields
            if "name" not in frontmatter or "description" not in frontmatter:
                print(f"⚠️  {skill_path} missing required fields (name or description)")
                return None

            # Get skill directory (parent of SKILL.md)
            skill_dir = skill_path.parent

            # Replace relative paths in content with absolute paths
            # This ensures scripts and resources can be found from any working directory
            processed_content = self._process_skill_paths(skill_content, skill_dir)
            allowed_tools = self._normalize_string_list(frontmatter.get("allowed-tools"))
            tools = self._unique_ordered_strings(
                [
                    *self._normalize_string_list(frontmatter.get("tools")),
                    *allowed_tools,
                ]
            )

            # Create Skill object
            skill = Skill(
                name=str(frontmatter["name"]).strip(),
                description=str(frontmatter["description"]).strip(),
                content=processed_content,
                license=frontmatter.get("license"),
                allowed_tools=allowed_tools or None,
                metadata=frontmatter.get("metadata"),
                skill_path=skill_path,
                tags=self._normalize_string_list(frontmatter.get("tags")),
                tools=tools,
                triggers=self._normalize_string_list(frontmatter.get("triggers")),
                platform=str(frontmatter.get("platform") or "").strip(),
                sections=self._parse_skill_sections(processed_content),
            )

            return skill

        except Exception as e:
            print(f"❌ Failed to load skill ({skill_path}): {e}")
            return None

    def _process_skill_paths(self, content: str, skill_dir: Path) -> str:
        """
        Process skill content to replace relative paths with absolute paths.

        Supports Progressive Disclosure Level 3+: converts relative file references
        to absolute paths so Agent can easily read nested resources.

        Args:
            content: Original skill content
            skill_dir: Skill directory path

        Returns:
            Processed content with absolute paths
        """
        import re
        skill_root = skill_dir.resolve(strict=False)

        def resolve_skill_reference(relative_path: str) -> Path | None:
            # Resolve symlinks before rewriting so references cannot escape the skill root.
            candidate = (skill_dir / relative_path).expanduser()
            try:
                resolved = candidate.resolve(strict=False)
            except OSError:
                return None

            try:
                resolved.relative_to(skill_root)
            except ValueError:
                return None

            return resolved if resolved.exists() else None

        # Pattern 1: Directory-based paths (scripts/, references/, assets/)
        # See https://agentskills.io/specification#optional-directories
        def replace_dir_path(match):
            prefix = match.group(1)  # e.g., "python " or "`"
            rel_path = match.group(2)  # e.g., "scripts/with_server.py"

            abs_path = resolve_skill_reference(rel_path)
            if abs_path is not None:
                return f"{prefix}{abs_path}"
            return match.group(0)

        pattern_dirs = r"(python\s+|`)((?:scripts|references|assets)/[^\s`\)]+)"
        content = re.sub(pattern_dirs, replace_dir_path, content)

        # Pattern 2: Direct markdown/document references (forms.md, reference.md, etc.)
        # Matches phrases like "see reference.md" or "read forms.md"
        def replace_doc_path(match):
            prefix = match.group(1)  # e.g., "see ", "read "
            filename = match.group(2)  # e.g., "reference.md"
            suffix = match.group(3)  # e.g., punctuation

            abs_path = resolve_skill_reference(filename)
            if abs_path is not None:
                # Add helpful instruction for Agent
                return f"{prefix}`{abs_path}` (use read_file to access){suffix}"
            return match.group(0)

        # Match patterns like: "see reference.md" or "read forms.md"
        pattern_docs = r"(see|read|refer to|check)\s+([a-zA-Z0-9_-]+\.(?:md|txt|json|yaml))([.,;\s])"
        content = re.sub(pattern_docs, replace_doc_path, content, flags=re.IGNORECASE)

        # Pattern 3: Markdown links - supports multiple formats:
        # - [`filename.md`](filename.md) - simple filename
        # - [text](./reference/file.md) - relative path with ./
        # - [text](scripts/file.js) - directory-based path
        # Matches patterns like: "Read [`docx-js.md`](docx-js.md)" or "Load [Guide](./reference/guide.md)"
        def replace_markdown_link(match):
            prefix = match.group(1) if match.group(1) else ""  # e.g., "Read ", "Load ", or empty
            link_text = match.group(2)  # e.g., "`docx-js.md`" or "Guide"
            filepath = match.group(3)  # e.g., "docx-js.md", "./reference/file.md", "scripts/file.js"

            # Remove leading ./ if present
            clean_path = filepath[2:] if filepath.startswith("./") else filepath

            abs_path = resolve_skill_reference(clean_path)
            if abs_path is not None:
                # Preserve the link text style (with or without backticks)
                return f"{prefix}[{link_text}](`{abs_path}`) (use read_file to access)"
            return match.group(0)

        # Match markdown link patterns with optional prefix words
        # Captures: (optional prefix word) [link text] (complete file path including ./)
        pattern_markdown = (
            r"(?:(Read|See|Check|Refer to|Load|View)\s+)?\[(`?[^`\]]+`?)\]\(((?:\./)?[^)]+\.(?:md|txt|json|yaml|js|py|html))\)"
        )
        content = re.sub(pattern_markdown, replace_markdown_link, content, flags=re.IGNORECASE)

        return content

    def discover_skills(self) -> List[Skill]:
        """
        Discover and load all skills in the skills directory

        Returns:
            List of Skills
        """
        skills = []
        self.loaded_skills = {}

        for skills_dir in self._iter_skill_dirs():
            if not skills_dir.exists():
                continue

            # Recursively find all SKILL.md files
            for skill_file in skills_dir.rglob("SKILL.md"):
                if self._should_ignore_skill_path(skill_file):
                    continue
                skill = self.load_skill(skill_file)
                if skill and skill.name not in self.loaded_skills:
                    skills.append(skill)
                    self.loaded_skills[skill.name] = skill

        return skills

    def _should_ignore_skill_path(self, skill_file: Path) -> bool:
        """Skip skill files stored in non-loadable internal directories."""
        return any(part in self.ignored_skill_dir_names for part in skill_file.parts)

    def _iter_skill_dirs(self) -> List[Path]:
        """Return skill directories in precedence order."""
        skill_dirs = [self.skills_dir, *self.extra_skills_dirs]
        unique_dirs = []
        seen = set()
        for skill_dir in skill_dirs:
            resolved = str(skill_dir.expanduser().resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_dirs.append(skill_dir.expanduser())
        return unique_dirs

    def select_relevant_skills(self, query: str, max_skills: int = 2) -> List[Skill]:
        """Select the most relevant skills for a user request."""
        if max_skills <= 0 or not query.strip():
            return []

        query_profile = self._build_query_profile(query)
        if not query_profile.tokens:
            return []

        scored_skills = []
        for skill in self.loaded_skills.values():
            score = self._score_skill(skill, query_profile)
            if score > 0:
                scored_skills.append((score, skill.name.lower(), skill))

        scored_skills.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _, _, skill in scored_skills[:max_skills]]

    def get_auto_skills_prompt(self, query: str, max_skills: int = 2, max_content_chars: int = 1200) -> str:
        """Build a prompt block for the skills selected for this request."""
        selected_skills = self.select_relevant_skills(query, max_skills=max_skills)
        if not selected_skills:
            return ""

        query_profile = self._build_query_profile(query)

        prompt_parts = [
            "## Auto-loaded Skills",
            "The following skills were selected automatically for the current request. Follow them as the primary guidance for this turn.",
        ]

        for index, skill in enumerate(selected_skills, 1):
            prompt_parts.append(
                self._render_skill_context(
                    index,
                    skill,
                    query_profile,
                    max_content_chars=max_content_chars,
                )
            )

        return "\n\n".join(prompt_parts)

    def get_skill(self, name: str) -> Optional[Skill]:
        """
        Get loaded skill

        Args:
            name: Skill name

        Returns:
            Skill object, or None if not found
        """
        return self.loaded_skills.get(name)

    def list_skills(self) -> List[str]:
        """
        List all loaded skill names

        Returns:
            List of skill names
        """
        return sorted(self.loaded_skills.keys())

    def get_skills_metadata_prompt(self) -> str:
        """
        Generate prompt containing ONLY metadata (name + description) for all skills.
        This implements Progressive Disclosure - Level 1.

        Returns:
            Metadata-only prompt string
        """
        if not self.loaded_skills:
            return ""

        prompt_parts = ["## Available Skills\n"]
        prompt_parts.append("You have access to specialized skills. Each skill provides expert guidance for specific tasks.\n")
        prompt_parts.append(
            "Use `list_skills` when you need the complete loaded skill inventory, including bundled and user-installed skills.\n"
        )
        prompt_parts.append("Load a skill's full content using the appropriate skill tool when needed.\n")

        # List all skills with their descriptions
        for skill in sorted(self.loaded_skills.values(), key=lambda item: item.name.lower()):
            prompt_parts.append(f"- `{skill.name}`: {skill.description}")

        return "\n".join(prompt_parts)
