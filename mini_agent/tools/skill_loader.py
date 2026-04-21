"""
Skill Loader - Load Claude Skills

Supports loading skills from SKILL.md files and providing them to Agent
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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

    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")

    def __init__(self, skills_dir: str = "./skills", extra_skills_dirs: Optional[List[str]] = None):
        """
        Initialize Skill Loader

        Args:
            skills_dir: Skills directory path
            extra_skills_dirs: Additional skill directories to scan
        """
        self.skills_dir = Path(skills_dir)
        self.extra_skills_dirs = [Path(path) for path in (extra_skills_dirs or [])]
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

    def _tokenize(self, text: str) -> set[str]:
        """Split text into normalized search tokens."""
        return {
            token.lower()
            for token in self._TOKEN_PATTERN.findall(text.lower())
            if len(token) > 1
        }

    def _build_search_text(self, skill: Skill) -> str:
        """Build the searchable text for a skill."""
        metadata_text = self._flatten_metadata(skill.metadata)
        return " ".join(
            part
            for part in [
                skill.name,
                skill.name,
                skill.description,
                skill.description,
                metadata_text,
                skill.content[:2000],
            ]
            if part
        )

    def _score_skill(self, skill: Skill, query_terms: set[str], query_text: str) -> int:
        """Score how relevant a skill is for the query."""
        search_text = self._build_search_text(skill)
        skill_terms = self._tokenize(search_text)
        score = len(query_terms & skill_terms)

        normalized_name = skill.name.lower().replace("-", " ")
        if normalized_name and normalized_name in query_text:
            score += 5

        normalized_description = skill.description.lower()
        if normalized_description and normalized_description in query_text:
            score += 3

        metadata_text = self._flatten_metadata(skill.metadata).lower()
        if metadata_text and metadata_text in query_text:
            score += 2

        return score

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

            # Required fields
            if "name" not in frontmatter or "description" not in frontmatter:
                print(f"⚠️  {skill_path} missing required fields (name or description)")
                return None

            # Get skill directory (parent of SKILL.md)
            skill_dir = skill_path.parent

            # Replace relative paths in content with absolute paths
            # This ensures scripts and resources can be found from any working directory
            processed_content = self._process_skill_paths(skill_content, skill_dir)

            # Create Skill object
            skill = Skill(
                name=frontmatter["name"],
                description=frontmatter["description"],
                content=processed_content,
                license=frontmatter.get("license"),
                allowed_tools=frontmatter.get("allowed-tools"),
                metadata=frontmatter.get("metadata"),
                skill_path=skill_path,
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

        # Pattern 1: Directory-based paths (scripts/, references/, assets/)
        # See https://agentskills.io/specification#optional-directories
        def replace_dir_path(match):
            prefix = match.group(1)  # e.g., "python " or "`"
            rel_path = match.group(2)  # e.g., "scripts/with_server.py"

            abs_path = skill_dir / rel_path
            if abs_path.exists():
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

            abs_path = skill_dir / filename
            if abs_path.exists():
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

            abs_path = skill_dir / clean_path
            if abs_path.exists():
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
                skill = self.load_skill(skill_file)
                if skill and skill.name not in self.loaded_skills:
                    skills.append(skill)
                    self.loaded_skills[skill.name] = skill

        return skills

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

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        query_text = query.lower()
        scored_skills = []
        for skill in self.loaded_skills.values():
            score = self._score_skill(skill, query_terms, query_text)
            if score > 0:
                scored_skills.append((score, skill.name.lower(), skill))

        scored_skills.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _, _, skill in scored_skills[:max_skills]]

    def get_auto_skills_prompt(self, query: str, max_skills: int = 2, max_content_chars: int = 1200) -> str:
        """Build a prompt block for the skills selected for this request."""
        selected_skills = self.select_relevant_skills(query, max_skills=max_skills)
        if not selected_skills:
            return ""

        prompt_parts = [
            "## Auto-loaded Skills",
            "The following skills were selected automatically for the current request. Follow them as the primary guidance for this turn.",
        ]

        for skill in selected_skills:
            prompt_parts.append(skill.to_prompt(max_content_chars=max_content_chars).strip())

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
        return list(self.loaded_skills.keys())

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
        prompt_parts.append("Load a skill's full content using the appropriate skill tool when needed.\n")

        # List all skills with their descriptions
        for skill in self.loaded_skills.values():
            prompt_parts.append(f"- `{skill.name}`: {skill.description}")

        return "\n".join(prompt_parts)
