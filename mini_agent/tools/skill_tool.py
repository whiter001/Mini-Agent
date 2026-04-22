"""
Skill Tool - Tools for Agent to load and inspect Skills on-demand

Implements Progressive Disclosure:
- Level 1: List available skills
- Level 2: Load full skill content when needed
"""

from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult
from .skill_loader import SkillLoader


class GetSkillTool(Tool):
    """Tool to get detailed information about a specific skill"""

    def __init__(self, skill_loader: SkillLoader):
        self.skill_loader = skill_loader

    @property
    def name(self) -> str:
        return "get_skill"

    @property
    def description(self) -> str:
        return "Get complete content and guidance for a specified skill, used for executing specific types of tasks"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to retrieve (use list_skills to view available skills)",
                }
            },
            "required": ["skill_name"],
        }

    async def execute(self, skill_name: str) -> ToolResult:
        """Get detailed information about specified skill"""
        skill = self.skill_loader.get_skill(skill_name)

        if not skill:
            available = ", ".join(self.skill_loader.list_skills())
            return ToolResult(
                success=False,
                content="",
                error=f"Skill '{skill_name}' does not exist. Available skills: {available}",
            )

        # Return complete skill content
        result = skill.to_prompt()
        return ToolResult(success=True, content=result)


class ListSkillsTool(Tool):
    """Tool to list all currently loaded skills"""

    def __init__(self, skill_loader: SkillLoader):
        self.skill_loader = skill_loader

    @property
    def name(self) -> str:
        return "list_skills"

    @property
    def description(self) -> str:
        return "List every loaded skill, including bundled and user-installed skills, with descriptions and source paths"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self) -> ToolResult:
        """List all currently loaded skills."""
        skills = sorted(self.skill_loader.loaded_skills.values(), key=lambda skill: skill.name.lower())

        if not skills:
            return ToolResult(success=True, content="No skills are currently loaded.")

        lines = [
            "Loaded Skills:",
            "The following skills are currently available, including bundled skills and user-installed skills.",
        ]

        for index, skill in enumerate(skills, 1):
            source = str(skill.skill_path.parent) if skill.skill_path else "unknown"
            lines.append(f"{index}. `{skill.name}` — {skill.description}")
            lines.append(f"   Source: `{source}`")

        lines.append(f"Total loaded skills: {len(skills)}")
        return ToolResult(success=True, content="\n".join(lines))


def create_skill_tools(
    skills_dir: str = "./skills",
    extra_skills_dirs: Optional[List[str]] = None,
) -> tuple[List[Tool], Optional[SkillLoader]]:
    """
    Create skill tools for Progressive Disclosure

    Provides list_skills and get_skill tools. The agent uses metadata in the
    system prompt to know what skills are available, then loads them on-demand.

    Args:
        skills_dir: Skills directory path
        extra_skills_dirs: Additional skill directories to scan

    Returns:
        Tuple of (list of tools, skill loader)
    """
    # Create skill loader
    loader = SkillLoader(skills_dir, extra_skills_dirs)

    # Discover and load skills
    skills = loader.discover_skills()
    print(f"✅ Discovered {len(skills)} Skills")

    # Create skill tools (Progressive Disclosure Level 1 + 2)
    tools = [
        ListSkillsTool(loader),
        GetSkillTool(loader),
    ]

    return tools, loader
