"""Core Agent implementation."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import tiktoken

from .llm import LLMClient
from .logger import AgentLogger
from .schema import Message
from .tool_output import ToolOutputStore
from .tools.base import Tool, ToolResult
from .utils import calculate_display_width


# ANSI color codes
class Colors:
    """Terminal color definitions"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


ANCHOR_SUMMARY_PREFIX = "[Conversation Summary Anchor]"
ROUND_SUMMARY_PREFIX = "[Assistant Execution Summary]"


@dataclass
class ConversationRound:
    """A single user turn and the execution that followed it."""

    user_message: Message
    execution_messages: list[Message]


class Agent:
    """Single agent with basic tools and MCP support."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int | None = None,
        workspace_dir: str = "./workspace",
        token_limit: int = 160000,  # Leave safe headroom below MiniMax-M2.7's 204,800-token context window.
        summary_recent_rounds: int = 2,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.token_limit = token_limit
        self.summary_recent_rounds = max(summary_recent_rounds, 0)
        self.workspace_dir = Path(workspace_dir)
        self._ephemeral_messages: list[Message] = []
        # Cancellation event for interrupting agent execution (set externally, e.g., by Esc key)
        self.cancel_event: Optional[asyncio.Event] = None

        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Inject workspace information into system prompt if not already present
        if "Current Workspace" not in system_prompt:
            workspace_info = f"\n\n## Current Workspace\nYou are currently working in: `{self.workspace_dir.absolute()}`\nAll relative paths will be resolved relative to this directory."
            system_prompt = system_prompt + workspace_info

        self.system_prompt = system_prompt

        # Initialize message history
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]

        # Initialize logger
        self.logger = AgentLogger()
        self.tool_output_store = ToolOutputStore(self.workspace_dir)

        # Token usage from last API response (updated after each LLM call)
        self.api_total_tokens: int = 0
        # Flag to skip token check right after summary (avoid consecutive triggers)
        self._skip_next_token_check: bool = False

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    def set_ephemeral_context(self, messages: list[Message]) -> None:
        """Set temporary messages that only apply to the next run."""
        self._ephemeral_messages = messages

    def clear_ephemeral_context(self) -> None:
        """Clear temporary run-only messages."""
        self._ephemeral_messages = []

    def _get_active_messages(self) -> list[Message]:
        """Return the message list used for the next LLM call."""
        return self.messages + self._ephemeral_messages

    def _check_cancelled(self) -> bool:
        """Check if agent execution has been cancelled.

        Returns:
            True if cancelled, False otherwise.
        """
        if self.cancel_event is not None and self.cancel_event.is_set():
            return True
        return False

    def _cleanup_incomplete_messages(self):
        """Remove the incomplete assistant message and its partial tool results.

        This ensures message consistency after cancellation by removing
        only the current step's incomplete messages, preserving completed steps.
        """
        # Find the index of the last assistant message
        last_assistant_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx == -1:
            # No assistant message found, nothing to clean
            return

        # Remove the last assistant message and all tool results after it
        removed_count = len(self.messages) - last_assistant_idx
        if removed_count > 0:
            self.messages = self.messages[:last_assistant_idx]
            print(f"{Colors.DIM}   Cleaned up {removed_count} incomplete message(s){Colors.RESET}")

    def _estimate_tokens(self) -> int:
        """Accurately calculate token count for message history using tiktoken

        Uses cl100k_base encoder (GPT-4/Claude/M2 compatible)
        """
        try:
            # Use cl100k_base encoder (used by GPT-4 and most modern models)
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback: if tiktoken initialization fails, use simple estimation
            return self._estimate_tokens_fallback()

        total_tokens = 0

        for msg in self.messages:
            # Count text content
            if isinstance(msg.content, str):
                total_tokens += len(encoding.encode(msg.content))
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        # Convert dict to string for calculation
                        total_tokens += len(encoding.encode(str(block)))

            # Count thinking
            if msg.thinking:
                total_tokens += len(encoding.encode(msg.thinking))

            # Count tool_calls
            if msg.tool_calls:
                total_tokens += len(encoding.encode(str(msg.tool_calls)))

            # Metadata overhead per message (approximately 4 tokens)
            total_tokens += 4

        return total_tokens

    def _estimate_tokens_fallback(self) -> int:
        """Fallback token estimation method (when tiktoken is unavailable)"""
        total_chars = 0
        for msg in self.messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        total_chars += len(str(block))

            if msg.thinking:
                total_chars += len(msg.thinking)

            if msg.tool_calls:
                total_chars += len(str(msg.tool_calls))

        # Rough estimation: average 2.5 characters = 1 token
        return int(total_chars / 2.5)

    @staticmethod
    def _stringify_message_content(content: str | list[dict[str, Any]] | Any) -> str:
        """Convert message content into a readable string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        parts.append(str(block["text"]))
                    elif "content" in block:
                        parts.append(str(block["content"]))
                    else:
                        parts.append(str(block))
                else:
                    parts.append(str(block))
            return "\n".join(part for part in parts if part)
        return str(content)

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """Trim verbose text while keeping enough detail for summaries."""
        normalized = " ".join(text.split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3] + "..."

    def _is_summary_artifact(self, message: Message) -> bool:
        """Return True when the message is an internal summary artifact."""
        if message.role != "user" or not isinstance(message.content, str):
            return False
        stripped = message.content.lstrip()
        return stripped.startswith(ANCHOR_SUMMARY_PREFIX) or stripped.startswith(ROUND_SUMMARY_PREFIX)

    def _is_anchor_summary_message(self, message: Message) -> bool:
        """Return True when the message is the rolling conversation anchor."""
        return message.role == "user" and isinstance(message.content, str) and message.content.lstrip().startswith(ANCHOR_SUMMARY_PREFIX)

    def _extract_summary_body(self, message: Message, prefix: str | None = None) -> str:
        """Extract the body text from an internal summary message."""
        content = self._stringify_message_content(message.content).strip()
        if prefix and content.startswith(prefix):
            return content[len(prefix) :].lstrip()
        for candidate in (ANCHOR_SUMMARY_PREFIX, ROUND_SUMMARY_PREFIX):
            if content.startswith(candidate):
                return content[len(candidate) :].lstrip()
        return content

    def _build_anchor_message(self, anchor_text: str) -> Message:
        """Create the rolling anchor summary message."""
        return Message(role="user", content=f"{ANCHOR_SUMMARY_PREFIX}\n\n{anchor_text}")

    def _build_round_summary_message(self, summary_text: str) -> Message:
        """Create a compact execution summary for the current round."""
        return Message(role="user", content=f"{ROUND_SUMMARY_PREFIX}\n\n{summary_text}")

    def _collect_conversation_rounds(self) -> tuple[str, list[ConversationRound]]:
        """Split message history into archived anchor + actual user rounds."""
        existing_anchor = ""
        working_messages = self.messages[1:]

        if working_messages and self._is_anchor_summary_message(working_messages[0]):
            existing_anchor = self._extract_summary_body(working_messages[0], prefix=ANCHOR_SUMMARY_PREFIX)
            working_messages = working_messages[1:]

        user_indices = [
            index
            for index, message in enumerate(working_messages)
            if message.role == "user" and not self._is_summary_artifact(message)
        ]

        rounds: list[ConversationRound] = []
        for offset, user_idx in enumerate(user_indices):
            next_user_idx = user_indices[offset + 1] if offset + 1 < len(user_indices) else len(working_messages)
            rounds.append(
                ConversationRound(
                    user_message=working_messages[user_idx],
                    execution_messages=working_messages[user_idx + 1 : next_user_idx],
                )
            )

        return existing_anchor, rounds

    def _collect_turn_trace(self, turn_messages: list[Message]) -> list[dict[str, str]]:
        """Match tool results back to their originating calls for summary generation."""
        trace: list[dict[str, str]] = []
        pending_indices: list[int] = []
        pending_by_call_id: dict[str, list[int]] = {}

        for message in turn_messages:
            if message.role == "assistant":
                tool_calls = message.tool_calls or []
                for call in tool_calls:
                    arguments = call.function.arguments if isinstance(call.function.arguments, dict) else {}
                    formatted_args = ", ".join(
                        f"{key}={self._truncate_text(self._stringify_message_content(value), 60)}"
                        for key, value in list(arguments.items())[:3]
                    )
                    trace.append(
                        {
                            "name": call.function.name,
                            "arguments": formatted_args,
                            "result": "",
                        }
                    )
                    index = len(trace) - 1
                    pending_indices.append(index)
                    if call.id:
                        pending_by_call_id.setdefault(call.id, []).append(index)
                continue

            if message.role != "tool" or not trace:
                continue

            result_text = self._stringify_message_content(message.content).strip()
            if not result_text:
                continue

            trace_index: int | None = None
            if message.tool_call_id:
                matching_indices = pending_by_call_id.get(message.tool_call_id) or []
                if matching_indices:
                    trace_index = matching_indices.pop(0)

            if trace_index is None and pending_indices:
                while pending_indices and trace[pending_indices[0]]["result"]:
                    pending_indices.pop(0)
                if pending_indices:
                    trace_index = pending_indices.pop(0)

            if trace_index is not None and not trace[trace_index]["result"]:
                trace[trace_index]["result"] = result_text

        return trace

    def _render_round_digest(self, round_data: ConversationRound, round_num: int) -> str:
        """Render a compact, structured digest for an archived round."""
        user_request = self._truncate_text(self._stringify_message_content(round_data.user_message.content), 240)
        lines = [f"Round {round_num}", f"User request: {user_request}"]

        trace = self._collect_turn_trace(round_data.execution_messages)
        if trace:
            lines.append("Tool trace:")
            for step in trace[:8]:
                args = f" ({step['arguments']})" if step["arguments"] else ""
                result = self._truncate_text(step["result"] or "(no recorded tool result)", 180)
                lines.append(f"- {step['name']}{args}: {result}")

        final_assistant = ""
        for message in reversed(round_data.execution_messages):
            if message.role == "assistant":
                final_assistant = self._stringify_message_content(message.content).strip()
                if final_assistant:
                    break
            elif self._is_summary_artifact(message):
                final_assistant = self._extract_summary_body(message)
                if final_assistant:
                    break

        if final_assistant:
            lines.append(f"Final assistant result: {self._truncate_text(final_assistant, 240)}")

        return "\n".join(lines)

    async def _create_anchor_summary(self, existing_anchor: str, archived_rounds: list[ConversationRound]) -> str:
        """Merge older rounds into a rolling summary anchor."""
        if not archived_rounds:
            return existing_anchor.strip()

        archived_text = "\n\n".join(
            self._render_round_digest(round_data, index + 1)
            for index, round_data in enumerate(archived_rounds)
        )

        try:
            summary_prompt = f"""You maintain a rolling conversation summary anchor for a coding-agent session.

Existing anchor summary:
{existing_anchor.strip() or "(none)"}

Older rounds to merge:
{archived_text}

Return an updated anchor summary with these sections:
- User goals and constraints
- Work completed and verified outcomes
- Important artifacts (files, commands, identifiers)
- Open issues / next steps

Requirements:
1. Preserve durable facts, decisions, and verified results.
2. Mention exact filenames, symbols, and commands when they matter.
3. Omit verbose tool output and transient logs.
4. Keep it concise, factual, and under 1200 words.
5. Do not invent information."""

            response = await self.llm.generate(
                messages=[
                    Message(
                        role="system",
                        content="You maintain concise rolling summaries for long coding-agent conversations.",
                    ),
                    Message(role="user", content=summary_prompt),
                ]
            )

            summary_text = self._stringify_message_content(response.content).strip()
            if summary_text:
                print(f"{Colors.BRIGHT_GREEN}✓ Updated rolling conversation anchor{Colors.RESET}")
                return summary_text
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Anchor summary generation failed: {e}{Colors.RESET}")

        fallback_sections: list[str] = []
        if existing_anchor.strip():
            fallback_sections.append(f"Previous anchor:\n{existing_anchor.strip()}")
        fallback_sections.extend(
            self._render_round_digest(round_data, index + 1)
            for index, round_data in enumerate(archived_rounds)
        )
        return "\n\n".join(fallback_sections).strip()

    async def _summarize_messages(self):
        """Compact message history when the context window gets too large.

        Strategy:
        - Merge older rounds into a single rolling anchor summary.
        - Keep the most recent N rounds verbatim when possible.
        - If there is only one round left, replace its execution trace with a compact round summary.

        Summary is triggered when EITHER:
        - Local token estimation exceeds limit
        - API reported total_tokens exceeds limit
        """
        # Skip check if we just completed a summary (wait for next LLM call to update api_total_tokens)
        if self._skip_next_token_check:
            self._skip_next_token_check = False
            return

        estimated_tokens = self._estimate_tokens()

        # Check both local estimation and API reported tokens
        should_summarize = estimated_tokens > self.token_limit or self.api_total_tokens > self.token_limit

        # If neither exceeded, no summary needed
        if not should_summarize:
            return

        print(
            f"\n{Colors.BRIGHT_YELLOW}📊 Token usage - Local estimate: {estimated_tokens}, API reported: {self.api_total_tokens}, Limit: {self.token_limit}{Colors.RESET}"
        )
        print(f"{Colors.BRIGHT_YELLOW}🔄 Triggering message history summarization...{Colors.RESET}")

        existing_anchor, conversation_rounds = self._collect_conversation_rounds()

        if not conversation_rounds:
            print(f"{Colors.BRIGHT_YELLOW}⚠️  Insufficient messages, cannot summarize{Colors.RESET}")
            return

        new_messages = [self.messages[0]]
        anchor_in_use = False
        retained_rounds = 0

        if len(conversation_rounds) == 1:
            round_data = conversation_rounds[0]
            summary_text = await self._create_summary(round_data.execution_messages, 1)
            if not summary_text:
                print(f"{Colors.BRIGHT_YELLOW}⚠️  Current round has nothing to summarize{Colors.RESET}")
                return

            if existing_anchor.strip():
                new_messages.append(self._build_anchor_message(existing_anchor.strip()))
                anchor_in_use = True
            new_messages.append(round_data.user_message)
            new_messages.append(self._build_round_summary_message(summary_text))
            retained_rounds = 1
        else:
            if len(conversation_rounds) > self.summary_recent_rounds:
                raw_recent_rounds = conversation_rounds[-self.summary_recent_rounds :] if self.summary_recent_rounds > 0 else []
                archived_rounds = conversation_rounds[: len(conversation_rounds) - len(raw_recent_rounds)]
            else:
                raw_recent_rounds = conversation_rounds[-1:]
                archived_rounds = conversation_rounds[:-1]

            anchor_text = await self._create_anchor_summary(existing_anchor, archived_rounds)
            if anchor_text:
                new_messages.append(self._build_anchor_message(anchor_text))
                anchor_in_use = True

            for round_data in raw_recent_rounds:
                new_messages.append(round_data.user_message)
                new_messages.extend(round_data.execution_messages)

            retained_rounds = len(raw_recent_rounds)

        self.messages = new_messages

        # Skip next token check to avoid consecutive summary triggers
        # (api_total_tokens will be updated after next LLM call)
        self._skip_next_token_check = True

        new_tokens = self._estimate_tokens()
        print(f"{Colors.BRIGHT_GREEN}✓ Summary completed, local tokens: {estimated_tokens} → {new_tokens}{Colors.RESET}")
        structure_parts = []
        if anchor_in_use:
            structure_parts.append("rolling anchor")
        if retained_rounds:
            structure_parts.append(f"{retained_rounds} recent round(s)")
        print(f"{Colors.DIM}  Structure: system + {' + '.join(structure_parts) if structure_parts else 'compressed history'}{Colors.RESET}")
        print(f"{Colors.DIM}  Note: API token count will update on next LLM call{Colors.RESET}")

    async def _create_summary(self, messages: list[Message], round_num: int) -> str:
        """Create summary for one execution round

        Args:
            messages: List of messages to summarize
            round_num: Round number

        Returns:
            Summary text
        """
        if not messages:
            return ""

        # Build summary content
        summary_content = f"Round {round_num} execution process:\n\n"
        for msg in messages:
            if msg.role == "assistant":
                content_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"Assistant: {content_text}\n"
                if msg.tool_calls:
                    tool_names = [tc.function.name for tc in msg.tool_calls]
                    summary_content += f"  → Called tools: {', '.join(tool_names)}\n"
            elif msg.role == "tool":
                result_preview = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"  ← Tool returned: {result_preview}...\n"

        # Call LLM to generate concise summary
        try:
            summary_prompt = f"""Please provide a concise summary of the following Agent execution process:

{summary_content}

Requirements:
1. Focus on what tasks were completed and which tools were called
2. Keep key execution results and important findings
3. Be concise and clear, within 1000 words
4. Use English
5. Do not include "user" related content, only summarize the Agent's execution process"""

            summary_msg = Message(role="user", content=summary_prompt)
            response = await self.llm.generate(
                messages=[
                    Message(
                        role="system",
                        content="You are an assistant skilled at summarizing Agent execution processes.",
                    ),
                    summary_msg,
                ]
            )

            summary_text = response.content
            print(f"{Colors.BRIGHT_GREEN}✓ Summary for round {round_num} generated successfully{Colors.RESET}")
            return summary_text

        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Summary generation failed for round {round_num}: {e}{Colors.RESET}")
            # Use simple text summary on failure
            return summary_content

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        """Execute agent loop until task completes, is cancelled, or reaches an optional step limit.

        Args:
            cancel_event: Optional asyncio.Event that can be set to cancel execution.
                          When set, the agent will stop at the next safe checkpoint
                          (after completing the current step to keep messages consistent).

        Returns:
            The final response content, or error message (including cancellation message).
        """
        # Set cancellation event (can also be set via self.cancel_event before calling run())
        if cancel_event is not None:
            self.cancel_event = cancel_event

        # Start new run, initialize log file
        self.logger.start_new_run()
        print(f"{Colors.DIM}📝 Log file: {self.logger.get_log_file_path()}{Colors.RESET}")

        step = 0
        run_start_time = perf_counter()

        while self.max_steps is None or step < self.max_steps:
            # Check for cancellation at start of each step
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                cancel_msg = "Task cancelled by user."
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {cancel_msg}{Colors.RESET}")
                return cancel_msg

            step_start_time = perf_counter()
            # Check and summarize message history to prevent context overflow
            await self._summarize_messages()

            # Step header with proper width calculation
            BOX_WIDTH = 58
            step_limit = str(self.max_steps) if self.max_steps is not None else "∞"
            step_text = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}💭 Step {step + 1}/{step_limit}{Colors.RESET}"
            step_display_width = calculate_display_width(step_text)
            padding = max(0, BOX_WIDTH - 1 - step_display_width)  # -1 for leading space

            print(f"\n{Colors.DIM}╭{'─' * BOX_WIDTH}╮{Colors.RESET}")
            print(f"{Colors.DIM}│{Colors.RESET} {step_text}{' ' * padding}{Colors.DIM}│{Colors.RESET}")
            print(f"{Colors.DIM}╰{'─' * BOX_WIDTH}╯{Colors.RESET}")

            # Get tool list for LLM call
            tool_list = list(self.tools.values())
            active_messages = self._get_active_messages()

            # Log LLM request and call LLM with Tool objects directly
            self.logger.log_request(messages=active_messages, tools=tool_list)

            try:
                response = await self.llm.generate(messages=active_messages, tools=tool_list)
            except Exception as e:
                # Check if it's a retry exhausted error
                from .retry import RetryExhaustedError

                if isinstance(e, RetryExhaustedError):
                    error_msg = f"LLM call failed after {e.attempts} retries\nLast error: {str(e.last_exception)}"
                    print(f"\n{Colors.BRIGHT_RED}❌ Retry failed:{Colors.RESET} {error_msg}")
                else:
                    error_msg = f"LLM call failed: {str(e)}"
                    print(f"\n{Colors.BRIGHT_RED}❌ Error:{Colors.RESET} {error_msg}")
                return error_msg

            # Accumulate API reported token usage
            if response.usage:
                self.api_total_tokens = response.usage.total_tokens

            # Log LLM response
            self.logger.log_response(
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
            )

            # Add assistant message
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
            )
            self.messages.append(assistant_msg)

            # Print thinking if present
            if response.thinking:
                print(f"\n{Colors.BOLD}{Colors.MAGENTA}🧠 Thinking:{Colors.RESET}")
                print(f"{Colors.DIM}{response.thinking}{Colors.RESET}")

            # Print assistant response
            if response.content:
                print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
                print(f"{response.content}")

            # Check if task is complete (no tool calls)
            if not response.tool_calls:
                step_elapsed = perf_counter() - step_start_time
                total_elapsed = perf_counter() - run_start_time
                print(f"\n{Colors.DIM}⏱️  Step {step + 1} completed in {step_elapsed:.2f}s (total: {total_elapsed:.2f}s){Colors.RESET}")
                return response.content

            # Check for cancellation before executing tools
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                cancel_msg = "Task cancelled by user."
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {cancel_msg}{Colors.RESET}")
                return cancel_msg

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments

                # Tool call header
                print(f"\n{Colors.BRIGHT_YELLOW}🔧 Tool Call:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}{function_name}{Colors.RESET}")

                # Arguments (formatted display)
                print(f"{Colors.DIM}   Arguments:{Colors.RESET}")
                # Truncate each argument value to avoid overly long output
                truncated_args = {}
                for key, value in arguments.items():
                    value_str = str(value)
                    if len(value_str) > 200:
                        truncated_args[key] = value_str[:200] + "..."
                    else:
                        truncated_args[key] = value
                args_json = json.dumps(truncated_args, indent=2, ensure_ascii=False)
                for line in args_json.split("\n"):
                    print(f"   {Colors.DIM}{line}{Colors.RESET}")

                # Execute tool
                if function_name not in self.tools:
                    result = ToolResult(
                        success=False,
                        content="",
                        error=f"Unknown tool: {function_name}",
                    )
                else:
                    try:
                        tool = self.tools[function_name]
                        result = await tool.execute(**arguments)
                    except Exception as e:
                        # Catch all exceptions during tool execution, convert to failed ToolResult
                        import traceback

                        error_detail = f"{type(e).__name__}: {str(e)}"
                        error_trace = traceback.format_exc()
                        result = ToolResult(
                            success=False,
                            content="",
                            error=f"Tool execution failed: {error_detail}\n\nTraceback:\n{error_trace}",
                        )

                prepared_content = self.tool_output_store.prepare(
                    result.content,
                    label=f"{function_name} output",
                )
                prepared_error = (
                    self.tool_output_store.prepare(result.error, label=f"{function_name} error")
                    if result.error
                    else None
                )
                tool_result_content = prepared_content.preview
                tool_result_error = prepared_error.preview if prepared_error else result.error

                # Log tool execution result
                self.logger.log_tool_result(
                    tool_name=function_name,
                    arguments=arguments,
                    result_success=result.success,
                    result_content=tool_result_content if result.success else None,
                    result_error=tool_result_error if not result.success else None,
                )

                # Print result
                if result.success:
                    result_text = tool_result_content
                    if len(result_text) > 300:
                        result_text = result_text[:300] + f"{Colors.DIM}...{Colors.RESET}"
                    print(f"{Colors.BRIGHT_GREEN}✓ Result:{Colors.RESET} {result_text}")
                else:
                    print(f"{Colors.BRIGHT_RED}✗ Error:{Colors.RESET} {Colors.RED}{tool_result_error}{Colors.RESET}")

                # Add tool result message
                tool_msg = Message(
                    role="tool",
                    content=tool_result_content if result.success else f"Error: {tool_result_error}",
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
                self.messages.append(tool_msg)

                # Check for cancellation after each tool execution
                if self._check_cancelled():
                    self._cleanup_incomplete_messages()
                    cancel_msg = "Task cancelled by user."
                    print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {cancel_msg}{Colors.RESET}")
                    return cancel_msg

            step_elapsed = perf_counter() - step_start_time
            total_elapsed = perf_counter() - run_start_time
            print(f"\n{Colors.DIM}⏱️  Step {step + 1} completed in {step_elapsed:.2f}s (total: {total_elapsed:.2f}s){Colors.RESET}")

            step += 1

        # Max steps reached
        error_msg = f"Task couldn't be completed after {self.max_steps} steps."
        print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {error_msg}{Colors.RESET}")
        return error_msg

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()

    async def cleanup(self) -> None:
        """Release runtime resources that may outlive the last turn on Windows."""
        from .tools.bash_tool import cleanup_background_shells

        await cleanup_background_shells()
