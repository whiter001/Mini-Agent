"""Runtime orchestration helpers for the Mini-Agent CLI."""

from __future__ import annotations

import asyncio
import platform
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.config import Config
from mini_agent.memory_store import MemoryStore
from mini_agent.retry import RetryConfig as RetryConfigBase
from mini_agent.schema import LLMProvider, Message
from mini_agent.tools.base import Tool
from mini_agent.tools.bash_tool import cleanup_background_shells
from mini_agent.tools.mcp_loader import cleanup_mcp_connections, set_mcp_timeout_config
from mini_agent.tools.auto_skill import build_auto_skill_context, maybe_create_auto_skill
from mini_agent.terminal_ui import Colors, print_help, print_stats, read_log_file, show_log_directory


@dataclass
class RuntimeContext:
    """Runtime state required by prompt and interactive modes."""

    config: Config
    agent: Agent
    workspace_dir: Path
    session_start: datetime
    skill_loader: Any | None = None
    memory_store: MemoryStore | None = None


async def build_runtime_context(
    workspace_dir: Path,
    *,
    initialize_base_tools: Callable[[Config], Awaitable[tuple[list[Tool], Any | None, MemoryStore | None]]],
    add_workspace_tools: Callable[[list[Tool], Config, Path], None],
) -> RuntimeContext | None:
    """Build the runtime context used by CLI prompt and interactive modes."""
    session_start = datetime.now()

    config_path = Config.get_default_config_path()
    if not config_path.exists():
        print(f"{Colors.RED}❌ Configuration file not found{Colors.RESET}")
        print()
        print(f"{Colors.BRIGHT_CYAN}📦 Configuration Search Path:{Colors.RESET}")
        print(f"  {Colors.DIM}1) mini_agent/config/config.yaml{Colors.RESET} (development)")
        print(f"  {Colors.DIM}2) ~/.mini-agent/config/config.yaml{Colors.RESET} (user)")
        print(f"  {Colors.DIM}3) <package>/config/config.yaml{Colors.RESET} (installed)")
        print()
        print(f"{Colors.BRIGHT_YELLOW}🚀 Quick Setup (Recommended):{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_GREEN}curl -fsSL https://raw.githubusercontent.com/MiniMax-AI/Mini-Agent/main/scripts/setup-config.sh | bash{Colors.RESET}"
        )
        print()
        print(f"{Colors.DIM}  This will automatically:{Colors.RESET}")
        print(f"{Colors.DIM}    • Create ~/.mini-agent/config/{Colors.RESET}")
        print(f"{Colors.DIM}    • Download configuration files{Colors.RESET}")
        print(f"{Colors.DIM}    • Guide you to add your API Key{Colors.RESET}")
        print()
        print(f"{Colors.BRIGHT_YELLOW}📝 Manual Setup:{Colors.RESET}")
        user_config_dir = Path.home() / ".mini-agent" / "config"
        example_config = Config.get_package_dir() / "config" / "config-example.yaml"
        print(f"  {Colors.DIM}mkdir -p {user_config_dir}{Colors.RESET}")
        print(f"  {Colors.DIM}cp {example_config} {user_config_dir}/config.yaml{Colors.RESET}")
        print(f"  {Colors.DIM}# Then edit {user_config_dir}/config.yaml to add your API Key{Colors.RESET}")
        print()
        return None

    try:
        config = Config.from_yaml(config_path)
    except FileNotFoundError:
        print(f"{Colors.RED}❌ Error: Configuration file not found: {config_path}{Colors.RESET}")
        return None
    except ValueError as exc:
        print(f"{Colors.RED}❌ Error: {exc}{Colors.RESET}")
        print(f"{Colors.YELLOW}Please check the configuration file format{Colors.RESET}")
        return None
    except Exception as exc:
        print(f"{Colors.RED}❌ Error: Failed to load configuration file: {exc}{Colors.RESET}")
        return None

    retry_config = RetryConfigBase(
        enabled=config.llm.retry.enabled,
        max_retries=config.llm.retry.max_retries,
        initial_delay=config.llm.retry.initial_delay,
        max_delay=config.llm.retry.max_delay,
        exponential_base=config.llm.retry.exponential_base,
        retryable_exceptions=(Exception,),
    )

    def on_retry(exception: Exception, attempt: int) -> None:
        print(f"\n{Colors.BRIGHT_YELLOW}⚠️  LLM call failed (attempt {attempt}): {str(exception)}{Colors.RESET}")
        next_delay = retry_config.calculate_delay(attempt - 1)
        print(f"{Colors.DIM}   Retrying in {next_delay:.1f}s (attempt {attempt + 1})...{Colors.RESET}")

    provider = LLMProvider.ANTHROPIC if config.llm.provider.lower() == "anthropic" else LLMProvider.OPENAI
    llm_client = LLMClient(
        api_key=config.llm.api_key,
        provider=provider,
        api_base=config.llm.api_base,
        model=config.llm.model,
        retry_config=retry_config if config.llm.retry.enabled else None,
        temperature=config.llm.temperature,
    )

    if config.llm.retry.enabled:
        llm_client.retry_callback = on_retry
        print(f"{Colors.GREEN}✅ LLM retry mechanism enabled (max {config.llm.retry.max_retries} retries){Colors.RESET}")

    tools, skill_loader, memory_store = await initialize_base_tools(config)
    add_workspace_tools(tools, config, workspace_dir)

    system_prompt_path = Config.find_config_file(config.agent.system_prompt_path)
    if system_prompt_path and system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text(encoding="utf-8")
        print(f"{Colors.GREEN}✅ Loaded system prompt (from: {system_prompt_path}){Colors.RESET}")
    else:
        system_prompt = "You are Mini-Agent, an intelligent assistant powered by MiniMax M2.5 that can help users complete various tasks."
        print(f"{Colors.YELLOW}⚠️  System prompt not found, using default{Colors.RESET}")

    if skill_loader:
        skills_metadata = skill_loader.get_skills_metadata_prompt()
        if skills_metadata:
            system_prompt = system_prompt.replace("{SKILLS_METADATA}", skills_metadata)
            print(f"{Colors.GREEN}✅ Injected {len(skill_loader.loaded_skills)} skills metadata into system prompt{Colors.RESET}")
        else:
            system_prompt = system_prompt.replace("{SKILLS_METADATA}", "")
    else:
        system_prompt = system_prompt.replace("{SKILLS_METADATA}", "")

    if memory_store:
        memory_prompt = memory_store.build_system_prompt()
        if memory_prompt:
            system_prompt = f"{system_prompt.rstrip()}\n\n{memory_prompt}"
            print(f"{Colors.GREEN}✅ Injected persistent memory into system prompt{Colors.RESET}")

    agent = Agent(
        llm_client=llm_client,
        system_prompt=system_prompt,
        tools=tools,
        max_steps=config.agent.max_steps,
        workspace_dir=str(workspace_dir),
    )

    return RuntimeContext(
        config=config,
        agent=agent,
        workspace_dir=workspace_dir,
        session_start=session_start,
        skill_loader=skill_loader,
        memory_store=memory_store,
    )


def build_turn_context(
    memory_store: MemoryStore | None,
    skill_loader: Any | None,
    query: str,
    max_skills: int,
    enable_auto_skills: bool = True,
) -> list[Message]:
    """Build temporary per-turn context from durable memory and auto-selected skills."""
    context: list[Message] = []
    if memory_store is not None:
        context.extend(memory_store.build_turn_context(query))
    if enable_auto_skills and skill_loader is not None:
        context.extend(build_auto_skill_context(skill_loader, query, max_skills=max_skills))
    return context


def persist_auto_skill_if_needed(
    context: RuntimeContext,
    turn_start_index: int,
    final_result: str,
) -> None:
    """Persist an auto-generated skill when the quality gate approves it."""
    config = context.config
    if not config.tools.enable_auto_skill_creation:
        return

    turn_messages = context.agent.get_history()[turn_start_index:]
    result = maybe_create_auto_skill(
        context.skill_loader,
        turn_messages,
        final_result,
        auto_skill_dir=config.tools.auto_skill_dir,
        min_tool_calls=config.tools.auto_skill_min_tool_calls,
        candidate_score_threshold=config.tools.auto_skill_candidate_score,
        approved_score_threshold=config.tools.auto_skill_approved_score,
    )
    if result.created:
        if context.skill_loader is not None and result.skill_path is not None and result.tier == "approved":
            loaded_skill = context.skill_loader.load_skill(result.skill_path)
            if loaded_skill is not None:
                context.skill_loader.loaded_skills[loaded_skill.name] = loaded_skill
            else:
                context.skill_loader.discover_skills()
        tier_label = result.tier or "approved"
        print(
            f"{Colors.BRIGHT_GREEN}🧠 Auto skill {tier_label} created:{Colors.RESET} "
            f"{result.skill_name} -> {result.skill_path} {Colors.DIM}(score={result.quality_score}){Colors.RESET}"
        )


async def execute_prompt(context: RuntimeContext, prompt: str) -> None:
    """Execute a single prompt and exit."""
    print(f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET} {Colors.DIM}Executing prompt...{Colors.RESET}\n")
    turn_start_index = len(context.agent.messages)
    context.agent.add_user_message(prompt)
    context.agent.set_ephemeral_context(
        build_turn_context(
            context.memory_store,
            context.skill_loader,
            prompt,
            context.config.tools.auto_skills_limit,
            enable_auto_skills=context.config.tools.enable_auto_skills,
        )
    )
    try:
        final_result = await context.agent.run()
        persist_auto_skill_if_needed(context, turn_start_index, final_result)
    except Exception as exc:
        print(f"\n{Colors.RED}❌ Error: {exc}{Colors.RESET}")
    finally:
        context.agent.clear_ephemeral_context()
        print_stats(context.agent, context.session_start)
        await quiet_cleanup()


async def run_interactive_session(context: RuntimeContext) -> None:
    """Run the interactive CLI session."""
    command_completer = WordCompleter(
        ["/help", "/clear", "/history", "/stats", "/log", "/exit", "/quit", "/q"],
        ignore_case=True,
        sentence=True,
    )

    prompt_style = Style.from_dict(
        {
            "prompt": "#00ff00 bold",
            "separator": "#666666",
        }
    )

    key_bindings = KeyBindings()

    @key_bindings.add("c-u")
    def _(event):
        event.current_buffer.reset()

    @key_bindings.add("c-l")
    def _(event):
        event.app.renderer.clear()

    @key_bindings.add("c-j")
    def _(event):
        event.current_buffer.insert_text("\n")

    history_file = Path.home() / ".mini-agent" / ".history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=command_completer,
        style=prompt_style,
        key_bindings=key_bindings,
    )

    while True:
        try:
            user_input = await session.prompt_async(
                [("class:prompt", "You"), ("", " › ")],
                multiline=False,
                enable_history_search=True,
            )
            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                command = user_input.lower()

                if command in ["/exit", "/quit", "/q"]:
                    print(f"\n{Colors.BRIGHT_YELLOW}👋 Goodbye! Thanks for using Mini Agent{Colors.RESET}\n")
                    print_stats(context.agent, context.session_start)
                    break
                if command == "/help":
                    print_help()
                    continue
                if command == "/clear":
                    old_count = len(context.agent.messages)
                    context.agent.messages = [context.agent.messages[0]]
                    print(f"{Colors.GREEN}✅ Cleared {old_count - 1} messages, starting new session{Colors.RESET}\n")
                    continue
                if command == "/history":
                    print(f"\n{Colors.BRIGHT_CYAN}Current session message count: {len(context.agent.messages)}{Colors.RESET}\n")
                    continue
                if command == "/stats":
                    print_stats(context.agent, context.session_start)
                    continue
                if command == "/log" or command.startswith("/log "):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) == 1:
                        show_log_directory(open_file_manager=True)
                    else:
                        filename = parts[1].strip("\"'")
                        read_log_file(filename)
                    continue

                print(f"{Colors.RED}❌ Unknown command: {user_input}{Colors.RESET}")
                print(f"{Colors.DIM}Type /help to see available commands{Colors.RESET}\n")
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"\n{Colors.BRIGHT_YELLOW}👋 Goodbye! Thanks for using Mini Agent{Colors.RESET}\n")
                print_stats(context.agent, context.session_start)
                break

            print(
                f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET} {Colors.DIM}Thinking... (Esc to cancel){Colors.RESET}\n"
            )
            turn_start_index = len(context.agent.messages)
            context.agent.add_user_message(user_input)
            context.agent.set_ephemeral_context(
                build_turn_context(
                    context.memory_store,
                    context.skill_loader,
                    user_input,
                    context.config.tools.auto_skills_limit,
                    enable_auto_skills=context.config.tools.enable_auto_skills,
                )
            )

            cancel_event = asyncio.Event()
            context.agent.cancel_event = cancel_event
            esc_listener_stop = threading.Event()
            esc_cancelled = [False]

            def esc_key_listener() -> None:
                if platform.system() == "Windows":
                    try:
                        import msvcrt

                        while not esc_listener_stop.is_set():
                            if msvcrt.kbhit():
                                char = msvcrt.getch()
                                if char == b"\x1b":
                                    print(f"\n{Colors.BRIGHT_YELLOW}⏹️  Esc pressed, cancelling...{Colors.RESET}")
                                    esc_cancelled[0] = True
                                    cancel_event.set()
                                    break
                            esc_listener_stop.wait(0.05)
                    except Exception:
                        pass
                    return

                try:
                    import select
                    import termios
                    import tty

                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setcbreak(fd)
                        while not esc_listener_stop.is_set():
                            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if rlist:
                                char = sys.stdin.read(1)
                                if char == "\x1b":
                                    print(f"\n{Colors.BRIGHT_YELLOW}⏹️  Esc pressed, cancelling...{Colors.RESET}")
                                    esc_cancelled[0] = True
                                    cancel_event.set()
                                    break
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

            esc_thread = threading.Thread(target=esc_key_listener, daemon=True)
            esc_thread.start()

            try:
                agent_task = asyncio.create_task(context.agent.run())
                while not agent_task.done():
                    if esc_cancelled[0]:
                        cancel_event.set()
                    await asyncio.sleep(0.1)

                final_result = agent_task.result()
                persist_auto_skill_if_needed(context, turn_start_index, final_result)
            except asyncio.CancelledError:
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  Agent execution cancelled{Colors.RESET}")
            finally:
                context.agent.cancel_event = None
                esc_listener_stop.set()
                esc_thread.join(timeout=0.2)
                context.agent.clear_ephemeral_context()

            print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.BRIGHT_YELLOW}👋 Interrupt signal detected, exiting...{Colors.RESET}\n")
            print_stats(context.agent, context.session_start)
            break
        except Exception as exc:
            print(f"\n{Colors.RED}❌ Error: {exc}{Colors.RESET}")
            print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

    await quiet_cleanup()


async def quiet_cleanup() -> None:
    """Clean up MCP connections without noisy asyncgen teardown tracebacks."""
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(lambda _loop, _ctx: None)
    try:
        await cleanup_background_shells()
    except Exception:
        pass
    try:
        await cleanup_mcp_connections()
    except Exception:
        pass
