# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the editing conversation with Claude or Codex through the Blender MCP tools.

Trust boundary: ``execute_blender_code`` runs whatever Python the model writes inside
Blender, with the permissions of the user running the edit (files, network, processes),
and is approved automatically. Codex's ``workspace_write`` sandbox and ``deny_all``
approval apply only to Codex's own shell and file tools, not to Blender. Claude gets no
built-in tools. See docs/agents.md#security.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from threading import Timer
from time import monotonic
from typing import cast

from .assets import AssetValidationError

BLENDER_TOOLS = ('execute_blender_code', 'get_scene_info', 'look')
DEFAULT_CODEX_TOKEN_BUDGET = 100_000
DEFAULT_MAX_TURNS = 30
DEFAULT_BUDGET_USD = 5.0

INSTRUCTIONS = """Edit the supplied I4H scene using Blender MCP. Inspect the scene and a
viewport screenshot before and after changes (use look for images). Only edit this session's scene.
Each execute_blender_code call has a fresh namespace: import dependencies and
resolve scenes/objects within that call, or save reusable functions in a workspace
module. Keep inspection images at max_size=768 or smaller unless detail requires
more. Save required reference/result comparison images in the workspace.
Batch related scene queries and geometry measurements in one Python call. Once
the target, placement and necessary reference evidence are understood, make the
first supported local edit. Further inspection should answer a concrete question
or verify a change. Reuse unchanged views; avoid repeated planning and scene dumps.
Keep the existing mesh objects and their mesh-data i4h_path properties: adding,
deleting, splitting or joining objects is unsupported. Geometry edits, modifiers,
UVs and USD Preview Surface compatible materials/textures are supported. Use
Principled BSDF and image textures; bake unsupported procedural shaders first.
Do not change scene units. Leave Object Mode active when finished. Do not export
or modify source files: the application validates and publishes the result.
Before finishing, run the supplied check_asset command through execute_blender_code.
Fix any reported scene errors and check again before finishing.
Use the available budget for bounded edits, not repeated inspection of unchanged views.
When budget reminders arrive, reserve the final quarter for required comparison images,
reports and check_asset. Report unresolved work honestly; do not relax validation to finish.
"""


def _edit_attempts(prompt: str, publish: Callable[[], None] | None, max_repairs: int) -> Iterator[str]:
    """Yield prompts and publish each completed attempt."""
    for attempt in range(max_repairs + 1):
        yield prompt
        if publish is None:
            return
        try:
            publish()
            return
        except AssetValidationError as exc:
            if attempt == max_repairs:
                raise
            print(f'Validation failed; repair {attempt + 1}/{max_repairs}: {exc}', flush=True)
            prompt = (f'The edited asset failed validation:\n{exc}\n'
                      'Repair the current scene while preserving the requested edits and original identities. '
                      'Run the scene check before finishing. The application will retry export and publication.')


def run_claude(prompt: str, workspace: Path, config: dict, model: str | None,
               max_turns: int, budget_usd: float, timeout_s: float = 900, *,
               publish: Callable[[], None] | None = None, max_repairs: int = 2) -> None:
    """Stream Claude turns, resuming the conversation with the remaining limits."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
    from claude_agent_sdk.types import McpStdioServerConfig

    options = ClaudeAgentOptions(
        cwd=str(workspace), system_prompt=INSTRUCTIONS, model=model,
        tools=[], allowed_tools=["mcp__blender__" + name for name in BLENDER_TOOLS], mcp_servers={"blender": cast(McpStdioServerConfig, config)},
        setting_sources=[], max_turns=max_turns, max_budget_usd=budget_usd,
    )

    async def turn():
        result = None
        async for message in query(prompt=prompt, options=options):
            for block in getattr(message, "content", []):
                if text := getattr(block, "text", None):
                    print(text, flush=True)
            if isinstance(message, ResultMessage):
                result = message
                if message.is_error or message.subtype != "success":
                    raise RuntimeError(f"Claude session ended: {message.subtype}")
        if result is None:
            raise RuntimeError("Claude session ended without a result")
        return result

    remaining_turns, spent = max_turns, 0.
    deadline = monotonic() + timeout_s
    for prompt in _edit_attempts(prompt, publish, max_repairs):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('Agent editing timed out')
        if remaining_turns <= 0 or spent >= budget_usd:
            raise RuntimeError('Claude session limits exhausted; saved edits retained')
        result = asyncio.run(asyncio.wait_for(turn(), remaining))
        remaining_turns -= result.num_turns
        # num_turns counts this invocation only, but total_cost_usd is cumulative for a
        # resumed session and is what the SDK compares with max_budget_usd. Keep the
        # original cap (verified with claude-agent-sdk 0.2.188); an unknown cost ends the loop.
        spent = result.total_cost_usd if result.total_cost_usd is not None else budget_usd
        options = replace(options, resume=result.session_id, max_turns=remaining_turns)


def run_codex(prompt: str, workspace: Path, config: dict, model: str | None, timeout_s: float = 900,
              token_budget: int = DEFAULT_CODEX_TOKEN_BUDGET, *,
              publish: Callable[[], None] | None = None, max_repairs: int = 2) -> None:
    """Stream the Codex Python SDK; credentials/config remain owned by Codex."""
    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

    reminders = sorted({max(1, token_budget // divisor) for divisor in (2, 4, 10)}, reverse=True)
    overrides = (
        'features.rollout_budget.enabled=true',
        f'features.rollout_budget.limit_tokens={token_budget}',
        f'features.rollout_budget.reminder_at_remaining_tokens={json.dumps(reminders)}',
        'features.rollout_budget.prefill_token_weight=1.0',
        'features.rollout_budget.sampling_token_weight=1.0',
        f'mcp_servers.blender.command={json.dumps(config["command"])}',
        f'mcp_servers.blender.args={json.dumps(config["args"])}',
        *(f'mcp_servers.blender.env.{key}={json.dumps(value)}' for key, value in config.get('env', {}).items()),
    )
    # Expose only the Blender tools Claude also allows, auto-approved, without changing user
    # policy or enabling other MCP servers. The sandbox and deny_all approval below govern
    # Codex's own shell and file tools; they do not contain execute_blender_code.
    overrides += (f'mcp_servers.blender.enabled_tools={json.dumps(BLENDER_TOOLS)}',)
    overrides += tuple(f'mcp_servers.blender.tools.{name}.approval_mode="approve"' for name in BLENDER_TOOLS)
    with Codex(config=CodexConfig(cwd=str(workspace), config_overrides=overrides)) as codex:
        thread = codex.thread_start(cwd=str(workspace), model=model,
                                    developer_instructions=INSTRUCTIONS,
                                    sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all)
        deadline = monotonic() + timeout_s
        for prompt in _edit_attempts(prompt, publish, max_repairs):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError('Agent editing timed out')
            turn = thread.turn(prompt)
            def interrupt():
                with contextlib.suppress(Exception):
                    turn.interrupt()
            timer = Timer(remaining, interrupt)
            timer.daemon = True
            timer.start()
            completed = False
            try:
                for event in turn.stream():
                    method = getattr(event, "method", "")
                    params = getattr(event, "payload", None)
                    if method == "item/agentMessage/delta":
                        print(getattr(params, "delta", ""), end="", flush=True)
                    if method == "turn/completed":
                        result = getattr(params, "turn", None)
                        status = getattr(result, "status", None)
                        if getattr(status, "value", status) != "completed":
                            error = getattr(result, "error", None)
                            raise RuntimeError(error.message if error else f"Codex turn ended: {status}")
                        completed = True
            except BaseException:
                interrupt()
                raise
            finally:
                timer.cancel()
            if not completed:
                raise RuntimeError("Codex session ended without a completed turn")
            print()
