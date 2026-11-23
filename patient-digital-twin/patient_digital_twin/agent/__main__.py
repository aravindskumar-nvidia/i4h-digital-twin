# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optionally edit pipeline assets through Claude/Codex and upstream Blender MCP."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from threading import Timer
from typing import cast

from .assets import finalize, prepare

BLENDER_TOOLS = ('execute_blender_code', 'get_scene_info', 'get_object_info', 'get_viewport_screenshot')
DEFAULT_CODEX_TOKEN_BUDGET = 100_000

INSTRUCTIONS = """Edit the supplied I4H scene using Blender MCP. Inspect the scene and a
viewport screenshot before and after changes. Only edit this session's scene.
Keep the existing mesh objects and their mesh-data i4h_path properties: adding,
deleting, splitting or joining objects is unsupported. Geometry edits, modifiers,
UVs and USD Preview Surface compatible materials/textures are supported. Use
Principled BSDF and image textures; bake unsupported procedural shaders first.
Do not change scene units. Leave Object Mode active when finished. Do not export
or modify source files: the application validates and publishes the result.
Before finishing, run the supplied check_asset command through execute_blender_code.
Fix any reported scene errors and check again before finishing.
"""


def mcp_config() -> dict:
    """Use the upstream server and the user's already-running Blender add-on."""
    return {"command": "uvx", "args": ["mcp-for-blender"],
            "env": {"BLENDER_MCP_DISABLE_TELEMETRY": "true",
                    **{key: os.environ[key] for key in ("BLENDER_HOST", "BLENDER_PORT") if key in os.environ}}}


async def blender_call(config: dict, function: str, workspace: Path) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    worker = Path(__file__).with_name('blender.py')
    marker = workspace / ('edited.usdc' if function == 'export_asset' else 'opened')
    marker.unlink(missing_ok=True)
    code = f"exec(compile(open({str(worker)!r}).read(), {str(worker)!r}, 'exec')); {function}({str(workspace)!r})"
    with (workspace / 'mcp.log').open('a') as log:
        async with stdio_client(StdioServerParameters(**config), errlog=log) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120)) as client:
                await client.initialize()
                result = await client.call_tool('execute_blender_code', {'code': code})
                if result.isError or not marker.is_file():
                    raise RuntimeError(f'Blender {function} failed: {result.content}')


async def run_claude(prompt: str, workspace: Path, config: dict, model: str | None,
                     max_turns: int, budget_usd: float) -> None:
    """Stream a bounded Claude Agent SDK session using its normal authentication."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
    from claude_agent_sdk.types import McpStdioServerConfig

    options = ClaudeAgentOptions(
        cwd=str(workspace), system_prompt=INSTRUCTIONS, model=model,
        tools=[], allowed_tools=["mcp__blender__" + name for name in BLENDER_TOOLS], mcp_servers={"blender": cast(McpStdioServerConfig, config)},
        setting_sources=[], max_turns=max_turns, max_budget_usd=budget_usd,
    )
    finished = False
    async for message in query(prompt=prompt, options=options):
        for block in getattr(message, "content", []):
            if text := getattr(block, "text", None):
                print(text, flush=True)
        if isinstance(message, ResultMessage):
            finished = True
            if message.is_error or message.subtype != "success":
                raise RuntimeError(f"Claude session ended: {message.subtype}")
    if not finished:
        raise RuntimeError("Claude session ended without a result")


def run_codex(prompt: str, workspace: Path, config: dict, model: str | None, timeout_s: float = 600,
              token_budget: int = DEFAULT_CODEX_TOKEN_BUDGET) -> None:
    """Stream the Codex Python SDK; credentials/config remain owned by Codex."""
    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

    overrides = (
        'features.rollout_budget.enabled=true',
        f'features.rollout_budget.limit_tokens={token_budget}',
        'features.rollout_budget.reminder_at_remaining_tokens=[]',
        'features.rollout_budget.prefill_token_weight=1.0',
        'features.rollout_budget.sampling_token_weight=1.0',
        f'mcp_servers.blender.command={json.dumps(config["command"])}',
        f'mcp_servers.blender.args={json.dumps(config["args"])}',
        *(f'mcp_servers.blender.env.{key}={json.dumps(value)}' for key, value in config.get('env', {}).items()),
    )
    # These are the capabilities exposed by this application (the same set
    # Claude permits through allowed_tools). Do not change user/global policy
    # or grant access to unrelated MCP servers or future unlisted tools.
    overrides += (f'mcp_servers.blender.enabled_tools={json.dumps(BLENDER_TOOLS)}',)
    overrides += tuple(f'mcp_servers.blender.tools.{name}.approval_mode="approve"' for name in BLENDER_TOOLS)
    with Codex(config=CodexConfig(cwd=str(workspace), config_overrides=overrides)) as codex:
        thread = codex.thread_start(cwd=str(workspace), model=model,
                                    developer_instructions=INSTRUCTIONS,
                                    sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all)
        turn = thread.turn(prompt)
        def interrupt():
            with contextlib.suppress(Exception):
                turn.interrupt()
        timer = Timer(timeout_s, interrupt)
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


def main(argv: list[str] | None = None) -> int:
    """Optionally edit an existing pipeline output; generation itself is unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('claude', 'codex'), required=True)
    parser.add_argument('--input', type=Path, required=True, help='Pipeline USD or patient_twin.yaml')
    parser.add_argument('--output', type=Path, required=True, help='New final asset directory')
    parser.add_argument('--mcp-config', type=Path, help='JSON stdio server config overriding upstream uvx default')
    parser.add_argument('--vessel-names', nargs='+', help='Vessels used to regenerate existing navigation arrays')
    parser.add_argument('--model')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--max-turns', type=int, help='Claude turn limit, default 30')
    parser.add_argument('--budget-usd', type=float, help='Claude dollar budget, default 5')
    parser.add_argument('--token-budget', type=int, help=f'Codex native token budget, default {DEFAULT_CODEX_TOKEN_BUDGET}')
    parser.add_argument('prompt')
    args = parser.parse_args(argv)
    if args.backend == 'codex' and (args.max_turns is not None or args.budget_usd is not None):
        parser.error('Codex uses --token-budget. --max-turns and --budget-usd apply only to Claude')
    if args.backend == 'claude' and args.token_budget is not None:
        parser.error('--token-budget applies only to Codex')
    for value in (args.timeout, args.max_turns, args.budget_usd, args.token_budget):
        if value is not None and not 0 < value < float('inf'):
            parser.error('Session limits must be positive and finite')
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('Use a new output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.mcp_config.read_text()) if args.mcp_config else mcp_config()
    # Same host/filesystem as Blender; the temporary files never replace the input.
    with tempfile.TemporaryDirectory(prefix='.asset-edit-', dir=output.parent) as temp:
        workspace = Path(temp)
        settings = prepare(args.input, workspace, vessel_names=args.vessel_names)
        asyncio.run(blender_call(config, 'open_asset', workspace))
        worker = str(Path(__file__).with_name('blender.py'))
        prompt = (f"Edit the active I4H scene (session {settings['token']}).\n{args.prompt}\n"
                  f"Scene check: exec(compile(open({worker!r}).read(), {worker!r}, 'exec')); "
                  f"check_asset({str(workspace)!r})")
        if args.backend == 'claude':
            async def bounded():
                await asyncio.wait_for(run_claude(prompt, workspace, config, args.model,
                                                 args.max_turns or 30, args.budget_usd or 5), args.timeout)
            asyncio.run(bounded())
        else:
            run_codex(prompt, workspace, config, args.model, args.timeout, args.token_budget or DEFAULT_CODEX_TOKEN_BUDGET)
        asyncio.run(blender_call(config, 'export_asset', workspace))
        print(finalize(workspace, output, vessel_names=settings['vessel_names']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
