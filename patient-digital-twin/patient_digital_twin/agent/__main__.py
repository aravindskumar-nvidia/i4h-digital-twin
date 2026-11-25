# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optionally edit pipeline assets through Claude/Codex and upstream Blender MCP."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Timer
from time import monotonic
from typing import cast

from . import AgentOptions
from .assets import AssetValidationError, finalize, load_workspace, prepare
from .runtime import MCP_PACKAGE, blender_session

BLENDER_TOOLS = ('execute_blender_code', 'get_scene_info', 'look')
DEFAULT_CODEX_TOKEN_BUDGET = 100_000

INSTRUCTIONS = """Edit the supplied I4H scene using Blender MCP. Inspect the scene and a
viewport screenshot before and after changes (use look for images). Only edit this session's scene.
Each execute_blender_code call has a fresh namespace: import dependencies and
resolve scenes/objects within that call, or save reusable functions in a workspace
module. Keep inspection images at max_size=768 or smaller unless detail requires
more. Save required reference/result comparison images in the workspace.
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


def mcp_config() -> dict:
    """Use the pinned upstream server; runtime selects an existing or managed Blender."""
    return {"command": "uvx", "args": ["--from", MCP_PACKAGE, "mcp-for-blender"],
            "env": {"BLENDER_MCP_DISABLE_TELEMETRY": "true",
                    **{key: os.environ[key] for key in ("BLENDER_HOST", "BLENDER_PORT") if key in os.environ}}}


async def blender_call(config: dict, function: str, workspace: Path) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    worker = Path(__file__).with_name('blender.py')
    marker = workspace / ('edited.usdc' if function == 'export_asset' else 'opened')
    marker.unlink(missing_ok=True)
    failure = workspace / 'validation-error.txt'
    failure.unlink(missing_ok=True)
    call = f"{function}({str(workspace)!r})"
    if function == 'export_asset':
        call = (f"try:\n    {call}\n"
                f"except ValueError as exc:\n    Path({str(failure)!r}).write_text(str(exc))\n")
    code = f"exec(compile(open({str(worker)!r}).read(), {str(worker)!r}, 'exec'))\n{call}"
    with (workspace / 'mcp.log').open('a') as log:
        async with stdio_client(StdioServerParameters(**config), errlog=log) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120)) as client:
                await client.initialize()
                result = await client.call_tool('execute_blender_code', {'code': code})
    if failure.is_file():
        raise AssetValidationError(failure.read_text())
    if result.isError or not marker.is_file():
        raise RuntimeError(f'Blender {function} failed: {result.content}')


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

    remaining_turns, remaining_budget = max_turns, budget_usd
    deadline = monotonic() + timeout_s
    for prompt in _edit_attempts(prompt, publish, max_repairs):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('Agent editing timed out')
        if remaining_turns <= 0 or remaining_budget <= 0:
            raise RuntimeError('Claude session limits exhausted; saved edits retained')
        result = asyncio.run(asyncio.wait_for(turn(), remaining))
        remaining_turns -= result.num_turns
        remaining_budget = budget_usd - (result.total_cost_usd if result.total_cost_usd is not None else budget_usd)
        options = replace(options, resume=result.session_id,
                          max_turns=remaining_turns, max_budget_usd=remaining_budget)


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
    # These are the capabilities exposed by this application (the same set
    # Claude permits through allowed_tools). Do not change user/global policy
    # or grant access to unrelated MCP servers or future unlisted tools.
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


def parser(command: str = "edit") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, epilog='Recovery commands use finalize or resume with --workspace PATH.')
    parser.set_defaults(command=command)
    if command == 'edit':
        parser.add_argument('--input', type=Path, required=True, help='Pipeline USD or patient_twin.yaml')
    else:
        parser.add_argument('--workspace', type=Path, required=True, help='Retained editing workspace')
    parser.add_argument('--output', type=Path, required=True, help='New final asset directory')
    parser.add_argument('--mcp-config', type=Path, help='JSON stdio server config overriding upstream uvx default')
    parser.add_argument('--blender-mode', choices=('auto', 'virtual', 'existing'),
                        help='Default: reuse/start Blender; virtual starts a private Xvfb GUI; existing never launches')
    parser.add_argument('--vessel-names', nargs='+', help='Vessels used to regenerate existing navigation arrays')
    if command != 'finalize':
        parser.add_argument('--backend', choices=('claude', 'codex'), required=True)
        parser.add_argument('--model')
        parser.add_argument('--timeout', type=float, default=900)
        parser.add_argument('--max-turns', type=int, help='Claude turn limit, default 30')
        parser.add_argument('--budget-usd', type=float, help='Claude dollar budget, default 5')
        parser.add_argument('--token-budget', type=int,
                            help=f'Codex native token budget, default {DEFAULT_CODEX_TOKEN_BUDGET}')
        parser.add_argument('--max-repairs', type=int, default=2, help='Validation repair attempts, default 2; 0 disables')
        parser.add_argument('prompt', nargs='?', default=None if command == 'resume' else '')
    if command == 'edit':
        from .prompts import TASKS
        parser.add_argument('--agent-task', choices=TASKS)
        parser.add_argument('--agent-target', nargs='+')
        parser.add_argument('--anatomy-reference', action='append', default=[], metavar='ORGAN=METADATA')
        parser.add_argument('--major-branching', dest='include_major_branches', action='store_true', default=False,
                            help='Include reference-covered major vessel branches (off by default)')
        parser.add_argument('--fragment-area-mm2', type=float, default=1.0)
    return parser


def _run_blender(args: argparse.Namespace, workspace: Path, settings: dict, output: Path) -> None:
    config = json.loads(args.mcp_config.read_text()) if args.mcp_config else mcp_config()
    mode = args.blender_mode or ('existing' if args.mcp_config else 'auto')
    print('Connecting to Blender…', flush=True)
    with blender_session(workspace, config, mode) as config:
        asyncio.run(blender_call(config, 'open_asset' if args.command == 'edit' else 'open_checkpoint', workspace))

        def publish():
            print('Exporting edited asset…', flush=True)
            asyncio.run(blender_call(config, 'export_asset', workspace))
            try:
                result = finalize(workspace, output, vessel_names=settings['vessel_names'])
            except ValueError as exc:
                raise AssetValidationError(str(exc)) from exc
            print(result)

        try:
            if args.command == 'finalize':
                publish()
                return
            print('Editing asset…', flush=True)
            worker = str(Path(__file__).with_name('blender.py'))
            prompt = (f"Edit the active I4H scene (session {settings['token']}).\n{args.prompt}\n"
                      f"Scene check: exec(compile(open({worker!r}).read(), {worker!r}, 'exec')); "
                      f"check_asset({str(workspace)!r})")
            if settings.get('cleanup_components'):
                prompt += (f"\nCleanup fragment history is in {str(workspace / 'edit.json')!r}, "
                           "under cleanup_components. Review it before editing. Cleanup removed tiny/distant "
                           "secondary islands and retained nearby fragments as separate geometry. Bounds are "
                           "in original structure-local millimeters; component IDs describe the pre-cleanup "
                           "mesh, not current vertex indices. Inspect the current mesh to identify surviving "
                           "pieces. Proximity is evidence for review, not proof of a connection. Preserve "
                           "meaningful nearby fragments; connect them only when the selected operation, "
                           "patient geometry and references support it. Refinement and texture tasks must "
                           "not change connectivity.\n")
            if args.backend == 'claude':
                run_claude(prompt, workspace, config, args.model, args.max_turns or 30,
                           args.budget_usd or 5, args.timeout, publish=publish, max_repairs=args.max_repairs)
            else:
                run_codex(prompt, workspace, config, args.model, args.timeout,
                          args.token_budget or DEFAULT_CODEX_TOKEN_BUDGET,
                          publish=publish, max_repairs=args.max_repairs)
        except BaseException:
            with contextlib.suppress(Exception):
                asyncio.run(blender_call(config, 'export_asset', workspace))
            raise


def edit_asset(source: str | Path, output: str | Path, options: AgentOptions, *, workspace: str | Path) -> Path:
    """Edit through the same SDK lifecycle and repair loop used by the CLI."""
    from ..refinement import prepare_references

    output, workspace = Path(output).resolve(), Path(workspace).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    workspace.mkdir(parents=True, exist_ok=False)
    settings = prepare(Path(source), workspace)
    from .prompts import build_prompt
    prompt = build_prompt(options.task, options.targets, options.prompt,
                          references=options.anatomy_references, major_branching=options.include_major_branches)
    for name in options.targets or options.anatomy_references:
        if '/HumanBody/Anatomy/' + name not in settings['paths']:
            raise ValueError(f'Agent target is not in the patient asset: {name}')
    if options.anatomy_references:
        prompt += prepare_references(workspace, options.anatomy_references,
                                     options.include_major_branches, options.fragment_area_mm2, operation=options.task)
        settings = load_workspace(workspace)
    (workspace / 'task_prompt.txt').write_text(prompt)
    args = argparse.Namespace(
        command='edit', mcp_config=None, blender_mode=options.blender_mode,
        prompt=prompt, backend=options.backend, model=options.model,
        max_turns=options.max_turns, budget_usd=options.budget_usd,
        token_budget=options.token_budget, timeout=options.timeout_s, max_repairs=options.max_repairs,
    )
    _run_blender(args, workspace, settings, output)
    source_path = Path(settings['source'])
    return output / ('patient_twin.yaml' if settings['manifest'] is not None else source_path.name)


def _resume_prompt(workspace: Path, continuation: str | None, fallback: str = '') -> str:
    saved = workspace / 'task_prompt.txt'
    prompt = saved.read_text() if saved.is_file() else fallback
    if continuation:
        prompt += '\n\nAdditional recovery instructions:\n' + continuation
    return prompt


def resume_asset(workspace: Path, output: Path, options: AgentOptions, *,
                 continuation: str | None = None) -> Path:
    """Resume the editor within a parent pipeline, then return to that pipeline."""
    settings = load_workspace(workspace)
    if not (workspace / 'edited.blend').is_file():
        raise ValueError('Agent resume requires edited.blend')
    args = argparse.Namespace(command='resume', mcp_config=None, blender_mode=options.blender_mode,
        prompt=_resume_prompt(workspace, continuation, options.prompt), backend=options.backend, model=options.model,
        max_turns=options.max_turns, budget_usd=options.budget_usd,
        token_budget=options.token_budget, timeout=options.timeout_s, max_repairs=options.max_repairs)
    _run_blender(args, workspace, settings, output)
    return output / ('patient_twin.yaml' if settings['manifest'] is not None else Path(settings['source']).name)


def main(argv: list[str] | None = None) -> int:
    """Optionally edit an existing pipeline output; generation itself is unchanged."""
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in {'finalize', 'resume'} else 'edit'
    cli = parser(command)
    args = cli.parse_args(argv)
    if command != 'finalize':
        if args.backend == 'codex' and (args.max_turns is not None or args.budget_usd is not None):
            cli.error('Codex uses --token-budget. --max-turns and --budget-usd apply only to Claude')
        if args.backend == 'claude' and args.token_budget is not None:
            cli.error('--token-budget applies only to Codex')
        if args.max_repairs < 0:
            cli.error('max-repairs must be nonnegative')
        for value in (args.timeout, args.max_turns, args.budget_usd, args.token_budget):
            if value is not None and not 0 < value < float('inf'):
                cli.error('Session limits must be positive and finite')
    output = args.output.expanduser().resolve()
    if output.exists():
        cli.error('Use a new output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mcp_config and args.blender_mode not in {None, 'existing'}:
        cli.error('--mcp-config requires --blender-mode existing')
    if command == 'edit':
        workspace = Path(tempfile.mkdtemp(prefix='.asset-edit-', dir=output.parent))
    else:
        workspace = args.workspace.expanduser().resolve()
    if command == 'resume' and (workspace.parent / 'pipeline.json').is_file():
        from ..workflow import PatientWorkflow
        overrides = {key: value for key, value in dict(backend=args.backend, model=args.model,
            prompt=args.prompt, timeout_s=args.timeout, max_repairs=args.max_repairs,
            max_turns=args.max_turns, budget_usd=args.budget_usd).items() if value is not None}
        print('Resuming parent patient pipeline, including textures and final export…', flush=True)
        print(PatientWorkflow.resume(workspace.parent, output=output, token_budget=args.token_budget,
                                    blender_mode=args.blender_mode, agent_overrides=overrides))
        return 0
    checkpoint, edited = workspace / 'edited.blend', workspace / 'edited.usdc'
    if command != 'edit':
        if not all((workspace / name).is_file() for name in ('edit.json', 'original.usdc')):
            cli.error('Workspace must contain edit.json and original.usdc')
        if not checkpoint.is_file() and (command == 'resume' or not edited.is_file()):
            cli.error('Resume requires edited.blend. Finalize requires edited.blend or edited.usdc')
    try:
        if command == 'edit':
            settings = prepare(args.input, workspace, vessel_names=args.vessel_names)
            from ..references import parse_references
            from ..refinement import prepare_references
            references = parse_references(args.anatomy_reference)
            from .prompts import build_prompt
            args.prompt = build_prompt(args.agent_task, args.agent_target, args.prompt,
                                       references=references, major_branching=args.include_major_branches)
            for name in args.agent_target or references:
                if '/HumanBody/Anatomy/' + name not in settings['paths']:
                    raise ValueError(f'Agent target is not in the patient asset: {name}')
            if references or args.include_major_branches:
                args.prompt += prepare_references(workspace, references, args.include_major_branches,
                                                  args.fragment_area_mm2, operation=args.agent_task)
                settings = load_workspace(workspace)
            (workspace / 'task_prompt.txt').write_text(args.prompt)
        else:
            settings = load_workspace(workspace, vessel_names=args.vessel_names)
            if command == 'resume':
                args.prompt = _resume_prompt(workspace, args.prompt)
        needs_export = (command != 'finalize' or not edited.is_file()
                        or (checkpoint.is_file() and checkpoint.stat().st_mtime_ns > edited.stat().st_mtime_ns))
        if needs_export:
            _run_blender(args, workspace, settings, output)
        else:
            print(finalize(workspace, output, vessel_names=settings['vessel_names']))
    except AssetValidationError as exc:
        print(f'Publication stopped. {exc}', file=sys.stderr)
        print(f'Saved edits are retained at {workspace}', file=sys.stderr)
        return 1
    except BaseException:
        print(f'Edit failed; recovery files retained at {workspace}', file=sys.stderr)
        raise
    else:
        if command == 'edit':
            shutil.rmtree(workspace)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
