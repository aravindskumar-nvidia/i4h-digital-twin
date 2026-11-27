# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One agent editing session: prepare a workspace, open it in Blender, edit, then publish."""
from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import timedelta
from pathlib import Path

from . import AgentOptions, SessionOptions
from .assets import AssetValidationError, finalize, load_workspace, prepare
from .backends import (
    DEFAULT_BUDGET_USD,
    DEFAULT_CODEX_TOKEN_BUDGET,
    DEFAULT_MAX_TURNS,
    run_claude,
    run_codex,
)
from .completion import prepare_references
from .prompts import build_prompt, session_prompt
from .runtime import MCP_PACKAGE, blender_session

WORKER = Path(__file__).with_name('blender.py')


def mcp_config() -> dict:
    """Use the pinned upstream server; runtime selects an existing or managed Blender."""
    return {"command": "uvx", "args": ["--from", MCP_PACKAGE, "mcp-for-blender"],
            "env": {"BLENDER_MCP_DISABLE_TELEMETRY": "true",
                    **{key: os.environ[key] for key in ("BLENDER_HOST", "BLENDER_PORT") if key in os.environ}}}


async def blender_call(config: dict, function: str, workspace: Path) -> None:
    """Run one ``agent/blender.py`` function in Blender, raising its validation errors."""
    # The MCP client is needed only once a session talks to Blender.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    marker = workspace / ('edited.usdc' if function == 'export_asset' else 'opened')
    marker.unlink(missing_ok=True)
    failure = workspace / 'validation-error.txt'
    failure.unlink(missing_ok=True)
    call = f"{function}({str(workspace)!r})"
    if function == 'export_asset':
        call = (f"try:\n    {call}\n"
                f"except ValueError as exc:\n    Path({str(failure)!r}).write_text(str(exc))\n")
    code = f"exec(compile(open({str(WORKER)!r}).read(), {str(WORKER)!r}, 'exec'))\n{call}"
    with (workspace / 'mcp.log').open('a') as log:
        async with stdio_client(StdioServerParameters(**config), errlog=log) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120)) as client:
                await client.initialize()
                result = await client.call_tool('execute_blender_code', {'code': code})
    if failure.is_file():
        raise AssetValidationError(failure.read_text())
    if result.isError or not marker.is_file():
        raise RuntimeError(f'Blender {function} failed: {result.content}')


def start(source: str | Path, workspace: Path, options: AgentOptions, *,
          vessel_names: list[str] | None = None) -> tuple[dict, str]:
    """Prepare a new workspace for ``options``; return its settings and saved task prompt."""
    settings = prepare(Path(source), workspace, vessel_names=vessel_names)
    for name in options.targets or options.anatomy_references:
        if '/HumanBody/Anatomy/' + name not in settings['paths']:
            raise ValueError(f'Agent target is not in the patient asset: {name}')
    prompt = build_prompt(options.task, options.targets, options.prompt,
                          references=options.anatomy_references, major_branching=options.include_major_branches)
    if options.anatomy_references:
        prompt += prepare_references(workspace, options.anatomy_references, options.include_major_branches,
                                     options.fragment_area_mm2, operation=options.task)
        settings = load_workspace(workspace)
    (workspace / 'task_prompt.txt').write_text(prompt)
    return settings, prompt


def resume_prompt(workspace: Path, continuation: str | None) -> str:
    """The workspace's saved task, followed by any recovery instructions."""
    prompt = (workspace / 'task_prompt.txt').read_text()
    if continuation:
        prompt += '\n\nAdditional recovery instructions:\n' + continuation
    return prompt


def run(workspace: Path, settings: dict, output: Path, options: SessionOptions, *, prompt: str | None,
        fresh: bool, mcp: dict | None = None) -> None:
    """Open the workspace in Blender and let the agent work on ``prompt``, then publish to ``output``.

    ``fresh`` opens the prepared input; otherwise the saved ``edited.blend``. With no
    prompt, only the scene is exported and published. Every failure first saves the
    current scene, so the edit can be recovered.
    """
    print('Connecting to Blender…', flush=True)
    with blender_session(workspace, mcp or mcp_config(), options.blender_mode) as config:
        asyncio.run(blender_call(config, 'open_asset' if fresh else 'open_checkpoint', workspace))

        def publish():
            print('Exporting edited asset…', flush=True)
            asyncio.run(blender_call(config, 'export_asset', workspace))
            try:
                result = finalize(workspace, output, vessel_names=settings['vessel_names'])
            except ValueError as exc:
                raise AssetValidationError(str(exc)) from exc
            print(result)

        try:
            if prompt is None:
                publish()
                return
            print('Editing asset…', flush=True)
            text = session_prompt(prompt, settings, workspace, WORKER)
            if options.backend == 'claude':
                run_claude(text, workspace, config, options.model, options.max_turns or DEFAULT_MAX_TURNS,
                           options.budget_usd or DEFAULT_BUDGET_USD, options.timeout_s,
                           publish=publish, max_repairs=options.max_repairs)
            else:
                run_codex(text, workspace, config, options.model, options.timeout_s,
                          options.token_budget or DEFAULT_CODEX_TOKEN_BUDGET,
                          publish=publish, max_repairs=options.max_repairs)
        except BaseException:
            # Save the agent's latest scene; the original error is the one to report.
            with contextlib.suppress(Exception):
                asyncio.run(blender_call(config, 'export_asset', workspace))
            raise


def published_path(output: Path, settings: dict) -> Path:
    """The manifest of a published bundle, or the published USD."""
    return output / ('patient_twin.yaml' if settings['manifest'] is not None else Path(settings['source']).name)


def edit_asset(source: str | Path, output: str | Path, options: AgentOptions, *, workspace: str | Path) -> Path:
    """Edit ``source`` with the agent and publish the result to a new ``output`` directory."""
    output, workspace = Path(output).resolve(), Path(workspace).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    workspace.mkdir(parents=True, exist_ok=False)
    settings, prompt = start(source, workspace, options)
    run(workspace, settings, output, options, prompt=prompt, fresh=True)
    return published_path(output, settings)


def resume_asset(workspace: Path, output: Path, options: SessionOptions, *,
                 continuation: str | None = None) -> Path:
    """Reopen the workspace's saved scene, continue its task, and publish to ``output``."""
    settings = load_workspace(workspace)
    if not (workspace / 'edited.blend').is_file():
        raise ValueError('Agent resume requires edited.blend')
    run(workspace, settings, output, options, prompt=resume_prompt(workspace, continuation), fresh=False)
    return published_path(output, settings)
