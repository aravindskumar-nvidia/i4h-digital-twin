# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edit a patient USD or bundle with Claude or Codex in Blender, or recover an interrupted edit.

The agent runs Python inside Blender with your permissions; use only inputs and prompts
you trust. See docs/agents.md#security.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from ..references import parse_references
from ..workflow import PatientWorkflow
from . import BACKENDS, BLENDER_MODES, AgentOptions, SessionOptions, session
from .assets import AssetValidationError, finalize, load_workspace
from .backends import DEFAULT_BUDGET_USD, DEFAULT_CODEX_TOKEN_BUDGET, DEFAULT_MAX_TURNS
from .prompts import TASKS

# Session flags and the SessionOptions fields they set; unset flags keep the defaults.
_SESSION_FLAGS = {'backend': 'backend', 'model': 'model', 'blender_mode': 'blender_mode', 'timeout': 'timeout_s',
                  'max_turns': 'max_turns', 'budget_usd': 'budget_usd', 'token_budget': 'token_budget',
                  'max_repairs': 'max_repairs'}


def parser(command: str = 'edit') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Recovery commands use finalize or resume with --workspace PATH.')
    parser.set_defaults(command=command)
    if command == 'edit':
        parser.add_argument('--input', type=Path, required=True, help='Pipeline USD or patient_twin.yaml')
    else:
        parser.add_argument('--workspace', type=Path, required=True, help='Retained editing workspace')
    parser.add_argument('--output', type=Path, required=True, help='New final asset directory')
    parser.add_argument('--mcp-config', type=Path, help='JSON stdio server config overriding upstream uvx default')
    parser.add_argument('--blender-mode', choices=BLENDER_MODES,
                        help='Default: reuse/start Blender; virtual starts a private Xvfb GUI; existing never launches')
    parser.add_argument('--vessel-names', nargs='+', help='Vessels used to regenerate existing navigation arrays')
    if command != 'finalize':
        parser.add_argument('--backend', choices=BACKENDS, required=True)
        parser.add_argument('--model')
        parser.add_argument('--timeout', type=float, help='Session seconds, default 900')
        parser.add_argument('--max-turns', type=int, help=f'Claude turn limit, default {DEFAULT_MAX_TURNS}')
        parser.add_argument('--budget-usd', type=float, help=f'Claude dollar budget, default {DEFAULT_BUDGET_USD:g}')
        parser.add_argument('--token-budget', type=int,
                            help=f'Codex native token budget, default {DEFAULT_CODEX_TOKEN_BUDGET}')
        parser.add_argument('--max-repairs', type=int, help='Validation repair attempts, default 2; 0 disables')
        parser.add_argument('prompt', nargs='?', help='Task instructions; for resume, recovery instructions')
    if command == 'edit':
        parser.add_argument('--agent-task', choices=TASKS)
        parser.add_argument('--agent-target', nargs='+')
        parser.add_argument('--anatomy-reference', action='append', default=[], metavar='ORGAN=REFERENCE',
                            help='Catalog ID, CATALOG.json#ID, or a reference metadata JSON path')
        parser.add_argument('--major-branching', dest='include_major_branches', action='store_true',
                            help='Include reference-covered major vessel branches (off by default)')
        parser.add_argument('--fragment-area-mm2', type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Edit, resume or finalize; a resumed pipeline workspace continues through export."""
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in {'finalize', 'resume'} else 'edit'
    cli = parser(command)
    args = cli.parse_args(argv)
    if args.mcp_config and args.blender_mode not in {None, 'existing'}:
        cli.error('--mcp-config requires --blender-mode existing')
    if args.mcp_config:
        args.blender_mode = 'existing'
    limits = {field: getattr(args, flag) for flag, field in _SESSION_FLAGS.items()
              if getattr(args, flag, None) is not None}
    output = args.output.expanduser().resolve()
    if output.exists():
        cli.error('Use a new output directory')
    if command == 'resume' and (args.workspace.expanduser().resolve().parent / 'pipeline.json').is_file():
        print('Resuming parent patient pipeline, including textures and final export…', flush=True)
        overrides = dict(limits, prompt=args.prompt) if args.prompt else limits
        print(PatientWorkflow.resume(args.workspace.expanduser().resolve().parent, output=output,
                                     agent_overrides=overrides))
        return 0
    try:
        options = (AgentOptions(prompt=args.prompt or '', task=args.agent_task,
                                targets=tuple(args.agent_target or ()),
                                anatomy_references=parse_references(args.anatomy_reference),
                                include_major_branches=args.include_major_branches,
                                fragment_area_mm2=args.fragment_area_mm2, **limits)
                   if command == 'edit' else SessionOptions(**limits))
    except ValueError as error:
        cli.error(str(error))
    mcp = json.loads(args.mcp_config.read_text()) if args.mcp_config else None
    output.parent.mkdir(parents=True, exist_ok=True)
    if command == 'edit':
        workspace = Path(tempfile.mkdtemp(prefix='.asset-edit-', dir=output.parent))
    else:
        workspace = args.workspace.expanduser().resolve()
        checkpoint, edited = workspace / 'edited.blend', workspace / 'edited.usdc'
        if not all((workspace / name).is_file() for name in ('edit.json', 'original.usdc')):
            cli.error('Workspace must contain edit.json and original.usdc')
        if not checkpoint.is_file() and (command == 'resume' or not edited.is_file()):
            cli.error('Resume requires edited.blend. Finalize requires edited.blend or edited.usdc')
    try:
        if command == 'edit':
            assert isinstance(options, AgentOptions)
            settings, prompt = session.start(args.input, workspace, options, vessel_names=args.vessel_names)
            session.run(workspace, settings, output, options, prompt=prompt, fresh=True, mcp=mcp)
        elif command == 'resume':
            settings = load_workspace(workspace, vessel_names=args.vessel_names)
            session.run(workspace, settings, output, options, prompt=session.resume_prompt(workspace, args.prompt),
                        fresh=False, mcp=mcp)
        elif not edited.is_file() or (checkpoint.is_file() and checkpoint.stat().st_mtime_ns > edited.stat().st_mtime_ns):
            # The saved scene is newer than its export, so export it again before publishing.
            settings = load_workspace(workspace, vessel_names=args.vessel_names)
            session.run(workspace, settings, output, options, prompt=None, fresh=False, mcp=mcp)
        else:
            print(finalize(workspace, output, vessel_names=args.vessel_names))
    except AssetValidationError as exc:
        print(f'Publication stopped. {exc}', file=sys.stderr)
        print(f'Saved edits are retained at {workspace}', file=sys.stderr)
        return 1
    except BaseException:
        print(f'Edit failed; recovery files retained at {workspace}', file=sys.stderr)
        raise
    if command == 'edit':
        shutil.rmtree(workspace)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
