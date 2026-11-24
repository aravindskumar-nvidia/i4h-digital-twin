# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared refinement flags for inference and existing-mesh entry points."""
from __future__ import annotations

import argparse
from pathlib import Path

from .agent import AgentOptions
from .references import TextureReference, parse_references
from .surface import CleanupOptions
from .texturing import TextureOptions, TextureRuntime
from .workflow import PatientWorkflow

# None distinguishes an omitted flag from a request for the refinement path.
_FIELDS = {
    'cleanup': None, 'smoothing_iterations': None, 'smoothing_pass_band': None,
    'hole_radius_mm': None, 'min_component_area_mm2': None,
    'agent_task': None, 'agent_target': None,
    'agent_prompt': None, 'agent_backend': None, 'agent_model': None,
    'agent_timeout': None, 'token_budget': None, 'budget_usd': None,
    'max_turns': None, 'max_repairs': None, 'blender_mode': None,
    'anatomy_reference': None, 'include_major_branches': None, 'fragment_area_mm2': None,
    'texture': None,
    'texture_timeout': None, 'default_textures': None, 'missing_texture': None,
    'texture_seed': None, 'texture_resolution': None, 'texture_size': None,
    'texture_steps': None, 'texture_roughness': None, 'match_reference_color': None,
    'workspace': None,
}


def add_refinement_arguments(parser: argparse.ArgumentParser) -> None:
    cleanup = parser.add_argument_group('Surface cleanup (enabled when refinement is requested)')
    cleanup.add_argument('--cleanup', action=argparse.BooleanOptionalAction, default=None)
    for flag, kind in [('smoothing-iterations', int), ('smoothing-pass-band', float),
                       ('hole-radius-mm', float), ('min-component-area-mm2', float)]:
        cleanup.add_argument('--' + flag, type=kind)
    agent = parser.add_argument_group('Optional SDK agent editing')
    from .agent.prompts import TASKS
    agent.add_argument('--agent-task', choices=TASKS, help='Named operation; no verbal prompt required')
    agent.add_argument('--agent-target', nargs='+', help='Structure names to edit; defaults to reference targets')
    agent.add_argument('--agent-prompt', help='Enable agent editing with these instructions')
    agent.add_argument('--agent-backend', choices=['codex', 'claude'])
    agent.add_argument('--agent-model')
    agent.add_argument('--agent-timeout', type=float, help='Seconds per agent session (default: 900)')
    agent.add_argument('--token-budget', type=int, help='Codex token budget (default: 100000)')
    agent.add_argument('--budget-usd', type=float, help='Claude dollar budget (default: 5)')
    agent.add_argument('--max-turns', type=int, help='Claude turn limit (default: 30)')
    agent.add_argument('--max-repairs', type=int)
    agent.add_argument('--blender-mode', choices=['auto', 'virtual', 'existing'])
    agent.add_argument('--anatomy-reference', action='append', metavar='ORGAN=METADATA_JSON')
    agent.add_argument('--major-branching', dest='include_major_branches', action='store_true', default=False,
                       help='Include reference-covered major vessel branches (off by default)')
    agent.add_argument('--fragment-area-mm2', type=float)
    texture = parser.add_argument_group('Optional CUDA texture generation')
    texture.add_argument('--texture', action='append', metavar='ORGAN=METADATA_JSON',
                         help='Override an organ tile; repeat for more organs')
    texture.add_argument('--texture-timeout', type=float, help='Seconds per worker (default: 900)')
    texture.add_argument('--default-textures', action=argparse.BooleanOptionalAction, default=None,
                         help='Generate default textures using TRELLIS.2 in the active environment')
    texture.add_argument('--missing-texture', choices=['error', 'retain'],
                         help='Missing default tile: retain material (default), or stop with error')
    for flag, kind in [('seed', int), ('resolution', int), ('size', int), ('steps', int), ('roughness', float)]:
        texture.add_argument('--texture-' + flag, type=kind)
    texture.add_argument('--match-reference-color', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--workspace', type=Path, help='Intermediate assets and recovery files')
    for action in parser._actions:
        if action.dest in _FIELDS:
            action.default = argparse.SUPPRESS


def pop_refinement_workflow(options: dict) -> PatientWorkflow | None:
    """Consume CLI-only keys; leave importer/exporter arguments in options."""
    values = {key: options.pop(key, default) for key, default in _FIELDS.items()}
    if all(value is None for value in values.values()):
        return None
    agent_fields = {
        'agent_task': 'task', 'agent_target': 'targets',
        'agent_prompt': 'prompt', 'agent_backend': 'backend', 'agent_model': 'model',
        'agent_timeout': 'timeout_s', 'token_budget': 'token_budget', 'budget_usd': 'budget_usd',
        'max_turns': 'max_turns', 'max_repairs': 'max_repairs', 'blender_mode': 'blender_mode',
        'include_major_branches': 'include_major_branches', 'fragment_area_mm2': 'fragment_area_mm2',
    }
    agent_values = {field: values[key] for key, field in agent_fields.items() if values[key] is not None}
    if (agent_values or values['anatomy_reference']) and not (values['agent_prompt'] or values['agent_task']):
        raise ValueError('Agent options require --agent-task or --agent-prompt')
    if values['anatomy_reference']:
        agent_values['anatomy_references'] = {
            name: Path(path).expanduser().resolve()
            for name, path in parse_references(values['anatomy_reference']).items()
        }
    if values['agent_task'] == 'texture' and (values['cleanup'] is True or
            any(values[key] is not None for key in ('smoothing_iterations', 'smoothing_pass_band',
                                                  'hole_radius_mm', 'min_component_area_mm2'))):
        raise ValueError('Texture-only edits cannot request geometry cleanup')
    cleanup_values = {key: values[key] for key in
                      ('smoothing_iterations', 'smoothing_pass_band', 'hole_radius_mm',
                       'min_component_area_mm2') if values[key] is not None}
    if values['cleanup'] is False and cleanup_values:
        raise ValueError('--no-cleanup cannot be combined with cleanup settings')
    runtime = None
    texture_settings = [value for key, value in values.items()
                        if key.startswith('texture_') or key in
                        {'texture', 'missing_texture', 'match_reference_color'}]
    texture_requested = any(value is not None for value in texture_settings) or values['default_textures'] is True
    if texture_requested and values['agent_task'] == 'texture':
        raise ValueError('Run texture generation before a texture-only agent edit; later generation would overwrite it')
    if texture_requested:
        runtime_options = {} if values['texture_timeout'] is None else {'timeout_s': values['texture_timeout']}
        runtime = TextureRuntime(**runtime_options)
    texture_fields = {'texture_seed': 'seed', 'texture_resolution': 'resolution',
                      'texture_size': 'texture_size', 'texture_steps': 'steps',
                      'texture_roughness': 'roughness', 'match_reference_color': 'match_reference_color'}
    return PatientWorkflow(
        cleanup=None if values['cleanup'] is False or (values['agent_task'] in {'texture', 'refine'}
            and values['cleanup'] is not True and not cleanup_values) else CleanupOptions(**cleanup_values),
        agent=AgentOptions(**agent_values) if agent_values else None,
        textures={name: TextureReference.load(path) for name, path in
                  parse_references(values['texture'] or [], '--texture').items()},
        texture_runtime=runtime,
        texture_options=TextureOptions(**{field: values[key] for key, field in texture_fields.items()
                                          if values[key] is not None}),
        default_textures=values['default_textures'] if values['default_textures'] is not None else True,
        missing_texture=values['missing_texture'] or 'retain', workspace=values['workspace'],
    )
