# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI settings reach typed options without a workflow configuration file."""
import argparse

import pytest
from patient_digital_twin._refinement_cli import (
    add_refinement_arguments,
    pop_refinement_workflow,
)


@pytest.fixture(autouse=True)
def installed_texture_runtime(tmp_path, monkeypatch):
    from functools import partial

    from patient_digital_twin import _refinement_cli
    from patient_digital_twin.texturing import TextureRuntime
    monkeypatch.setattr(_refinement_cli, 'TextureRuntime', partial(TextureRuntime, checkout=tmp_path))


def parse(flags):
    parser = argparse.ArgumentParser()
    add_refinement_arguments(parser)
    return pop_refinement_workflow(vars(parser.parse_args(flags)))


def test_agent_texture_and_cleanup_settings(tmp_path, monkeypatch):
    from patient_digital_twin import references
    monkeypatch.setattr(references, 'load_references', lambda *args, **kwargs: {})
    run = parse([
        '--agent-prompt', 'Complete only the colon', '--agent-model', 'gpt-6.1-terra',
        '--agent-backend', 'codex', '--agent-timeout', '900', '--token-budget', '100000',
        '--blender-mode', 'virtual', '--anatomy-reference', f'colon={tmp_path}/reference.json',
        '--default-textures',
        '--texture-timeout', '900', '--missing-texture', 'retain',
        '--texture-size', '1024', '--texture-steps', '8', '--no-match-reference-color',
        '--smoothing-iterations', '40', '--workspace', str(tmp_path / 'work'),
        '--min-component-area-mm2', '30', '--max-component-distance-mm', '40',
    ])
    assert run.agent.model == 'gpt-6.1-terra'
    assert run.agent.timeout_s == 900 and run.agent.token_budget == 100000
    assert run.agent.anatomy_references == {'colon': tmp_path / 'reference.json'}
    # Catalog IDs are kept as written; only file paths are made absolute.
    by_id = parse(['--agent-prompt', 'Complete only the colon',
                 '--anatomy-reference', 'colon=voxel-man-colon-male-15882019'])
    assert by_id.agent.anatomy_references == {'colon': 'voxel-man-colon-male-15882019'}
    assert run.texture_runtime.timeout_s == 900
    assert run.texture_options.texture_size == 1024 and run.texture_options.steps == 8
    assert run.texture_options.match_reference_color is False
    assert run.default_textures and run.missing_texture == 'retain'
    assert run.cleanup.smoothing_iterations == 40
    assert run.cleanup.min_component_area_mm2 == 30 and run.cleanup.max_component_distance_mm == 40
    assert run.workspace == tmp_path / 'work'


@pytest.mark.parametrize('flags,message', [
    (['--agent-model', 'gpt-6.1-terra'], '--agent-prompt'),
    (['--no-cleanup', '--smoothing-iterations', '1'], '--no-cleanup'),
    (['--agent-prompt', 'Edit', '--budget-usd', '2'], 'Codex uses'),
])
def test_invalid_settings_fail_before_inference(flags, message):
    with pytest.raises(ValueError, match=message):
        parse(flags)


def test_explicit_tile_overrides_and_default_optout(tmp_path):
    import hashlib
    import json
    image = tmp_path / 'tile.png'
    image.write_bytes(b'tile')
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps(dict(file='tile.png', creator='test', source='test',
                                       surface='capsule', changes='none',
                                       sha256=hashlib.sha256(image.read_bytes()).hexdigest())))
    run = parse(['--texture', f'liver={metadata}', '--no-default-textures'])
    assert set(run.textures) == {'liver'} and not run.default_textures
    assert run.agent is None
    assert run.missing_texture == 'retain'
    assert parse(['--default-textures', '--missing-texture', 'error']).missing_texture == 'error'
    with pytest.raises(ValueError, match='unique'):
        parse(['--texture', f'liver={metadata}', '--texture', f'liver={metadata}'])


def test_missing_trellis_installation_fails_without_path_guessing(monkeypatch):
    from patient_digital_twin import texturing
    monkeypatch.setattr(texturing.importlib.util, 'find_spec', lambda name: None)
    with pytest.raises(ImportError, match='active Python environment'):
        texturing.TextureRuntime()


def test_major_branching_is_explicit_in_both_clis():
    from patient_digital_twin.agent.__main__ import parser as agent_parser
    pipeline = argparse.ArgumentParser()
    add_refinement_arguments(pipeline)
    assert not getattr(pipeline.parse_args([]), 'include_major_branches', False)
    assert pipeline.parse_args(['--major-branching']).include_major_branches is True
    editor = agent_parser()
    args = ['--input', 'input.usdc', '--output', 'out', '--backend', 'codex', 'Repair']
    assert editor.parse_args(args).include_major_branches is False
    assert editor.parse_args(args + ['--major-branching']).include_major_branches is True


def test_named_operations_without_verbal_prompt(tmp_path, monkeypatch):
    from patient_digital_twin import references
    from patient_digital_twin.agent.prompts import build_prompt
    monkeypatch.setattr(references, 'load_references', lambda *args, **kwargs: {})
    for task in ('complete', 'repair'):
        run = parse(['--agent-task', task, '--anatomy-reference', f'colon={tmp_path}/ref.json'])
        assert run.agent.task == task and run.agent.prompt == ''
        prompt = build_prompt(task, references=run.agent.anatomy_references)
        assert 'Targets: colon.' in prompt
        assert 'check_asset' in prompt
    for task in ('refine', 'texture'):
        run = parse(['--agent-task', task, '--agent-target', 'liver'])
        assert run.agent.task == task and run.cleanup is None
    assert parse(['--agent-task', 'refine', '--agent-target', 'liver',
                  '--agent-prompt', 'Preserve this scar']).agent.prompt == 'Preserve this scar'


@pytest.mark.parametrize('flags,message', [
    (['--agent-task', 'refine'], 'require --agent-target'),
    (['--agent-task', 'complete', '--agent-target', 'colon'], 'reference for every target'),
    (['--agent-task', 'texture', '--agent-target', 'liver', '--major-branching'], 'only to complete or repair'),
    (['--agent-task', 'texture', '--agent-target', 'liver', '--cleanup'], 'cannot request geometry cleanup'),
    (['--agent-task', 'texture', '--agent-target', 'liver', '--default-textures'], 'overwrite'),
])
def test_named_operations_reject_conflicting_scope(flags, message):
    with pytest.raises(ValueError, match=message):
        parse(flags)
