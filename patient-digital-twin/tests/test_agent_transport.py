# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK adapter contracts; model services are never called."""
import asyncio
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('budget_exhausted', [False, True])
def test_codex_streams_actual_sdk_notification_shape(monkeypatch, tmp_path, capsys, budget_exhausted):
    sdk = pytest.importorskip('openai_codex')
    from patient_digital_twin.agent.__main__ import mcp_config, run_codex
    captured = {}

    class Turn:
        def stream(self):
            yield SimpleNamespace(method='item/agentMessage/delta', payload=SimpleNamespace(delta='done'))
            result = SimpleNamespace(status='completed', error=None)
            if budget_exhausted:
                result = SimpleNamespace(status='failed', error=SimpleNamespace(message='shared rollout token budget exhausted'))
            yield SimpleNamespace(method='turn/completed', payload=SimpleNamespace(turn=result))

        def interrupt(self):
            captured['interrupted'] = True

    class FakeCodex:
        def __init__(self, config):
            captured['config'] = config

        def __enter__(self):
            return self

        def __exit__(self, *args):
            captured['closed'] = True

        def thread_start(self, **kwargs):
            captured['thread'] = kwargs
            return SimpleNamespace(turn=lambda prompt: Turn())

    monkeypatch.setattr(sdk, 'Codex', FakeCodex)
    monkeypatch.setenv('DISPLAY', ':77')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'never-forward-this')
    config = mcp_config()
    assert 'ANTHROPIC_API_KEY' not in config['env']
    if budget_exhausted:
        with pytest.raises(RuntimeError, match='token budget exhausted'):
            run_codex('inspect anatomy', tmp_path, config, None, token_budget=1000)
    else:
        run_codex('inspect anatomy', tmp_path, config, None, token_budget=1000)
    assert 'done' in capsys.readouterr().out
    assert captured['closed']
    assert captured.get('interrupted', False) == budget_exhausted
    assert captured['thread']['sandbox'] == sdk.Sandbox.workspace_write
    assert any('mcp_servers.blender.command' in item for item in captured['config'].config_overrides)
    assert any('mcp_servers.blender.args=' in item for item in captured['config'].config_overrides)
    assert 'mcp_servers.blender.tools.look.approval_mode="approve"' in captured['config'].config_overrides
    assert 'features.rollout_budget.enabled=true' in captured['config'].config_overrides
    assert 'features.rollout_budget.limit_tokens=1000' in captured['config'].config_overrides
    assert 'features.rollout_budget.reminder_at_remaining_tokens=[]' in captured['config'].config_overrides
    assert not any('default_tools_approval_mode' in item for item in captured['config'].config_overrides)


@pytest.mark.parametrize(('backend', 'flag', 'value'), [
    ('codex', '--max-turns', '10'),
    ('codex', '--budget-usd', '5'),
    ('claude', '--token-budget', '1000'),
    ('codex', '--token-budget', '0'),
])
def test_backend_limits_rejected_before_editing(tmp_path, backend, flag, value):
    from patient_digital_twin.agent.__main__ import main

    with pytest.raises(SystemExit) as error:
        main(['--backend', backend, '--input', str(tmp_path / 'unused.usdc'),
              '--output', str(tmp_path / 'output'), flag, value, 'Edit the aorta.'])
    assert error.value.code == 2
    assert not (tmp_path / 'output').exists()


def test_claude_receives_blender_tools(monkeypatch, tmp_path):
    sdk = pytest.importorskip('claude_agent_sdk')
    from patient_digital_twin.agent.__main__ import run_claude
    captured = {}

    async def query(**kwargs):
        captured.update(kwargs)
        yield sdk.ResultMessage(subtype='success', duration_ms=1, duration_api_ms=1,
                                is_error=False, num_turns=1, session_id='test')

    monkeypatch.setattr(sdk, 'query', query)
    asyncio.run(run_claude('inspect anatomy', tmp_path, {'command': sys.executable, 'args': []}, None, 3, 1.))
    options = captured['options']
    assert options.tools == []
    assert 'mcp__blender__execute_blender_code' in options.allowed_tools
    assert options.mcp_servers['blender']['command'] == sys.executable
    assert options.max_turns == 3 and options.max_budget_usd == 1.
