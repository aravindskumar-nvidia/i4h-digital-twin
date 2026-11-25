# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installer integrity, process ownership, and an opt-in real virtual desktop."""
import asyncio
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from patient_digital_twin.agent import runtime


@pytest.mark.parametrize('valid', [True, False])
def test_installer_checks_hash_and_reuses_cached_binary(tmp_path, monkeypatch, valid):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:xz') as package:
        info = tarfile.TarInfo(f'blender-{runtime.BLENDER_VERSION}-linux-x64/blender')
        data = f'#!/bin/sh\necho "Blender {runtime.BLENDER_VERSION}"\n'.encode()
        info.size, info.mode = len(data), 0o755
        package.addfile(info, io.BytesIO(data))
    payload = archive.getvalue()
    monkeypatch.delenv('BLENDER_EXECUTABLE', raising=False)
    # A symlinked cache (for example an NFS home) must not trip the archive path check.
    (tmp_path / 'real').mkdir()
    (tmp_path / 'cache').symlink_to(tmp_path / 'real')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setattr(runtime.shutil, 'which', lambda _: None)
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(runtime.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(runtime, 'BLENDER_SHA256', hashlib.sha256(payload).hexdigest() if valid else '0' * 64)
    monkeypatch.setattr(runtime.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(payload))
    prompts = []
    monkeypatch.setattr('builtins.input', lambda prompt: prompts.append(prompt) or 'y')
    if not valid:
        with pytest.raises(ValueError, match='SHA-256'):
            runtime.ensure_blender()
        assert not list(tmp_path.rglob('blender-*/blender'))
        return
    path = runtime.ensure_blender()
    assert Path(path).is_file() and os.access(path, os.X_OK)
    monkeypatch.setattr(runtime.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('Cached Blender was downloaded again'))
    assert runtime.ensure_blender() == path
    assert len(prompts) == 1
    Path(path).write_text('#!/bin/sh\necho "Blender 2.93.0"\n')
    with pytest.raises(RuntimeError, match='incompatible'):
        runtime.ensure_blender()
    Path(path).write_text('#!/bin/sh\nexit 46\n')
    with pytest.raises(RuntimeError, match='--version failed.*BLENDER_EXECUTABLE'):
        runtime.ensure_blender()
    Path(path).unlink()
    monkeypatch.setattr('builtins.input', lambda _: 'n')
    with pytest.raises(RuntimeError, match='cancelled'):
        runtime.ensure_blender()

    def no_input(_):
        raise EOFError

    monkeypatch.setattr('builtins.input', no_input)
    with pytest.raises(RuntimeError, match='I4H_INSTALL_BLENDER=1'):
        runtime.ensure_blender()
    monkeypatch.setenv('I4H_INSTALL_BLENDER', '1')
    monkeypatch.setattr(runtime.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(payload))
    assert runtime.ensure_blender() == path


def test_existing_blender_is_not_installed_or_owned(tmp_path, monkeypatch):
    config = {'command': 'unused', 'args': [], 'env': {'BLENDER_PORT': '9876'}}
    monkeypatch.setattr(runtime, '_responding', lambda *a: True)
    monkeypatch.setattr(runtime, 'ensure_blender', lambda: pytest.fail('Existing Blender must not trigger installation'))
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Existing Blender must not launch a process'))
    with pytest.raises(RuntimeError, match='agent failure'):
        with runtime.blender_session(tmp_path, config) as connection:
            assert connection is config
            raise RuntimeError('agent failure')
    remote = {**config, 'env': {**config['env'], 'BLENDER_HOST': '10.0.0.5'}}
    with pytest.raises(RuntimeError, match='must run on this machine'):
        with runtime.blender_session(tmp_path, remote, 'existing'):
            pytest.fail('Remote Blender cannot access workspace files')


def test_real_virtual_desktop_and_cleanup_on_failure(tmp_path, monkeypatch):
    executable = os.environ.get('I4H_TEST_BLENDER')
    if not executable:
        pytest.skip('Set I4H_TEST_BLENDER and provide xvfb-run/xauth for the real runtime check')
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from patient_digital_twin.agent.__main__ import BLENDER_TOOLS, mcp_config

    monkeypatch.setenv('BLENDER_EXECUTABLE', executable)
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    config = mcp_config()
    config['env']['UV_CACHE_DIR'] = os.environ.get('UV_CACHE_DIR', '/tmp/i4h-uv-cache')
    with pytest.raises(RuntimeError, match='agent failure'):
        with runtime.blender_session(tmp_path, config, 'virtual') as connection:
            ready = json.loads((tmp_path / 'blender-ready.json').read_text())
            assert ready['software_gl_requested'] == os.environ.get('LIBGL_ALWAYS_SOFTWARE', '1')
            assert ready['viewport_renderer'] and ready['blender_version']
            assert runtime._responding('127.0.0.1', ready['port'])
            with monkeypatch.context() as incompatible:
                incompatible.setattr(runtime, 'MCP_PROTOCOL', -1)
                assert not runtime._responding('127.0.0.1', ready['port'])
            display = Path('/tmp/.X11-unix/X' + ready['display'].split(':')[-1].split('.')[0])
            assert display.exists()
            with runtime.blender_session(tmp_path, connection) as reused:
                assert reused is connection
            assert runtime._responding('127.0.0.1', ready['port'])

            async def screenshot():
                async with stdio_client(StdioServerParameters(**connection)) as (read, write):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        assert set(BLENDER_TOOLS) <= {t.name for t in (await client.list_tools()).tools}
                        result = await client.call_tool('look', {'mode': 'viewport', 'shading': 'solid', 'max_size': 256})
                        assert not result.isError and any(c.type == 'image' for c in result.content)
                        # Reference inspection changes scenes between offscreen
                        # captures. Exercise the path that failed in the live run.
                        for name in ('reference', 'patient', 'reference'):
                            code = ("import bpy\n"
                                    f"scene=bpy.data.scenes.get({name!r}) or bpy.data.scenes.new({name!r})\n"
                                    "scene.render.engine='BLENDER_WORKBENCH'\n"
                                    "bpy.context.window.scene=scene\n"
                                    "if not scene.objects: bpy.ops.mesh.primitive_uv_sphere_add()\n"
                                    "bpy.context.view_layer.update()")
                            result = await client.call_tool('execute_blender_code', {
                                'code': code, 'user_prompt': 'Verify reference scene capture stability'})
                            assert not result.isError
                            result = await client.call_tool('look', {
                                'mode': 'angles', 'views': ['front', 'right'],
                                'shading': 'solid', 'max_size': 512})
                            assert not result.isError and any(c.type == 'image' for c in result.content)

            asyncio.run(screenshot())
            raise RuntimeError('agent failure')
    assert not runtime._responding('127.0.0.1', ready['port'])
    assert not display.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(ready['pid'], 0)
