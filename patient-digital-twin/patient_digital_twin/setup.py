# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Blender or TRELLIS.2 in the active patient environment."""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence

from .agent.runtime import ensure_blender


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='patient-digital-twin setup', description=__doc__)
    parser.add_argument('component', nargs='?', choices=('editing', 'texture'), default='editing')
    args = parser.parse_args(argv)
    if args.component == 'texture':
        from ._texture_setup import setup_texture
        try:
            checkout = setup_texture()
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Texture setup failed: {exc}', file=sys.stderr)
            return 1
        print(f'Texture setup complete. TRELLIS.2: {checkout}\n'
              'Model weights download automatically on first use.')
        return 0
    required = ('mcp', 'openai_codex', 'claude_agent_sdk', 'pxr', 'vtk', 'trimesh', 'PIL')
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print('Missing editing dependencies: ' + ', '.join(missing) + '.\n'
              "Run: uv pip install -e './patient-digital-twin[editing]'", file=sys.stderr)
        return 1
    headless = platform.system() == 'Linux' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    if headless and any(shutil.which(name) is None for name in ('xvfb-run', 'xauth')):
        print('Headless editing needs Xvfb and xauth. On Ubuntu, run:\n'
              '  sudo apt install xvfb xauth\nThen rerun: patient-digital-twin setup editing', file=sys.stderr)
        return 1
    try:
        executable = ensure_blender(install=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'Editing setup failed: {exc}', file=sys.stderr)
        return 1
    print(f'Editing setup complete. Blender: {executable}\n'
          'Codex and Claude SDKs are installed. Authenticate your chosen backend before editing.')
    return 0
