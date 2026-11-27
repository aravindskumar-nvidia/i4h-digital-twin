# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Blender or TRELLIS.2 in the active patient environment."""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Sequence

from ._texture_setup import setup_texture
from .agent.runtime import (
    VIRTUAL_DISPLAY_TOOLS,
    find_blender,
    headless,
    install_blender,
)

# Import names of the editing extra's packages.
_EDITING_MODULES = ('mcp', 'openai_codex', 'claude_agent_sdk', 'pxr', 'vtk', 'trimesh', 'PIL')


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='patient-digital-twin setup', description=__doc__)
    parser.add_argument('component', nargs='?', choices=('editing', 'texture'), default='editing')
    args = parser.parse_args(argv)
    if args.component == 'texture':
        try:
            checkout = setup_texture()
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Texture setup failed: {exc}', file=sys.stderr)
            return 1
        print(f'Texture setup complete. TRELLIS.2: {checkout}\n'
              'Model weights download automatically on first use.')
        return 0
    if missing := [name for name in _EDITING_MODULES if importlib.util.find_spec(name) is None]:
        print(f'Missing editing dependencies: {", ".join(missing)}. '
              "Install the 'editing' extra of patient-digital-twin first.", file=sys.stderr)
        return 1
    if headless() and any(shutil.which(tool) is None for tool in VIRTUAL_DISPLAY_TOOLS):
        print('Headless editing needs Xvfb and xauth. On Ubuntu, run:\n'
              '  sudo apt install xvfb xauth\nThen rerun: patient-digital-twin setup editing', file=sys.stderr)
        return 1
    try:
        executable = find_blender() or install_blender()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'Editing setup failed: {exc}', file=sys.stderr)
        return 1
    print(f'Editing setup complete. Blender: {executable}\n'
          'Authenticate your chosen agent backend before editing. The agent runs Python in Blender '
          'with your permissions; see docs/agents.md#security.')
    return 0
