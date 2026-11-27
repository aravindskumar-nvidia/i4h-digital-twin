# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build TRELLIS.2's CUDA extensions in the active environment at pinned commits.

The ``texture`` extra installs PyTorch and every Python dependency. This step only
builds the sources that have no usable wheel, against that PyTorch and the local
CUDA toolkit.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

from ._cache import cache_directory
from .texturing import TextureRuntime, check_runtime


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    commit: str
    package: str = '.'  # Directory inside the checkout that is installed.


# Tested with PyTorch 2.6.0+cu124 and the CUDA 12.4 toolkit.
SOURCES = (
    Source('utils3d', 'https://github.com/EasternJournalist/utils3d.git', '9a4eb15e4021b67b12c460c7057d642626897ec8'),
    Source('CuMesh', 'https://github.com/JeffreyXiang/CuMesh.git', '12289e1062f0603f2f0d0771b02e1395d247f26f'),
    Source('FlexGEMM', 'https://github.com/JeffreyXiang/FlexGEMM.git', '6dd94a859c26ee8246888502eada3dd8ad85532e'),
    Source('nvdiffrast', 'https://github.com/NVlabs/nvdiffrast.git', '253ac4fcea7de5f396371124af597e6cc957bfae'),  # v0.4.0
    Source('TRELLIS.2', 'https://github.com/microsoft/TRELLIS.2.git', '75fbf0183001ed9876c8dbb35de6b68552ee08bd',
           package='o-voxel'),
)
FLASH_ATTENTION = 'flash-attn==2.7.3'
INSTALL_TEXTURE = "uv pip install -e './patient-digital-twin[texture]'"


def setup_texture() -> Path:
    """Check out, build and register the pinned sources; return the TRELLIS.2 checkout."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Activate a virtual environment first: source .venv/bin/activate')
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Texture setup supports Linux x86_64 with an NVIDIA GPU')
    environment = _build_environment()
    if missing := [tool for tool in ('uv', 'git', 'c++') if shutil.which(tool, path=environment['PATH']) is None]:
        raise RuntimeError(f'Missing system tools: {", ".join(missing)}. Install them and rerun setup.')
    root = cache_directory('texture')
    trellis = root / 'TRELLIS.2'
    # The .pth puts the TRELLIS.2 checkout on sys.path; its first line records the built commits.
    registration = Path(sysconfig.get_path('purelib')) / 'trellis2-source.pth'
    pins = {source.name: source.commit for source in SOURCES}
    if _registered_pins(registration) == pins:
        try:
            check_runtime(TextureRuntime(checkout=trellis))
            return trellis
        except (RuntimeError, FileNotFoundError) as error:
            print(f'Rebuilding: the installed texture runtime failed its check. {error}', flush=True)
    for source in SOURCES:
        checkout = _checkout(root / source.name, source, environment)
        _install(environment, str(checkout / source.package))
    _install(environment, FLASH_ATTENTION)
    check_runtime(TextureRuntime(checkout=trellis))
    registration.write_text(f'# {json.dumps(pins)}\n{trellis}\n')
    return trellis


def _build_environment() -> dict[str, str]:
    """Return a build environment for the installed PyTorch, or explain the mismatch."""
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except ModuleNotFoundError as error:
        raise RuntimeError(f'Install the texture dependencies first: {INSTALL_TEXTURE}') from error
    if torch.version.cuda is None:
        raise RuntimeError(f'PyTorch {torch.__version__} has no CUDA support. Install a CUDA build of PyTorch.')
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot reach a GPU. Check the NVIDIA driver with nvidia-smi.')
    if CUDA_HOME is None:
        raise RuntimeError('No CUDA toolkit found. Install one and set CUDA_HOME to its directory.')
    nvcc = Path(CUDA_HOME) / 'bin/nvcc'
    release = re.search(r'release (\d+)\.(\d+)', subprocess.check_output([str(nvcc), '--version'], text=True))
    if release is None:
        raise RuntimeError(f'Cannot read the CUDA toolkit version from {nvcc} --version')
    if release[1] != torch.version.cuda.split('.')[0]:
        raise RuntimeError(
            f'The CUDA {release[1]}.{release[2]} toolkit at {CUDA_HOME} cannot build extensions for PyTorch '
            f'{torch.__version__} (CUDA {torch.version.cuda}). Install a toolkit with the same major version, '
            'or a PyTorch build for this toolkit.')
    environment = dict(os.environ, CUDA_HOME=CUDA_HOME)
    # Builds run the environment's ninja and the toolkit's nvcc.
    environment['PATH'] = os.pathsep.join([str(Path(sys.executable).parent), str(nvcc.parent), environment['PATH']])
    return environment


def _checkout(destination: Path, source: Source, environment: dict[str, str]) -> Path:
    git = ['git', '-C', str(destination)]
    if not destination.exists():
        subprocess.run(['git', 'clone', source.url, str(destination)], check=True, env=environment)
    elif subprocess.check_output([*git, 'remote', 'get-url', 'origin'], text=True).strip() != source.url:
        raise RuntimeError(f'{destination} is not a checkout of {source.url}. Remove it and rerun setup.')
    if subprocess.run([*git, 'cat-file', '-e', source.commit + '^{commit}'], env=environment).returncode:
        subprocess.run([*git, 'fetch', 'origin'], check=True, env=environment)
    subprocess.run([*git, 'checkout', '--detach', source.commit], check=True, env=environment)
    subprocess.run([*git, 'submodule', 'update', '--init', '--recursive'], check=True, env=environment)
    return destination


def _install(environment: dict[str, str], requirement: str) -> None:
    # Extensions compile against the installed PyTorch, so builds must not be isolated.
    subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--no-build-isolation', requirement],
                   check=True, env=environment)


def _registered_pins(registration: Path) -> dict | None:
    if not registration.is_file():
        return None
    first = registration.read_text().splitlines()[0]
    return json.loads(first[2:]) if first.startswith('# {') else None
