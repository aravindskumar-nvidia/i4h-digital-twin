# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the native TRELLIS.2 backend in the active Python environment."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    candidates = [Path(env[key]) for key in ('CUDA_HOME', 'CUDA_PATH') if env.get(key)]
    candidates.append(Path('/usr/local/cuda'))
    cuda = next((path for path in candidates if (path / 'bin/nvcc').is_file()), None)
    paths = [str(Path(sys.executable).parent)]
    if cuda:
        env['CUDA_HOME'] = str(cuda)
        paths.append(str(cuda / 'bin'))
    env['PATH'] = os.pathsep.join([*paths, env.get('PATH', '')])
    env['VIRTUAL_ENV'] = sys.prefix
    return env


def _install(*requirements: str) -> None:
    command = ['uv', 'pip', 'install', '--python', sys.executable]
    try:
        torch_version = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        subprocess.run([*command, *requirements], check=True, env=_environment())
    else:
        # Native extensions must build against the active CUDA/PyTorch stack.
        with tempfile.TemporaryDirectory(prefix='i4h-torch-') as temporary:
            constraint = Path(temporary) / 'constraints.txt'
            constraint.write_text(f'torch=={torch_version}\n')
            subprocess.run([*command, '--constraint', str(constraint), *requirements], check=True, env=_environment())


def _checkout(root: Path, name: str, url: str, revision: str | None = None) -> Path:
    destination = root / name
    if destination.exists():
        origin = subprocess.check_output(
            ['git', '-C', str(destination), 'remote', 'get-url', 'origin'], text=True, env=_environment()).strip()
        if origin != url:
            raise ValueError(f'{destination} is not the expected checkout: {url}')
    else:
        subprocess.run(['git', 'clone', '--recursive', url, str(destination)], check=True, env=_environment())
    if revision:
        subprocess.run(['git', '-C', str(destination), 'checkout', revision], check=True, env=_environment())
        subprocess.run(['git', '-C', str(destination), 'submodule', 'update', '--init', '--recursive'], check=True, env=_environment())
    return destination


def _check_runtime(checkout: Path) -> None:
    code = (
        'import sys; sys.path.insert(0, sys.argv[1]); '
        'import torch, cumesh, flex_gemm, o_voxel, nvdiffrast.torch, flash_attn; '
        'from trellis2.pipelines import Trellis2TexturingPipeline; '
        'assert torch.cuda.is_available(), "CUDA-enabled PyTorch and a working NVIDIA driver are required"'
    )
    subprocess.run([sys.executable, '-c', code, str(checkout)], check=True, env=_environment())


def setup_texture() -> Path:
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Activate your virtual environment first: source .venv/bin/activate')
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'AMD64'}:
        raise RuntimeError('Automatic texture setup supports Linux x86_64 with an NVIDIA GPU')
    spec = importlib.util.find_spec('trellis2')
    if spec is not None and spec.origin is not None:
        checkout = Path(spec.origin).resolve().parent.parent
        try:
            _check_runtime(checkout)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            pass
        else:
            return checkout
    required = ('uv', 'git', 'nvcc', 'c++')
    missing = [name for name in required if shutil.which(name, path=_environment()['PATH']) is None]
    if missing:
        raise RuntimeError('Missing system tools: ' + ', '.join(missing) +
                           '. Install the CUDA development toolkit (nvcc), a C++ compiler, git and uv, then rerun setup.')
    root = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'i4h/texture'
    root.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec('torch') is None:
        _install('torch==2.6.0', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu124')
    subprocess.run([sys.executable, '-c',
                    "import torch; assert torch.cuda.is_available(), 'CUDA-enabled PyTorch and a working NVIDIA driver are required'"],
                   check=True, env=_environment())
    _install('setuptools>=68', 'wheel', 'packaging', 'ninja', 'psutil', 'numpy', 'scipy',
             'trimesh', 'Pillow>=10', 'imageio', 'imageio-ffmpeg', 'tqdm', 'easydict',
             'opencv-python-headless', 'transformers>=4.56,<5', 'huggingface_hub',
             'safetensors', 'kornia', 'timm', 'zstandard', 'plyfile')
    _install('git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8')
    trellis = _checkout(root, 'TRELLIS.2', 'https://github.com/microsoft/TRELLIS.2.git')
    for name, url, revision in (
        ('CuMesh', 'https://github.com/JeffreyXiang/CuMesh.git', None),
        ('FlexGEMM', 'https://github.com/JeffreyXiang/FlexGEMM.git', None),
        ('nvdiffrast', 'https://github.com/NVlabs/nvdiffrast.git', 'v0.4.0'),
    ):
        source = _checkout(root, name, url, revision)
        _install('--no-build-isolation', str(source))
    _install('--no-build-isolation', str(trellis / 'o-voxel'))
    _install('--no-build-isolation', 'flash-attn==2.7.3')
    # Publish source discovery only after native imports and CUDA checks succeed.
    _check_runtime(trellis)
    (Path(sysconfig.get_path('purelib')) / 'trellis2-source.pth').write_text(str(trellis) + '\n')
    return trellis
