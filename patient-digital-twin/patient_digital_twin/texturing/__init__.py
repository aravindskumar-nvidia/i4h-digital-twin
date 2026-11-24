# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-mesh texturing functions with an isolated, optional CUDA worker."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..anatomy import MeshGeometry, SurfaceMaterial
from ..geometry import validate_triangles
from ..references import TextureReference


@dataclass(frozen=True)
class TextureOptions:
    seed: int = 42
    resolution: int = 640
    texture_size: int = 2048
    steps: int = 12
    roughness: float = 0.35
    match_reference_color: bool = True

    def __post_init__(self):
        integers = (self.seed, self.resolution, self.texture_size, self.steps)
        if (not np.isfinite([*integers, self.roughness]).all()
                or any(int(value) != value for value in integers)
                or self.seed < 0 or self.resolution < 256 or self.resolution % 16
                or self.texture_size < 64 or self.steps < 1 or not 0 <= self.roughness <= 1
                or not isinstance(self.match_reference_color, bool)):
            raise ValueError('Invalid texture seed, resolution, size, steps, or roughness')


def _installed_trellis_checkout() -> Path:
    spec = importlib.util.find_spec('trellis2')
    if spec is None or spec.origin is None:
        raise ImportError('Install TRELLIS.2 in the active Python environment before requesting textures')
    return Path(spec.origin).resolve().parent.parent


@dataclass(frozen=True)
class TextureRuntime:
    checkout: Path = field(default_factory=_installed_trellis_checkout)
    timeout_s: float = 900

    def __post_init__(self):
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError('Texture timeout must be positive and finite')


@dataclass(frozen=True)
class TextureResult:
    mesh: MeshGeometry
    vertex_map: np.ndarray
    workspace: Path
    metadata: dict


def check_runtime(runtime: TextureRuntime) -> dict:
    """Check the external environment and identify the worker/model dependencies."""
    root = Path(runtime.checkout).expanduser().resolve()
    if not (root / 'trellis2/pipelines/trellis2_texturing.py').is_file():
        raise FileNotFoundError(f'Expected a TRELLIS.2 checkout: {root}')
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name('_trellis.py')), '--check', str(root)],
        capture_output=True, text=True, check=False, timeout=min(runtime.timeout_s, 180),
    )
    if result.returncode:
        raise RuntimeError('TRELLIS.2 preflight failed:\n' + result.stderr[-4000:])
    return json.loads(result.stdout.splitlines()[-1])


def texture_mesh(
    mesh: MeshGeometry, reference: TextureReference, options: TextureOptions | None = None, *,
    runtime: TextureRuntime, workspace: str | Path,
) -> TextureResult:
    """Generate appearance on an existing mesh without modifying its coordinates.

    Completed inference is reused for identical geometry, tile, options and model
    identity. Each attempt has its own directory; failures retain their diagnostics.
    UV seams may duplicate vertices. The returned vertex map proves triangle identity.
    """
    from PIL import Image

    options = options or TextureOptions()
    reference.validate()
    if mesh.vertices is None or mesh.faces is None:
        raise ValueError("Texturing requires a nonempty mesh")
    vertices, faces = validate_triangles(mesh.vertices, mesh.faces)
    with Image.open(reference.image) as image:
        if image.width != image.height:
            raise ValueError('Texture conditioning requires a square image')
    identity = check_runtime(runtime)
    config = dict(asdict(options), texture_square=str(reference.image),
                  trellis_root=str(Path(runtime.checkout).expanduser().resolve()),
                  model='microsoft/TRELLIS.2-4B')
    digest = hashlib.sha256(json.dumps([config, identity], sort_keys=True).encode())
    for array in (vertices, faces):
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    digest.update(reference.image.read_bytes())
    root = Path(workspace).expanduser().resolve() / digest.hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    for marker in sorted(root.glob('attempt-*/complete.json')):
        return _load_result(vertices, faces, marker.parent, options, reference)
    work = Path(tempfile.mkdtemp(prefix='attempt-', dir=root))
    np.savez(work / 'mesh.npz', vertices=vertices, faces=faces)
    (work / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    with (work / 'texture.log').open('w') as log:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name('_trellis.py')), str(work)],
            cwd=config['trellis_root'], stdout=log, stderr=subprocess.STDOUT,
            check=False, timeout=runtime.timeout_s,
        )
    if result.returncode:
        raise RuntimeError(f'TRELLIS.2 texturing failed; diagnostics: {work / "texture.log"}')
    output = _load_result(vertices, faces, work, options, reference)
    (work / 'complete.json').write_text(json.dumps(output.metadata, indent=2) + '\n')
    return output


def _load_result(vertices, faces, work, options, reference) -> TextureResult:
    with np.load(work / 'textured.npz', allow_pickle=False) as arrays:
        vertex_map, output_faces = arrays['vertex_map'], arrays['faces']
        if (vertex_map.ndim != 1 or not len(vertex_map)
                or not np.issubdtype(vertex_map.dtype, np.integer)
                or vertex_map.min() < 0 or vertex_map.max() >= len(vertices)):
            raise ValueError('Invalid texture source vertex mapping')
        _, output_faces = validate_triangles(vertices[vertex_map], output_faces)
        if not np.array_equal(vertex_map[output_faces], faces):
            raise ValueError('Texture unwrapping changed the source triangles')
        uv, normals = arrays['uv'].copy(), arrays['normals'].copy()
        if (uv.shape != (len(vertex_map), 2) or normals.shape != (len(vertex_map), 3)
                or not np.isfinite(uv).all() or not np.isfinite(normals).all()):
            raise ValueError('Invalid texture UVs or normals')
        from PIL import Image
        for name in ('base_color.png', 'roughness.png'):
            if not (work / name).is_file():
                raise ValueError(f'Missing generated texture: {name}')
            with Image.open(work / name) as image:
                image.verify()
        material = SurfaceMaterial(str(work / 'base_color.png'), str(work / 'roughness.png'), options.roughness)
        geometry = MeshGeometry(vertices[vertex_map].copy(), output_faces.copy(), uv, normals, material)
        metadata = dict(options=asdict(options), reference=reference.metadata,
                        generation=json.loads((work / 'config.json').read_text()))
        return TextureResult(geometry, vertex_map.copy(), work, metadata)
