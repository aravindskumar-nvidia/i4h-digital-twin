# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download pinned reference sources and prepare local GLBs on demand."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _checksum(path: Path, algorithm: str = 'sha256') -> str:
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _download(source: dict, cache: Path, suffix: str) -> Path:
    algorithm, expected = source['checksum'].split(':', 1)
    if algorithm not in {'sha256', 'md5'} or not re.fullmatch(r'[0-9a-f]+', expected):
        raise ValueError('Reference download requires a supported pinned checksum')
    destination = cache / 'downloads' / (algorithm + '-' + expected + suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if _checksum(destination, algorithm) != expected:
            raise ValueError(f'Cached reference checksum mismatch: {destination}')
        return destination
    temporary = None
    try:
        request = urllib.request.Request(source['url'], headers={'User-Agent': 'i4h-digital-twin'})
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    stream.write(chunk)
        if _checksum(temporary, algorithm) != expected:
            raise ValueError(f'Downloaded reference checksum mismatch: {source["url"]}')
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def materialize(metadata: dict, directory: Path, cache: Path | None = None) -> tuple[Path, dict]:
    """Resolve a recipe into a checksum-verified model and effective metadata.

    Local schema-1 assets remain supported by the caller. Recipe sources and
    generated GLBs live outside the repository. A completed manifest is published
    last, allowing interrupted and concurrent attempts to retain consistent caches.
    """
    default = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'i4h/references'
    cache = Path(cache or os.environ.get('I4H_REFERENCE_CACHE', str(default))).expanduser().resolve()
    recipe = metadata['preparation']
    kind = recipe['kind']
    if kind == 'download-glb':
        model = _download(recipe, cache, '.glb')
        if _checksum(model) != metadata['sha256']:
            raise ValueError('Prepared reference differs from its manifest')
        return model, metadata
    if kind != 'voxel-man-labels':
        raise ValueError(f'Unsupported reference preparation: {kind}')
    labels = json.loads((directory / metadata['structures']).read_text())
    identity = dict(recipe=recipe, extraction=metadata['extraction'], labels=labels,
                    anatomy=metadata['anatomy'], converter_version=1,
                    libraries={name: version(name) for name in ('numpy', 'scipy', 'scikit-image', 'trimesh', 'Pillow')})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = cache / 'prepared' / key
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'complete.json'
    if not manifest.is_file():
        archive = _download(recipe, cache, '.zip')
        with tempfile.TemporaryDirectory(dir=folder, prefix='attempt-') as temporary:
            work = Path(temporary)
            model = work / 'reference.glb'
            _convert_labels(archive, labels, metadata, model)
            digest = _checksum(model)
            record: dict = dict(file=digest + '.glb', sha256=digest, bytes=model.stat().st_size,
                          preparation_identity=identity)
            os.replace(model, folder / record['file'])
            marker = work / 'complete.json'
            marker.write_text(json.dumps(record, indent=2) + '\n')
            os.replace(marker, manifest)
    record = json.loads(manifest.read_text())
    model = folder / record['file']
    if _checksum(model) != record['sha256']:
        raise ValueError(f'Prepared reference checksum mismatch: {model}')
    return model, dict(metadata, **record)


def _convert_labels(archive: Path, labels: dict, metadata: dict, output: Path) -> None:
    """Reproduce the documented 1 mm VOXEL-MAN surface extraction in glTF Y-up."""
    import trimesh
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    recipe = metadata['preparation']
    extraction = metadata['extraction']
    ids = [label for group in labels.values() for label in group]
    with zipfile.ZipFile(archive) as source:
        members = [name for name in source.namelist()
                   if name.startswith(recipe['label_directory'] + '/')
                   and re.fullmatch(r'labels\d+\.tif', Path(name).name)]
        members.sort(key=lambda name: int(Path(name).stem.removeprefix('labels')))
        if len(members) != recipe['slice_count']:
            raise ValueError('Reference archive has unexpected label slices')
        numbers = [int(Path(name).stem.removeprefix('labels')) for name in members]
        if numbers != list(range(recipe['first_slice'], recipe['first_slice'] + recipe['slice_count'])):
            raise ValueError('Reference archive has missing or duplicate label slices')
        # Keep only the selected binary mask in memory, not the full label volume.
        slices = []
        for member in members:
            with source.open(member) as stream, Image.open(stream) as image:
                slices.append(np.isin(np.asarray(image), ids))
    mask = np.stack(slices)
    coordinates = np.nonzero(mask)
    if not len(coordinates[0]):
        raise ValueError('Selected reference labels are absent')
    lower = np.maximum([axis.min() - 3 for axis in coordinates], 0)
    upper = np.minimum([axis.max() + 4 for axis in coordinates], mask.shape)
    region = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    field = gaussian_filter(mask[region].astype(np.float32), sigma=extraction['gaussian_sigma_voxels'])
    vertices, faces, _, _ = marching_cubes(field, extraction['marching_cubes_level'])
    vertices += lower
    vertices = vertices[:, [2, 1, 0]]
    vertices[:, 2] *= -1
    vertices *= np.asarray(extraction['voxel_spacing_mm']) * .001
    vertices = vertices[:, [0, 2, 1]] * [1, 1, -1]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces[:, [0, 2, 1]], process=False)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=metadata['anatomy'], geom_name=metadata['anatomy'])
    output.write_bytes(scene.export(file_type='glb'))
