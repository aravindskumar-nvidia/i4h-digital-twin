# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference recipes produce verified reusable assets without repository GLBs."""
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from patient_digital_twin import _reference_assets as assets
from PIL import Image


def test_download_cache_and_failed_checksum(tmp_path, monkeypatch):
    data = b'fixture GLB'
    source = {'url': 'https://example.org/reference.glb', 'checksum': 'sha256:' + hashlib.sha256(data).hexdigest()}
    calls = []
    def download(request, **kwargs):
        calls.append(request.full_url)
        return io.BytesIO(data)
    monkeypatch.setattr(assets.urllib.request, 'urlopen', download)
    first = assets._download(source, tmp_path, '.glb')
    assert assets._download(source, tmp_path, '.glb') == first
    assert calls == [source['url']]
    bad = dict(source, checksum='sha256:' + '0' * 64)
    with pytest.raises(ValueError, match='Downloaded reference checksum'):
        assets._download(bad, tmp_path, '.glb')
    assert list((tmp_path / 'downloads').iterdir()) == [first]
    first.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Cached reference checksum'):
        assets._download(source, tmp_path, '.glb')


def test_interrupted_download_is_retryable(tmp_path, monkeypatch):
    class Interrupted(io.BytesIO):
        def read(self, *args):
            raise OSError('connection lost')
    source = {'url': 'https://example.org/reference.glb', 'checksum': 'sha256:' + '0' * 64}
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *args, **kwargs: Interrupted())
    with pytest.raises(OSError, match='connection lost'):
        assets._download(source, tmp_path, '.glb')
    assert not list((tmp_path / 'downloads').iterdir())


def test_label_recipe_conversion_and_cache(tmp_path, monkeypatch):
    import trimesh

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for index in range(6):
            labels = np.zeros((10, 12), np.uint16)
            if 1 <= index <= 4:
                labels[2:8, 3:9] = 511
            image = io.BytesIO()
            Image.fromarray(labels).save(image, format='TIFF')
            archive.writestr(f'source/labels/labels{100 + index}.tif', image.getvalue())
    data = stream.getvalue()
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(data))
    metadata = {'anatomy': 'aorta', 'structures': 'structures.json',
                'preparation': {'kind': 'voxel-man-labels', 'url': 'https://example.org/source.zip',
                                'checksum': 'md5:' + hashlib.md5(data).hexdigest(),
                                'label_directory': 'source/labels', 'first_slice': 100, 'slice_count': 6},
                'extraction': {'gaussian_sigma_voxels': .6, 'marching_cubes_level': .5,
                               'voxel_spacing_mm': [1, 1, 1]}}
    (tmp_path / 'structures.json').write_text(json.dumps({'aorta': [511]}))
    model, prepared = assets.materialize(metadata, tmp_path, tmp_path / 'cache')
    assert hashlib.sha256(model.read_bytes()).hexdigest() == prepared['sha256']
    scene = trimesh.load(model, force='scene', process=False)
    assert 'aorta' in scene.graph.nodes_geometry
    assert np.isfinite(scene.bounds).all()
    assert scene.bounds[0, 0] > 0 and scene.bounds[1, 1] < 0 and scene.bounds[1, 2] < 0
    assert np.max(np.abs(scene.bounds)) < .02
    def unexpected(*args, **kwargs):
        pytest.fail('Completed reference was downloaded or converted again')
    monkeypatch.setattr(assets, '_convert_labels', unexpected)
    monkeypatch.setattr(assets.urllib.request, 'urlopen', unexpected)
    assert assets.materialize(metadata, tmp_path, tmp_path / 'cache')[0] == model
    assert not (tmp_path / 'reference.glb').exists()
