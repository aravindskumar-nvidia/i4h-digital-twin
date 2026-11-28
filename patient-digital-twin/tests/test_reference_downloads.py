# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference recipes produce verified reusable assets without repository GLBs."""
import hashlib
import io
import json
import urllib.error
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
    # A corrupt cache entry is replaced, not reported on every run.
    first.write_bytes(b'corrupt')
    assert assets._download(source, tmp_path, '.glb') == first
    assert first.read_bytes() == data and len(calls) == 3


def test_interrupted_download_is_retryable(tmp_path, monkeypatch):
    class Interrupted(io.BytesIO):
        def read(self, *args):
            raise OSError('connection lost')
    source = {'url': 'https://example.org/reference.glb', 'checksum': 'sha256:' + '0' * 64}
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *args, **kwargs: Interrupted())
    with pytest.raises(OSError, match='connection lost'):
        assets._download(source, tmp_path, '.glb')
    assert not list((tmp_path / 'downloads').iterdir())


@pytest.mark.parametrize('recover', [True, False])
def test_transient_download_retries_alternate_pinned_source(tmp_path, monkeypatch, recover):
    data = b'verified reference archive'
    source = {'url': 'https://example.org/api/content',
              'alternate_urls': ['https://example.org/files/reference.zip'],
              'checksum': 'sha256:' + hashlib.sha256(data).hexdigest()}
    calls = []
    def download(request, **kwargs):
        calls.append(request.full_url)
        if recover and len(calls) == 2:
            return io.BytesIO(data)
        raise urllib.error.HTTPError(request.full_url, 504, 'Gateway Time-out', {}, None)
    monkeypatch.setattr(assets.urllib.request, 'urlopen', download)
    monkeypatch.setattr(assets.time, 'sleep', lambda _: None)
    if recover:
        result = assets._download(source, tmp_path, '.zip')
        assert result.read_bytes() == data
        assert calls == [source['url'], *source['alternate_urls']]
    else:
        with pytest.raises(RuntimeError, match='after 3 attempts'):
            assets._download(source, tmp_path, '.zip')
        assert len(calls) == 3
        assert not list((tmp_path / 'downloads').iterdir())


@pytest.mark.parametrize('wrong_range', [False, True])
def test_known_size_archive_ranges_are_reassembled_or_streamed(tmp_path, monkeypatch, wrong_range):
    data = b'archive split across several ranges'
    source = {'url': 'https://example.org/source.zip', 'size_bytes': len(data),
              'checksum': 'sha256:' + hashlib.sha256(data).hexdigest()}
    calls = []
    class Response(io.BytesIO):
        status = 206
    def download(request, **kwargs):
        if request.get_header('Range') is None:
            calls.append('stream')
            return io.BytesIO(data)
        start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
        calls.append((start, end))
        response = Response(data[start:end + 1])
        response.headers = {'Content-Range': 'bytes 0-0/1' if wrong_range
                            else f'bytes {start}-{end}/{len(data)}'}
        return response
    monkeypatch.setattr(assets, '_RANGE_CHUNK_BYTES', 4)
    monkeypatch.setattr(assets.urllib.request, 'urlopen', download)
    result = assets._download(source, tmp_path, '.zip')
    assert result.read_bytes() == data
    if wrong_range:
        # A server that ignores Range is downloaded as one stream.
        assert calls[-1] == 'stream'
    else:
        assert sorted(calls) == [(i, min(i + 4, len(data)) - 1) for i in range(0, len(data), 4)]


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
    metadata['preparation']['alternate_urls'] = ['https://example.org/alternate.zip']
    metadata['preparation']['size_bytes'] = len(data)
    assert assets.materialize(metadata, tmp_path, tmp_path / 'cache')[0] == model
    assert not (tmp_path / 'reference.glb').exists()
