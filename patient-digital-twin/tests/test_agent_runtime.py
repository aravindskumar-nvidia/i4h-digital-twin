# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Managed Blender installation verifies official downloads and reuses installs."""
import hashlib
import io
import tarfile
from pathlib import Path

import pytest
from patient_digital_twin.agent import runtime


@pytest.mark.parametrize('valid_hash', [True, False])
def test_managed_blender_checksum_and_reuse(tmp_path, monkeypatch, valid_hash):
    version = runtime.BLENDER_VERSION
    archive = f'blender-{version}-linux-x64.tar.xz'
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:xz') as package:
        entry = tarfile.TarInfo(f'blender-{version}-linux-x64/blender')
        payload = b'#!/bin/sh\n'
        entry.size = len(payload)
        entry.mode = 0o755
        package.addfile(entry, io.BytesIO(payload))
    payload = data.getvalue()
    checksum = hashlib.sha256(payload).hexdigest() if valid_hash else '0' * 64
    responses = {
        runtime.RELEASES + f'blender-{version}.sha256': f'{checksum}  {archive}'.encode(),
        runtime.RELEASES + archive: payload,
    }
    monkeypatch.delenv('BLENDER_EXECUTABLE', raising=False)
    monkeypatch.setattr(runtime.shutil, 'which', lambda _: None)
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(runtime.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(runtime.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(runtime.urllib.request, 'urlopen', lambda url, **kwargs: io.BytesIO(responses[url]))
    if not valid_hash:
        with pytest.raises(ValueError, match='SHA-256'):
            runtime.ensure_blender()
        assert not list(tmp_path.rglob('blender-*/blender'))
    else:
        path = runtime.ensure_blender()
        assert Path(path).is_file()
        monkeypatch.setattr(runtime.urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('Unexpected download'))
        assert runtime.ensure_blender() == path
