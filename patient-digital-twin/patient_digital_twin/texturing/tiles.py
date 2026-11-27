# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tissue tiles: checksum-verified images that condition TRELLIS.2 texturing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..anatomy import canonical_name

DEFAULT_TILES = Path(__file__).resolve().parents[1] / 'data' / 'textures'


@dataclass(frozen=True)
class TextureReference:
    """A checksum-verified tile and its tissue surface; physical scale is descriptive."""
    image: Path
    metadata: dict

    def validate(self) -> None:
        """Verify provenance and image content; license metadata is optional."""
        for key in ('creator', 'source', 'changes', 'surface', 'sha256'):
            if not self.metadata.get(key):
                raise ValueError(f'Texture metadata requires {key}')
        if hashlib.sha256(self.image.read_bytes()).hexdigest() != self.metadata['sha256']:
            raise ValueError(f'Texture checksum mismatch: {self.image}')

    @classmethod
    def load(cls, path: str | Path) -> TextureReference:
        path = Path(path).expanduser().resolve()
        metadata = json.loads(path.read_text())
        reference = cls((path.parent / metadata['file']).resolve(), metadata)
        reference.validate()
        return reference


class MissingTextureDefault(ValueError):
    """No registered organ or tissue-family tile exists for this structure."""


def default_texture_reference(organ: str) -> TextureReference:
    """Load the packaged organ tile; never substitute an unrelated tissue."""
    records = json.loads((DEFAULT_TILES / 'defaults.json').read_text())
    name = canonical_name(organ)
    key = name if name in records else next(
        (key for key, record in records.items() if name in record.get('applies_to', [])), None
    )
    if key is None:
        raise MissingTextureDefault(
            f'No default texture for {organ!r}; supply a texture override or '
            'set default_textures=False to texture only explicitly selected organs'
        )
    metadata = dict(records[key], selection={'structure': name, 'default': key,
                                           'match': 'organ' if key == name else 'tissue_family'})
    reference = TextureReference(DEFAULT_TILES / metadata['file'], metadata)
    reference.validate()
    return reference
