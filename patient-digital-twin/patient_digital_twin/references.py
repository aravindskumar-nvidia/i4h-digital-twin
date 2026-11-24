# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified anatomy references and tissue texture tiles."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAJOR_BRANCHES = ('brachiocephalic_artery', 'left_common_carotid_artery', 'left_subclavian_artery',
                  'celiac_trunk', 'superior_mesenteric_artery', 'inferior_mesenteric_artery',
                  'left_renal_artery', 'right_renal_artery', 'left_common_iliac_artery', 'right_common_iliac_artery')



def is_repair_vessel(name):
    from .anatomy import is_vessel
    return is_vessel(name) or name in {
        'portal_vein_and_splenic_vein', 'pulmonary_artery', 'celiac_trunk',
        'superior_mesenteric_artery', 'inferior_mesenteric_artery',
        'renal_artery_left', 'renal_artery_right',
    }


def parse_references(items, option="--anatomy-reference"):
    result = {}
    for item in items:
        organ, separator, path = item.partition('=')
        if not separator or not organ or not path or organ in result:
            raise ValueError(f'{option} requires unique ORGAN=METADATA_JSON entries')
        result[organ] = path
    return result


def load_references(references, include_major_branches=False, fragment_area_mm2=1., *, cache: Path | None = None):
    if not np.isfinite(fragment_area_mm2) or fragment_area_mm2 < 0:
        raise ValueError('Fragment area must be finite and nonnegative')
    if include_major_branches and not any(is_repair_vessel(name) for name in references):
        raise ValueError('--major-branching requires a vessel reference')
    loaded = {}
    for organ, path in references.items():
        path = Path(path).expanduser().resolve()
        metadata = json.loads(path.read_text())
        if metadata['license']['spdx'] != 'CC-BY-4.0':
            raise ValueError('Anatomy references require verified CC BY 4.0 metadata')
        for required in (metadata['license']['text'], metadata['source']['evidence'], 'ATTRIBUTION.txt'):
            if not (path.parent / required).is_file():
                raise ValueError(f'Missing reference credit/license evidence: {required}')
        if metadata['units'] != 'meters' or metadata['up_axis'] != 'Y':
            raise ValueError('Reference GLBs must use meters and Y-up')
        selections = metadata.get('selections', {})
        nodes = selections.get(organ + '_main', selections.get(organ + '_regions', []))
        if not nodes and metadata['structures'].endswith('.csv'):
            from .anatomy import canonical_name
            with (path.parent / metadata['structures']).open() as stream:
                nodes = [row['node_name'] for row in csv.DictReader(stream)
                         if canonical_name(row['label']) == organ]
        branches = []
        if is_repair_vessel(organ) and include_major_branches:
            branches = selections.get(organ + '_major_branches', [])
            if organ == 'aorta' and not branches:
                branches = [n for n in selections.get('available_arteries', [])
                            if any(b in n for b in MAJOR_BRANCHES)]
            if not branches:
                raise ValueError(f'This reference has no named major branches for {organ}')
        if not nodes:
            raise ValueError(f'Reference has no explicit node selection for {organ}')
        if 'preparation' in metadata:
            from ._reference_assets import materialize
            model, metadata = materialize(metadata, path.parent, cache)
        else:
            model = path.parent / metadata['file']
            if model.suffix != '.glb' or hashlib.sha256(model.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError(f'Reference GLB checksum mismatch: {model}')
        loaded[organ] = dict(metadata=metadata, directory=path.parent, model=model,
                             nodes=list(dict.fromkeys(nodes + branches)), branches=branches)
    return loaded


@dataclass(frozen=True)
class TextureReference:
    """A checksum-verified tile and its tissue surface; physical scale is descriptive."""
    image: Path
    metadata: dict

    def validate(self) -> None:
        """Verify provenance and image content; license metadata is optional."""
        metadata = self.metadata
        for key in ('creator', 'source', 'changes', 'surface'):
            if not metadata.get(key):
                raise ValueError(f'Texture metadata requires {key}')
        if hashlib.sha256(self.image.read_bytes()).hexdigest() != metadata.get('sha256'):
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
    from .anatomy import canonical_name

    directory = Path(__file__).parent / 'data' / 'textures'
    records = json.loads((directory / 'defaults.json').read_text())
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
    reference = TextureReference(directory / metadata['file'], metadata)
    reference.validate()
    return reference
