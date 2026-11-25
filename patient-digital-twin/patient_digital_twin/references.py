# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified anatomy references and tissue texture tiles."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
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
            raise ValueError(f'{option} requires unique ORGAN=REFERENCE entries')
        result[organ] = path
    return result


def default_catalog() -> Path:
    """The repository catalog, or the file named by I4H_ANATOMY_CATALOG."""
    override = os.environ.get('I4H_ANATOMY_CATALOG')
    return (Path(override).expanduser() if override
            else Path(__file__).resolve().parents[1] / 'assets/anatomy/catalog.json')


def load_catalog(path: str | Path | None = None) -> dict:
    """Read a reference catalog: shared ``sources`` plus one entry per reference."""
    path = Path(path).expanduser() if path else default_catalog()
    catalog = json.loads(path.read_text(encoding='utf-8'))
    if catalog.get('schema_version') != 2 or not isinstance(catalog.get('references'), dict):
        raise ValueError(f'Unsupported anatomy reference catalog: {path}')
    return catalog


def _merge(base: dict, extra: dict) -> dict:
    """Entry values override shared ones; nested objects merge and limitations accumulate."""
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        elif key == 'limitations':
            result[key] = [*result.get(key, []), *value]
        else:
            result[key] = value
    return result


def catalog_reference(reference_id: str, catalog: str | Path | None = None) -> tuple[dict, Path, dict | None]:
    """Expand one catalog entry into ``(metadata, catalog directory, structures)``.

    The metadata has the same fields as a standalone reference metadata file.
    License text and source evidence paths are relative to the catalog directory.
    """
    path = (Path(catalog).expanduser() if catalog else default_catalog()).resolve()
    data = load_catalog(path)
    if reference_id not in data['references']:
        raise ValueError(f'Unknown anatomy reference {reference_id!r} in {path}. '
                         f'Available: {", ".join(sorted(data["references"]))}')
    entry = dict(data['references'][reference_id])
    shared = entry.pop('source_ref', None)
    if shared is not None and shared not in data.get('sources', {}):
        raise ValueError(f'Reference {reference_id!r} names an unknown source: {shared!r}')
    structures = entry.pop('structures', None)
    download = entry.pop('download', None)
    metadata = _merge(_merge(data.get('defaults', {}), data['sources'][shared] if shared else {}), entry)
    metadata.update(id=reference_id, schema_version=2)
    metadata.setdefault('file', 'reference.glb')
    if download:
        # One pinned original GLB: the checksum identifies the source and the prepared model.
        metadata.update(sha256=download['sha256'], preparation=dict(
            kind='download-glb', url=download['url'], checksum='sha256:' + download['sha256']))
        if 'bytes' in download:
            metadata['bytes'] = download['bytes']
        metadata['source'] = dict(metadata.get('source', {}), download_url=download['url'], sha256=download['sha256'])
    if structures:
        metadata['structures'] = structures['file']
    return metadata, path.parent, structures


def _structures_text(structures: dict) -> str:
    """Render the catalog's node table or label mapping as its standalone file."""
    if 'rows' in structures:
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(structures['columns'])
        writer.writerows(structures['rows'])
        return stream.getvalue()
    return json.dumps(structures['labels'], indent=2) + '\n'


def _resolve(spec) -> tuple[dict, Path, dict | None, bool]:
    """Accept a catalog ID, CATALOG.json#ID, or a standalone metadata JSON path."""
    text = str(spec)
    location, separator, reference_id = text.rpartition('#')
    if separator and Path(location).expanduser().is_file():
        return (*catalog_reference(reference_id, location), True)
    path = Path(text).expanduser().resolve()
    if path.is_file():
        metadata = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(metadata.get('references'), dict):
            raise ValueError(f'Select one catalog entry as {path}#REFERENCE_ID. '
                             f'Available: {", ".join(sorted(metadata["references"]))}')
        return metadata, path.parent, None, False
    packaged = default_catalog()
    if packaged.is_file():
        if text in load_catalog(packaged)['references']:
            return (*catalog_reference(text, packaged), True)
        # Earlier releases addressed each catalog entry as GROUP/NAME/metadata.json.
        if path.name == 'metadata.json' and len(path.parents) > 2 and (path.parents[2] / 'catalog.json').is_file():
            legacy = path.parents[1].name + '-' + path.parent.name
            if legacy in load_catalog(path.parents[2] / 'catalog.json')['references']:
                return (*catalog_reference(legacy, path.parents[2] / 'catalog.json'), True)
        available = ', '.join(sorted(load_catalog(packaged)['references']))
        raise ValueError(f'Anatomy reference not found: {text}. Use a catalog ID ({available}), '
                         'CATALOG.json#ID, or a metadata JSON path')
    raise ValueError(f'Anatomy reference not found: {text}. No catalog at {packaged}; '
                     'use CATALOG.json#ID or a metadata JSON path')


def load_references(references, include_major_branches=False, fragment_area_mm2=1., *, cache: Path | None = None):
    if not np.isfinite(fragment_area_mm2) or fragment_area_mm2 < 0:
        raise ValueError('Fragment area must be finite and nonnegative')
    if include_major_branches and not any(is_repair_vessel(name) for name in references):
        raise ValueError('--major-branching requires a vessel reference')
    loaded = {}
    for organ, spec in references.items():
        metadata, directory, structures, cataloged = _resolve(spec)
        if metadata['license']['spdx'] != 'CC-BY-4.0':
            raise ValueError('Anatomy references require verified CC BY 4.0 metadata')
        # Catalog entries generate their attribution; standalone folders must supply it.
        required_files = [metadata['license']['text'], metadata['source']['evidence']]
        if cataloged:
            if not metadata.get('attribution') or not metadata.get('changes'):
                raise ValueError(f'Catalog reference requires attribution and changes: {metadata["id"]}')
        else:
            required_files.append('ATTRIBUTION.txt')
        for required in required_files:
            if not (directory / required).is_file():
                raise ValueError(f'Missing reference credit/license evidence: {required}')
        structures_text: str | None
        if structures is not None:
            structures_text = _structures_text(structures)
        else:
            listing = directory / metadata.get('structures', '')
            structures_text = listing.read_text(encoding='utf-8') if metadata.get('structures') and listing.is_file() else None
        if metadata['units'] != 'meters' or metadata['up_axis'] != 'Y':
            raise ValueError('Reference GLBs must use meters and Y-up')
        selections = metadata.get('selections', {})
        nodes = selections.get(organ + '_main', selections.get(organ + '_regions', []))
        if not nodes and structures_text is not None and metadata['structures'].endswith('.csv'):
            from .anatomy import canonical_name
            nodes = [row['node_name'] for row in csv.DictReader(io.StringIO(structures_text))
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
            model, metadata = materialize(metadata, directory, cache,
                                          labels=structures.get('labels') if structures else None)
        else:
            model = directory / metadata['file']
            if model.suffix != '.glb' or hashlib.sha256(model.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError(f'Reference GLB checksum mismatch: {model}')
        loaded[organ] = dict(metadata=metadata, directory=directory, model=model,
                             nodes=list(dict.fromkeys(nodes + branches)), branches=branches,
                             structures_text=structures_text)
    return loaded


def attribution_text(metadata: dict) -> str:
    """Credit, source, license and change notice for one prepared reference."""
    lines = [metadata['attribution'], 'Source: ' + metadata['source']['url'],
             'License: CC BY 4.0 (LICENSE.txt)', 'Changes: ' + metadata['changes']]
    if metadata.get('library_credits'):
        lines.append('Library credits: ' + ', '.join(metadata['library_credits']))
    lines.append('Source evidence: ' + metadata['source']['evidence'])
    return '\n'.join(lines) + '\n'


def write_reference_files(item: dict, folder: Path) -> dict:
    """Generate a self-contained reference folder and return its portable metadata.

    Writes the model, license text, source evidence, attribution, structure listing
    and metadata.json so the folder carries complete provenance without the catalog.
    """
    folder.mkdir(parents=True)
    shutil.copy2(item['model'], folder / 'reference.glb')
    metadata = dict(item['metadata'], file='reference.glb')
    evidence = 'source' + Path(metadata['source']['evidence']).suffix
    shutil.copy2(item['directory'] / metadata['license']['text'], folder / 'LICENSE.txt')
    shutil.copy2(item['directory'] / metadata['source']['evidence'], folder / evidence)
    metadata['license'] = dict(metadata['license'], text='LICENSE.txt')
    metadata['source'] = dict(metadata['source'], evidence=evidence)
    # Credits contain non-ASCII names; never depend on the locale encoding.
    (folder / 'ATTRIBUTION.txt').write_text(attribution_text(metadata), encoding='utf-8')
    if item.get('structures_text') is not None:
        (folder / Path(metadata['structures']).name).write_text(item['structures_text'], encoding='utf-8')
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


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
