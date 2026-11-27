# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verified CC BY 4.0 anatomy references that guide agent shape edits.

A reference is named by an ID from the packaged catalog, by ``CATALOG.json#ID`` for
another catalog, or by the path of a standalone metadata JSON kept beside its GLB,
license text, source evidence and ``ATTRIBUTION.txt``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from ._reference_assets import materialize
from .anatomy import canonical_name, is_vessel

CATALOG = Path(__file__).resolve().parent / 'data' / 'anatomy' / 'catalog.json'
MAJOR_BRANCHES = ('brachiocephalic_artery', 'left_common_carotid_artery', 'left_subclavian_artery',
                  'celiac_trunk', 'superior_mesenteric_artery', 'inferior_mesenteric_artery',
                  'left_renal_artery', 'right_renal_artery', 'left_common_iliac_artery', 'right_common_iliac_artery')
# Vessel-like targets that are not vessel kinds in the anatomy catalog.
_REPAIR_VESSELS = frozenset({
    'portal_vein_and_splenic_vein', 'pulmonary_artery', 'celiac_trunk', 'superior_mesenteric_artery',
    'inferior_mesenteric_artery', 'renal_artery_left', 'renal_artery_right',
})


def is_repair_vessel(name: str) -> bool:
    return is_vessel(name) or name in _REPAIR_VESSELS


def parse_references(items, option='--anatomy-reference') -> dict[str, str]:
    """Parse repeated ``ORGAN=REFERENCE`` command-line values."""
    result: dict[str, str] = {}
    for item in items:
        organ, separator, reference = item.partition('=')
        if not separator or not organ or not reference or organ in result:
            raise ValueError(f'{option} requires unique ORGAN=REFERENCE entries')
        result[organ] = reference
    return result


def load_catalog(path: str | Path = CATALOG) -> dict:
    """Read a reference catalog: shared ``sources`` plus one entry per reference."""
    catalog = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
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


def catalog_reference(reference_id: str, catalog: str | Path = CATALOG) -> tuple[dict, Path, dict | None]:
    """Expand one catalog entry into ``(metadata, catalog directory, structures)``.

    The metadata has the same fields as a standalone reference metadata file.
    License text and source evidence paths are relative to the catalog directory.
    """
    path = Path(catalog).expanduser().resolve()
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


def _resolve(spec: str | Path) -> tuple[dict, Path, dict | None, bool]:
    """Return ``(metadata, directory, catalog structures, from a catalog)`` for one reference."""
    text = str(spec)
    location, separator, reference_id = text.rpartition('#')
    if separator:
        return (*catalog_reference(reference_id, location), True)
    if not text.endswith('.json'):
        return (*catalog_reference(text), True)
    path = Path(text).expanduser().resolve()
    metadata = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(metadata.get('references'), dict):
        raise ValueError(f'Select one catalog entry as {path}#REFERENCE_ID. '
                         f'Available: {", ".join(sorted(metadata["references"]))}')
    return metadata, path.parent, None, False


def resolve_references(references: Mapping[str, str | Path], include_major_branches: bool = False) -> dict[str, dict]:
    """Validate license, evidence and node selections without downloading any model."""
    if include_major_branches and not any(is_repair_vessel(name) for name in references):
        raise ValueError('--major-branching requires a vessel reference')
    resolved = {}
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
        resolved[organ] = dict(metadata=metadata, directory=directory, nodes=list(dict.fromkeys(nodes + branches)),
                               branches=branches, structures_text=structures_text,
                               labels=structures.get('labels') if structures else None)
    return resolved


def load_references(references: Mapping[str, str | Path], include_major_branches: bool = False, *,
                    cache: Path | None = None) -> dict[str, dict]:
    """Resolve references and add a checksum-verified local ``model`` for each one."""
    loaded = resolve_references(references, include_major_branches)
    for item in loaded.values():
        metadata = item['metadata']
        if 'preparation' in metadata:
            item['model'], item['metadata'] = materialize(metadata, item['directory'], cache, labels=item['labels'])
        else:
            model = item['directory'] / metadata['file']
            if model.suffix != '.glb' or hashlib.sha256(model.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError(f'Reference GLB checksum mismatch: {model}')
            item['model'] = model
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
    if item['structures_text'] is not None:
        (folder / Path(metadata['structures']).name).write_text(item['structures_text'], encoding='utf-8')
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
