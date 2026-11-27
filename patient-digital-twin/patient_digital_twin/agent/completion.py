# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference preparation, completion-report validation and provenance for agent edits."""
from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..references import load_references, write_reference_files
from .prompts import reference_prompt


def prepare_references(workspace: Path, references: Mapping, include_major_branches: bool = False,
                       fragment_area_mm2: float = 1., *, operation: str | None = None) -> str:
    """Copy each reference with its provenance into the workspace and return its task instructions."""
    loaded = load_references(references, include_major_branches)
    settings = json.loads((workspace / 'edit.json').read_text())
    records = {}
    for organ, item in loaded.items():
        target = '/HumanBody/Anatomy/' + organ
        if target not in settings['paths']:
            raise ValueError(f'Reference target is not in the patient asset: {organ}')
        folder = workspace / 'references' / organ
        # Per-reference credit and listing files are generated here, not stored in the catalog.
        metadata = write_reference_files(item, folder)
        records[organ] = dict(target=target, file=str(folder / 'reference.glb'), nodes=item['nodes'],
                              branches=item['branches'], metadata=metadata)
    settings['references'] = records
    settings['completion'] = dict(include_major_branches=include_major_branches,
                                  fragment_area_mm2=fragment_area_mm2, require_reference_review=True,
                                  require_colon_shape='colon' in loaded and operation not in {'refine', 'texture'},
                                  require_connected=operation not in {'refine', 'texture'})
    (workspace / 'edit.json').write_text(json.dumps(settings))
    return reference_prompt(records, operation, str(workspace / 'completion.json'), fragment_area_mm2,
                            include_major_branches)


def validate_completion(stage, workspace, settings):
    import trimesh
    from pxr import UsdGeom

    report_path = workspace / 'completion.json'
    if not report_path.is_file():
        raise ValueError('Write completion.json describing the reference-guided edits')
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict):
        raise ValueError('Completion report must be keyed by target structure')
    result = {}
    for organ, reference in settings['references'].items():
        mesh = UsdGeom.Mesh.Get(stage, reference['target'])
        if not mesh or mesh.ComputeVisibility() == 'invisible':
            raise ValueError(f'Completed target must be present and visible: {organ}')
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        # Exact coordinate welding ignores UV seam duplication without bridging gaps.
        points, inverse = np.unique(vertices, axis=0, return_inverse=True)
        surface = trimesh.Trimesh(points, inverse[faces], process=False)
        components = surface.split(only_watertight=False, repair=False)
        if len(components) != 1 and settings['completion'].get('require_connected', True):
            raise ValueError(f'{organ}: completion requires one connected surface, found {len(components)}')
        entry = report.get(organ, {})
        if not isinstance(entry, dict):
            raise ValueError(f'{organ}: completion report must be an object')
        for key in ('generated_regions', 'removed_fragments', 'retained_fragments', 'uncertainties'):
            if not isinstance(entry.get(key), list):
                raise ValueError(f'{organ}: completion report requires {key}')
        validation = {'connected_components': len(components), 'watertight': bool(surface.is_watertight),
                      'euler_characteristic': int(surface.euler_number),
                      'anatomical_plausibility': 'not_automatically_verified'}
        if organ == 'colon' and settings['completion'].get('require_colon_shape'):
            if not surface.is_watertight or not surface.is_winding_consistent or surface.euler_number != 2:
                raise ValueError('colon: shape correction requires a watertight, consistently oriented '
                                 f'genus-zero surface; repair holes/handles before publishing '
                                 f'(Euler characteristic {surface.euler_number})')
        result[organ] = dict(reference=reference['metadata']['id'], agent_report=entry,
                             validation=validation)
        if settings['completion'].get('require_reference_review'):
            result[organ]['reference_review'] = _check_reference_review(entry, workspace, organ)
    return dict(settings=settings['completion'], structures=result)


def _check_reference_review(entry, workspace, organ):
    """Require inspectable evidence; never equate an agent claim with anatomy validation."""
    from PIL import Image

    review = entry.get('reference_review')
    if not isinstance(review, dict):
        raise ValueError(f'{organ}: completion requires reference_review and comparison images')
    for key in ('observed_features', 'preserved_features', 'generated_features', 'unresolved_features'):
        values = review.get(key)
        if (not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values)
                or (key == 'observed_features' and not values)):
            raise ValueError(f'{organ}: reference_review requires {key}')
    for key in ('reference_images', 'result_images'):
        paths = review.get(key)
        if not isinstance(paths, list) or not paths:
            raise ValueError(f'{organ}: reference_review requires {key}')
        for relative in paths:
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError(f'{organ}: comparison images must be workspace-relative')
            path = (workspace / relative).resolve()
            if not path.is_relative_to(workspace.resolve()) or not path.is_file():
                raise ValueError(f'{organ}: missing or external comparison image: {relative}')
            with Image.open(path) as image:
                image.verify()
    return dict(review)


def publish_provenance(workspace, folder, report):
    destination = folder / 'refinement'
    # Keep each older report with its relative reference/inspection paths intact.
    # The latest report stays at refinement/report.json for existing consumers.
    if destination.exists():
        previous = [path for path in destination.iterdir() if path.name != 'history']
        if previous:
            archive = destination / 'history' / uuid4().hex
            archive.mkdir(parents=True)
            for path in previous:
                shutil.move(str(path), archive / path.name)
    shutil.copytree(workspace / 'references', destination / 'references',
                    ignore=shutil.ignore_patterns('*.glb'))
    for path in destination.glob('references/*/metadata.json'):
        metadata = json.loads(path.read_text())
        metadata['source_filename'] = metadata.pop('file')
        metadata['geometry_included'] = False
        path.write_text(json.dumps(metadata, indent=2) + '\n')
    for entry in report['structures'].values():
        review = entry.get('reference_review')
        if review:
            for key in ('reference_images', 'result_images'):
                copied = []
                for relative in review[key]:
                    # Validation already checked that the resolved source is local.
                    source = (workspace / relative).resolve()
                    target = destination / 'inspection' / source.relative_to(workspace.resolve())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied.append(str(target.relative_to(destination)))
                review[key] = copied
    (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
