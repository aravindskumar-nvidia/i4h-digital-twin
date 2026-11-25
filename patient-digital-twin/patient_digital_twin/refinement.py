# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-guided completion instructions and publication checks."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from .references import MAJOR_BRANCHES, load_references, write_reference_files

COLON_REPAIR = """For COLON, work in this order: (1) correct overall course, caliber and topology;
(2) remove segmentation stair-stepping; (3) preserve or restore regional haustra.
Use the initial overview to locate the first demonstrated course/topology defect. Correct
flattened/collapsed spans, implausible narrow bridges, accidental fusions and
shortcuts between neighboring limbs before adding surface decoration. Preserve
patient placement and the course supported by surviving anatomy, not inherited
segmentation or previous-repair errors. Local movement limits from a previous
cosmetic pass are not a reason to leave an incorrect shape unchanged.
The completed colon must be one watertight, consistently oriented, genus-zero
surface (Euler characteristic 2 with capped ends). Repair pre-existing handles;
do not merely list them as unresolved and publish an otherwise decorated mesh.
These topology checks do not prove an anatomically correct or unbranched lumen.
Record course/caliber corrections and residual shape uncertainty in the feature
review, and save whole-organ views as well as close-ups of haustra.
Inspect surface morphology as well as overall course. For colon, explicitly compare
haustral pouches and intervening folds; scan stair-stepping is not haustral detail.
Preserve visible patient folds and use the reference to guide missing surface detail
without replacing the patient course. Do not smooth the result into a featureless tube.
"""


def vessel_repair_guidance(targets, include_major_branches=False):
    """Scope vascular repair instructions to explicitly selected reference targets."""
    from .references import is_repair_vessel

    vessels = [name for name in targets if is_repair_vessel(name)]
    if not vessels:
        return ''
    prompt = f"""For selected VESSELS ({', '.join(vessels)}), correct course, caliber and
branch topology before smoothing. Locate gaps, collapsed spans, false bridges,
and branch junctions in the initial geometry check. Join fragments only when surviving
patient anatomy and the target reference support the same vessel course.
Preserve patient placement, left/right identity, meaningful taper and supported
branch origins. Do not merge arteries with veins or connect neighboring vessels
because they are close. Keep separately labeled vessels as separate objects;
do not duplicate their geometry inside another target. Preserve scan-boundary
endpoints; do not extend anatomy beyond available coverage without reporting it
as inferred. Do not impose the colon's unbranched topology on vessel trees.
Except for the explicit main-aorta-only mode below, preserve existing supported
branches. Remove only demonstrated segmentation
artifacts; never treat a real branch as an artifact solely because it is small.
Inspect the selected reference GLB for the first planned repair before editing.
Add reference views when needed to resolve a specific course or branch ambiguity.
Compare each selected branch origin and junction against the reference and surviving
patient anatomy. For veins, major branches include tributaries. Reference coverage
is guidance, not proof of patient anatomy; report missing or ambiguous junctions.
Record repaired gaps, caliber changes, branch decisions and unresolved coverage.
"""
    if include_major_branches:
        prompt += ('Include only the major branch junctions explicitly selected in each target reference. '
                   'Report missing coverage; do not invent unreferenced branches.\n')
    if 'aorta' in vessels:
        prompt += ('For aorta: ' + ('include selected major branch junctions. Expected groups: '
                   + ', '.join(MAJOR_BRANCHES) if include_major_branches else
                   'keep the main aortic course only; remove side branches without shortening the main course')
                   + '.\n')
    return prompt


def prepare_references(workspace, references, include_major_branches=False, fragment_area_mm2=1., *, operation=None):
    loaded = load_references(references, include_major_branches, fragment_area_mm2)
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
    report = str(workspace / 'completion.json')
    if operation in {'refine', 'texture'}:
        return f'''\nInspect these read-only reference scenes: {json.dumps(records)}.
Use them only for the requested {operation} operation, not anatomical completion.
Do not copy reference geometry or infer tissue appearance from geometry alone.
Save reference and result comparison images. Write {report!r} as JSON keyed by
target structure. Each entry must include lists generated_regions, removed_fragments,
retained_fragments, uncertainties, plus reference_review containing observed_features
(nonempty), preserved_features, generated_features, unresolved_features, reference_images
and result_images (nonempty workspace-relative image paths). Record unresolved defects
without repairing topology outside the requested scope.
'''
    return f'''\nComplete the target structures using these anatomical references: {json.dumps({n: {'nodes': r['nodes'], 'major_branches': r['branches'], 'description': r['metadata'].get('description', ''), 'limitations': r['metadata'].get('limitations', [])} for n, r in records.items()})}.
Reference scenes are named I4H Reference <organ> <session token>. Inspect the shape/topology relevant to the first planned edit, then act when supported. Expand inspection as subsequent edits require it.
{COLON_REPAIR if 'colon' in records else ''}
{vessel_repair_guidance(records, include_major_branches)}
If a reference lacks a requested feature, report that limitation instead of treating
connectivity as success. Reference sex and proportions are guidance, not a patient match.
They are read-only guidance: do not copy/duplicate their mesh into the patient or replace patient geometry.
Register the reference conceptually to the target's surviving anatomy; preserve target placement, scale and meaningful geometry.
Complete missing sections and connect meaningful fragments with smooth, plausible tissue geometry.
Remove only isolated artifacts smaller than {fragment_area_mm2} square millimeters of surface area; retain larger meaningful pieces and incorporate them into the continuous structure.
Each target must be one connected surface, not disconnected islands in one object. Preserve the existing target object and i4h_path.
For colon, produce a continuous unbranched colon; do not add appendix or small intestine from the reference.
Do not connect unrelated nearby vessels or shortcut adjacent colon loops. Keep UVs/materials usable after edits; no automatic retexturing occurs.
Before finishing, use Blender Python to write {report!r} as JSON keyed by target organ. Each entry must contain lists named generated_regions, removed_fragments, retained_fragments, and uncertainties, describing what you actually changed.
Each entry must also contain reference_review with nonempty observed_features (list of strings),
preserved_features, generated_features, unresolved_features (lists of strings), and
reference_images and result_images (nonempty lists of workspace-relative image paths).
Save those actual comparison images with look before reporting completion; reuse saved before/reference views when unchanged. Describe
the features visible in those images; do not claim a feature is present just because
the mesh is connected. For colon explicitly record whether haustra were preserved,
inferred, or remain unresolved. This is review evidence, not automatic anatomical validation.
Generated completions are inferred anatomy, not observations from the scan. Do not claim anatomical validation.
Return to the target I4H scene in Object Mode. The application rejects disconnected targets before publishing.
'''


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
