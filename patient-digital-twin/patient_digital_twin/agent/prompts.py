# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every instruction sent to the editing agent; CLI and Python callers share them."""
from __future__ import annotations

import json
from pathlib import Path

from ..references import MAJOR_BRANCHES, is_repair_vessel

TASKS = {
    'complete': """Complete missing sections of the selected anatomy using the supplied anatomical
reference. First map surviving regions, gaps and scan boundaries. Establish the
patient-supported course and scale before creating geometry. Bridge a gap only
when its endpoints and reference anatomy support that connection. Preserve
meaningful fragments and branch origins. Do not replace the patient with a reference
mesh. Distinguish missing segmentation from anatomy outside scan coverage; do not
extend scan-boundary endpoints just to resemble a whole reference organ.
Work from overall shape to topology to surface morphology. Record each inferred
region and the evidence used. If the reference cannot resolve a connection, report
that uncertainty instead of inventing certainty. Verify whole-organ course and
edited junctions; add viewing angles where a connection remains ambiguous.""",
    'repair': """Repair demonstrated defects in the selected anatomy. Inspect connected components,
open boundaries, non-manifold edges, winding, self-intersections, collapsed spans
and suspicious bridges. Classify defects before editing: a branch, fold, opening
or narrow segment is not an artifact merely because it is unusual. Use the supplied
reference and surviving patient geometry to resolve ambiguous connections.
Prioritize wrong connections and course/caliber defects, then holes and orientation,
then local surface quality. Use the smallest repair that resolves the actual defect;
do not enforce a tube or genus-zero topology on every organ. Recheck the defect
list after editing and report which checks passed, which defects remain and where
repair required inferred anatomy. Do not claim visual plausibility proves accuracy.""",
    'refine': """Refine the surface of the selected anatomy without completing missing regions or
changing connectivity, branches or overall course. Identify scan stair-stepping,
faceting and localized noise. Smooth these selectively while preserving caliber,
organ volume, thin structures, landmarks and real folds such as colon haustra.
Compare pre/post bounds, surface area and volume when the surface is closed;
inspect silhouettes and close-ups at the same camera settings. Reduce or undo a
smoothing change that erases anatomical detail or materially shrinks the anatomy.
Do not conceal gaps or incorrect connections with smoothing. Report structural
defects that need a separate repair operation. Preserve existing UVs and materials.""",
    'texture': """Correct materials and texture mapping on the selected anatomy; do not move vertices,
change topology, smooth surfaces, or alter patient placement. Inspect existing maps,
UV seams, stretching, texel scale, color space, roughness and material bindings.
Determine which tissue surface is visible (for example exterior serosa versus
interior mucosa) before using any supplied tissue reference. Do not use a convenient
but anatomically different tissue tile. Preserve anatomical detail and avoid baked
lighting, repeated conspicuous patterns and unjustified glossy or saturated surfaces.
Use supplied tissue references as evidence. Anatomy reference geometry alone does
not establish accurate tissue color. Without suitable appearance evidence, limit
changes to demonstrable mapping/material defects and explicitly report that tissue
appearance accuracy could not be established. Do not invent pathology or claim
patient-specific color from a CT. Compare before/after under identical neutral
lighting. Keep export-compatible image textures and Principled BSDF materials.""",
}

COMMON = """Make useful edits early. Start with one whole-target before view and a batched
geometry check for the requested operation. Inspect the reference features needed
for the first proposed edit and capture its before-detail view. State a brief plan,
then apply that bounded edit once
the evidence supports it. Resolve uncertainty affecting that edit with focused
inspection; avoid an exhaustive survey or repeated planning before starting.
Edit only the named targets; leave
other anatomy, CT registration, units, object identities and patient placement intact.
Use references as read-only guidance. Keep arteries and veins distinct. Do not merge
separately labeled structures or manufacture branch extensions unless major branching
was explicitly enabled. Preserve existing supported branches unless the supplied
reference-completion instructions explicitly request a main-aorta-only result.
Batch related measurements and compatible local edits in one Python call. Check
the relevant geometry metrics after each batch; capture another image when the
result needs visual verification or a specific uncertainty must be resolved.
Reuse unchanged before/reference views. Stop iterating when the requested work
is complete and checks pass; do not spend the remaining budget on optional polish.
Reserve time for comparison images, the required report and check_asset.
Before finishing, save comparable whole-target and detail views, run check_asset,
and resolve reported failures. Summarize actual changes, checks performed, remaining
defects and inferred anatomy. An unresolved defect is not a successful repair.
"""


def build_prompt(task: str | None, targets=(), custom: str = '', *, references=(),
                 major_branching: bool = False) -> str:
    """Build explicit scope and operation instructions without interpreting free text."""
    if task is None:
        if not custom.strip():
            raise ValueError('Choose --agent-task or provide --agent-prompt')
        return custom
    if task not in TASKS:
        raise ValueError(f'Unknown agent task: {task}')
    names = tuple(dict.fromkeys(targets or references))
    if not names:
        raise ValueError('Named tasks require --agent-target or --anatomy-reference targets')
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError('Agent targets must be nonempty structure names')
    if task in {'complete', 'repair'} and set(names) - set(references):
        raise ValueError('Completion and repair require an anatomy reference for every target')
    if set(references) - set(names):
        raise ValueError('Anatomy references must belong to the selected agent targets')
    if major_branching and task not in {'complete', 'repair'}:
        raise ValueError('--major-branching applies only to complete or repair')
    return (f'Operation: {task}.\nTargets: {", ".join(names)}.\n'
            + TASKS[task] + '\n\n' + COMMON
            + ('\nAdditional user instructions:\n' + custom if custom.strip() else ''))


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


def reference_prompt(records: dict, operation: str | None, report: str, fragment_area_mm2: float,
                     include_major_branches: bool) -> str:
    """Instructions for using prepared reference scenes in a refine, texture or completion task."""
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


def session_prompt(task: str, settings: dict, workspace: Path, worker: Path) -> str:
    """Wrap the task with this session's scene, its check command and any cleanup history."""
    prompt = (f"Edit the active I4H scene (session {settings['token']}).\n{task}\n"
              f"Scene check: exec(compile(open({str(worker)!r}).read(), {str(worker)!r}, 'exec')); "
              f"check_asset({str(workspace)!r})")
    if settings.get('cleanup_components'):
        prompt += (f"\nCleanup fragment history is in {str(workspace / 'edit.json')!r}, "
                   "under cleanup_components. Review it before editing. Cleanup removed tiny/distant "
                   "secondary islands and retained nearby fragments as separate geometry. Bounds are "
                   "in original structure-local millimeters; component IDs describe the pre-cleanup "
                   "mesh, not current vertex indices. Inspect the current mesh to identify surviving "
                   "pieces. Proximity is evidence for review, not proof of a connection. Preserve "
                   "meaningful nearby fragments; connect them only when the selected operation, "
                   "patient geometry and references support it. Refinement and texture tasks must "
                   "not change connectivity.\n")
    return prompt
