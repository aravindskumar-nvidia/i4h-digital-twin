# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named editing operations; CLI and Python callers share the same instructions."""
from __future__ import annotations

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
