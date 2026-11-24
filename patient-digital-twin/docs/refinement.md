# Patient refinement

Refine imported patient anatomy before export. `PatientWorkflow` combines surface
cleanup, optional agent editing and texture generation with the existing exporters.

```mermaid
flowchart LR
    A[Cleanup] --> B[Optional agent refinement]
    B --> C[Texture generation]
    C --> D[Centerlines and export]
```

Agent editing runs only when `agent=AgentOptions(...)` is supplied. Surface cleanup runs by default using 60 iterations of windowed-sinc smoothing
with pass band 0.01. For sparse meshes, the filter increases the pass band if
needed to keep volume change within 2% (surface area for open meshes); requested
and effective settings are recorded in the report. Set `cleanup=None`  to skip it,
including when working with an already textured mesh. Other omitted operations
are skipped. The input `HumanBody` is copied for the run. Failed runs
retain their intermediate assets under the workspace.

## Named agent operations

Use `--agent-task` for routine edits. No verbal prompt is required.

| Task | Intended changes | Preserved |
| --- | --- | --- |
| `complete` | Infer missing sections using a matching anatomy reference | Surviving patient anatomy and placement |
| `repair` | Correct demonstrated gaps, bad connections and topology defects | Meaningful anatomy and reference-supported branches |
| `refine` | Reduce stepping and surface noise | Course, connectivity, caliber and anatomical folds |
| `texture` | Correct existing materials, UV mapping and appearance | Geometry and placement |

For example, add these flags to a generation command:

```bash
--agent-task complete \
--anatomy-reference colon=patient-digital-twin/assets/anatomy/voxel-man/colon-male-15882019/metadata.json
```

Targets default to the reference keys. Otherwise use `--agent-target liver` (or
several space-separated names). `complete` and `repair` require references for
every target. `--major-branching` remains an explicit opt-in for these two tasks.
`--agent-prompt` remains available alone, or adds case-specific instructions to a task.

For an existing textured USD, use the editor directly:

```bash
patient-digital-twin-agent --input patient.usdc --output corrected \
  --backend codex --agent-task texture --agent-target liver
```

The texture operation corrects existing materials and maps. It does not invoke
Trellis/Pixal texture generation. Use the existing `--default-textures` / `--texture`
options for tile-conditioned generation first, then edit that output. Later texture
generation would overwrite agent changes, so combining it with the texture task is
rejected. Anatomy geometry references are not evidence of tissue color; the prompt
requires uncertainty to be reported when suitable appearance evidence is absent.

Saved instructions live in `agent/prompts.py`. Each run writes the composed task
and reference instructions to its agent workspace's `task_prompt.txt`; resume reuses
that text. Task instructions guide the model; they are not additional automatic
anatomical-accuracy checks. Surface and texture tasks do not trigger colon completion
requirements. Budget defaults and the existing SDK validation loop still apply.

## Python

```python
from patient_digital_twin import CleanupOptions, PatientWorkflow

# body is an existing HumanBody from any supported importer.
workflow = PatientWorkflow(cleanup=CleanupOptions(hole_radius_mm=2.0))
output = workflow.run(body, "refined.usdc")
```

For reference-guided completion, supply an explicit agent configuration.

```python
from pathlib import Path
from patient_digital_twin import AgentOptions, PatientWorkflow

workflow = PatientWorkflow(
    agent=AgentOptions(
        task="complete",
        backend="codex",
        token_budget=100_000,
        anatomy_references={
            "colon": Path("assets/anatomy/voxel-man/colon-male-15882019/metadata.json"),
        },
    ),
    workspace="refinement-work",
)
output = workflow.run(body, "refined-bundle", format="bundle")
```

Reference metadata resolves on demand. HRA models are downloaded directly, while
VOXEL-MAN label images are converted into GLB. Files are reused from the local
`I4H_REFERENCE_CACHE` directory (default `~/.cache/i4h/references`, honoring
`XDG_CACHE_HOME`). The first VOXEL-MAN request downloads approximately 395 MB.

Completion operates on existing structures. Requests for entirely absent meshes
raise an error. The completion report records inferred regions and uncertainty.
Connectedness is checked before publication. New reference-guided colon runs
prioritize course, caliber, and topology before surface detail; publication requires
a watertight, consistently oriented genus-zero surface (Euler characteristic 2).
This catches residual handles but does not certify an anatomically unbranched lumen. New reference-guided runs also require
reference/result images and a report of observed, preserved, generated, and unresolved
features. Colon guidance explicitly distinguishes haustra from scan stair-stepping.
Choose a reference that contains the requested detail: the selected HRA female colon is a smooth
outline, while the VOXEL-MAN colon shows haustral morphology. Anatomical plausibility
still requires review. The existing SDK session owns timeout, token/dollar limits and validation
repairs. Claude accepts `budget_usd` and `max_turns`; Codex accepts `token_budget`.

## Saved repair guidance

The SDK combines your `--agent-prompt` with saved Blender instructions in
`agent/__main__.py` (`INSTRUCTIONS`). When you supply anatomy references,
`refinement.py` adds the applicable repair guidance:

| Target | Saved guidance |
| --- | --- |
| Colon | `COLON_REPAIR`: course and caliber, then topology, smoothing, and haustra. |
| Aorta | `vessel_repair_guidance`: vascular repair plus main-aorta or major-branch mode. |
| Other major vessels | Same vascular repair guidance: preserve identity, taper, junctions and patient placement; never join nearby arteries and veins. |
| Other organs | General reference comparison, fragment completion and evidence requirements. |

Vascular guidance covers the catalog's vessels, including vena cava, carotid,
subclavian, brachiocephalic, iliac and pulmonary veins. It also covers explicitly
named pulmonary artery, renal arteries, mesenteric arteries, celiac trunk and the
combined portal/splenic-vein label. Only selected reference targets receive it.
It does not automatically select every vessel or supply missing references.

`--major-branching` enables major branch expansion for any supported vessel reference.
It is off by default. Non-aortic references must list exact GLB nodes under
`selections.<target>_major_branches`. Missing branch coverage stops the request.
Existing aortic branch selections still work. Without the flag, aorta uses its
main-course-only mode; other vessels preserve existing supported branches without
requesting branch expansion.

These are prompt instructions, not new automatic anatomical checks. Every target
still requires a connected surface and reference/result review evidence. New
instructions apply to new edits; they do not change an already-running SDK session.

## Texture generation

When a texture runtime is configured, every enabled, nonempty organ automatically
uses its packaged default tile. Defaults cover aorta, colon, liver, gallbladder,
urinary bladder, left/right kidneys, pancreas, spleen, and stomach.
Generated tissue-family tiles also cover both lungs and their lobes, heart and
left atrial appendage, skeletal muscles, cortical bone structures, small bowel,
and duodenum. Explicit organ defaults take priority over family mappings.
Selection metadata records the requested structure and matched default.
No per-organ configuration is needed. The existing colon default depicts mucosa,
not exterior serosa; override it when another tissue surface is needed.

Explicit `textures` entries override defaults for those organs. Set
`default_textures=False`  to use only explicit
entries; with an empty mapping this disables texturing. Without a texture runtime,
cleanup/agent workflows continue without texture generation. An organ without a default keeps its existing material and is recorded in
`retained_materials_no_default`. Use `--missing-texture error` for strict runs. Checksum and worker
failures still stop the run. No unrelated tissue is substituted. Defaults are packaged with the Python
distribution and checked against recorded checksums.


Install `.[texture]` and run `patient-digital-twin setup texture` in the active
environment. The worker uses native TRELLIS.2 texture weights, its shape encoder,
and DINOv3 conditioning. See [backend setup](environment.md#texture-generation).

```python
from pathlib import Path
from patient_digital_twin import PatientWorkflow, TextureRuntime
from patient_digital_twin.references import default_texture_reference

workflow = PatientWorkflow(
    textures={"aorta": default_texture_reference("aorta")},
    texture_runtime=TextureRuntime(),
    workspace="refinement-work",
)
output = workflow.run(body, "textured.usdc")
```

Tiles condition appearance generation on the final mesh. By default,
`TextureOptions.match_reference_color` anchors the generated diffuse map to the
tile’s median palette using per-channel gains in linear RGB. This corrects color
drift while retaining generated spatial variation. Raw generated maps and the
color transform are retained in the inference workspace; set the option to
`False` to keep the model’s uncorrected palette. Physical repeat scale
is not controlled by this backend. Choose the correct tissue surface explicitly.
The supplied colon tile depicts internal mucosa. The aorta tile depicts adventitia.
Texture references record their creator, source, tissue surface, changes, and matching
checksum. Their metadata is included in the output report.

`prepare_surface(mesh, options)` and `texture_mesh(mesh, reference, options,
runtime=..., workspace=...)` are independently usable functions. They return
`CleanupResult` and `TextureResult`, respectively. Texture results include a
source-vertex map to verify that UV unwrapping preserves triangles. Neither
function mutates its input mesh.

Completed texture inference is cached by mesh content, tile content, options,
and runtime identity. Each attempt uses a separate directory and becomes reusable
only after output validation. Agent runs always start explicitly and reuse the
existing editing recovery files on failure.

## CLI

Both CLI entry points accept direct refinement flags. Paths resolve from the
current working directory. No workflow configuration file is used.

```bash
python -m patient_digital_twin \
  --source nvsegment --input ct.nii.gz --bundle-root /path/to/NV-Segment-CTMR \
  --classes all \
  --agent-prompt "Complete only the colon. Correct course and caliber before haustra." \
  --agent-model gpt-6.1-terra --blender-mode virtual \
  --agent-timeout 900 --token-budget 100000 \
  --anatomy-reference colon=patient-digital-twin/assets/anatomy/voxel-man/colon-male-15882019/metadata.json \
  --default-textures \
  --texture-timeout 900 \
  --format bundle --output output/new-patient
```

- No refinement flags: retain the original import/export path.
- `--cleanup`: enable deterministic cleanup alone. `--no-cleanup` skips cleanup.
- Any refinement request enables cleanup unless `--no-cleanup` is supplied.
- `--agent-prompt`: enable SDK editing. `--agent-model`, `--agent-backend`,
  `--agent-timeout`, `--token-budget`, `--budget-usd`, `--max-turns`, and
  `--max-repairs` control the session. Backend-specific budgets are validated.
- Repeat `--anatomy-reference ORGAN=METADATA_JSON` for shape references.
- `--default-textures`: enable GPU texturing with TRELLIS.2 installed in the active environment.
- Repeat `--texture ORGAN=METADATA_JSON` for tile overrides. These JSON files
  describe reference assets; they do not configure the workflow.
- `--no-default-textures`: use only explicit tiles.
- `--missing-texture error`: stop when a structure has no default tile; retaining its material is the default.
- `--smoothing-iterations`, `--smoothing-pass-band`, `--hole-radius-mm`, and
  `--min-component-area-mm2`: control cleanup.
- `--texture-seed`, `--texture-resolution`, `--texture-size`, `--texture-steps`,
  `--texture-roughness`, and `--no-match-reference-color`: control texture generation.
- `--workspace`: select where intermediate assets and recovery files are retained.

Run either entry point with `--help` for the complete option list. Python callers
can construct `PatientWorkflow` with typed options and pass it to `run_pipeline`.

Bundles include both `human_body.usdc` (presentation stage with embedded CT) and
`patient_anatomy.usdc` (patient-frame anatomy), along with `patient_twin.yaml`,
texture dependencies, requested navigation arrays, and `workflow.json`. Standalone USD outputs
have a sibling `<filename>.assets` directory containing maps and reports. Keep
that directory beside the USD when moving the asset. Reference-guided edits also
include source attribution and a completion report under `refinement/`.

The existing `examples/pipeline.py` accepts the same refinement flags. It retains
its anatomy selection, topology extraction, and dual-USD output behavior.
`PatientWorkflow.run()` forwards `patient_id`, `world_from_patient_m`,
`ct_exterior`, and `skin_opacity` to the existing bundle exporter.
Stored centerlines are retained or regenerated at their original spacing when
geometric changes invalidate them, including structures outside the navigation
selection. `topology_names` requests additional per-structure graphs independently
of the bundle's `vessel_names` navigation selection.

Virtual Blender sessions use software OpenGL by default for Xvfb viewports; CUDA
texture generation still uses the GPU. Set `LIBGL_ALWAYS_SOFTWARE=0` only with a
tested display/driver configuration. `blender-ready.json` records Blender and GL
renderer versions for crash diagnosis. Patient and reference scenes use Workbench
for inspection.

The inference CLI accepts `--classes all` to import every detected supported
structure. Absent labels are omitted. Explicit class lists still require each
requested structure. Structures without tiles retain their materials by default. Navigation arrays are requested for detected vessels.

### Defaults

Agent and texture-worker timeouts are 900 seconds each. Codex uses 100,000
tokens per session. Claude uses $5 and 30 turns per session. Omit those flags
unless you want different limits. Cleanup uses 60 iterations and pass band 0.01.
Blender mode is auto. Texture settings use the typed TextureOptions defaults.
Workers always use the active Python. TRELLIS.2 is located through its installed
module. Missing dependencies fail clearly. Model data paths are independent of
the Python environment. Missing tiles retain existing materials. Use `--missing-texture error` to stop
instead. Invalid checksums and worker failures still stop the run.

## Resume an interrupted pipeline

The workflow saves an internal checkpoint after cleanup, before the agent or
texture workers start. It includes the meshes, native CT and transforms, selected
tiles, export options, and progress. This is automatic run state, not a user
workflow configuration file.

```bash
patient-digital-twin resume --workspace output/.patient-work/run-EXAMPLE
```

Resume uses the original output destination. Use `--output NEW_DIRECTORY` to
change it. If the agent stopped, resume continues its Blender checkpoint, then
textures and exports the patient. `--blender-mode virtual --token-budget 200000`
can override the next agent session's limits and display mode. If the agent
already completed, it is not run again. Matching validated texture-cache results
are reused. An incomplete texture attempt is not accepted as a complete result.

For an editor result already published separately, pass
`--edited-asset /path/to/published/input.usdc`. The pipeline then completes the
remaining texture and export steps. Keep the published refinement report beside
that asset. An editor `resume` command also delegates to the parent pipeline when
its checkpoint exists, so it cannot silently end after editing.

Runs created before this checkpoint support need a one-time recovery from their
saved input USD and original CT. Missing cleanup measurements cannot be recovered
and must be reported as unavailable.

Whole-patient CLI runs (`--classes all`) export anatomy and CT without vessel
navigation. Explicit vessel selections, such as `--classes aorta`, request navigation
for those vessels. The Python API keeps `vessel_names` explicit.

The HRA reference at `assets/anatomy/hra/blood-vasculature-female-v1.3/metadata.json`
has explicit main/branch selections for aorta, inferior vena cava (tributaries),
pulmonary artery, celiac trunk and superior/inferior mesenteric arteries. Supply it
as `--anatomy-reference TARGET=PATH`. This is partial named coverage, not a complete
vascular tree. Missing branches and source limitations must be recorded in the
agent report. Other vessels require a verified reference with explicit branch
selections; no unrelated asset is substituted. The agent must inspect the GLB and
compare branch origins before editing. No existing patient is rerun by this change.

## More detail

- [Run commands](run-commands.md)
- [Backend setup](environment.md)
