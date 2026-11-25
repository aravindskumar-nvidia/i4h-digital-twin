# Run commands

One patient, one command per step, in order. Each step writes a new folder under
`$OUT`, so you can compare results. Steps 6, 8 and 11 read the output of an earlier
step; every other step runs on its own.

## Before you start

Follow the [patient install](../README.md#install) and install the
[optional backends](environment.md) for the steps you want: NV-Segment for all
steps, agent editing for steps 4 to 9, texture generation for steps 3 and 5.
Run from the repository root.

```bash
source .venv/bin/activate
CT=/path/to/patient/ct.nii.gz
OUT=output/my-patient
```

Output paths must not exist yet. Change `OUT` or delete a step's folder to rerun it.

## 1. Export the original anatomy

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CT" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --output "$OUT/baseline"
```

## 2. Clean up the surfaces

Removes tiny and distant fragments and smooths voxel steps.

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CT" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --cleanup \
  --output "$OUT/cleanup"
```

## 3. Generate textures on the original geometry

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CT" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --no-cleanup \
  --default-textures \
  --output "$OUT/textures"
```

## 4. Clean up, then refine the liver with the agent

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CT" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --cleanup \
  --agent-task refine \
  --agent-target liver \
  --blender-mode virtual \
  --output "$OUT/refine"
```

## 5. Clean up, refine the liver, then generate textures

`--workspace` keeps the checkpoint that step 11 resumes.

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CT" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --cleanup \
  --agent-task refine \
  --agent-target liver \
  --blender-mode virtual \
  --default-textures \
  --workspace "$OUT/full-work" \
  --output "$OUT/full"
```

## 6. Repair the colon in an existing bundle

Edits the bundle from step 2. No segmentation or texturing runs.

```bash
patient-digital-twin-agent \
  --input "$OUT/cleanup/patient_twin.yaml" \
  --output "$OUT/repair" \
  --backend codex \
  --blender-mode virtual \
  --agent-task repair \
  --agent-target colon \
  --anatomy-reference colon=voxel-man-colon-male-15882019
```

## 7. Complete missing colon anatomy

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CT" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --cleanup \
  --agent-task complete \
  --agent-target colon \
  --anatomy-reference colon=voxel-man-colon-male-15882019 \
  --blender-mode virtual \
  --output "$OUT/complete"
```

## 8. Edit materials and UVs on the textured bundle

Edits the bundle from step 3. Geometry stays unchanged.

```bash
patient-digital-twin-agent \
  --input "$OUT/textures/patient_twin.yaml" \
  --output "$OUT/materials" \
  --backend codex \
  --blender-mode virtual \
  --agent-task texture \
  --agent-target liver gallbladder colon aorta
```

## 9. Export aorta navigation, then add major branches

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CT" \
  --classes aorta \
  --format bundle \
  --output "$OUT/aorta"

patient-digital-twin-agent \
  --input "$OUT/aorta/patient_twin.yaml" \
  --output "$OUT/aorta-branches" \
  --backend codex \
  --blender-mode virtual \
  --agent-task complete \
  --agent-target aorta \
  --major-branching \
  --anatomy-reference aorta=hra-blood-vasculature-female-v1.3
```

## 10. View a result in Isaac Sim

Close the window to return to the terminal.

```bash
/path/to/isaac-sim/python.sh patient-digital-twin/examples/isaac_sim.py view \
  "$OUT/full/patient_anatomy.usdc"
```

## 11. Resume the run from step 5

Continues an interrupted run. If the run already finished, it prints the result path.

```bash
patient-digital-twin resume \
  --workspace "$OUT"/full-work/run-*
```

## More options

Agent limits, models, custom prompts, whole-patient runs and texture overrides are
described in [refinement](refinement.md).
