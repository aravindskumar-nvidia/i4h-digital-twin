# Run commands

Start with the [patient install](../README.md#install), then install only the [optional backends](environment.md) used by your command. Run from the repository root. Replace `/path/to/...` with your own files. Examples use **GPT-6 Astra**, virtual Blender and a **300,000-token budget**. Agent timeout: **15 minutes**. Output paths must be new.

## Activate

```bash
source .venv/bin/activate
```

## Resume a saved run

Use a checkpoint from a new run. Resume keeps its saved model and continues editing, textures and export.

```bash
patient-digital-twin resume \
  --workspace /path/to/run-directory \
  --token-budget 300000 \
  --blender-mode virtual
```

## Four organs: complete colon and aorta, include branches, texture all four

```bash
patient-digital-twin \
  --source nvsegment \
  --input /path/to/patient/ct.nii.gz \
  --classes liver gallbladder colon aorta \
  --agent-task complete \
  --major-branching \
  --agent-model gpt-6-astra \
  --token-budget 300000 \
  --blender-mode virtual \
  --anatomy-reference colon=patient-digital-twin/assets/anatomy/voxel-man/colon-male-15882019/metadata.json \
  --anatomy-reference aorta=patient-digital-twin/assets/anatomy/hra/blood-vasculature-female-v1.3/metadata.json \
  --default-textures \
  --format bundle \
  --output "output/s0878-four-organs-$(date +%Y%m%d-%H%M%S)"
```

Includes aorta navigation. Remove `--major-branching` for main-aorta-only completion.

## Whole patient: complete colon and texture available structures

```bash
patient-digital-twin \
  --source nvsegment \
  --input /path/to/patient/ct.nii.gz \
  --classes all \
  --agent-task complete \
  --agent-model gpt-6-astra \
  --token-budget 300000 \
  --blender-mode virtual \
  --anatomy-reference colon=patient-digital-twin/assets/anatomy/voxel-man/colon-male-15882019/metadata.json \
  --default-textures \
  --format bundle \
  --output "output/s0878-whole-patient-$(date +%Y%m%d-%H%M%S)"
```

`all` includes every detected supported structure. Only colon is agent-edited. No navigation.

## Whole patient: textures, no agent

```bash
patient-digital-twin \
  --source nvsegment \
  --input /path/to/patient/ct.nii.gz \
  --classes all \
  --default-textures \
  --format bundle \
  --output "output/s0878-textured-$(date +%Y%m%d-%H%M%S)"
```

## Aorta only: completion, branches, texture and navigation

```bash
patient-digital-twin \
  --source nvsegment \
  --input /path/to/patient/ct.nii.gz \
  --classes aorta \
  --agent-task complete \
  --major-branching \
  --agent-model gpt-6-astra \
  --token-budget 300000 \
  --blender-mode virtual \
  --anatomy-reference aorta=patient-digital-twin/assets/anatomy/hra/blood-vasculature-female-v1.3/metadata.json \
  --default-textures \
  --format bundle \
  --output "output/s0878-aorta-$(date +%Y%m%d-%H%M%S)"
```

## Existing patient: repair colon

```bash
patient-digital-twin-agent \
  --input /path/to/patient/human_body.usdc \
  --output "output/s0878-colon-repair-$(date +%Y%m%d-%H%M%S)" \
  --backend codex \
  --model gpt-6-astra \
  --token-budget 300000 \
  --blender-mode virtual \
  --agent-task repair \
  --anatomy-reference colon=patient-digital-twin/assets/anatomy/voxel-man/colon-male-15882019/metadata.json
```

Standalone editing: no segmentation or new texture generation. Replace `repair` with `complete` to fill missing anatomy.

## Existing patient: refine liver and gallbladder surfaces

```bash
patient-digital-twin-agent \
  --input /path/to/patient/human_body.usdc \
  --output "output/s0878-surface-refinement-$(date +%Y%m%d-%H%M%S)" \
  --backend codex \
  --model gpt-6-astra \
  --token-budget 300000 \
  --blender-mode virtual \
  --agent-task refine \
  --agent-target liver gallbladder
```

## Existing patient: correct materials and UVs

```bash
patient-digital-twin-agent \
  --input /path/to/patient/human_body.usdc \
  --output "output/s0878-material-correction-$(date +%Y%m%d-%H%M%S)" \
  --backend codex \
  --model gpt-6-astra \
  --token-budget 300000 \
  --blender-mode virtual \
  --agent-task texture \
  --agent-target liver gallbladder colon aorta
```

Keeps geometry unchanged. Corrects existing maps; does not generate new tissue tiles.

## Optional changes

| Need | Change |
| --- | --- |
| Different patient | Replace `--input` with its CT path |
| Terra model | Use `gpt-6.1-terra` instead of `gpt-6-astra` |
| Repair instead of completion | Use `--agent-task repair` |
| No branch expansion | Omit `--major-branching` |
| No texture generation | Omit `--default-textures` |
| Custom instructions, pipeline | Add `--agent-prompt "Preserve the visible scar"` |
| Custom instructions, standalone editor | Append `"Preserve the visible scar"` before `--agent-target`, if present |
| Resume another pipeline | Use its printed `Pipeline checkpoint` directory as `--workspace` |

Structures without a default tile keep their existing material. Resume uses the saved model, targets and references.
