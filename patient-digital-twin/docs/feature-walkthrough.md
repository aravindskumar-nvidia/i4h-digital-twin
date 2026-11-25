# s1384 sequential commands

## Activate the prepared environment (Bash; run once in this terminal)

```bash
cd /home/aravikumar/workspace/i4h-digital-twin
source .venv/bin/activate
CASE=output/nvsegment-ctmr-vessel-breaks/cases/s1384
```

## 1. Export the supplied real mask (existing main behavior)

```bash
python patient-digital-twin/examples/pipeline.py \
  --source segmentation \
  --input "$CASE/reference" \
  --ct "$CASE/ct.nii.gz" \
  --output output/s1384-review/masks-original
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/masks-original/human_body.usdc \
  --output output/s1384-review/renders/masks-original.png \
  --engine workbench
```

## 2. Clean up the supplied mask and save before/after renders

```bash
python patient-digital-twin/examples/pipeline.py \
  --source segmentation \
  --input "$CASE/reference" \
  --ct "$CASE/ct.nii.gz" \
  --cleanup \
  --output output/s1384-review/masks-cleanup
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/masks-original/human_body.usdc \
  output/s1384-review/masks-cleanup/human_body.usdc \
  --output output/s1384-review/renders/reference-cleanup \
  --engine workbench
```

## 3. Run NV-Segment and export the original four-organ bundle

```bash
export NV_SEGMENT_CTMR_ROOT=/home/aravikumar/workspace/NV-Segment-CTMR/NV-Segment-CTMR
```

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CASE/ct.nii.gz" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --output output/s1384-review/baseline
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/baseline/patient_anatomy.usdc \
  --output output/s1384-review/renders/baseline.png
```

## 4. Remove tiny/distant islands and clean up the inferred anatomy

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CASE/ct.nii.gz" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --cleanup \
  --output output/s1384-review/cleanup
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/baseline/patient_anatomy.usdc \
  output/s1384-review/cleanup/patient_anatomy.usdc \
  --output output/s1384-review/renders/cleanup \
  --engine workbench
```

## 5. Generate TRELLIS.2 textures while keeping the original geometry

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CASE/ct.nii.gz" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --no-cleanup \
  --default-textures \
  --output output/s1384-review/textures
```

Note: Add default textures for all things covered by NV Segment

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/baseline/patient_anatomy.usdc \
  output/s1384-review/textures/patient_anatomy.usdc \
  --output output/s1384-review/renders/textures
```

## 6. Clean up, then refine the liver with the agent

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CASE/ct.nii.gz" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --cleanup \
  --agent-task refine \
  --agent-target liver \
  --token-budget 300000 --agent-timeout 1800 \
  --blender-mode virtual \
  --output output/s1384-review/refine
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/baseline/patient_anatomy.usdc \
  output/s1384-review/refine/patient_anatomy.usdc \
  --output output/s1384-review/renders/refine \
  --engine workbench
```

## 7. Clean up, refine the liver, then generate textures

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CASE/ct.nii.gz" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --cleanup \
  --agent-task refine \
  --agent-target liver \
  --token-budget 300000 --agent-timeout 1800 \
  --blender-mode virtual \
  --default-textures \
  --workspace output/s1384-review/full-work \
  --output output/s1384-review/full
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/baseline/patient_anatomy.usdc \
  output/s1384-review/full/patient_anatomy.usdc \
  --output output/s1384-review/renders/full
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/refine/patient_anatomy.usdc \
  output/s1384-review/full/patient_anatomy.usdc \
  --output output/s1384-review/renders/refine-to-full
```

## 8. Repair the colon in the original bundle

```bash
patient-digital-twin-agent \
  --input output/s1384-review/cleanup/patient_twin.yaml \
  --output output/s1384-review/repair \
  --backend codex \
  --token-budget 300000 --timeout 1800 \
  --blender-mode virtual \
  --agent-task repair \
  --agent-target colon \
  --anatomy-reference colon=patient-digital-twin/assets/anatomy/voxel-man/colon-male-15882019/metadata.json
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/cleanup/patient_anatomy.usdc \
  output/s1384-review/repair/patient_anatomy.usdc \
  --output output/s1384-review/renders/repair \
  --engine workbench
```

## 9. Complete reference-supported missing colon anatomy

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CASE/ct.nii.gz" \
  --classes liver gallbladder colon aorta \
  --format bundle \
  --cleanup \
  --agent-task complete \
  --agent-target colon \
  --anatomy-reference colon=patient-digital-twin/assets/anatomy/voxel-man/colon-male-15882019/metadata.json \
  --token-budget 300000 --agent-timeout 1800 \
  --blender-mode virtual \
  --output output/s1384-review/complete
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/baseline/patient_anatomy.usdc \
  output/s1384-review/complete/patient_anatomy.usdc \
  --output output/s1384-review/renders/complete \
  --engine workbench
```

## 10. Edit materials and UVs on the textured bundle

```bash
patient-digital-twin-agent \
  --input output/s1384-review/textures/patient_twin.yaml \
  --output output/s1384-review/materials \
  --backend codex \
  --token-budget 300000 --timeout 1800 \
  --blender-mode virtual \
  --agent-task texture \
  --agent-target liver gallbladder colon aorta
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/textures/patient_anatomy.usdc \
  output/s1384-review/materials/patient_anatomy.usdc \
  --output output/s1384-review/renders/materials
```

## 11. Export aorta navigation and try reference-supported branches

```bash
patient-digital-twin \
  --source nvsegment \
  --input "$CASE/ct.nii.gz" \
  --classes aorta \
  --format bundle \
  --output output/s1384-review/aorta

patient-digital-twin-agent \
  --input output/s1384-review/aorta/patient_twin.yaml \
  --output output/s1384-review/aorta-branches \
  --backend codex \
  --token-budget 300000 --timeout 1800 \
  --blender-mode virtual \
  --agent-task complete \
  --agent-target aorta \
  --major-branching \
  --anatomy-reference aorta=patient-digital-twin/assets/anatomy/hra/blood-vasculature-female-v1.3/metadata.json
```

```bash
python patient-digital-twin/examples/render.py \
  output/s1384-review/aorta/patient_anatomy.usdc \
  output/s1384-review/aorta-branches/patient_anatomy.usdc \
  --output output/s1384-review/renders/aorta-branches \
  --engine workbench
```

## 12. Open results interactively (close each window to continue)

```bash
DISPLAY=:1 /home/aravikumar/isaacsim/python.sh \
  patient-digital-twin/examples/isaac_sim.py view \
  output/s1384-review/full/patient_anatomy.usdc
```

```bash
DISPLAY=:1 /home/aravikumar/isaacsim/python.sh \
  patient-digital-twin/examples/isaac_sim.py view \
  output/s1384-review/baseline/patient_anatomy.usdc
```

## 13. Resume the checkpoint from step 7 (returns the result if already complete)

```bash
patient-digital-twin resume \
  --workspace output/s1384-review/full-work/run-* \
  --token-budget 300000 \
  --blender-mode virtual
```
