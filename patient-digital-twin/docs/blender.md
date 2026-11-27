# Blender asset editing

Install `./patient-digital-twin[pipeline]` and Blender (tested with 5.1.2).
Prepare an existing patient USD or bundle in your Python environment:

```python
from pathlib import Path
from patient_digital_twin.agent import blender
from patient_digital_twin.agent.assets import prepare, finalize

workspace = Path("output/edit").resolve()
prepare(Path("output/patient_twin.yaml"), workspace)
print(f"exec(open({blender.__file__!r}).read()); open_asset({str(workspace)!r})")
print(f"export_asset({str(workspace)!r})")
```

In Blender's Python Console, run the first printed command to open a separate
scene. Edit the meshes/materials, return to Object Mode, then run the second
printed command to export. Blender must share the same machine/filesystem.
Back in your original Python environment, publish the final asset:

```python
result = finalize(workspace, Path("output/edited-patient").resolve())
print(result)  # Use this USD or patient_twin.yaml in Isaac.
```

The output directory must be new. Manifest-listed bundle files, an updated
`human_body.usdc` when present, and required textures are published; source files, CT data and original prim identities are
preserved. Geometry edits regenerate existing USD centerlines using grid spacing
estimated from the original graph. If extraction fails, publication stops and the edit files stay in
the workspace. Navigation uses the bundle's saved vessel selection. For older
bundles with several vessels, pass `vessel_names=["aorta", ...]` to `prepare()`.
The selection is validated before editing and reused by `finalize()`.

Supported: existing meshes, transforms, UVs and USD Preview Surface materials/image
textures. Add/delete/split/join, rigs and animation are unsupported.

The printed commands run this repository's helper in Blender's Python, with your
permissions. [Agent editing](agents.md) runs model-written Python in Blender the same
way; read its [security notes](agents.md#security) before using it.
