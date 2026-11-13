# Blender MCP tools

Use these optional MCP tools to view and edit patient anatomy in Blender.
`HumanBody` owns the patient data. Blender provides a separate editing view.
Native MCP clients, including Claude Code and Codex, can use the same server.

## Setup

From the repository root, in an active Python environment:

```bash
uv pip install -e './patient-digital-twin[agent,pipeline]'
patient-digital-twin-setup
```

Setup finds Blender through `BLENDER_EXECUTABLE` or PATH. If neither provides it,
Linux x86_64 setup downloads Blender 4.5.16 LTS into `~/.cache/i4h/blender` and
checks its published SHA-256. Other platforms need an existing Blender install.
Aggregate-package extras are named `patient-agent` and `patient-pipeline`.

Configure this command in your MCP client, using absolute paths:

```text
/path/to/venv/bin/python -m patient_digital_twin.agent.server \
  --workspace /path/to/output/blender
```

The first Blender tool call starts a private Blender process. It closes when the
MCP session ends. Save named screenshots and `.blend` files before closing.
Desktop clients must forward `DISPLAY`/`XAUTHORITY` or the Wayland environment.

## Servers without a screen

On Linux, install Xvfb and xauth, then add `--headless` to the server command.
This starts normal Blender on a virtual display. It uses no visible desktop
window. It does not use Blender's unsupported `--background` mode for this bridge.
The MCP process manages its own Blender and virtual display.
Viewport previews use offscreen rendering. A working OpenGL driver is required;
servers without 3D acceleration can use Mesa with `LIBGL_ALWAYS_SOFTWARE=1`.

## View, edit, apply

1. Call `load_patient` with `source="segmentation"`, an absolute mask directory,
   and `names=["aorta"]`. Use `ct_path` to attach matching NIfTI CT.
2. Call `blender_show`, then `blender_screenshot` to inspect the result.
3. Use `blender_execute` for Blender Python edits. Edit only the returned patient
   collection. Leave Edit Mode and apply object scale to mesh vertices.
4. Call `blender_apply_edits` to update the patient. Refresh the Blender view
   after any patient-side change.
5. Use `export_patient` for a new USD or bundle. Use `blender_save_scene` for a
   `.blend` file and `blender_screenshot` with a new `.png` path to retain an image.

`inspect_patient`, `set_visibility` and `extract_centerlines` use the existing
patient API. `load_patient` also supports STL/OBJ maps, NV-Segment and NV-Generate;
model inference requires its own checkout, weights and environment.

## Editing contract

- Geometry and Blender poses use meters. CT registration stays fixed.
- The apply step rejects stale, incomplete or invalid snapshots before committing.
  Keep structure names and the patient object set intact.
- Mesh edits clear affected centerlines. Visibility and rigid poses retain graphs.
- Bundles keep imported placement. Export rejects differing display poses;
  standalone USD supports those poses. Materials and lighting stay in `.blend`.
- `blender_open_usd` opens exported assets for presentation edits. It cannot
  reconstruct the original `HumanBody`; reload source inputs for patient edits.
- Export tools require new paths inside the workspace. Temporary exchanges are
  removed after successful sessions. Failed startup logs remain for diagnosis.

`blender_execute` runs trusted Python with the Blender process's user permissions.
This integration is for trusted local agents. It does not provide a CT volume
renderer or tissue simulation.
