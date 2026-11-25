# Isaac for Healthcare — Digital Twin

Pipelines and packages for building patient, hospital, and robot digital twins for NVIDIA Isaac Sim and healthcare robotics.

This repository turns medical imaging, robot descriptions, and hospital scenes into simulation-ready OpenUSD assets and preprocessing artifacts that feed Isaac Lab / IsaacLab-Arena workflows.

## Repository Layout

```text
i4h-digital-twin/
├── patient-digital-twin/     # Imaging → anatomy twin (CT, vessels, USD)
├── hospital-digital-twin/    # Hospital scene authoring, teleop, sim2real
├── robot-digital-twin/       # Bring-your-own robot (CAD / URDF → USD)
└── sim-ready-assets/         # Asset catalog helper for i4h sim assets
```

## Patient Digital Twin

Convert clinical or synthetic imaging into vessel/anatomy artifacts and OpenUSD meshes.

[Blender asset editing](patient-digital-twin/docs/blender.md)
modifies exported geometry, materials and textures while preserving the patient asset contract.
[Claude/Codex orchestration](patient-digital-twin/docs/agents.md) automates this editing step.

| Component | Status | Purpose |
| --- | --- | --- |
| [`patient_digital_twin.export`](./patient-digital-twin/patient_digital_twin/export.py) | Patient module | USD, native HU scan bundles, vessel masks and centerlines |
| [`patient_digital_twin.geometry`](./patient-digital-twin/patient_digital_twin/geometry.py) | Patient module | Binary masks → NumPy vertices and faces; skeleton centerlines |
| [`patient_digital_twin.importers`](./patient-digital-twin/examples/README.md) | Patient module | Synthetic CT/MR masks, segmentation and mesh import |
| [`vasculature_digital_twin`](./patient-digital-twin/README.md#deprecated-vasculature_digital_twin) | **Deprecated** | CT HU→μ preprocessing and vessel masks; kept for migration only |

### Quick start — installable packages

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

```bash
# Run from the repository root (Python 3.10+).
uv venv .venv
source .venv/bin/activate
uv pip install -e './patient-digital-twin[editing]'
patient-digital-twin setup editing
```

This installs anatomy import/export and both Codex and Claude agent SDKs.
`setup` reuses a compatible Blender or installs the verified version on Linux x86_64;
subsequent runs find it automatically. Authenticate your chosen agent backend before editing.
On headless Ubuntu, setup reports if `sudo apt install xvfb xauth` is needed.

For basic import/export only, install `./patient-digital-twin[pipeline]` instead and skip `setup`.
Activate the environment once per terminal; run `python` and `patient-digital-twin` normally afterward.
Continue with the [patient walkthrough](patient-digital-twin/README.md).
To add generated textures in the same environment:

```bash
uv pip install -e './patient-digital-twin[texture]'
patient-digital-twin setup texture
```

Texture setup needs an NVIDIA GPU and the CUDA development toolkit. See
[backend setup](patient-digital-twin/docs/environment.md) for system prerequisites and model inference.

Python API example (segmentation → patient anatomy meshes):

```python
from patient_digital_twin import HumanBody, SegmentationImporter

anatomy = SegmentationImporter("patient_label.nii.gz", "label_dict.json").to_anatomy_collection()
body = HumanBody(anatomy)
for structure in body.anatomy.select(include_empty=False):
    print(structure.name, structure.world_vertices.shape, structure.faces.shape)
```

## Hospital Digital Twin

Tools for authoring hospital simulation environments and collecting / augmenting robot demonstration data.

| Component | Purpose |
| --- | --- |
| [`setup_from_assets`](./hospital-digital-twin/setup_from_assets/README.md) | Assemble operating-room scenes from catalog assets |
| [`reconstruct_from_video`](./hospital-digital-twin/reconstruct_from_video/README.md) | NuRec / neural reconstruction of hospital spaces |

## Robot Digital Twin

Bring-your-own-robot guides for converting CAD / URDF into articulated USD assets and integrating them into Isaac Sim scenes.

See [`robot-digital-twin/README.md`](./robot-digital-twin/README.md).

## Sim-Ready Assets

Catalog of robots, anatomy, equipment, and hospital environments used across i4h simulations, plus the `sim_ready_assets` download helper.

See [`sim-ready-assets/README.md`](./sim-ready-assets/README.md).

## Requirements

Shared / typical prerequisites (exact versions depend on the component):

| Requirement | Notes |
| --- | --- |
| OS | Linux (x86_64) recommended |
| Python | 3.10+ for installable packages (`patient_digital_twin` and the other twin modules) |
| GPU | Required for the documented model-inference and texture workflows; not required for basic mask import and USD export |
| Tooling | `uv`; Isaac Sim when loading USD in simulation |

Use `uv` for Python environments, package installation, development commands and builds.
Installable packages do **not** require Conda. The 3DGUT reconstruction container retains
its upstream Conda-based installer; this is internal to the container.
Hospital / robot twin guides may assume Isaac Sim, Isaac Lab, or XR runtimes — see each component README.

## Python Packages

With the virtual environment above active, install every top-level digital twin module from the repository root:

```bash
uv pip install .
```

Install one top-level module by supplying its directory instead:

```bash
uv pip install ./patient-digital-twin
uv pip install ./hospital-digital-twin
uv pip install ./sim-ready-assets
uv pip install ./robot-digital-twin
```

The corresponding Python imports are `patient_digital_twin`, `hospital_digital_twin`, `sim_ready_assets`, and `robot_digital_twin`.

For the aggregate install with patient export support, use `uv pip install -e '.[patient-pipeline]'`.
The aggregate package prefixes patient extras with `patient-`; the standalone patient package uses `pipeline`, `dicom`, etc.

## Development / CI

Installable packages under `patient-digital-twin/` include unit tests and can be exercised with:

```bash
cd patient-digital-twin
uv run --extra dev pytest
```

Repository GitHub Actions cover copyright headers, markdown link checks, pre-commit linting, and package build/test for the installable modules.

## Security

See [SECURITY.md](./SECURITY.md). Do not report security vulnerabilities through public GitHub issues.

## Support

This repository is under active development (experimental). For questions and support, open an issue in the GitHub repository.

## License

Licensing varies by component and asset source. Check each package or asset directory for SPDX / LICENSE files. Lightwheel SimReady assets under `sim-ready-assets/` are for non-commercial R&D use only unless otherwise stated.
