# Optional patient backends

The [patient install](../README.md#install) is sufficient for importing masks,
extracting meshes and centerlines, and exporting USD or bundles. Choose only the
sections needed for your workflow; they are independent additions to that install.

Run commands from the repository root with the same environment active:

```bash
source .venv/bin/activate
```

If you used another environment for the patient install, keep using it instead.
Backend source checkouts and downloads below live in `.deps`.

## Choose a backend

| Workflow | Requirements |
| --- | --- |
| [Masks, meshes and USD export](../README.md#install) | Python 3.10+; no backend setup |
| [Model inference](#model-inference) | NVIDIA GPU and a compatible driver |
| [Agent editing](#agent-editing) | Codex authentication; Blender; `xvfb` and `xauth` for virtual display |
| [Texture generation](#texture-generation) | NVIDIA GPU, a CUDA toolkit matching PyTorch, and a C++ compiler |

The commands below target Linux x86_64.

## PyTorch

Model inference and texture generation run on a CUDA build of PyTorch. The extras
require PyTorch 2.4 or later and keep an installed build, so install the one for your
NVIDIA driver first. The tested build:

```bash
uv pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

## Model inference

For NV-Segment, install [PyTorch](#pytorch) and the inference dependencies, then clone
the model source. For DICOM input, also install `./patient-digital-twin[dicom]`.

```bash
mkdir -p .deps
export HF_HOME="$PWD/.deps/cache/huggingface"
uv pip install -e './patient-digital-twin[nvsegment]'

git clone https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git .deps/NV-Segment-CTMR
export NV_SEGMENT_CTMR_ROOT="$PWD/.deps/NV-Segment-CTMR/NV-Segment-CTMR"
```

Acquire the model weights using the cloned repository's instructions in
`.deps/NV-Segment-CTMR/NV-Segment-CTMR/docs/README.md`.
Keep downloads in the `HF_HOME` set above.

```bash
python -c 'import torch, monai; print(torch.__version__); assert torch.cuda.is_available()'
```

For synthetic patients, install the `nvgenerate` extra and clone
[NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR).
Pass that new checkout to `--source-root`.

## Agent editing

The README's `editing` install includes both Codex and Claude SDKs. If you chose
the smaller `pipeline` install, add editing support in the same active environment:

```bash
uv pip install -e './patient-digital-twin[editing]'
patient-digital-twin setup editing
```

Setup reuses compatible Blender or installs the verified Linux x86_64 release in
the managed cache. Later runs find it automatically. On other platforms, install
Blender separately. Headless Ubuntu also needs `xvfb` and `xauth`; setup prints the
system-package command when they are missing.

Authenticate your chosen agent backend before running a task.
See [agent editing](agents.md) for running edits and recovering interrupted work.

## Texture generation

Install [PyTorch](#pytorch), then in the same active environment run:

```bash
uv pip install -e './patient-digital-twin[texture]'
patient-digital-twin setup texture
```

The `texture` extra installs TRELLIS.2's Python dependencies. Setup then builds what
has no usable wheel against the installed PyTorch: utils3d, CuMesh, FlexGEMM,
nvdiffrast and TRELLIS.2's o-voxel at the commits pinned in
`patient_digital_twin/_texture_setup.py`, plus flash-attn 2.7.3. It registers the
TRELLIS.2 checkout in the environment. Checkouts are cached under
`~/.cache/i4h/texture` (or `$XDG_CACHE_HOME/i4h/texture`); when a pin changes, setup
moves the checkout to it and rebuilds. Rerunning setup reuses a working build at the
current pins.

Requirements, checked before anything is built:

- Linux x86_64 and an NVIDIA GPU that PyTorch can use
- git, uv and a C++ compiler
- a CUDA toolkit (`nvcc`) whose major version matches `torch.version.cuda`; PyTorch
  finds it through `CUDA_HOME` or `/usr/local/cuda`

Tested with PyTorch 2.6.0+cu124 and the CUDA 12.9 toolkit on an RTX 6000 Ada. Native
builds can take several minutes.

The worker downloads the TRELLIS.2 texturing weights and DINOv3 image-conditioning
weights on first use. Hugging Face may require authentication and access approval
for the official DINOv3 model. Tissue tiles do not need a background-removal model.

## Verify and run

```bash
patient-digital-twin --help
uv pip check
```

For agent editing, also check `patient-digital-twin-agent --help`.

Use the [run commands](run-commands.md) for generation, completion, repair,
texturing and recovery. Supply your own CT or use the bundled sample.

For subsequent terminals, activate the same environment and restore any environment
variables you set above. Optionally set `I4H_REFERENCE_CACHE` to
`$PWD/.deps/cache/references` to keep downloaded anatomy references in the repository's
cache as well.
