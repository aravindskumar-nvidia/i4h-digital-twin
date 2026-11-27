# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-mesh TRELLIS.2 texturing in a worker using the active Python environment."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

MODEL = 'microsoft/TRELLIS.2-4B'
IMAGE_ENCODER = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
# Native modules built by `patient-digital-twin setup texture`.
NATIVE_MODULES = ('cumesh', 'flex_gemm', 'o_voxel', 'nvdiffrast.torch', 'flash_attn', 'trellis2.pipelines')


def match_tile_palette(generated, reference):
    """Anchor diffuse color to the tile while retaining generated variation.

    DINO conditioning does not guarantee color fidelity. A global per-channel
    gain in linear RGB matches the median palette without changing UVs or
    painting new spatial features. Both images are diffuse sRGB color maps.
    """
    from PIL import Image

    def linear(image):
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
        return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)

    source, target = linear(generated), linear(reference)
    source_median = np.median(source.reshape(-1, 3), axis=0)
    target_median = np.median(target.reshape(-1, 3), axis=0)
    gain = target_median / np.maximum(source_median, 1e-6)
    corrected = np.clip(source * gain, 0, 1)
    srgb = np.where(corrected <= .0031308, corrected * 12.92,
                    1.055 * corrected ** (1 / 2.4) - .055)
    result = Image.fromarray(np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8))
    return result, dict(method='median_gain_linear_rgb', gain=gain.tolist(),
                        generated_median_linear=source_median.tolist(),
                        reference_median_linear=target_median.tolist())


def load_pipeline(model: str):
    """Load the released texture denoiser/decoder without unused shape generators."""
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import Trellis2TexturingPipeline, samplers
    from trellis2.pipelines.base import Pipeline

    class TexturePipeline(Trellis2TexturingPipeline):
        model_names_to_load = ["shape_slat_encoder", "tex_slat_decoder", "tex_slat_flow_model_1024"]

    pipeline = Pipeline.from_pretrained.__func__(TexturePipeline, model, "texturing_pipeline.json")
    config = pipeline._pretrained_args
    pipeline.tex_slat_sampler = getattr(samplers, config["tex_slat_sampler"]["name"])(
        **config["tex_slat_sampler"]["args"]
    )
    pipeline.tex_slat_sampler_params = config["tex_slat_sampler"]["params"]
    pipeline.shape_slat_normalization = config["shape_slat_normalization"]
    pipeline.tex_slat_normalization = config["tex_slat_normalization"]
    conditioning = config["image_cond_model"]
    pipeline.image_cond_model = getattr(image_feature_extractor, conditioning["name"])(**conditioning["args"])
    # Tissue tiles are already cropped; no background-removal model is needed.
    pipeline.rembg_model = None
    pipeline.low_vram = True
    pipeline.pbr_attr_layout = {
        "base_color": slice(0, 3), "metallic": slice(3, 4),
        "roughness": slice(4, 5), "alpha": slice(5, 6),
    }
    pipeline._device = "cpu"
    return pipeline


def run(work: Path) -> None:
    config = json.loads((work / "config.json").read_text())
    sys.path.insert(0, config["trellis_root"])
    import torch
    import trimesh
    from cumesh.cumesh import Atlas
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 requires CUDA; use its configured Python environment")
    torch.set_grad_enabled(False)
    torch.manual_seed(config["seed"])
    with np.load(work / "mesh.npz", allow_pickle=False) as arrays:
        source = trimesh.Trimesh(vertices=arrays["vertices"], faces=arrays["faces"], process=False)
    pipeline = load_pipeline(config["model"])
    pipeline.cuda()
    mesh = pipeline.preprocess_mesh(source.copy())
    with Image.open(config["texture_square"]) as image:
        reference = image.convert("RGB")
    reference.save(work / "conditioning.png")
    resolution = config["resolution"]
    shape = pipeline.encode_shape_slat(mesh, resolution)
    cond = pipeline.get_cond([reference], 1024)
    # CPU Xatlas avoids the GPU chart kernel and preserves a source vertex map.
    atlas = Atlas()
    atlas.add_mesh(torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32),
                   torch.tensor(np.asarray(mesh.faces), dtype=torch.int32))
    atlas.compute_charts()
    atlas.pack_charts()
    vmap, faces, uv = (value.numpy() for value in atlas.get_mesh(0))
    if not np.array_equal(vmap[faces], source.faces):
        raise ValueError("UV atlas changed triangle identity")
    uv[:, 1] = 1 - uv[:, 1]
    class TexturingMesh(trimesh.Trimesh):
        @property
        def vertex_normals(self):
            # TRELLIS rotates normals in place; modern Trimesh caches are immutable.
            return super().vertex_normals.copy()

    unwrapped = TexturingMesh(
        vertices=mesh.vertices[vmap], faces=faces, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv),
    )
    latent = pipeline.sample_tex_slat(
        cond, pipeline.models["tex_slat_flow_model_1024"], shape, {"steps": config["steps"]}
    )
    # This samples one 3-D PBR field onto every UV chart and inpaints uncovered
    # texels, avoiding black chart borders during bilinear texture filtering.
    textured = pipeline.postprocess_mesh(
        unwrapped, pipeline.decode_tex_slat(latent), resolution, config["texture_size"]
    )
    material = textured.visual.material
    base_color = material.baseColorTexture.convert("RGB")
    # Keep raw inference output inspectable as well as the final color map.
    base_color.save(work / "base_color_generated.png")
    if config['match_reference_color']:
        with Image.open(config['texture_square']) as tile:
            base_color, config['palette_match'] = match_tile_palette(base_color, tile)
    base_color.save(work / "base_color.png")
    packed = np.asarray(material.metallicRoughnessTexture)
    Image.fromarray(packed[..., 1]).save(work / "roughness.png")
    # Preserve source coordinates and smooth normals exactly across UV seams.
    np.savez(work / "textured.npz", vertex_map=vmap, faces=faces,
             uv=textured.visual.uv, normals=source.vertex_normals[vmap])
    (work / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def cache_identity(root: Path) -> dict:
    from huggingface_hub import try_to_load_from_cache

    models = {
        MODEL: ['texturing_pipeline.json',
            'ckpts/shape_enc_next_dc_f16c32_fp16.json', 'ckpts/shape_enc_next_dc_f16c32_fp16.safetensors',
            'ckpts/tex_dec_next_dc_f16c32_fp16.json', 'ckpts/tex_dec_next_dc_f16c32_fp16.safetensors',
            'ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json', 'ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors'],
        IMAGE_ENCODER: ['config.json', 'model.safetensors'],
    }
    files = [Path(__file__), *sorted((root / 'trellis2').rglob('*.py'))]
    missing = False
    for repo, names in models.items():
        for name in names:
            cached = try_to_load_from_cache(repo, name)
            if isinstance(cached, str):
                files.append(Path(cached).resolve())
            else:
                missing = True
    # Unknown weights cannot safely share an inference checkpoint.
    return {'files': [(str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files],
            'uncached': uuid4().hex if missing else None}


def check(root: Path) -> dict:
    """Import every native module, require a GPU, and return the cache identity."""
    sys.path.insert(0, str(root))
    import torch

    for module in NATIVE_MODULES:
        importlib.import_module(module)
    if not torch.cuda.is_available():
        raise RuntimeError('TRELLIS.2 requires a CUDA-capable environment')
    return cache_identity(root)


if __name__ == "__main__":
    if sys.argv[1] == '--check':
        print(json.dumps(check(Path(sys.argv[2]))))
    else:
        run(Path(sys.argv[1]))
