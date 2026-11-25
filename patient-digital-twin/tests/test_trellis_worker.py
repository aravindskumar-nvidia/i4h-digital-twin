# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native texture checkpoints and conditioning preserve the source mesh map."""
import json
import sys
from types import SimpleNamespace

import numpy as np
from patient_digital_twin.texturing import _trellis
from PIL import Image


def test_load_native_texture_components_without_background_model(monkeypatch):
    config = {
        'tex_slat_sampler': {'name': 'Sampler', 'args': {}, 'params': {'steps': 12}},
        'shape_slat_normalization': {}, 'tex_slat_normalization': {},
        'image_cond_model': {'name': 'DinoV3FeatureExtractor', 'args': {'model_name': 'native-dino'}},
    }
    loaded = []

    class BasePipeline:
        @classmethod
        def from_pretrained(cls, model, config_file):
            loaded.append((model, config_file, cls.model_names_to_load))
            pipeline = cls()
            pipeline._pretrained_args = config
            return pipeline

    class TexturePipeline:
        pass

    monkeypatch.setitem(sys.modules, 'trellis2', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'trellis2.pipelines', SimpleNamespace(
        Trellis2TexturingPipeline=TexturePipeline, samplers=SimpleNamespace(Sampler=lambda: 'sampler')))
    monkeypatch.setitem(sys.modules, 'trellis2.pipelines.base', SimpleNamespace(Pipeline=BasePipeline))
    monkeypatch.setitem(sys.modules, 'trellis2.modules', SimpleNamespace(image_feature_extractor=SimpleNamespace(
        DinoV3FeatureExtractor=lambda model_name: model_name)))

    pipeline = _trellis.load_pipeline('microsoft/TRELLIS.2-4B')
    assert loaded == [('microsoft/TRELLIS.2-4B', 'texturing_pipeline.json',
                       ['shape_slat_encoder', 'tex_slat_decoder', 'tex_slat_flow_model_1024'])]
    assert pipeline.image_cond_model == 'native-dino'
    assert pipeline.rembg_model is None
    assert pipeline.low_vram


def test_worker_conditions_on_tile_and_preserves_uv_vertex_mapping(tmp_path, monkeypatch):
    import trimesh

    vertices = np.array([[0., 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    vertex_map = np.array([0, 1, 2, 0, 2, 3])
    atlas_faces = np.array([[0, 1, 2], [3, 4, 5]])
    np.savez(tmp_path / 'mesh.npz', vertices=vertices, faces=faces)
    tile = Image.new('RGB', (8, 8), (180, 100, 90))
    tile.save(tmp_path / 'tile.png')
    config = dict(trellis_root=str(tmp_path), model='microsoft/TRELLIS.2-4B', seed=42,
                  texture_square=str(tmp_path / 'tile.png'), resolution=640, steps=12, texture_size=64)
    (tmp_path / 'config.json').write_text(json.dumps(config))

    class Atlas:
        def add_mesh(self, *_args):
            pass

        def compute_charts(self):
            pass

        def pack_charts(self):
            pass

        def get_mesh(self, _index):
            return [SimpleNamespace(numpy=lambda array=array: array.copy()) for array in
                    (vertex_map, atlas_faces, np.zeros((6, 2)))]

    class NativePipeline:
        models = {'tex_slat_flow_model_1024': 'texture-model'}

        def cuda(self):
            pass

        def preprocess_mesh(self, mesh):
            mesh.vertices = mesh.vertices * .25 - .25
            return mesh

        def encode_shape_slat(self, mesh, resolution):
            assert resolution == 640
            return 'shape'

        def get_cond(self, images, resolution):
            np.testing.assert_array_equal(np.asarray(images[0]), np.asarray(tile))
            assert resolution == 1024
            return 'native-conditioning'

        def sample_tex_slat(self, cond, model, shape, params):
            assert (cond, model, shape, params) == ('native-conditioning', 'texture-model', 'shape', {'steps': 12})
            return 'latent'

        def decode_tex_slat(self, latent):
            assert latent == 'latent'
            return 'pbr-field'

        def postprocess_mesh(self, mesh, field, resolution, size):
            assert (field, resolution, size) == ('pbr-field', 640, 64)
            np.testing.assert_array_equal(mesh.faces, atlas_faces)
            return SimpleNamespace(visual=SimpleNamespace(uv=mesh.visual.uv, material=SimpleNamespace(
                baseColorTexture=tile.copy(), metallicRoughnessTexture=Image.new('RGB', (8, 8), (0, 80, 0)))))

    def load_pipeline(model):
        assert model == 'microsoft/TRELLIS.2-4B'
        return NativePipeline()

    monkeypatch.setattr(_trellis, 'load_pipeline', load_pipeline)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True), set_grad_enabled=lambda _: None,
        manual_seed=lambda _: None, tensor=lambda array, dtype: array, float32=np.float32, int32=np.int32))
    monkeypatch.setitem(sys.modules, 'cumesh', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'cumesh.cumesh', SimpleNamespace(Atlas=Atlas))
    _trellis.run(tmp_path)

    with np.load(tmp_path / 'textured.npz') as output:
        np.testing.assert_array_equal(output['vertex_map'][output['faces']], faces)
        np.testing.assert_array_equal(output['normals'], trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False).vertex_normals[vertex_map])
    assert (tmp_path / 'base_color.png').is_file()
    assert (tmp_path / 'roughness.png').is_file()
