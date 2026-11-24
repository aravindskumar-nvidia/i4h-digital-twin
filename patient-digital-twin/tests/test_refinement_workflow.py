# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for optional editing, operation order and portable outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from patient_digital_twin import (
    CleanupOptions,
    HumanBody,
    MeshGeometry,
    SegmentationImporter,
    TextureOptions,
    TextureReference,
    TextureRuntime,
    prepare_surface,
)
from patient_digital_twin.scan_volume import from_array


@pytest.fixture
def body():
    labels = np.zeros((12, 12, 12), np.uint8)
    labels[2:10, 2:10, 2:10] = 1
    affine = np.diag([.001, .001, .001, 1])
    patient = HumanBody(SegmentationImporter(labels, {1: 'liver'},
                        affine_xyz_to_imaging_m=affine).to_anatomy_collection())
    patient.attach_scan(from_array(labels, np.eye(4), world_unit='mm'))
    return patient



def test_cleanup_is_independent_and_does_not_mutate(body):
    mesh = body.anatomy.structures['liver'].mesh
    before = mesh.vertices.copy()
    result = prepare_surface(mesh, CleanupOptions(smoothing_iterations=2))
    np.testing.assert_array_equal(mesh.vertices, before)
    assert result.measurements['after']['watertight']
    assert result.mesh is not mesh



def test_texture_cache_reuses_completed_inference_and_rejects_changed_triangles(tmp_path, monkeypatch):
    from patient_digital_twin import texturing
    from PIL import Image
    tile = tmp_path / 'tile.png'
    Image.new('RGB', (8, 8)).save(tile)
    metadata = {'creator': 'test', 'source': 'test', 'changes': 'none', 'surface': 'serosa',
                'file': tile.name, 'sha256': hashlib.sha256(tile.read_bytes()).hexdigest()}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    reference = TextureReference.load(tmp_path / 'metadata.json')
    monkeypatch.setattr(texturing, 'check_runtime', lambda runtime: {'revision': 'test'})
    calls = []
    def worker(args, **kwargs):
        import sys
        from types import SimpleNamespace
        assert args[0] == sys.executable
        calls.append(args)
        work = Path(args[-1])
        with np.load(work / 'mesh.npz') as source:
            np.savez(work / 'textured.npz', vertex_map=np.arange(3), faces=source['faces'],
                     uv=np.zeros((3, 2)), normals=np.ones((3, 3)))
        for name in ('base_color.png', 'roughness.png'):
            (work / name).write_bytes(tile.read_bytes())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(texturing.subprocess, 'run', worker)
    mesh = MeshGeometry(np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
    arguments = dict(runtime=TextureRuntime(tmp_path), workspace=tmp_path / 'cache')
    first = texturing.texture_mesh(mesh, reference, **arguments)
    second = texturing.texture_mesh(mesh, reference, **arguments)
    assert len(calls) == 1 and first.workspace == second.workspace
    texturing.texture_mesh(mesh, reference, TextureOptions(seed=43), **arguments)
    assert len(calls) == 2
    np.savez(first.workspace / 'textured.npz', vertex_map=np.arange(3), faces=[[0, 2, 1]],
             uv=np.zeros((3, 2)), normals=np.ones((3, 3)))
    with pytest.raises(ValueError, match='source triangles'):
        texturing.texture_mesh(mesh, reference, **arguments)



def test_texture_palette_matches_reference_without_flattening_detail():
    from patient_digital_twin.texturing._trellis import match_tile_palette
    from PIL import Image

    # Regression: generated yellow tissue despite a pink conditioning tile.
    values = np.tile(np.array([154, 115, 19], dtype=np.uint8), (9, 9, 1))
    values[0, 0] = [120, 90, 15]
    generated = Image.fromarray(values)
    reference = Image.new('RGB', (9, 9), (230, 130, 110))
    corrected, record = match_tile_palette(generated, reference)
    actual = np.asarray(corrected)
    np.testing.assert_allclose(np.median(actual.reshape(-1, 3), axis=0), [230, 130, 110], atol=1)
    assert np.all(actual[0, 0] < actual[4, 4])
    assert corrected.size == generated.size
    assert record['method'] == 'median_gain_linear_rgb'
    np.testing.assert_array_equal(np.asarray(generated), values)



def test_default_smoothing_reduces_voxel_steps_without_large_volume_drift():
    z, y, x = np.indices((25, 65, 65))
    labels = (((x - 32) ** 2 + (y - 32) ** 2 + ((z - 12) * 3) ** 2) < 24 ** 2).astype(np.uint8)
    anatomy = SegmentationImporter(labels, {1: 'liver'},
                                   affine_xyz_to_imaging_m=np.diag([.001, .001, .003, 1])).to_anatomy_collection()
    result = prepare_surface(anatomy.structures['liver'].mesh)
    before, after = result.measurements['before'], result.measurements['after']
    assert after['p95_adjacent_normal_angle_deg'] < .55 * before['p95_adjacent_normal_angle_deg']
    assert abs(after['volume_mm3'] / before['volume_mm3'] - 1) < .02
    assert after['watertight'] and after['euler_characteristic'] == before['euler_characteristic'] == 2



def test_strong_cleanup_does_not_collapse_coarse_mesh():
    import trimesh
    cylinder = trimesh.creation.cylinder(radius=.005, height=.04, sections=16)
    result = prepare_surface(MeshGeometry(cylinder.vertices, cylinder.faces))
    surface = trimesh.Trimesh(result.mesh.vertices, result.mesh.faces, process=False)
    assert surface.is_watertight and len(surface.faces)
    assert abs(surface.volume / cylinder.volume - 1) <= .02
    assert result.measurements['smoothing_pass_band'] > .01



def test_packaged_texture_defaults():
    from patient_digital_twin.references import default_texture_reference
    for name in ('aorta', 'colon', 'liver', 'gallbladder', 'bladder',
                 'left kidney', 'right kidney', 'pancreas', 'spleen', 'stomach'):
        reference = default_texture_reference(name)
        reference.validate()
        assert reference.image.is_file()
    assert default_texture_reference('colon').metadata['surface'] == 'mucosa'
    with pytest.raises(ValueError, match='No default texture'):
        default_texture_reference('unknown')



def test_tissue_family_defaults_are_explicit_and_preserve_exact_overrides():
    from patient_digital_twin.anatomy import CATALOG
    from patient_digital_twin.references import default_texture_reference
    from PIL import Image

    examples = {'lung_upper_lobe_right': 'lung', 'atrial_appendage_left': 'heart',
                'iliopsoas_left': 'skeletal_muscle', 'rib_left_3': 'cortical_bone',
                'small_bowel': 'bowel_serosa'}
    for organ, family in examples.items():
        reference = default_texture_reference(organ)
        assert reference.metadata['selection'] == {
            'structure': organ, 'default': family, 'match': 'tissue_family'}
        assert reference.metadata['synthetic']
        with Image.open(reference.image) as image:
            assert image.width == image.height
    assert default_texture_reference('colon').metadata['selection']['default'] == 'colon'
    for unsupported in ('brain', 'costal_cartilages', 'kidney_cyst_left', 'lung_fake'):
        with pytest.raises(ValueError, match='No default texture'):
            default_texture_reference(unsupported)
    directory = default_texture_reference('heart').image.parent
    records = json.loads((directory / 'defaults.json').read_text())
    assigned = []
    for record in records.values():
        names = record.get('applies_to', [])
        assert set(names) <= CATALOG.keys()
        assigned.extend(names)
    assert len(assigned) == len(set(assigned))



def test_non_aortic_major_branches_require_explicit_selection(tmp_path):
    from patient_digital_twin.references import load_references
    (tmp_path / 'reference.glb').write_bytes(b'fixture')
    for name in ('LICENSE.txt', 'source.json', 'ATTRIBUTION.txt'):
        (tmp_path / name).write_text('fixture')
    metadata = dict(file='reference.glb', units='meters', up_axis='Y',
                    sha256=hashlib.sha256(b'fixture').hexdigest(),
                    license={'spdx': 'CC-BY-4.0', 'text': 'LICENSE.txt'},
                    source={'evidence': 'source.json'}, structures='structures.json',
                    selections={'inferior_vena_cava_main': ['ivc'],
                                'inferior_vena_cava_major_branches': ['renal_vein_junction']})
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(metadata))
    assert load_references({'inferior_vena_cava': path})['inferior_vena_cava']['branches'] == []
    result = load_references({'inferior_vena_cava': path}, include_major_branches=True)
    assert result['inferior_vena_cava']['nodes'] == ['ivc', 'renal_vein_junction']
    del metadata['selections']['inferior_vena_cava_major_branches']
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='no named major branches for inferior_vena_cava'):
        load_references({'inferior_vena_cava': path}, include_major_branches=True)
