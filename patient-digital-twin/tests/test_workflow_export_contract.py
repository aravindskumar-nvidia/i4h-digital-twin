# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare refinement outputs with the reviewed standalone and schema-2 exporters."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from patient_digital_twin import (
    AgentOptions,
    AnatomicalStructure,
    CleanupOptions,
    HumanBody,
    Kind,
    PatientWorkflow,
    SegmentationImporter,
)
from patient_digital_twin.agent import session
from patient_digital_twin.agent.assets import finalize
from patient_digital_twin.geometry import centerline_spacing
from patient_digital_twin.scan_volume import from_array
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


@pytest.fixture
def patient(request):
    frame = getattr(request, 'param', 'RAS')
    labels = np.zeros((32, 26, 30), np.uint8)
    labels[4:28, 8:13, 7:12] = 1
    labels[4:28, 8:13, 19:24] = 2
    angle = .3
    affine = np.eye(4)
    affine[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    affine[:3, 3] = [20, 30, 40]
    hu = np.where(labels.transpose(2, 1, 0), 100, -1000).astype(np.float32)
    scan = from_array(hu, affine, world_unit='mm', world_frame=frame)
    body = HumanBody(SegmentationImporter(labels, {1: 'aorta', 2: 'inferior_vena_cava'},
                     affine_xyz_to_imaging_m=scan.ijk_to_ras_m).to_anatomy_collection())
    body.attach_scan(scan, source_path='patient/ct.nii.gz')
    body.anatomy.structures['inferior_vena_cava'].enabled = False
    body.anatomy.structures['spleen'] = AnatomicalStructure('spleen', Kind.ORGAN)
    body.extract_topology(spacing_m=.0005)
    body.anatomy.structures['aorta'].local_to_world[:3, 3] += [.1, .2, .3]
    return body


def simulate_edit(workspace, settings, output, options, **kwargs):
    """Exercise the real preparation/finalization boundary with a controlled edit."""
    stage = Usd.Stage.Open(str(workspace / 'input.usdc'))
    mesh = UsdGeom.Mesh.Get(stage, '/HumanBody/Anatomy/aorta')
    mesh.GetPointsAttr().Set([point + Gf.Vec3f(2, 0, 0) for point in mesh.GetPointsAttr().Get()])
    mesh.MakeInvisible()
    UsdGeom.Mesh.Get(stage, '/HumanBody/Anatomy/inferior_vena_cava').MakeVisible()
    material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    shader = material.ComputeSurfaceSource()[0]
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(.8, .1, .2))
    texture = workspace / 'color.png'
    from PIL import Image
    Image.new('RGB', (4, 4), 'red').save(texture)
    sampler = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('ColorTexture'))
    sampler.CreateIdAttr('UsdUVTexture')
    sampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    sampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    shader.GetInput('diffuseColor').ConnectToSource(sampler.ConnectableAPI(), 'rgb')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(
        [(0., 0.)] * len(mesh.GetPointsAttr().Get()))
    stage.Export(str(workspace / 'edited.usdc'))
    finalize(workspace, output)


@pytest.mark.parametrize('patient', ['RAS', 'LPS'], indirect=True)
@pytest.mark.parametrize('editing', [False, True])
def test_bundle_matches_reviewed_contract(patient, tmp_path, monkeypatch, editing):
    placement = np.eye(4)
    placement[:3, 3] = [1, 2, 3]
    options = dict(patient_id='contract-patient', world_from_patient_m=placement,
                   vessel_names=['aorta'], ct_exterior=True, skin_opacity=.27)
    baseline = patient.export_patient_twin(tmp_path / 'baseline', **options)
    patient.export_to_usd(baseline.parent / 'human_body.usdc')
    monkeypatch.setattr(session, 'run', simulate_edit)
    result = PatientWorkflow(cleanup=None, agent=AgentOptions(prompt='edit') if editing else None).run(
        patient, tmp_path / 'refined', format='bundle', **options)
    original = yaml.safe_load(baseline.read_text())
    manifest = yaml.safe_load(result.read_text())
    assert manifest == original
    assert set(original['artifacts'].values()) <= {path.name for path in result.parent.iterdir()}
    for name in ('volume.npy', 'volume.yaml'):
        assert (result.parent / name).read_bytes() == (baseline.parent / name).read_bytes()
    for filename in ('human_body.usdc', 'patient_anatomy.usdc'):
        stage = Usd.Stage.Open(str(result.parent / filename))
        old = Usd.Stage.Open(str(baseline.parent / filename))
        assert str(stage.GetDefaultPrim().GetPath()) == '/HumanBody'
        assert UsdGeom.GetStageUpAxis(stage) == 'Z'
        assert UsdGeom.GetStageMetersPerUnit(stage) == .001
        assert bool(stage.GetPrimAtPath('/HumanBody/Imaging/CT')) == (filename == 'human_body.usdc')
        assert bool(stage.GetPrimAtPath('/HumanBody/Exterior/CT')) == (filename == 'patient_anatomy.usdc')
        if filename == 'human_body.usdc':
            ct = old.GetPrimAtPath('/HumanBody/Imaging/CT')
            for attr in ct.GetAttributes():
                assert stage.GetPrimAtPath(ct.GetPath()).GetAttribute(attr.GetName()).Get() == attr.Get()
        for name, item in manifest['anatomy']['structures'].items():
            prim, previous = stage.GetPrimAtPath(item['prim_path']), old.GetPrimAtPath(item['prim_path'])
            assert prim.GetCustomDataByKey('anatomy:name') == name
            assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Imageable(previous).ComputeVisibility()
            assert UsdGeom.XformCache().GetLocalToWorldTransform(prim) == UsdGeom.XformCache().GetLocalToWorldTransform(previous)
            offset = [2, 0, 0] if editing and name == 'aorta' else [0, 0, 0]
            np.testing.assert_allclose(prim.GetAttribute('points').Get(),
                                       np.asarray(previous.GetAttribute('points').Get()) + offset, atol=1e-5)
            np.testing.assert_allclose(prim.GetAttribute('centerline:points').Get(),
                                       np.asarray(previous.GetAttribute('centerline:points').Get()) + offset, atol=1e-5)
            assert prim.GetAttribute('centerline:coordinateFrame').Get() == 'structure_local'
        if editing:
            aorta = UsdGeom.Mesh.Get(stage, '/HumanBody/Anatomy/aorta')
            assert UsdGeom.PrimvarsAPI(aorta).GetPrimvar('st').Get()
            material = UsdShade.MaterialBindingAPI(aorta).ComputeBoundMaterial()[0]
            assert material.ComputeSurfaceSource()[0].GetInput('diffuseColor').GetConnectedSource()
            assets = [attr.Get() for prim in stage.Traverse() for attr in prim.GetAttributes()
                      if attr.GetTypeName() == Sdf.ValueTypeNames.Asset]
            assert assets and all(Path(asset.resolvedPath).is_file() for asset in assets)
            assert all(not Path(asset.path).is_absolute() for asset in assets)
    mask = np.load(result.parent / 'vessel_mask.npy')
    if editing:
        assert not np.array_equal(mask, np.load(baseline.parent / 'vessel_mask.npy'))
    else:
        for key in ('vessel_mask', 'centerline_points', 'centerline_edges', 'centerline_radii'):
            np.testing.assert_array_equal(np.load(result.parent / manifest['artifacts'][key]),
                                          np.load(baseline.parent / original['artifacts'][key]))
    assert 'spleen' in manifest['anatomy']['missing_meshes']
    assert not manifest['anatomy']['structures']['inferior_vena_cava']['enabled']
    assert manifest['navigation']['vessel_names'] == ['aorta']
    assert not (result.parent / 'input.usdc').exists()
    assert not (result.parent / 'edited.blend').exists()


def test_cleanup_restores_stored_graph_without_navigation_selection(patient, tmp_path):
    result = PatientWorkflow(cleanup=CleanupOptions(smoothing_iterations=2, hole_radius_mm=0)).run(
        patient, tmp_path / 'refined', format='bundle')
    manifest = yaml.safe_load(result.read_text())
    assert 'navigation' not in manifest
    for filename in ('human_body.usdc', 'patient_anatomy.usdc'):
        stage = Usd.Stage.Open(str(result.parent / filename))
        for name in ('aorta', 'inferior_vena_cava'):
            prim = stage.GetPrimAtPath('/HumanBody/Anatomy/' + name)
            spacing = centerline_spacing(prim.GetAttribute('centerline:points').Get(),
                                         prim.GetAttribute('centerline:edges').Get()) * .001
            assert spacing == pytest.approx(.0005, abs=1e-8)


def test_companion_export_failure_leaves_output_unpublished(patient, tmp_path, monkeypatch):
    export = HumanBody.export_to_usd
    def fail_companion(self, path):
        if Path(path).name == 'human_body.usdc':
            raise RuntimeError('companion export failed')
        return export(self, path)
    monkeypatch.setattr(HumanBody, 'export_to_usd', fail_companion)
    target = tmp_path / 'refined'
    with pytest.raises(RuntimeError, match='companion export failed'):
        PatientWorkflow().run(patient, target, format='bundle')
    assert not target.exists()
