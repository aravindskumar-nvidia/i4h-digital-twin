# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify final artifacts; optionally exercise actual Blender import/export."""
import asyncio
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import yaml
from patient_digital_twin import HumanBody, SegmentationImporter
from patient_digital_twin.agent import __main__ as cli
from patient_digital_twin.agent import session
from patient_digital_twin.agent.assets import finalize, prepare
from patient_digital_twin.scan_volume import from_array
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


@pytest.fixture
def asset(tmp_path):
    labels = np.zeros((100, 50, 90), np.uint8)
    labels[5:95, 15:35, 15:35] = 1
    labels[48:50, 24:26, 30:80] = 1  # Thin side branch requires the original fine grid.
    affine = np.diag([.0005, .0005, .0005, 1.])
    affine[:3, 3] = [.04, -.02, .08]
    body = HumanBody(SegmentationImporter(labels, {1: 'aorta'}, affine_xyz_to_imaging_m=affine).to_anatomy_collection())
    body.attach_scan(from_array(labels, affine * np.array([1000, 1000, 1000, 1])[:, None], world_unit='mm'))
    body.extract_topology(spacing_m=.0005)
    source = body.export_to_usd(tmp_path / 'patient.usdc')
    work = tmp_path / 'work'
    prepare(source, work)
    return source, work


@pytest.mark.parametrize('deform', [False, True])
def test_commit_preserves_ct_texture_and_structure_centerline(asset, tmp_path, deform):
    source, work = asset
    edited = Usd.Stage.Open(str(work / 'input.usdc'))
    mesh = UsdGeom.Mesh(edited.GetPrimAtPath('/HumanBody/Anatomy/aorta'))
    mesh.MakeInvisible()
    # A tiny round-off shift should count as unchanged geometry.
    shift = Gf.Vec3f(6 if deform else .005, 0, 0)
    mesh.GetPointsAttr().Set([p + shift for p in mesh.GetPointsAttr().Get()])
    texture = work / 'color.png'
    texture.write_bytes(b'texture-dependency')
    material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    shader = UsdShade.Shader.Define(edited, material.GetPath().AppendChild('Texture'))
    shader.CreateIdAttr('UsdUVTexture')
    shader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    edited.Export(str(work / 'edited.usdc'))
    before = source.read_bytes()
    snapshot = Usd.Stage.Open(str(work / 'original.usdc'))
    baseline = snapshot.GetRootLayer().ExportToString()
    texture.unlink()
    with pytest.raises(ValueError, match='Unresolved texture'):
        finalize(work, tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists()
    assert snapshot.GetRootLayer().ExportToString() == baseline
    texture.write_bytes(b'texture-dependency')
    output = finalize(work, tmp_path / 'final')
    assert source.read_bytes() == before
    stage = Usd.Stage.Open(str(output))
    original = Usd.Stage.Open(str(source))
    assert stage.GetDefaultPrim().GetPath() == original.GetDefaultPrim().GetPath()
    assert UsdGeom.GetStageMetersPerUnit(stage) == .001
    ct = '/HumanBody/Imaging/CT'
    for attr in original.GetPrimAtPath(ct).GetAttributes():
        assert stage.GetPrimAtPath(ct).GetAttribute(attr.GetName()).Get() == attr.Get()
    prim = stage.GetPrimAtPath('/HumanBody/Anatomy/aorta')
    assert prim.GetCustomDataByKey('anatomy:name') == 'aorta'
    old = original.GetPrimAtPath(prim.GetPath())
    assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Imageable(old).ComputeVisibility()
    offset = [6, 0, 0] if deform else [0, 0, 0]
    np.testing.assert_allclose(prim.GetAttribute('centerline:points').Get(),
                               np.asarray(old.GetAttribute('centerline:points').Get()) + offset, atol=1e-5)
    if not deform:
        assert prim.GetAttribute('points').Get() == old.GetAttribute('points').Get()
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    assert material
    path = stage.GetPrimAtPath(material.GetPath().AppendChild('Texture')).GetAttribute('inputs:file').Get()
    assert Path(path.resolvedPath).read_bytes() == texture.read_bytes()


def test_missing_mesh_never_publishes(asset, tmp_path):
    _, work = asset
    edited = Usd.Stage.Open(str(work / 'input.usdc'))
    edited.RemovePrim('/HumanBody/Anatomy/aorta')
    edited.Export(str(work / 'edited.usdc'))
    with pytest.raises(ValueError, match='identity'):
        finalize(work, tmp_path / 'final')
    assert not (tmp_path / 'final').exists()


@pytest.mark.parametrize('saved', ['usd', 'blend', 'resume'])
def test_recovery_reuses_saved_edit(asset, tmp_path, monkeypatch, saved):
    source, work = asset
    calls = []
    if saved == 'usd':
        (work / 'edited.usdc').write_bytes((work / 'input.usdc').read_bytes())
    else:
        (work / 'edited.blend').touch()

    async def blender_call(config, function, workspace):
        calls.append(function)
        assert workspace == work
        if function == 'export_asset':
            (work / 'edited.usdc').write_bytes((work / 'input.usdc').read_bytes())

    monkeypatch.setattr(session, 'blender_call', blender_call)
    monkeypatch.setattr(session, 'blender_session', lambda *args: nullcontext({}))
    def run_codex(*args, publish, **kwargs):
        calls.append('agent')
        publish()

    monkeypatch.setattr(session, 'run_codex', run_codex)
    command = ['resume' if saved == 'resume' else 'finalize',
               '--workspace', str(work), '--output', str(tmp_path / 'final')]
    if saved == 'resume':
        (work / 'task_prompt.txt').write_text('Saved task')
        command += ['--backend', 'codex', 'Repair the saved edit.']
    assert cli.main(command) == 0
    expected = {
        'usd': [],
        'blend': ['open_checkpoint', 'export_asset'],
        'resume': ['open_checkpoint', 'agent', 'export_asset'],
    }
    assert calls == expected[saved]
    assert (tmp_path / 'final' / source.name).is_file()
    assert (work / 'original.usdc').is_file()


@pytest.mark.parametrize('hidden', [False, True])
def test_real_blender_geometry_and_textured_material(asset, tmp_path, hidden):
    blender = os.environ.get('I4H_TEST_BLENDER')
    if not blender:
        pytest.skip('Set I4H_TEST_BLENDER to exercise an installed Blender')
    from patient_digital_twin.agent import blender as worker
    source, work = asset
    if hidden:
        stage = Usd.Stage.Open(str(source))
        UsdGeom.Imageable(stage.GetPrimAtPath('/HumanBody/Anatomy')).MakeInvisible()
        stage.GetRootLayer().Save()
        prepare(source, work)
    script = tmp_path / 'edit.py'
    script.write_text(f'''exec(compile(open({worker.__file__!r}).read(), {worker.__file__!r}, 'exec'))
open_asset({str(work)!r})
import bpy
obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
obj.hide_viewport = False
obj.hide_set(False)
obj.location.x += .001
obj.data.update()
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project()
try:
    check_asset({str(work)!r})
except ValueError as exc:
    assert 'Edit Mode' in str(exc)
else:
    raise AssertionError('Scene check accepted Edit Mode')
bpy.ops.object.mode_set(mode='OBJECT')
check_asset({str(work)!r})
material = obj.data.materials[0]
material.use_nodes = True
bsdf = next(n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
texture = material.node_tree.nodes.new('ShaderNodeTexImage')
image = bpy.data.images.new('test_color', width=4, height=4)
image.generated_color = (0.8, 0.1, 0.2, 1)
image.filepath_raw = {str(work / 'color.png')!r}
image.file_format = 'PNG'
image.save()
texture.image = image
material.node_tree.links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])
obj.hide_render = {not hidden!r}
obj.hide_viewport = {not hidden!r}
obj.hide_set({not hidden!r})
export_asset({str(work)!r})
open_checkpoint({str(work)!r})
export_asset({str(work)!r})
''')
    subprocess.run([blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python', str(script)], check=True, capture_output=True, text=True)
    output = finalize(work, tmp_path / 'final')
    stage = Usd.Stage.Open(str(output))
    original = Usd.Stage.Open(str(source))
    prim = stage.GetPrimAtPath('/HumanBody/Anatomy/aorta')
    old = original.GetPrimAtPath(prim.GetPath())
    assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Imageable(old).ComputeVisibility()
    delta = np.asarray(prim.GetAttribute('points').Get()) - np.asarray(old.GetAttribute('points').Get())
    np.testing.assert_allclose(delta, np.broadcast_to([1., 0., 0.], delta.shape), atol=1e-4)
    assert len(prim.GetAttribute('centerline:points').Get()) > 1
    ct = '/HumanBody/Imaging/CT'
    assert stage.GetPrimAtPath(ct).GetAttribute('ct:hu').Get() == original.GetPrimAtPath(ct).GetAttribute('ct:hu').Get()
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    shader = material.ComputeSurfaceSource()[0]
    assert shader.GetIdAttr().Get() == 'UsdPreviewSurface'
    assert shader.GetInput('diffuseColor').HasConnectedSource()
    assert UsdGeom.PrimvarsAPI(prim).GetPrimvar('st').HasValue()
    textures = [a.Get() for p in stage.Traverse() for a in p.GetAttributes() if a.GetTypeName() == Sdf.ValueTypeNames.Asset and a.Get()]
    assert textures and all(Path(t.resolvedPath).is_file() for t in textures)


def test_real_filtered_reference_survives_idle(asset, monkeypatch):
    """Opt-in regression for deleted HRA IDs retained in a render dependency graph."""
    blender = os.environ.get('I4H_TEST_BLENDER')
    reference = os.environ.get('I4H_TEST_REFERENCE_IDLE')
    if not blender or not reference:
        pytest.skip('Set I4H_TEST_BLENDER and I4H_TEST_REFERENCE_IDLE to HRA aorta metadata (75s idle check)')
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from patient_digital_twin.agent.completion import prepare_references
    from patient_digital_twin.agent.runtime import blender_session

    _, work = asset
    prepare_references(work, {'aorta': Path(reference).resolve()}, include_major_branches=True)
    monkeypatch.setenv('BLENDER_EXECUTABLE', blender)
    capture = """import bpy
for scene in bpy.data.scenes:
    if not (scene.get('i4h_session') or scene.get('i4h_reference')):
        continue
    camera = next((o for o in scene.objects if o.type == 'CAMERA'), None)
    if camera is None:
        camera = bpy.data.objects.new('capture', bpy.data.cameras.new('capture'))
        scene.collection.objects.link(camera)
    camera.location = (0, -2, 0)
    camera.rotation_euler = (1.5707963, 0, 0)
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 3
    scene.camera = camera
    scene.render.resolution_x = scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
    bpy.ops.render.render(scene=scene.name)
    scene.camera = None
"""

    async def inspect(config):
        await session.blender_call(config, 'open_asset', work)
        async with stdio_client(StdioServerParameters(**config)) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                for delay in (0, 75):
                    await asyncio.sleep(delay)
                    result = await client.call_tool('execute_blender_code', {'code': capture})
                    assert not result.isError
                    assert not any(c.type == 'text' and c.text.startswith('Error') for c in result.content)
                    result = await client.call_tool('look', {'image': 'Render Result', 'max_size': 256})
                    assert not result.isError and any(c.type == 'image' for c in result.content)

    with blender_session(work, session.mcp_config(), 'virtual') as config:
        asyncio.run(inspect(config))


@pytest.mark.parametrize('selection', ['saved', 'single', 'ambiguous'])
def test_bundle_regenerates_navigation_on_original_ct_grid(tmp_path, selection):
    labels = np.zeros((30, 20, 25), np.uint8)
    labels[3:27, 8:13, 9:14] = 1
    labelmap = {1: 'aorta'}
    if selection != 'single':
        labels[3:27, 8:13, 17:22] = 2
        labelmap[2] = 'inferior_vena_cava'
    body = HumanBody(SegmentationImporter(labels, labelmap).to_anatomy_collection())
    body.attach_scan(from_array(labels.transpose(2, 1, 0), np.eye(4), world_unit='mm'))
    body.extract_topology()
    source = body.export_patient_twin(tmp_path / 'source', vessel_names=['aorta'])
    body.anatomy.structures['aorta'].local_to_world[:3, 3] += [.1, .2, .3]
    companion = body.export_to_usd(source.parent / 'human_body.usdc') if selection != 'single' else None
    before = companion.read_bytes() if companion else None
    manifest = yaml.safe_load(source.read_text())
    assert manifest['navigation'] == {'vessel_names': ['aorta']}
    if selection != 'saved':
        del manifest['navigation']
        source.write_text(yaml.safe_dump(manifest))
    for name in ('view.png', 'inference_summary.json'):
        (source.parent / name).write_bytes(b'stale output from the original run')
    work = tmp_path / 'work'
    if selection == 'ambiguous':
        with pytest.raises(ValueError, match='Supply --vessel-names'):
            prepare(source, work)
        with pytest.raises(ValueError, match='Unknown'):
            prepare(source, work, vessel_names=['missing_vessel'])
        assert not work.exists()
    prepare(source, work, vessel_names=['aorta'] if selection == 'ambiguous' else None)
    edited = Usd.Stage.Open(str(work / 'input.usdc'))
    mesh = UsdGeom.Mesh(edited.GetPrimAtPath('/HumanBody/Anatomy/aorta'))
    points = mesh.GetPointsAttr().Get()
    mesh.GetPointsAttr().Set([p + Gf.Vec3f(2, 0, 0) for p in points])
    mesh.MakeInvisible()
    material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    shader = material.ComputeSurfaceSource()[0]
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(.8, .1, .2))
    edited.Export(str(work / 'edited.usdc'))
    output = finalize(work, tmp_path / 'final')
    assert yaml.safe_load(output.read_text())['navigation'] == {'vessel_names': ['aorta']}
    assert yaml.safe_load(output.read_text())['anatomy'] == manifest['anatomy']
    assert not any((output.parent / name).exists() for name in
                   ('view.png', 'inference_summary.json'))
    assert (output.parent / 'human_body.usdc').exists() == (companion is not None)
    if companion:
        assert companion.read_bytes() == before
        old = Usd.Stage.Open(str(companion))
        stage = Usd.Stage.Open(str(output.parent / companion.name))
        prim = stage.GetPrimAtPath('/HumanBody/Anatomy/aorta')
        original = old.GetPrimAtPath(prim.GetPath())
        assert UsdGeom.XformCache().GetLocalToWorldTransform(prim) == UsdGeom.XformCache().GetLocalToWorldTransform(original)
        for name in ('points', 'centerline:points'):
            np.testing.assert_allclose(prim.GetAttribute(name).Get(),
                                       np.asarray(original.GetAttribute(name).Get()) + [2, 0, 0], atol=1e-5)
        ct = '/HumanBody/Imaging/CT'
        for attr in old.GetPrimAtPath(ct).GetAttributes():
            assert stage.GetPrimAtPath(ct).GetAttribute(attr.GetName()).Get() == attr.Get()
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        assert material.ComputeSurfaceSource()[0].GetInput('diffuseColor').Get() == Gf.Vec3f(.8, .1, .2)
    assert (output.parent / 'volume.npy').read_bytes() == (source.parent / 'volume.npy').read_bytes()
    assert (output.parent / 'volume.yaml').read_bytes() == (source.parent / 'volume.yaml').read_bytes()
    np.testing.assert_allclose(np.load(output.parent / 'centerline_points.npy')[:, 0], 13., atol=1e-6)
    np.testing.assert_array_equal(np.load(output.parent / 'vessel_mask.npy'),
                                  np.roll(np.load(source.parent / 'vessel_mask.npy'), 2, axis=0))
