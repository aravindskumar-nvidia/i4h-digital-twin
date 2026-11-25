# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for optional editing, operation order and portable outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from patient_digital_twin import (
    AgentOptions,
    CleanupOptions,
    HumanBody,
    MeshGeometry,
    PatientWorkflow,
    SegmentationImporter,
    SurfaceMaterial,
    TextureOptions,
    TextureReference,
    TextureResult,
    TextureRuntime,
    prepare_surface,
    workflow,
)
from patient_digital_twin.agent import __main__ as agent_cli
from patient_digital_twin.scan_volume import from_array
from pxr import Sdf, Usd, UsdGeom


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


@pytest.fixture
def fragmented_surface():
    import trimesh

    pieces = []
    for size, x in [(.04, 0), (.01, .03), (.008, .085), (.001, .024), (.012, .25)]:
        part = trimesh.creation.box(extents=[size] * 3)
        part.apply_translation([x, 0, 0])
        pieces.append(part)
    surface = trimesh.util.concatenate(pieces)
    return MeshGeometry(surface.vertices.copy(), surface.faces.copy())


@pytest.mark.parametrize('filtering', [True, False])
def test_cleanup_filters_tiny_and_distant_islands_but_keeps_nearby_chains(
    fragmented_surface, filtering,
):
    import trimesh

    options = CleanupOptions() if filtering else CleanupOptions(min_component_area_mm2=0, max_component_distance_mm=0)
    result = prepare_surface(fragmented_surface, options)
    assert result.measurements['after']['components'] == (3 if filtering else 5)
    records = result.measurements['component_filter']['components']
    assert records[0]['decision'] == 'keep_primary'
    if filtering:
        assert [r['decision'] for r in records].count('remove_small') == 1
        assert [r['decision'] for r in records].count('remove_distant') == 1
        assert [r['decision'] for r in records].count('retain_nearby') == 2
        assert sorted(round(r['gap_mm']) for r in records if r['decision'] == 'retain_nearby') == [5, 46]
        parts = trimesh.Trimesh(result.mesh.vertices, result.mesh.faces, process=False).split(only_watertight=False)
        assert sorted(abs(part.volume) for part in parts) == pytest.approx(
            sorted([.04 ** 3, .01 ** 3, .008 ** 3]), rel=.02)


def test_cleanup_uses_surface_gap_and_preserves_small_primary():
    import trimesh

    large = trimesh.creation.box(extents=[.4, .3, .3])
    near_face = trimesh.creation.box(extents=[.01] * 3)
    near_face.apply_translation([.21, 0, 0])
    surface = trimesh.util.concatenate([large, near_face])
    result = prepare_surface(MeshGeometry(surface.vertices, surface.faces),
                             CleanupOptions(smoothing_iterations=0, hole_radius_mm=0))
    assert result.measurements['after']['components'] == 2
    fragment = result.measurements['component_filter']['components'][1]
    assert fragment['decision'] == 'retain_nearby'
    assert fragment['gap_mm'] == pytest.approx(5)
    small = trimesh.creation.box(extents=[.001] * 3)
    result = prepare_surface(MeshGeometry(small.vertices, small.faces),
                             CleanupOptions(smoothing_iterations=0, hole_radius_mm=0))
    assert result.measurements['after']['components'] == 1


def test_cleanup_history_reaches_immediate_and_later_agents(body, fragmented_surface, tmp_path, monkeypatch):
    import shutil
    from patient_digital_twin.agent.assets import prepare

    body.anatomy.structures['liver'].mesh = fragmented_surface
    captures = []
    def edit(args, workspace, settings, output):
        captures.append(settings['cleanup_components'])
        output.mkdir()
        shutil.copy2(workspace / 'input.usdc', output / 'input.usdc')
    monkeypatch.setattr(agent_cli, '_run_blender', edit)
    result = PatientWorkflow(
        cleanup=CleanupOptions(smoothing_iterations=0, hole_radius_mm=0),
        agent=AgentOptions(task='refine', targets=('liver',)),
    ).run(body, tmp_path / 'bundle', format='bundle')
    assert len(captures) == 1
    records = captures[0]['liver']['components']
    assert sum(r['decision'] == 'retain_nearby' for r in records) == 2
    later = prepare(result, tmp_path / 'later-agent')
    assert later['cleanup_components'] == captures[0]
    for filename in ['human_body.usdc', 'patient_anatomy.usdc']:
        stage = Usd.Stage.Open(str(result.parent / filename))
        assert json.loads(stage.GetPrimAtPath('/HumanBody').GetAttribute('cleanup:components').Get()) == captures[0]


@pytest.mark.parametrize('format', ['usd', 'bundle'])
def test_default_workflow_never_starts_agent(body, tmp_path, monkeypatch, format):
    def unexpected(*args, **kwargs):
        pytest.fail('Default workflow started an agent')
    monkeypatch.setattr(agent_cli, 'edit_asset', unexpected)
    target = tmp_path / ('patient.usdc' if format == 'usd' else 'bundle')
    output = PatientWorkflow().run(body, target, format=format)
    assert output.is_file()
    report = (output.parent / 'workflow.json' if format == 'bundle'
              else output.parent / (output.name + '.assets') / 'workflow.json')
    cleanup = json.loads(report.read_text())['cleanup']['liver']
    assert cleanup['smoothing_algorithm'] == 'windowed_sinc'
    assert cleanup['smoothing_iterations'] == 60 and cleanup['requested_smoothing_pass_band'] == .01
    with pytest.raises(FileExistsError):
        PatientWorkflow().run(body, target, format=format)


@pytest.mark.parametrize('format', ['usd', 'bundle'])
@pytest.mark.parametrize('reflected', [False, True])
def test_agent_precedes_texturing_and_output_is_portable(body, tmp_path, monkeypatch, format, reflected):
    events = []
    original = body.anatomy.structures['liver'].mesh.vertices.copy()
    tile = tmp_path / 'tile.png'
    tile.write_bytes(b'fixture image')
    reference = TextureReference(tile, {'surface': 'serosa', 'license': 'CC-BY-4.0'})

    def edit(source, output, options, *, workspace):
        events.append('agent')
        assert options.token_budget == 1234
        output.mkdir()
        stage = Usd.Stage.Open(str(source))
        mesh = UsdGeom.Mesh.Get(stage, '/HumanBody/Anatomy/liver')
        mesh.GetPointsAttr().Set([point + (1, 0, 0) for point in mesh.GetPointsAttr().Get()])
        if reflected:
            mesh.GetOrientationAttr().Set('leftHanded')
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            mesh.GetFaceVertexIndicesAttr().Set(faces[:, ::-1].ravel().tolist())
        stage.Export(str(output / 'input.usdc'))
        return output / 'input.usdc'

    def texture(mesh, reference, options, **kwargs):
        events.append('texture')
        np.testing.assert_allclose(mesh.vertices, original + [.001, 0, 0], atol=1e-9)
        material = SurfaceMaterial(str(tile), str(tile))
        geometry = MeshGeometry(mesh.vertices, mesh.faces, np.zeros((len(mesh.vertices), 2)), None, material)
        return TextureResult(geometry, np.arange(len(mesh.vertices)), tmp_path, {'surface': 'serosa'})

    monkeypatch.setattr(agent_cli, 'edit_asset', edit)
    monkeypatch.setattr(workflow, 'texture_mesh', texture)
    target = tmp_path / ('patient.usdc' if format == 'usd' else 'bundle')
    output = PatientWorkflow(cleanup=None, agent=AgentOptions('refine', token_budget=1234), textures={'liver': reference},
                             texture_runtime=TextureRuntime(tmp_path)).run(body, target, format=format)
    assert events == ['agent', 'texture']
    np.testing.assert_array_equal(body.anatomy.structures['liver'].mesh.vertices, original)
    tile.unlink()
    stage = Usd.Stage.Open(str(output if format == 'usd' else output.parent / 'patient_anatomy.usdc'))
    assets = [attr.Get() for prim in stage.Traverse() for attr in prim.GetAttributes()
              if attr.GetTypeName() == Sdf.ValueTypeNames.Asset]
    assert assets and all(Path(asset.resolvedPath).is_file() for asset in assets)
    assert all(not Path(asset.path).is_absolute() for asset in assets)


def test_texture_failure_does_not_publish_or_mutate(body, tmp_path, monkeypatch):
    original = body.anatomy.structures['liver'].mesh.vertices.copy()
    def fail(*args, **kwargs):
        raise RuntimeError('inference failed')
    monkeypatch.setattr(workflow, 'texture_mesh', fail)
    pipeline = PatientWorkflow(cleanup=CleanupOptions(), textures={'liver': TextureReference(tmp_path, {})},
                               texture_runtime=TextureRuntime(tmp_path))
    with pytest.raises(RuntimeError, match='inference failed'):
        pipeline.run(body, tmp_path / 'patient.usdc')
    assert not (tmp_path / 'patient.usdc').exists()
    np.testing.assert_array_equal(body.anatomy.structures['liver'].mesh.vertices, original)


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


def test_reference_completion_validation_and_provenance(body, tmp_path):
    import shutil

    from patient_digital_twin.agent.assets import finalize, prepare
    from patient_digital_twin.refinement import prepare_references

    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'reference.glb').write_bytes(b'fixture reference')
    (reference / 'LICENSE.txt').write_text('CC BY 4.0 fixture')
    (reference / 'source.json').write_text('{}')
    (reference / 'ATTRIBUTION.txt').write_text('Fixture author')
    (reference / 'structures.json').write_text('{}')
    metadata = dict(id='fixture', file='reference.glb', units='meters', up_axis='Y',
                    sha256=hashlib.sha256(b'fixture reference').hexdigest(),
                    license={'spdx': 'CC-BY-4.0', 'text': 'LICENSE.txt'},
                    source={'evidence': 'source.json', 'url': 'https://example.org/reference'},
                    structures='structures.json', selections={'liver_main': ['liver']},
                    attribution='Fixture author', changes='Fixture generated')
    (reference / 'metadata.json').write_text(json.dumps(metadata))
    source = body.export_to_usd(tmp_path / 'input.usdc')
    workspace = tmp_path / 'edit'
    prepare(source, workspace)
    prompt = prepare_references(workspace, {'liver': reference / 'metadata.json'})
    assert 'liver' in prompt
    shutil.copy2(workspace / 'input.usdc', workspace / 'edited.usdc')
    with pytest.raises(ValueError, match="completion.json"):
        finalize(workspace, tmp_path / 'missing-report')
    (workspace / 'completion.json').write_text(json.dumps({'liver': {
        'generated_regions': [], 'removed_fragments': [], 'retained_fragments': [], 'uncertainties': []}}))
    with pytest.raises(ValueError, match='reference_review'):
        finalize(workspace, tmp_path / 'missing-inspection')
    from PIL import Image
    for name in ('reference.png', 'result.png'):
        Image.new('RGB', (16, 16)).save(workspace / name)
    entry = json.loads((workspace / 'completion.json').read_text())
    entry['liver']['reference_review'] = dict(
        observed_features=['Fixture contour'], preserved_features=['Fixture contour'],
        generated_features=[], unresolved_features=[],
        reference_images=['reference.png'], result_images=['../external.png'])
    (workspace / 'completion.json').write_text(json.dumps(entry))
    with pytest.raises(ValueError, match='missing or external comparison image'):
        finalize(workspace, tmp_path / 'external-inspection')
    entry['liver']['reference_review']['result_images'] = ['result.png']
    (workspace / 'completion.json').write_text(json.dumps(entry))
    output = finalize(workspace, tmp_path / 'final')
    report = json.loads((output.parent / 'refinement/report.json').read_text())
    assert report['structures']['liver']['validation']['connected_components'] == 1
    assert (output.parent / 'refinement/references/liver/ATTRIBUTION.txt').is_file()
    review = report['structures']['liver']['reference_review']
    for relative in review['reference_images'] + review['result_images']:
        assert (output.parent / 'refinement' / relative).is_file()


def test_inference_cli_accepts_nonagentic_refinement_flags(body, tmp_path, monkeypatch):
    from patient_digital_twin import __main__ as cli
    def generate(self, **kwargs):
        self.ct_scan = body.imaging.scan
        return body.anatomy
    monkeypatch.setattr(cli.NVGenerateImporter, 'to_anatomy_collection', generate)
    result = cli.main(['--source', 'nvgenerate', '--classes', 'liver',
                       '--output', str(tmp_path / 'patient.usdc'), '--smoothing-iterations', '0',
                       '--workspace', str(tmp_path / 'work')])
    assert result == 0
    report = json.loads((tmp_path / 'patient.usdc.assets/workflow.json').read_text())
    assert report['agent'] is False and 'liver' in report['cleanup']
    assert (tmp_path / 'work').is_dir()


def test_navigation_is_derived_from_agent_edited_geometry(tmp_path, monkeypatch):
    import yaml
    z, y, x = np.indices((35, 25, 25))
    labels = (((x - 10) ** 2 + (y - 12) ** 2 < 16) & (z > 3) & (z < 31)).astype(np.uint8)
    patient = HumanBody(SegmentationImporter(labels, {1: 'aorta'},
                        affine_xyz_to_imaging_m=np.diag([.001, .001, .001, 1])).to_anatomy_collection())
    patient.attach_scan(from_array(labels.transpose(2, 1, 0), np.eye(4), world_unit='mm'))

    def edit(source, output, options, *, workspace):
        stage = Usd.Stage.Open(str(source))
        mesh = UsdGeom.Mesh.Get(stage, '/HumanBody/Anatomy/aorta')
        assert not mesh.GetPrim().HasAttribute('centerline:points')
        mesh.GetPointsAttr().Set([point + (3, 0, 0) for point in mesh.GetPointsAttr().Get()])
        output.mkdir()
        stage.Export(str(output / 'input.usdc'))
        return output / 'input.usdc'

    monkeypatch.setattr(agent_cli, 'edit_asset', edit)
    output = PatientWorkflow(cleanup=None, agent=AgentOptions('shift')).run(
        patient, tmp_path / 'bundle', format='bundle', vessel_names=['aorta'])
    manifest = yaml.safe_load(output.read_text())
    mask = np.load(output.parent / manifest['artifacts']['vessel_mask'])
    assert abs(np.argwhere(mask)[:, 0].mean() - 13) < .5
    stage = Usd.Stage.Open(str(output.parent / manifest['artifacts']['anatomy_usd']))
    assert stage.GetPrimAtPath('/HumanBody/Anatomy/aorta').HasAttribute('centerline:points')


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


def test_cleanup_can_be_explicitly_disabled_with_flags():
    import argparse

    from patient_digital_twin._refinement_cli import (
        add_refinement_arguments,
        pop_refinement_workflow,
    )
    parser = argparse.ArgumentParser()
    add_refinement_arguments(parser)
    assert pop_refinement_workflow(vars(parser.parse_args([]))) is None
    assert pop_refinement_workflow(vars(parser.parse_args(['--cleanup']))).cleanup == CleanupOptions()
    assert pop_refinement_workflow(vars(parser.parse_args(['--no-cleanup']))).cleanup is None


def test_colon_shape_gate_rejects_connected_surface_with_handles(tmp_path):
    import trimesh
    from patient_digital_twin.refinement import validate_completion

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, '/HumanBody/Anatomy/colon')
    settings = {'references': {'colon': {'target': str(mesh.GetPath()), 'metadata': {'id': 'fixture'}}},
                'completion': {'require_colon_shape': True}}
    (tmp_path / 'completion.json').write_text(json.dumps({'colon': {
        'generated_regions': [], 'removed_fragments': [], 'retained_fragments': [], 'uncertainties': []}}))
    for surface in (trimesh.creation.torus(major_radius=.02, minor_radius=.005),
                    trimesh.creation.icosphere(subdivisions=2, radius=.02)):
        mesh.CreatePointsAttr(surface.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(surface.faces))
        mesh.CreateFaceVertexIndicesAttr(surface.faces.ravel().tolist())
        if surface.euler_number != 2:
            with pytest.raises(ValueError, match='repair holes/handles'):
                validate_completion(stage, tmp_path, settings)
        else:
            report = validate_completion(stage, tmp_path, settings)
            assert report['structures']['colon']['validation']['euler_characteristic'] == 2


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


@pytest.mark.parametrize('mode', ['default', 'override', 'disabled'])
def test_workflow_resolves_texture_defaults(body, tmp_path, monkeypatch, mode):
    calls = []
    override = TextureReference(tmp_path / 'custom.png', {'surface': 'custom'})

    def texture(mesh, reference, options, **kwargs):
        calls.append(reference)
        return TextureResult(mesh, np.arange(len(mesh.vertices)), tmp_path,
                             {'surface': reference.metadata['surface']})

    monkeypatch.setattr(workflow, 'texture_mesh', texture)
    pipeline = PatientWorkflow(cleanup=None, texture_runtime=TextureRuntime(tmp_path),
                               default_textures=mode != 'disabled')
    if mode == 'override':
        pipeline.textures['liver'] = override
    output = pipeline.run(body, tmp_path / 'patient.usdc')
    assert output.is_file()
    if mode == 'disabled':
        assert calls == []
    elif mode == 'override':
        assert calls == [override]
    else:
        assert len(calls) == 1
        assert calls[0].metadata['organ'] == 'liver'
        assert pipeline.textures == {}  # Resolved defaults belong to this run.


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


def test_retain_missing_texture_does_not_hide_checksum_errors(body, tmp_path, monkeypatch):
    from patient_digital_twin.references import MissingTextureDefault
    def missing(name):
        raise MissingTextureDefault(name)
    monkeypatch.setattr(workflow, 'default_texture_reference', missing)
    pipeline = PatientWorkflow(cleanup=None,
                               texture_runtime=TextureRuntime(tmp_path))
    output = pipeline.run(body, tmp_path / 'retained.usdc')
    report = json.loads((output.parent / (output.name + '.assets') / 'workflow.json').read_text())
    assert report['retained_materials_no_default'] == ['liver']
    assert report['textures'] == {}
    pipeline.missing_texture = 'error'
    with pytest.raises(MissingTextureDefault):
        pipeline.run(body, tmp_path / 'strict.usdc')
    pipeline.missing_texture = 'retain'
    def corrupted(name):
        raise ValueError('Texture checksum mismatch')
    monkeypatch.setattr(workflow, 'default_texture_reference', corrupted)
    with pytest.raises(ValueError, match='checksum'):
        pipeline.run(body, tmp_path / 'corrupted.usdc')


def test_pipeline_resume_continues_textures_and_export_without_reediting(body, tmp_path, monkeypatch):
    from patient_digital_twin._workflow_checkpoint import load_body
    counts = {'agent': 0, 'texture': 0, 'cleanup': 0}
    original_cleanup = workflow.prepare_surface
    def clean(*args):
        counts['cleanup'] += 1
        return original_cleanup(*args)
    def edit(source, output, options, *, workspace):
        import shutil
        counts['agent'] += 1
        output.mkdir()
        shutil.copy2(source, output / source.name)
        return output / source.name
    def texture(mesh, reference, options, **kwargs):
        counts['texture'] += 1
        if counts['texture'] == 1:
            raise RuntimeError('GPU interrupted')
        return TextureResult(mesh, np.arange(len(mesh.vertices)), tmp_path, {'completed': True})
    monkeypatch.setattr(workflow, 'prepare_surface', clean)
    monkeypatch.setattr(agent_cli, 'edit_asset', edit)
    monkeypatch.setattr(workflow, 'texture_mesh', texture)
    pipeline = PatientWorkflow(agent=AgentOptions('Edit'), texture_runtime=TextureRuntime(tmp_path),
                               workspace=tmp_path / 'work')
    with pytest.raises(RuntimeError, match='GPU interrupted'):
        pipeline.run(body, tmp_path / 'bundle', format='bundle', patient_id='resume-test')
    run = next((tmp_path / 'work').glob('run-*'))
    restored = load_body(run / 'body')
    np.testing.assert_array_equal(restored.imaging.scan.values, body.imaging.scan.values)
    np.testing.assert_array_equal(restored.imaging.scan.ijk_to_ras_m, body.imaging.scan.ijk_to_ras_m)
    output = PatientWorkflow.resume(run)
    assert output.is_file()
    assert (output.parent / 'human_body.usdc').is_file()
    assert (output.parent / 'patient_anatomy.usdc').is_file()
    report = json.loads((output.parent / 'workflow.json').read_text())
    assert report['textures']['liver']['completed']
    assert counts == {'agent': 1, 'texture': 2, 'cleanup': 1}
    assert PatientWorkflow.resume(run) == output
    assert counts['texture'] == 2


def test_pipeline_resume_after_agent_failure_runs_remaining_pipeline(body, tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise RuntimeError('token budget exhausted')
    def resumed(workspace, output, options):
        import shutil
        assert options.token_budget == 200000 and options.blender_mode == 'virtual'
        output.mkdir()
        shutil.copy2(workspace.parent / 'input.usdc', output / 'input.usdc')
        return output / 'input.usdc'
    monkeypatch.setattr(agent_cli, 'edit_asset', interrupted)
    monkeypatch.setattr(agent_cli, 'resume_asset', resumed)
    run_options = PatientWorkflow(cleanup=None, agent=AgentOptions('Edit'), workspace=tmp_path / 'work')
    with pytest.raises(RuntimeError, match='token budget'):
        run_options.run(body, tmp_path / 'patient.usdc')
    run = next((tmp_path / 'work').glob('run-*'))
    result = PatientWorkflow.resume(run, token_budget=200000, blender_mode='virtual')
    assert result.is_file()
    assert json.loads((run / 'pipeline.json').read_text())['stage'] == 'complete'


def test_editor_resume_routes_back_to_parent_pipeline(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    workspace = run / 'agent'
    workspace.mkdir(parents=True)
    (run / 'pipeline.json').write_text('{}')
    calls = []
    def resume(path, **options):
        calls.append((path, options))
        return tmp_path / 'final/patient_twin.yaml'
    monkeypatch.setattr(PatientWorkflow, 'resume', resume)
    assert agent_cli.main(['resume', '--workspace', str(workspace), '--output', str(tmp_path / 'final'),
                           '--backend', 'codex', '--model', 'gpt-6.1-terra', '--blender-mode', 'virtual',
                           '--token-budget', '200000', 'Finish the colon']) == 0
    assert calls[0][0] == run
    assert calls[0][1]['token_budget'] == 200000
    assert calls[0][1]['agent_overrides']['prompt'] == 'Finish the colon'


def test_checkpoint_preserves_patient_contract(body, tmp_path):
    from patient_digital_twin._workflow_checkpoint import load_body, save_body
    from patient_digital_twin.geometry import CenterlineGraph
    structure = body.anatomy.structures['liver']
    structure.local_to_world[:3, 3] = [.1, .2, .3]
    structure.local_to_body[:3, 3] = [.01, .02, .03]
    structure.enabled = False
    structure.mesh.uv = np.zeros((len(structure.mesh.vertices), 2))
    structure.mesh.normals = np.ones_like(structure.mesh.vertices)
    structure.mesh.material = SurfaceMaterial('base.png', 'rough.png', .4)
    structure.centerline = CenterlineGraph(points=np.array([[0., 0, 0], [0, 0, .01]]),
                                         radii=np.array([.001, .002]), edges=np.array([[0, 1]]))
    save_body(body, tmp_path / 'snapshot')
    restored = load_body(tmp_path / 'snapshot')
    other = restored.anatomy.structures['liver']
    assert other.enabled is False and other.mesh.material == structure.mesh.material
    for name in ['vertices', 'faces', 'uv', 'normals']:
        np.testing.assert_array_equal(getattr(other.mesh, name), getattr(structure.mesh, name))
    np.testing.assert_array_equal(other.local_to_body, structure.local_to_body)
    np.testing.assert_array_equal(other.local_to_world, structure.local_to_world)
    np.testing.assert_array_equal(other.centerline.radii, structure.centerline.radii)
    np.testing.assert_array_equal(restored.imaging.scan.values, body.imaging.scan.values)
    assert restored.imaging.scan.metadata == body.imaging.scan.metadata


def test_vessel_repair_guidance_is_scoped():
    from patient_digital_twin.refinement import vessel_repair_guidance
    assert vessel_repair_guidance(['colon', 'liver']) == ''
    for name in ('aorta', 'inferior_vena_cava', 'superior_vena_cava',
                 'iliac_artery_left', 'iliac_vena_right', 'common_carotid_artery_left',
                 'subclavian_artery_right', 'pulmonary_vein', 'portal_vein_and_splenic_vein'):
        prompt = vessel_repair_guidance([name], True)
        assert name in prompt
        assert 'Do not merge arteries with veins' in prompt
        assert 'explicitly selected' in prompt
        assert ('Expected groups:' in prompt) == (name == 'aorta')


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


def test_vessel_branches_default_and_hra_coverage():
    import csv

    from patient_digital_twin.agent import AgentOptions
    from patient_digital_twin.refinement import vessel_repair_guidance
    assert AgentOptions(prompt='Inspect').include_major_branches is False
    assert 'Expected groups:' not in vessel_repair_guidance(['aorta'])
    assert 'Expected groups:' in vessel_repair_guidance(['aorta'], True)
    assert 'explicitly selected' not in vessel_repair_guidance(['inferior_vena_cava'])
    assert 'explicitly selected' in vessel_repair_guidance(['inferior_vena_cava'], True)
    path = Path(__file__).parents[1] / 'assets/anatomy/hra/blood-vasculature-female-v1.3'
    metadata = json.loads((path / 'metadata.json').read_text())
    nodes = {row['node_name'] for row in csv.DictReader((path / 'structures.csv').open())}
    for name in ('inferior_vena_cava', 'pulmonary_artery', 'celiac_trunk',
                 'superior_mesenteric_artery', 'inferior_mesenteric_artery'):
        for suffix in ('_main', '_major_branches'):
            selected = metadata['selections'][name + suffix]
            assert selected and set(selected) <= nodes


def test_named_prompt_is_saved_and_sent_to_sdk(body, tmp_path, monkeypatch):
    from patient_digital_twin.agent import AgentOptions
    from patient_digital_twin.agent import __main__ as editor
    captured = []
    monkeypatch.setattr(editor, '_run_blender', lambda args, *rest: captured.append(args.prompt))
    source = body.export_to_usd(tmp_path / 'input.usdc')
    workspace = tmp_path / 'agent'
    editor.edit_asset(source, tmp_path / 'out',
                      AgentOptions(task='refine', targets=('liver',)), workspace=workspace)
    saved = (workspace / 'task_prompt.txt').read_text()
    assert captured == [saved]
    assert 'Operation: refine.' in saved and 'Targets: liver.' in saved
    assert 'without completing missing regions' in saved
    (workspace / 'edited.blend').touch()
    editor.resume_asset(workspace, tmp_path / 'out', AgentOptions(prompt='Continue'))
    assert captured[-1] == saved
