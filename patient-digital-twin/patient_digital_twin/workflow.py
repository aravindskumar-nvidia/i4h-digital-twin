# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose optional refinement operations around the existing patient exporters."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .agent import AgentOptions
from .anatomy import MeshGeometry
from .body import HumanBody
from .references import (
    MissingTextureDefault,
    TextureReference,
    default_texture_reference,
)
from .surface import CleanupOptions, prepare_surface
from .texturing import TextureOptions, TextureRuntime, texture_mesh


class PatientWorkflow:
    """Own a run's options and artifacts. Cleanup is on; agent editing is opt-in.

    Pass cleanup=None to retain the input surface without deterministic cleanup.
    A configured texture runtime enables packaged organ defaults; explicit textures
    override them. Set default_textures=False to use only explicit textures.

    The input body is copied so failed or repeated runs do not partially modify
    caller-owned anatomy. Texture inference checkpoints may be shared across runs.
    """

    def __init__(
        self, *, cleanup: CleanupOptions | None = CleanupOptions(), agent: AgentOptions | None = None,
        textures: Mapping[str, TextureReference] | None = None,
        texture_options: TextureOptions | None = None, texture_runtime: TextureRuntime | None = None,
        workspace: str | Path | None = None, default_textures: bool = True,
        missing_texture: str = "retain",
    ):
        if agent and agent.task == 'texture' and (cleanup is not None or texture_runtime is not None or textures):
            raise ValueError('Texture-only agent edits require cleanup=None and no subsequent texture generation')
        self.cleanup = cleanup
        self.agent = agent
        self.textures = dict(textures or {})
        if not isinstance(default_textures, bool):
            raise ValueError("default_textures must be a boolean")
        self.default_textures = default_textures
        if missing_texture not in {"error", "retain"}:
            raise ValueError("missing_texture must be error or retain")
        self.missing_texture = missing_texture
        self.texture_options = texture_options or TextureOptions()
        self.texture_runtime = texture_runtime
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None
        if self.textures and self.texture_runtime is None:
            raise ValueError('Texturing requires a TextureRuntime')

    def run(self, body: HumanBody, output: str | Path, *, format: str = 'usd',
            vessel_names: Sequence[str] = (), centerline_spacing_mm: float = 1.5,
            ct_exterior: bool = False, topology_names: Sequence[str] = (),
            patient_id: str | None = None, world_from_patient_m=None,
            skin_opacity: float = 0.15, _resume_run: Path | None = None) -> Path:
        from .export import USD_SUFFIXES

        output = Path(output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f'Use a new output path: {output}')
        if format not in {'usd', 'bundle'} or (format == 'usd' and output.suffix not in USD_SUFFIXES):
            raise ValueError('Use format bundle or usd with a USD output extension')
        if not np.isfinite(centerline_spacing_mm) or centerline_spacing_mm <= 0:
            raise ValueError('Centerline spacing must be positive and finite')
        if unknown := set(topology_names) - body.anatomy.structures.keys():
            raise ValueError(f'Unknown topology structures: {sorted(unknown)}')
        textures = dict(self.textures)
        retained_materials = []
        if self.default_textures and self.texture_runtime is not None:
            for name, structure in body.anatomy.structures.items():
                if not structure.is_empty and name not in textures:
                    try:
                        textures[name] = default_texture_reference(name)
                    except MissingTextureDefault:
                        if self.missing_texture == 'error':
                            raise
                        retained_materials.append(name)
        selected = set(textures) | set(vessel_names)
        if self.agent:
            selected.update(self.agent.anatomy_references)
        if missing := [name for name in selected if name not in body.anatomy.structures
                       or body.anatomy.structures[name].is_empty]:
            raise ValueError(f'Requested structures have no enabled mesh: {missing}')
        root = self.workspace or output.parent / ('.' + output.name + '-work')
        if root == output or output in root.parents:
            raise ValueError('Workspace must be outside the output')
        root.mkdir(parents=True, exist_ok=True)
        run = _resume_run or Path(tempfile.mkdtemp(prefix='run-', dir=root))
        from ._workflow_checkpoint import save_body, write_state
        state_path = run / 'pipeline.json'
        state = json.loads(state_path.read_text()) if _resume_run else None
        body = copy.copy(body)
        body.anatomy = copy.deepcopy(body.anatomy)
        original_graphs = {name: structure.centerline for name, structure in body.anatomy.structures.items()
                           if structure.centerline is not None}
        if _resume_run and (run / 'original_graphs.npz').is_file():
            assert state is not None
            from .geometry import CenterlineGraph
            with np.load(run / 'original_graphs.npz', allow_pickle=False) as arrays:
                original_graphs = {name: CenterlineGraph(**{key: arrays[name + '/' + key].copy()
                    for key in ('points', 'edges', 'radii')}) for name in state['original_graph_names']}
        report: dict = {'cleanup': {}, 'textures': {}, 'agent': self.agent is not None,
                        'retained_materials_no_default': sorted(retained_materials)}
        if self.cleanup and not _resume_run:
            for name, structure in body.anatomy.structures.items():
                if structure.is_empty:
                    continue
                result = prepare_surface(structure.mesh, self.cleanup)
                structure.mesh, structure.centerline = result.mesh, None
                report['cleanup'][name] = result.measurements
        if state is None:
            save_body(body, run / 'body')
            if original_graphs:
                np.savez(run / 'original_graphs.npz', **{name + '/' + key: getattr(graph, key)
                    for name, graph in original_graphs.items() for key in ('points', 'edges', 'radii')})
            state = dict(schema_version=1, original_graph_names=list(original_graphs), output=str(output), stage='agent' if self.agent else 'textures',
                agent=asdict(self.agent) if self.agent else None,
                textures={name: {'image': str(ref.image.resolve()), 'metadata': ref.metadata} for name, ref in textures.items()},
                texture_options=asdict(self.texture_options),
                texture_runtime=asdict(self.texture_runtime) if self.texture_runtime else None,
                report=report, export=dict(format=format, vessel_names=list(vessel_names),
                    centerline_spacing_mm=centerline_spacing_mm, ct_exterior=ct_exterior,
                    topology_names=list(topology_names), patient_id=patient_id,
                    world_from_patient_m=world_from_patient_m, skin_opacity=skin_opacity))
            write_state(state_path, state)
        else:
            report = state['report']
            state['output'] = str(output)
            write_state(state_path, state)
        print(f'Pipeline checkpoint: {run}', flush=True)
        edited = Path(state['edited']) if state.get('edited') else None
        if self.agent and edited is None:
            from .agent.__main__ import edit_asset

            _restore_topology(body, original_graphs, (), centerline_spacing_mm)
            if _resume_run:
                from .agent.__main__ import resume_asset
                published = run / 'edited' / 'input.usdc'
                edited = published if published.is_file() else resume_asset(run / 'agent', run / 'edited', self.agent)
            else:
                source = body.export_to_usd(run / 'input.usdc')
                edited = edit_asset(source, run / 'edited', self.agent, workspace=run / 'agent')
            state.update(edited=str(edited), stage='textures')
            write_state(state_path, state)
        if edited:
            _read_edited_geometry(body, edited)
        for index, (name, reference) in enumerate(textures.items(), 1):
            print(f'Texturing {index}/{len(textures)}: {name}', flush=True)
            assert self.texture_runtime is not None
            structure = body.anatomy.structures[name]
            textured = texture_mesh(structure.mesh, reference, self.texture_options,
                                    runtime=self.texture_runtime, workspace=root / 'textures')
            structure.mesh = textured.mesh
            report['textures'][name] = textured.metadata
            write_state(state_path, state)
        _restore_topology(body, original_graphs, set(vessel_names) | set(topology_names), centerline_spacing_mm)
        state['stage'] = 'export'
        write_state(state_path, state)
        package = Path(tempfile.mkdtemp(prefix='package-', dir=run))
        package.rmdir()
        if format == 'bundle':
            manifest = body.export_patient_twin(
                package, vessel_names=vessel_names, ct_exterior=ct_exterior,
                patient_id=patient_id, world_from_patient_m=world_from_patient_m, skin_opacity=skin_opacity,
            )
            companion = body.export_to_usd(package / 'human_body.usdc')
            import yaml
            anatomy = package / yaml.safe_load(manifest.read_text())['artifacts']['anatomy_usd']
            result_name = manifest.name
        else:
            package.mkdir()
            anatomy = body.export_to_usd(package / output.name)
            result_name = output.name
        if edited:
            _restore_appearance(anatomy, edited, set(textures))
            if format == 'bundle':
                _restore_appearance(companion, edited, set(textures))
            provenance = edited.parent / 'refinement'
            if provenance.exists():
                shutil.copytree(provenance, package / 'refinement')
        (package / 'workflow.json').write_text(json.dumps(report, indent=2) + '\n')
        published_result = _publish_package(package, output, result_name, format)
        state.update(stage='complete', result=str(published_result))
        write_state(state_path, state)
        return published_result


    @classmethod
    def resume(cls, workspace: str | Path, *, output: str | Path | None = None,
               edited_asset: str | Path | None = None, token_budget: int | None = None,
               blender_mode: str | None = None, agent_overrides: Mapping | None = None) -> Path:
        """Continue the saved pipeline, including textures and final bundle export."""
        from ._workflow_checkpoint import load_body, write_state
        run = Path(workspace).expanduser().resolve()
        state = json.loads((run / 'pipeline.json').read_text())
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported pipeline checkpoint schema')
        if state['stage'] == 'complete':
            result = Path(state['result'])
            if not result.is_file():
                raise FileNotFoundError(f'Completed pipeline output is missing: {result}')
            return result
        options = state['agent']
        if options:
            options.update(agent_overrides or {})
            options['anatomy_references'] = {name: Path(path) for name, path in options['anatomy_references'].items()}
            if token_budget is not None:
                options['token_budget'] = token_budget
            if blender_mode is not None:
                options['blender_mode'] = blender_mode
        from .agent.prompts import build_prompt
        agent = AgentOptions(**dict(options, anatomy_references={}, task=None, targets=(),
            include_major_branches=False, prompt=build_prompt(options.get('task'),
                options.get('targets', ()), options['prompt'], references=options['anatomy_references'],
                major_branching=options['include_major_branches']))) if options else None
        if agent is not None:
            # The editor workspace already contains the resolved reference assets.
            agent.anatomy_references.update(options['anatomy_references'])
        if edited_asset is not None:
            edited = Path(edited_asset).expanduser().resolve()
            if not edited.is_file():
                raise FileNotFoundError(edited)
            # Accept only an already published editor result with its validation report.
            if options and options['anatomy_references'] and not (edited.parent / 'refinement/report.json').is_file():
                raise ValueError('Edited asset requires its published refinement report')
            state.update(edited=str(edited), stage='textures')
            write_state(run / 'pipeline.json', state)
        runtime = state['texture_runtime']
        workflow = cls(cleanup=None, agent=agent, default_textures=False,
            textures={name: TextureReference(Path(record['image']), record['metadata'])
                      for name, record in state['textures'].items()},
            texture_options=TextureOptions(**state['texture_options']),
            texture_runtime=TextureRuntime(checkout=Path(runtime['checkout']), timeout_s=runtime['timeout_s']) if runtime else None,
            workspace=run.parent)
        return workflow.run(load_body(run / 'body'), output or state['output'],
                            **state['export'], _resume_run=run)


def _restore_topology(body, original_graphs, requested, spacing_mm):
    """Restore invalidated graphs and compute explicitly requested new ones."""
    from .geometry import centerline_spacing

    for name in sorted(set(original_graphs) | set(requested)):
        if body.anatomy.structures[name].centerline is not None:
            continue
        spacing = spacing_mm * .001
        if name in original_graphs:
            graph = original_graphs[name]
            spacing = centerline_spacing(graph.points, graph.edges)
        body.extract_topology(names=[name], spacing_m=spacing)


def _read_edited_geometry(body: HumanBody, edited: Path) -> None:
    """Read local geometry from the validated handoff, retaining patient transforms."""
    from pxr import Usd, UsdGeom

    from .geometry import CenterlineGraph, validate_triangles

    stage = Usd.Stage.Open(str(edited))
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = prim.GetCustomDataByKey('anatomy:name')
        if name not in body.anatomy.structures:
            continue
        structure = body.anatomy.structures[name]
        mesh = UsdGeom.Mesh(prim)
        _normalize_winding(mesh)
        points, faces = validate_triangles(
            np.asarray(mesh.GetPointsAttr().Get(), dtype=float) * UsdGeom.GetStageMetersPerUnit(stage),
            np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
        )
        previous = np.asarray(structure.mesh.vertices)
        changed = (points.shape != previous.shape
                   or not np.allclose(points, previous, rtol=0, atol=1e-5)
                   or not np.array_equal(faces, np.asarray(structure.mesh.faces)))
        if changed:
            structure.mesh, structure.centerline = MeshGeometry(points, faces), None
            if prim.HasAttribute('centerline:points'):
                units = UsdGeom.GetStageMetersPerUnit(stage)
                structure.centerline = CenterlineGraph(
                    points=np.asarray(prim.GetAttribute('centerline:points').Get(), dtype=float) * units,
                    edges=np.asarray(prim.GetAttribute('centerline:edges').Get(), dtype=np.int64),
                    radii=np.asarray(prim.GetAttribute('centerline:radii').Get(), dtype=float) * units,
                )
    stage.GetRootLayer().Save()


def _normalize_winding(mesh) -> None:
    """Normalize reflected edits and their per-corner data before mesh processing."""
    from pxr import UsdGeom

    if mesh.GetOrientationAttr().Get() != UsdGeom.Tokens.leftHanded:
        return
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    mesh.GetFaceVertexIndicesAttr().Set(faces[:, ::-1].ravel().tolist())
    order = np.arange(faces.size).reshape(-1, 3)[:, ::-1].ravel()
    for variable in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if variable.GetInterpolation() == UsdGeom.Tokens.faceVarying:
            values = variable.ComputeFlattened()
            variable.Set([values[int(index)] for index in order])
            variable.BlockIndices()
    if mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying:
        normals = mesh.GetNormalsAttr().Get()
        mesh.GetNormalsAttr().Set([normals[int(index)] for index in order])
    mesh.GetOrientationAttr().Set(UsdGeom.Tokens.rightHanded)


def _restore_appearance(target: Path, edited: Path, textured: set[str]) -> None:
    """Carry agent materials and UVs onto final geometry unless explicitly retextured."""
    from pxr import Sdf, Usd, UsdGeom

    from .agent.assets import _copy_textures, _merge_materials

    stage, source = Usd.Stage.Open(str(target)), Usd.Stage.Open(str(edited))
    meshes = [prim for prim in source.Traverse() if prim.IsA(UsdGeom.Mesh)
              and prim.GetCustomDataByKey('anatomy:name') not in textured
              and stage.GetPrimAtPath(prim.GetPath())]
    for mesh in meshes:
        destination = stage.GetPrimAtPath(mesh.GetPath())
        for prop in mesh.GetAuthoredProperties():
            if prop.GetName().startswith(('primvars:', 'material:binding')) or prop.GetName() == 'normals':
                Sdf.CopySpec(source.GetRootLayer(), prop.GetPath(), stage.GetRootLayer(),
                             destination.GetPath().AppendProperty(prop.GetName()))
        for child in mesh.GetChildren():
            if child.IsA(UsdGeom.Subset):
                Sdf.CopySpec(source.GetRootLayer(), child.GetPath(), stage.GetRootLayer(), child.GetPath())
    _merge_materials(stage, source, meshes, [str(mesh.GetPath()) for mesh in meshes], 'refinement')
    _copy_textures(stage, edited.parent, target.parent)
    stage.GetRootLayer().Save()


def _publish_package(package: Path, output: Path, result_name: str, format: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name('.' + output.name + '.publishing')
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if output.exists():
            raise FileExistsError(f'Use a new output path: {output}')
        return _write_package(package, output, result_name, format)
    finally:
        os.close(descriptor)
        lock.unlink()


def _write_package(package: Path, output: Path, result_name: str, format: str) -> Path:
    from pxr import Sdf, Usd

    from .export import _publish

    if format == 'bundle':
        _publish(output, '.refined-patient-', lambda folder: shutil.copytree(package, folder))
        return output / result_name
    # One sidecar per output keeps different USD publications independent.
    sidecar = output.with_name(output.name + '.assets')
    stage = Usd.Stage.Open(str(package / result_name))
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset and attr.Get():
                asset = attr.Get()
                attr.Set(Sdf.AssetPath(sidecar.name + '/' + asset.path))
    stage.GetRootLayer().Save()
    _publish(sidecar, '.patient-assets-', lambda folder: shutil.copytree(package, folder,
             ignore=shutil.ignore_patterns(result_name)))
    try:
        # A hard link publishes the complete file without replacing a concurrent output.
        with tempfile.TemporaryDirectory(dir=output.parent, prefix='.patient-usd-') as temporary:
            staged = Path(temporary) / result_name
            shutil.copy2(package / result_name, staged)
            os.link(staged, output)
    except BaseException:
        shutil.rmtree(sidecar)
        raise
    return output
