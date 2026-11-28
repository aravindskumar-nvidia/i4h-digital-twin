# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose cleanup, optional agent editing and texturing around the patient exporters."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from ._workflow_checkpoint import Checkpoint
from .agent import AgentOptions
from .anatomy import Kind, canonical_name
from .body import HumanBody
from .export import SKIN_OPACITY, USD_SUFFIXES, publish_directory, publish_usd
from .geometry import CENTERLINE_SPACING_M, CenterlineGraph, centerline_spacing
from .surface import CleanupOptions, prepare_surface
from .texturing import (
    MissingTextureDefault,
    TextureOptions,
    TextureReference,
    TextureRuntime,
    check_runtime,
    default_texture_reference,
    texture_mesh,
)

# Catalog labels that are several separate bodies by definition, like grouped kinds.
_MULTIBODY_LABELS = frozenset({'pulmonary_vein'})


class PatientWorkflow:
    """Own a run's options and artifacts. Cleanup is on; agent editing is opt-in.

    Pass cleanup=None to retain the input surface without deterministic cleanup.
    Cleanup's distance filter is not applied to labels that are several bodies by
    definition (grouped and finding kinds, pulmonary_vein), and structures that
    already carry a material are left as supplied.
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
        missing_texture: str = 'retain',
    ):
        if agent and agent.task == 'texture' and (cleanup is not None or texture_runtime is not None or textures):
            raise ValueError('Texture-only agent edits require cleanup=None and no subsequent texture generation')
        if missing_texture not in {'error', 'retain'}:
            raise ValueError('missing_texture must be error or retain')
        if textures and texture_runtime is None:
            raise ValueError('Texturing requires a TextureRuntime')
        self.cleanup = cleanup
        self.agent = agent
        self.textures = dict(textures or {})
        self.default_textures = default_textures
        self.missing_texture = missing_texture
        self.texture_options = texture_options or TextureOptions()
        self.texture_runtime = texture_runtime
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None

    def run(self, body: HumanBody, output: str | Path, *, format: str = 'usd',
            vessel_names: Sequence[str] = (), centerline_spacing_mm: float = CENTERLINE_SPACING_M * 1000,
            ct_exterior: bool = False, topology_names: Sequence[str] = (),
            patient_id: str | None = None, world_from_patient_m=None,
            skin_opacity: float = SKIN_OPACITY) -> Path:
        """Clean, optionally edit and texture a copy of ``body``, then publish it to a new ``output``.

        Each run is checkpointed under the workspace; ``PatientWorkflow.resume`` continues it.
        """
        output = _new_output(output, format)
        if not np.isfinite(centerline_spacing_mm) or centerline_spacing_mm <= 0:
            raise ValueError('Centerline spacing must be positive and finite')
        if unknown := set(topology_names) - body.anatomy.structures.keys():
            raise ValueError(f'Unknown topology structures: {sorted(unknown)}')
        textures, retained = self._resolve_textures(body)
        self._check_selection(body, set(textures) | set(vessel_names))
        root = self._workspace_root(output)
        body = copy.copy(body)
        body.anatomy = copy.deepcopy(body.anatomy)
        original_graphs = {name: structure.centerline for name, structure in body.anatomy.structures.items()
                           if structure.centerline is not None}
        report = {'cleanup': self._clean(body) if self.cleanup else {}, 'textures': {},
                  'agent': self.agent is not None, 'retained_materials_no_default': sorted(retained)}
        checkpoint = Checkpoint.create(
            Path(tempfile.mkdtemp(prefix='run-', dir=root)), body, original_graphs,
            output=str(output), stage='agent' if self.agent else 'textures',
            agent=asdict(self.agent) if self.agent else None,
            textures={name: {'image': str(ref.image.resolve()), 'metadata': ref.metadata}
                      for name, ref in textures.items()},
            texture_options=asdict(self.texture_options),
            texture_runtime=asdict(self.texture_runtime) if self.texture_runtime else None,
            report=report, export=dict(
                format=format, vessel_names=list(vessel_names), centerline_spacing_mm=centerline_spacing_mm,
                ct_exterior=ct_exterior, topology_names=list(topology_names), patient_id=patient_id,
                world_from_patient_m=world_from_patient_m, skin_opacity=skin_opacity))
        return self._complete(checkpoint, body, original_graphs, textures, root, resuming=False)

    @classmethod
    def resume(cls, workspace: str | Path, *, output: str | Path | None = None,
               edited_asset: str | Path | None = None, agent_overrides: Mapping | None = None) -> Path:
        """Continue a saved run from its last completed stage through texturing and publication.

        ``agent_overrides`` replaces saved ``AgentOptions`` fields; its ``prompt`` is
        appended to the saved task as recovery instructions.
        """
        checkpoint = Checkpoint.load(Path(workspace).expanduser().resolve())
        state = checkpoint.state
        if state['stage'] == 'complete':
            result = Path(state['result'])
            if not result.is_file():
                raise FileNotFoundError(f'Completed pipeline output is missing: {result}')
            return result
        overrides = dict(agent_overrides or {})
        continuation = overrides.pop('prompt', None)
        agent = None
        if state['agent']:
            options = dict(state['agent'], targets=tuple(state['agent']['targets']))
            if overrides.get('backend', options['backend']) != options['backend']:
                # Saved limits belong to the previous backend and are invalid for the new one.
                options.update(max_turns=None, budget_usd=None, token_budget=None)
            agent = AgentOptions(**dict(options, **overrides))
        if edited_asset is not None:
            edited = Path(edited_asset).expanduser().resolve()
            if not edited.is_file():
                raise FileNotFoundError(edited)
            # Accept only an already published editor result with its validation report.
            if agent and agent.anatomy_references and not (edited.parent / 'refinement/report.json').is_file():
                raise ValueError('Edited asset requires its published refinement report')
            checkpoint.update(edited=str(edited), stage='textures')
        if output is not None:
            checkpoint.update(output=str(_new_output(output, state['export']['format'])))
        runtime = state['texture_runtime']
        workflow = cls(
            cleanup=None, agent=agent, default_textures=False, workspace=checkpoint.folder.parent,
            textures={name: TextureReference(Path(record['image']), record['metadata'])
                      for name, record in state['textures'].items()},
            texture_options=TextureOptions(**state['texture_options']),
            texture_runtime=TextureRuntime(checkout=Path(runtime['checkout']), timeout_s=runtime['timeout_s'])
            if runtime else None)
        body, original_graphs = checkpoint.load_body()
        return workflow._complete(checkpoint, body, original_graphs, workflow.textures, checkpoint.folder.parent,
                                  resuming=True, continuation=continuation)

    def _complete(self, checkpoint: Checkpoint, body: HumanBody, original_graphs: dict[str, CenterlineGraph],
                  textures: Mapping[str, TextureReference], root: Path, *, resuming: bool,
                  continuation: str | None = None) -> Path:
        """Run every stage after cleanup: agent edit, texturing, topology and publication."""
        # Agent assets need OpenUSD, which the package import must not require.
        from .agent import assets

        print(f'Pipeline checkpoint: {checkpoint.folder}', flush=True)
        state, export = checkpoint.state, checkpoint.state['export']
        output = _new_output(state['output'], export['format'])
        spacing_m = export['centerline_spacing_mm'] * .001
        edited = Path(state['edited']) if state.get('edited') else None
        if self.agent and edited is None:
            edited = self._edit(checkpoint, body, original_graphs, spacing_m, resuming, continuation)
        edited_stage = assets.read_edited_geometry(body, edited) if edited else None
        if textures:
            assert self.texture_runtime is not None
            identity = check_runtime(self.texture_runtime)
            for index, (name, reference) in enumerate(textures.items(), 1):
                print(f'Texturing {index}/{len(textures)}: {name}', flush=True)
                structure = body.anatomy.structures[name]
                textured = texture_mesh(structure.mesh, reference, self.texture_options, runtime=self.texture_runtime,
                                        workspace=root / 'textures', identity=identity)
                structure.mesh = textured.mesh
                state['report']['textures'][name] = textured.metadata
        _restore_topology(body, original_graphs, set(export['vessel_names']) | set(export['topology_names']),
                          spacing_m)
        checkpoint.update(stage='export')
        package = checkpoint.folder / 'package'
        # A previous attempt's package is regenerated from the checkpoint.
        shutil.rmtree(package, ignore_errors=True)
        if export['format'] == 'bundle':
            manifest = body.export_patient_twin(
                package, vessel_names=export['vessel_names'], ct_exterior=export['ct_exterior'],
                patient_id=export['patient_id'], world_from_patient_m=export['world_from_patient_m'],
                skin_opacity=export['skin_opacity'])
            usd_files = [package / yaml.safe_load(manifest.read_text())['artifacts']['anatomy_usd'],
                         body.export_to_usd(package / 'human_body.usdc')]
        else:
            package.mkdir()
            usd_files = [body.export_to_usd(package / output.name)]
        for usd in usd_files:
            if edited_stage is not None:
                assets.restore_appearance(usd, edited_stage, set(textures))
            assets.annotate_cleanup(usd, state['report']['cleanup'])
        if edited is not None and (edited.parent / 'refinement').exists():
            shutil.copytree(edited.parent / 'refinement', package / 'refinement')
        (package / 'workflow.json').write_text(json.dumps(state['report'], indent=2) + '\n')
        if export['format'] == 'bundle':
            result = publish_directory(output, '.refined-patient-', lambda folder: shutil.copytree(package, folder))
            result /= manifest.name
        else:
            result = publish_usd(usd_files[0], output)
        checkpoint.update(stage='complete', result=str(result))
        return result

    def _resolve_textures(self, body: HumanBody) -> tuple[dict[str, TextureReference], list[str]]:
        """Explicit textures plus packaged defaults; also the structures without a default."""
        textures, retained = dict(self.textures), []
        if self.default_textures and self.texture_runtime is not None:
            for name, structure in body.anatomy.structures.items():
                if structure.is_empty or name in textures:
                    continue
                try:
                    textures[name] = default_texture_reference(name)
                except MissingTextureDefault:
                    if self.missing_texture == 'error':
                        raise
                    retained.append(name)
        return textures, retained

    def _check_selection(self, body: HumanBody, selected: set[str]) -> None:
        # A mistyped name must fail here, before cleanup and checkpointing.
        if self.agent:
            selected = selected | set(self.agent.anatomy_references) | set(self.agent.targets)
        if missing := [name for name in selected if name not in body.anatomy.structures
                       or body.anatomy.structures[name].is_empty]:
            raise ValueError(f'Requested structures have no enabled mesh: {missing}')

    def _workspace_root(self, output: Path) -> Path:
        root = self.workspace or output.parent / ('.' + output.name + '-work')
        if root == output or output in root.parents:
            raise ValueError('Workspace must be outside the output')
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _clean(self, body: HumanBody) -> dict:
        """Clean every enabled surface in place and return the per-structure measurements."""
        assert self.cleanup is not None
        report = {}
        for name, structure in body.anatomy.structures.items():
            if structure.is_empty:
                continue
            if structure.mesh.material is not None:
                # Cleanup would invalidate existing UVs; keep textured input as supplied.
                print(f'Cleanup skipped for textured structure: {name}', flush=True)
                report[name] = {'skipped': 'already_textured'}
                continue
            options = self.cleanup
            multibody = structure.kind in {Kind.GROUP, Kind.FINDING} or canonical_name(name) in _MULTIBODY_LABELS
            if multibody:
                # Separate bodies are expected here; distance says nothing about validity.
                options = replace(options, max_component_distance_mm=0)
            try:
                result = prepare_surface(structure.mesh, options)
            except ValueError as error:
                raise ValueError(f'Cleanup failed for {name}: {error}') from error
            structure.mesh, structure.centerline = result.mesh, None
            report[name] = result.measurements
            if multibody and self.cleanup.max_component_distance_mm:
                result.measurements['distance_filter'] = 'disabled_for_multibody_label'
            distant = [record for record in result.measurements['component_filter']['components']
                       if record['decision'] == 'remove_distant']
            if distant:
                print(f'Warning: cleanup removed {len(distant)} distant component(s) of {name} '
                      f'({sum(record["area_mm2"] for record in distant):.0f} mm2, more than '
                      f'{options.max_component_distance_mm:g} mm from the retained surface). '
                      'Set max_component_distance_mm=0 to keep them.', flush=True)
        return report

    def _edit(self, checkpoint: Checkpoint, body: HumanBody, original_graphs: dict[str, CenterlineGraph],
              spacing_m: float, resuming: bool, continuation: str | None) -> Path:
        """Run or resume the agent session and record its published USD."""
        # The agent session needs the editing extra, which other runs do not install.
        from .agent import assets
        from .agent.session import edit_asset, resume_asset

        assert self.agent is not None
        run = checkpoint.folder
        _restore_topology(body, original_graphs, (), spacing_m)
        published = run / 'edited' / 'input.usdc'
        if resuming and published.is_file():
            # The previous attempt published its edit but stopped before recording it.
            edited = published
        elif resuming:
            edited = resume_asset(run / 'agent', run / 'edited', self.agent, continuation=continuation)
        else:
            source = body.export_to_usd(run / 'input.usdc')
            assets.annotate_cleanup(source, checkpoint.state['report']['cleanup'])
            edited = edit_asset(source, run / 'edited', self.agent, workspace=run / 'agent')
        checkpoint.update(edited=str(edited), stage='textures')
        return edited


def _new_output(output: str | Path, format: str) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output path: {output}')
    if format not in {'usd', 'bundle'} or (format == 'usd' and output.suffix not in USD_SUFFIXES):
        raise ValueError('Use format bundle or usd with a USD output extension')
    return output


def _restore_topology(body: HumanBody, original_graphs: dict[str, CenterlineGraph], requested,
                      spacing_m: float) -> None:
    """Restore centerlines invalidated by cleanup or edits, and compute requested new ones."""
    for name in sorted(set(original_graphs) | set(requested)):
        structure = body.anatomy.structures[name]
        if structure.centerline is not None:
            continue
        if name not in original_graphs:
            body.extract_topology(names=[name], spacing_m=spacing_m)
            continue
        try:
            regenerated = body.extract_topology(names=[name], spacing_m=_grid_spacing(original_graphs[name]) or spacing_m)
        except ValueError as error:
            regenerated, reason = {}, f'regeneration failed ({error})'
        else:
            reason = 'it cannot be regenerated for this structure kind'
        if not regenerated:
            # Only vessels and airways can be regenerated; never drop a supplied graph.
            print(f'Warning: retained the supplied centerline of {name}: {reason}. '
                  'It may not match the processed surface.', flush=True)
            structure.centerline = original_graphs[name]


def _grid_spacing(graph: CenterlineGraph) -> float | None:
    """Spacing of a pipeline grid skeleton, or None for single-node or resampled graphs."""
    try:
        spacing = centerline_spacing(graph.points, graph.edges)
        points, edges = np.asarray(graph.points, dtype=float), np.asarray(graph.edges)
        steps = (points[edges[:, 1]] - points[edges[:, 0]]) / spacing
    except (ValueError, IndexError):
        return None
    on_grid = np.allclose(steps, np.round(steps), rtol=0, atol=.05) and np.abs(steps).max() < 1.05
    return spacing if on_grid else None
