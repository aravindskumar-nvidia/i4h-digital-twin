# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic mesh cleanup, independent of patient workflows and agent SDKs.

VTK and trimesh (the ``pipeline`` extra) are imported when cleanup runs, so importing
the package does not require them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anatomy import MeshGeometry
from .geometry import polydata_arrays, triangle_polydata, validate_triangles

# Pass bands tried, in order, until smoothing keeps volume (or open-surface area) within 2%.
_FALLBACK_PASS_BANDS = (.03, .1, .3, .8, 1.5, 2.)


@dataclass(frozen=True)
class CleanupOptions:
    """Cleanup settings; zero disables hole filling, smoothing, and each fragment threshold."""
    smoothing_iterations: int = 60
    hole_radius_mm: float = 2.0
    min_component_area_mm2: float = 25.0
    smoothing_pass_band: float = 0.01
    max_component_distance_mm: float = 50.0

    def __post_init__(self):
        values = [self.smoothing_iterations, self.hole_radius_mm, self.min_component_area_mm2,
                  self.smoothing_pass_band, self.max_component_distance_mm]
        if (not np.isfinite(values).all() or min(values) < 0
                or int(self.smoothing_iterations) != self.smoothing_iterations
                or not 0 < self.smoothing_pass_band <= 2):
            raise ValueError("Iterations/thresholds must be nonnegative and finite; "
                             "smoothing pass band must be in (0, 2]")


@dataclass(frozen=True)
class CleanupResult:
    mesh: MeshGeometry
    measurements: dict


def _filter_components(source, options):
    """Keep the main surface and nearby significant fragments; record every decision."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    parts = sorted(source.split(only_watertight=False, repair=False),
                   key=lambda part: (-part.area, tuple(part.bounds.ravel())))
    if not parts:
        raise ValueError('Structure has no nondegenerate surface')
    records = [dict(id=i, area_mm2=float(part.area * 1e6), faces=len(part.faces),
                    bounds_local_mm=(part.bounds * 1000).tolist(),
                    decision='remove_small', nearest_component=None, gap_mm=None)
               for i, part in enumerate(parts)]
    # An entire small labeled structure must not disappear because of a fragment threshold.
    records[0].update(decision='keep_primary', gap_mm=0.0)
    retained, pending = [0], [i for i in range(1, len(parts))
                              if records[i]['area_mm2'] >= options.min_component_area_mm2]
    for i in pending:
        records[i]['decision'] = 'remove_distant'
    if not options.max_component_distance_mm:
        retained.extend(pending)
        for i in pending:
            records[i]['decision'] = 'retain_distance_filter_disabled'
    else:
        # Expand from the main surface through nearby fragments. Tiny removed islands
        # cannot act as stepping stones to retain otherwise distant false positives.
        def surface_distance(part):
            distance = vtk.vtkImplicitPolyDataDistance()
            distance.SetInput(triangle_polydata(part.vertices, part.faces))
            return distance

        def gap_mm(distance, part):
            values = vtk.vtkDoubleArray()
            values.SetNumberOfTuples(len(part.vertices))
            distance.FunctionValue(numpy_to_vtk(np.asarray(part.vertices), deep=True), values)
            return float(np.min(np.abs(vtk_to_numpy(values))) * 1000)

        limit = options.max_component_distance_mm
        for parent in retained:
            if not pending:
                break
            distance = surface_distance(parts[parent])
            for i in pending[:]:
                gap = gap_mm(distance, parts[i])
                # Sampling only the fragment's vertices overestimates the gap by up to one
                # fragment edge. Near the limit, also measure from the retained surface so a
                # coarse fragment is not removed on that error.
                if limit < gap <= limit + float(parts[i].edges_unique_length.max() * 1000):
                    gap = min(gap, gap_mm(surface_distance(parts[i]), parts[parent]))
                previous = records[i]['gap_mm']
                if previous is None or gap < previous:
                    records[i].update(gap_mm=gap, nearest_component=parent)
                if gap <= limit:
                    records[i]['decision'] = 'retain_nearby'
                    retained.append(i)
                    pending.remove(i)
    return [parts[i] for i in retained], dict(
        coordinate_frame='original_structure_local_mm',
        distance_method='minimum_vertex_to_surface_checked_in_both_directions_near_the_limit',
        components=records, retained_component_ids=retained,
    )


def _quality(mesh) -> dict:
    counts = np.bincount(mesh.edges_unique_inverse)
    return {
        "components": len(mesh.split(only_watertight=False, repair=False)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "watertight": bool(mesh.is_watertight),
        "euler_characteristic": int(mesh.euler_number),
        "p95_adjacent_normal_angle_deg": (float(np.percentile(np.degrees(mesh.face_adjacency_angles), 95))
                                           if len(mesh.face_adjacency) else None),
        "area_mm2": float(mesh.area * 1e6),
        "volume_mm3": float(abs(mesh.volume) * 1e9) if mesh.is_watertight else None,
    }


def _fill_holes(surface, radius_mm: float):
    import vtk

    repair = vtk.vtkFillHolesFilter()
    repair.SetInputData(surface)
    repair.SetHoleSize(radius_mm * 0.001)
    repair.Update()
    return repair.GetOutput()


def _smooth(surface, options: CleanupOptions):
    """Windowed-sinc smoothing at the requested pass band, backed off until the surface survives.

    A densely meshed scan supports strong filtering; a coarse CAD cylinder with only two
    vertex rings can collapse at that setting. Raising the pass band keeps such inputs
    instead of silently destroying them. Returns the surface and the pass band used.
    """
    import trimesh
    import vtk

    vertices, faces = polydata_arrays(surface)
    unsmoothed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    baseline = abs(unsmoothed.volume) if unsmoothed.is_watertight else unsmoothed.area
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(surface)
    smooth.SetNumberOfIterations(options.smoothing_iterations)
    smooth.NormalizeCoordinatesOn()
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOff()
    requested = options.smoothing_pass_band
    for pass_band in sorted({requested, *(max(requested, band) for band in _FALLBACK_PASS_BANDS)}):
        smooth.SetPassBand(pass_band)
        smooth.Update()
        candidate = trimesh.Trimesh(vertices=polydata_arrays(smooth.GetOutput())[0], faces=faces, process=False)
        measured = abs(candidate.volume) if unsmoothed.is_watertight else candidate.area
        if (baseline > 0 and np.isfinite(candidate.vertices).all() and abs(measured / baseline - 1) <= .02
                and np.all(candidate.area_faces > 0)):
            return smooth.GetOutput(), pass_band
    raise ValueError('Smoothing could not preserve the input surface; disable or reduce smoothing')


def _repair(component, options: CleanupOptions):
    """Fill small holes in and smooth one retained component; return it and its pass band."""
    import trimesh

    pass_band = options.smoothing_pass_band
    if options.hole_radius_mm or options.smoothing_iterations:
        surface = triangle_polydata(component.vertices, component.faces)
        if options.hole_radius_mm:
            surface = _fill_holes(surface, options.hole_radius_mm)
        if options.smoothing_iterations:
            surface, pass_band = _smooth(surface, options)
        vertices, faces = polydata_arrays(surface)
        component = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
    component.fix_normals(multibody=True)
    return component, pass_band


def prepare_surface(mesh: MeshGeometry, options: CleanupOptions | None = None) -> CleanupResult:
    """Return a repaired, smoothed local-meter mesh without modifying the input.

    Keeps the largest component, removes tiny and distant secondary components, and
    retains nearby fragments separately for agent review. Distance assumes one anatomical
    body: disable it for structures that are several separate bodies by definition.
    VTK's hole radius bounds a boundary loop; it is not a gap-bridging distance. No spline
    or connection between components is inferred. Windowed-sinc smoothing suppresses
    voxel stepping with limited volume drift; lower pass bands remove more fine-scale
    variation. Returns before/after topology and surface measurements. Run before
    applying textures.
    """
    import trimesh

    options = options or CleanupOptions()
    if mesh.material is not None:
        raise ValueError("Prepare geometry before texturing")
    if mesh.vertices is None or mesh.faces is None:
        raise ValueError("Structure has no mesh")
    vertices, faces = validate_triangles(mesh.vertices, mesh.faces)
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
    components, component_filter = _filter_components(source, options)
    prepared, pass_bands = [], []
    for component_id, component in zip(component_filter['retained_component_ids'], components):
        component, pass_band = _repair(component, options)
        prepared.append(component)
        pass_bands.append(pass_band)
        component_filter['components'][component_id]['smoothing_pass_band'] = pass_band
    surface = trimesh.util.concatenate(prepared)
    assert isinstance(surface, trimesh.Trimesh)
    report = {"before": _quality(source), "after": _quality(surface),
              "smoothing_iterations": options.smoothing_iterations, "hole_radius_mm": options.hole_radius_mm,
              "min_component_area_mm2": options.min_component_area_mm2,
              "max_component_distance_mm": options.max_component_distance_mm,
              "component_filter": component_filter,
              "smoothing_algorithm": "windowed_sinc", "smoothing_pass_band": max(pass_bands),
              "requested_smoothing_pass_band": options.smoothing_pass_band}
    geometry = MeshGeometry(np.asarray(surface.vertices), np.asarray(surface.faces),
                            normals=np.asarray(surface.vertex_normals))
    return CleanupResult(geometry, report)
