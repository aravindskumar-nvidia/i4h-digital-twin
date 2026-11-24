# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic mesh cleanup, independent of patient workflows and agent SDKs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anatomy import MeshGeometry
from .geometry import validate_triangles


@dataclass(frozen=True)
class CleanupOptions:
    smoothing_iterations: int = 60
    hole_radius_mm: float = 2.0
    min_component_area_mm2: float = 0.0
    smoothing_pass_band: float = 0.01

    def __post_init__(self):
        validate_cleanup(smoothing_iterations=self.smoothing_iterations,
                         hole_radius_mm=self.hole_radius_mm,
                         min_component_area_mm2=self.min_component_area_mm2,
                         smoothing_pass_band=self.smoothing_pass_band)


@dataclass(frozen=True)
class CleanupResult:
    mesh: MeshGeometry
    measurements: dict


def validate_cleanup(*, smoothing_iterations=60, hole_radius_mm=2., min_component_area_mm2=0.,
                     smoothing_pass_band=.01):
    values = [smoothing_iterations, hole_radius_mm, min_component_area_mm2, smoothing_pass_band]
    if (not np.isfinite(values).all() or min(values) < 0
            or int(smoothing_iterations) != smoothing_iterations or not 0 < smoothing_pass_band <= 2):
        raise ValueError("Iterations/thresholds must be nonnegative and finite; smoothing pass band must be in (0, 2]")


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


def prepare_surface(mesh: MeshGeometry, options: CleanupOptions | None = None) -> CleanupResult:
    """Return a repaired, smoothed local-meter mesh without modifying the input.

    Retains all components unless an explicit area threshold is supplied. VTK's
    hole radius bounds the circumsphere of a boundary loop; it is not a gap-
    bridging distance. No spline or connection between components is inferred.
    Windowed-sinc smoothing suppresses voxel stepping with limited volume drift.
    Lower pass bands remove more fine-scale variation. Returns before/after
    topology and surface measurements. Run before applying textures.
    """
    import trimesh
    import vtk
    from vtk.util.numpy_support import (
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )

    options = options or CleanupOptions()
    smoothing_iterations = options.smoothing_iterations
    hole_radius_mm = options.hole_radius_mm
    min_component_area_mm2 = options.min_component_area_mm2
    effective_pass_band = options.smoothing_pass_band
    validate_cleanup(smoothing_iterations=smoothing_iterations, hole_radius_mm=hole_radius_mm,
                     min_component_area_mm2=min_component_area_mm2,
                     smoothing_pass_band=options.smoothing_pass_band)
    if mesh.material is not None:
        raise ValueError("Prepare geometry before texturing")
    if mesh.vertices is None or mesh.faces is None:
        raise ValueError("Structure has no mesh")
    vertices, faces = validate_triangles(mesh.vertices, mesh.faces)
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
    before = _quality(source)
    components = [part for part in source.split(only_watertight=False, repair=False)
                  if part.area * 1e6 >= min_component_area_mm2]
    if not components:
        raise ValueError("Component threshold removed the entire surface")
    surface_mesh = trimesh.util.concatenate(components)
    assert isinstance(surface_mesh, trimesh.Trimesh)
    if hole_radius_mm or smoothing_iterations:
        points, cells, surface = vtk.vtkPoints(), vtk.vtkCellArray(), vtk.vtkPolyData()
        points.SetData(numpy_to_vtk(np.asarray(surface_mesh.vertices), deep=True))
        cells.SetData(
            numpy_to_vtkIdTypeArray(np.arange(0, 3 * (len(surface_mesh.faces) + 1), 3, dtype=np.int64), deep=True),
            numpy_to_vtkIdTypeArray(np.asarray(surface_mesh.faces, dtype=np.int64).ravel(), deep=True),
        )
        surface.SetPoints(points)
        surface.SetPolys(cells)
        result = surface
        if hole_radius_mm:
            repair = vtk.vtkFillHolesFilter()
            repair.SetInputData(result)
            repair.SetHoleSize(hole_radius_mm * 0.001)
            repair.Update()
            result = repair.GetOutput()
        if smoothing_iterations:
            unsmoothed = trimesh.Trimesh(
                vertices=vtk_to_numpy(result.GetPoints().GetData()),
                faces=vtk_to_numpy(result.GetPolys().GetConnectivityArray()).reshape(-1, 3),
                process=False,
            )
            smooth = vtk.vtkWindowedSincPolyDataFilter()
            smooth.SetInputData(result)
            smooth.SetNumberOfIterations(smoothing_iterations)
            smooth.NormalizeCoordinatesOn()
            smooth.BoundarySmoothingOff()
            smooth.FeatureEdgeSmoothingOff()
            smooth.NonManifoldSmoothingOff()
            # A densely meshed scan supports strong filtering; a coarse CAD
            # cylinder with only two vertex rings can collapse at that setting.
            # Back off the frequency cutoff until volume (or open-surface area)
            # stays within 2%, rather than silently destroying such inputs.
            bands = sorted({options.smoothing_pass_band,
                            *(max(options.smoothing_pass_band, band) for band in (.03, .1, .3, .8, 1.5, 2.))})
            for effective_pass_band in bands:
                smooth.SetPassBand(effective_pass_band)
                smooth.Update()
                candidate = trimesh.Trimesh(
                    vertices=vtk_to_numpy(smooth.GetOutput().GetPoints().GetData()),
                    faces=unsmoothed.faces, process=False,
                )
                baseline = abs(unsmoothed.volume) if unsmoothed.is_watertight else unsmoothed.area
                measured = abs(candidate.volume) if unsmoothed.is_watertight else candidate.area
                if (baseline > 0 and np.isfinite(candidate.vertices).all()
                        and abs(measured / baseline - 1) <= .02
                        and np.all(candidate.area_faces > 0)):
                    break
            else:
                raise ValueError('Smoothing could not preserve the input surface; disable or reduce smoothing')
            result = smooth.GetOutput()
        surface_mesh = trimesh.Trimesh(
            vertices=vtk_to_numpy(result.GetPoints().GetData()),
            faces=vtk_to_numpy(result.GetPolys().GetConnectivityArray()).reshape(-1, 3),
            process=True, validate=True,
        )
    surface_mesh.fix_normals(multibody=True)
    report = {"before": before, "after": _quality(surface_mesh),
              "smoothing_iterations": smoothing_iterations, "hole_radius_mm": hole_radius_mm,
              "min_component_area_mm2": min_component_area_mm2,
              "smoothing_algorithm": "windowed_sinc", "smoothing_pass_band": effective_pass_band,
              "requested_smoothing_pass_band": options.smoothing_pass_band}
    geometry = MeshGeometry(np.asarray(surface_mesh.vertices), np.asarray(surface_mesh.faces),
                            normals=np.asarray(surface_mesh.vertex_normals))
    return CleanupResult(geometry, report)
