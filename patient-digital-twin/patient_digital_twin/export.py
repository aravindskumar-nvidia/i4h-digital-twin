# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exporters: write a ``HumanBody`` as OpenUSD, as an i4h-workflows bundle, or as raw arrays.

Usually reached through ``HumanBody.export_to_usd`` and ``HumanBody.export_patient_twin``.

- ``export_to_usd``: one standalone stage with anatomy in its current pose, centerlines
  as custom prim attributes, and attached CT as native HU attributes. The stage uses the
  scan's units (meters without CT), Z-up, and default prim ``/HumanBody``.
- ``export_patient_twin``: a new schema-2 bundle directory: ``patient_twin.yaml``
  (manifest), ``patient_anatomy.usdc`` (anatomy in the scan's physical frame), and with
  CT ``volume.npy``/``volume.yaml``; ``vessel_names`` adds ``vessel_mask.npy`` and
  ``centerline_{points,edges,radii}.npy`` on the CT grid for navigation.
- ``write_artifacts``: just the native CT and optional vessel mask/centerline arrays.
- ``write_usd``: the shared stage writer behind both exporters.

Nothing here resamples, reorients, or recenters the source scan. Directory outputs are
written to a sibling staging folder and moved into place only when complete.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from .geometry import (
    mask_to_mesh,
    native_centerline,
    rigid_transform,
    transform_points,
    validate_triangles,
    voxelize_mesh,
)

if TYPE_CHECKING:
    from .anatomy import MeshGeometry
    from .body import HumanBody
    from .scan_volume import ScanVolume

USD_SUFFIXES = {".usd", ".usda", ".usdc"}
SKIN_OPACITY = 0.15
# The CT exterior is meshed from voxels above this HU value on a grid strided by EXTERIOR_STRIDE.
EXTERIOR_HU = -300
EXTERIOR_STRIDE = 3


def _surface_material(prim: Any, geometry: MeshGeometry, output: Path) -> None:
    """Write portable texture references and preserve normals across UV seams."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    mesh = UsdGeom.Mesh(prim)
    assert geometry.vertices is not None
    count = len(geometry.vertices)
    if geometry.normals is not None:
        normals = np.asarray(geometry.normals, dtype=np.float32)
        if normals.shape != (count, 3) or not np.isfinite(normals).all():
            raise ValueError("Surface normals must be finite (N, 3)")
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if geometry.material is None:
        return
    uv = np.asarray(geometry.uv, dtype=np.float32)
    if uv.shape != (count, 2) or not np.isfinite(uv).all():
        raise ValueError("Textured meshes require finite per-vertex UV coordinates (N, 2)")
    material = geometry.material
    if not np.isfinite(material.roughness) or not 0 <= material.roughness <= 1:
        raise ValueError("Material roughness must be in [0, 1]")
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec2fArray.FromNumpy(uv))
    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    stage, root = prim.GetStage(), bound.GetPath()
    shader = UsdShade.Shader.Get(stage, root.AppendChild("Surface"))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, root.AppendChild("UV"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for name, source, color_space, channel, kind in (
        ("diffuseColor", material.base_color, "sRGB", "rgb", Sdf.ValueTypeNames.Color3f),
        ("roughness", material.roughness_map, "raw", "r", Sdf.ValueTypeNames.Float),
    ):
        source_path = Path(source).resolve()
        data = source_path.read_bytes()
        relative = Path("textures") / (hashlib.sha256(data).hexdigest() + source_path.suffix.lower())
        target = output.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        texture = UsdShade.Shader.Define(stage, root.AppendChild(name + "Texture"))
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(relative.as_posix()))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        if name == "roughness":
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(material.roughness))
        texture.CreateOutput(channel, kind)
        shader.CreateInput(name, kind).ConnectToSource(texture.ConnectableAPI(), channel)


def _scan_from_body(body: HumanBody) -> NDArray[np.float64]:
    """Body frame (meters) to the scan's physical frame (still meters)."""
    registration = body.anatomy.body_to_imaging if body.imaging is None else body.imaging.body_to_imaging
    if registration is None:
        return np.eye(4)
    flip = np.diag([-1.0, -1.0, 1.0, 1.0]) if body.imaging is not None and body.imaging.scan.frame == "LPS" else np.eye(4)
    return flip @ registration


def _write_scan(scan: ScanVolume, folder: Path, vessel_mask: ArrayLike | None = None) -> dict[str, str]:
    """Native HU volume + YAML, plus an optional scan-grid vessel mask and its centerline arrays."""
    scan.save(folder)
    paths = {"hu_volume": "volume.npy", "volume_metadata": "volume.yaml"}
    if vessel_mask is not None:
        mask = np.asarray(vessel_mask)
        if mask.shape != scan.values.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("Vessel mask must be binary and match the scan grid")
        names = ("centerline_points", "centerline_edges", "centerline_radii")
        arrays: dict[str, np.ndarray] = {**dict(zip(names, native_centerline(mask, scan))), "vessel_mask": mask.astype(np.uint8)}
        for name, values in arrays.items():
            np.save(Path(folder) / f"{name}.npy", values, allow_pickle=False)
            paths[name] = f"{name}.npy"
    return paths


def publish_directory(output: str | Path, prefix: str, write: Callable[[Path], Any]) -> Path:
    """Run write(folder) in a sibling staging folder, then rename it to a new output path.

    The rename fails if anything has appeared at the output meanwhile, so concurrent
    runs cannot merge into or replace each other's results.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=prefix) as temp:
        write(Path(temp) / "out")
        try:
            os.rename(Path(temp) / "out", output)
        except OSError as error:
            if error.errno in {errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR}:
                raise FileExistsError(f"Use a new output directory: {output}") from error
            raise
    return output


def publish_usd(staged: Path, output: Path) -> Path:
    """Publish a staged USD as ``output`` and the files beside it as ``<output>.assets``.

    Asset paths are rewritten into that sidecar, which is published first; the USD is
    then hard-linked into place, so neither step can replace a concurrent output.
    """
    from pxr import Sdf, Usd

    sidecar = output.with_name(output.name + ".assets")
    stage = Usd.Stage.Open(str(staged))
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset and attr.Get():
                attr.Set(Sdf.AssetPath(sidecar.name + "/" + attr.Get().path))
    stage.GetRootLayer().Save()
    publish_directory(sidecar, ".patient-assets-",
             lambda folder: shutil.copytree(staged.parent, folder, ignore=shutil.ignore_patterns(staged.name)))
    try:
        with TemporaryDirectory(dir=output.parent, prefix=".patient-usd-") as temporary:
            linked = Path(temporary) / staged.name
            shutil.copy2(staged, linked)
            os.link(linked, output)
    except BaseException:
        shutil.rmtree(sidecar)
        raise
    return output


def write_artifacts(scan: ScanVolume, output: str | Path, *, vessel_mask: ArrayLike | None = None) -> Path:
    """Write native HU ``volume.npy`` + ``volume.yaml`` and an optional vessel mask with its centerline.

    Args:
        scan: Source CT, e.g. ``scan_volume.from_nifti("ct.nii.gz")``.
        output: New directory (must not exist); written atomically.
        vessel_mask: Binary array with ``scan.values``' shape and axis order.

    Returns:
        The output directory.
    """
    return publish_directory(output, ".ct-artifacts-", lambda folder: _write_scan(scan, folder, vessel_mask))


def _vessel_mask(body: HumanBody, scan: ScanVolume, body_to_imaging: NDArray[np.floating],
                 names: Sequence[str]) -> NDArray[np.bool_]:
    """Rasterize the requested vessel meshes onto the scan grid, exactly as the bundle USD places them.

    Each mesh goes through ``body_to_imaging @ local_to_body``, so a custom registration
    passed to ``attach_scan`` (or an edited structure placement) moves the navigation
    mask and centerline together with the exported vessel. Callers check the meshes exist.
    """
    scan_from_body = np.linalg.inv(scan.ijk_to_ras_m) @ body_to_imaging
    mask = np.zeros(scan.values_kji.shape, bool)
    for name in names:
        structure = body.anatomy.structures[name]
        vertices, faces = structure.mesh.vertices, structure.mesh.faces
        assert vertices is not None and faces is not None
        try:
            mask |= voxelize_mesh(
                transform_points(vertices, scan_from_body @ structure.local_to_body), faces,
                shape_zyx=mask.shape, spacing_zyx_m=(1.0, 1.0, 1.0), origin_xyz_m=(0.0, 0.0, 0.0),
            )
        except ValueError as exc:
            raise ValueError(f"Cannot rasterize vessel {name!r} onto the scan grid: {exc}") from exc
    if not mask.any():
        raise ValueError("Selected vessels have no foreground in the scan")
    return mask.transpose(["kji".index(c) for c in scan.array_axes])


def export_to_usd(body: HumanBody, path: str | Path) -> Path:
    """Write a standalone USD of ``body``: current pose, stored centerlines, and attached CT (native HU).

    The stage uses the scan's units (meters without CT), Z-up, and default prim ``/HumanBody``.
    An existing file is replaced only after the new layer is complete. Returns the resolved path.

    Example:
        ``export_to_usd(body, "patient.usdc")`` (same as ``body.export_to_usd("patient.usdc")``)
    """
    scan = None if body.imaging is None else body.imaging.scan
    units = scan.meters_per_unit if scan is not None else 1.0
    root = _scan_from_body(body)
    root[:3, 3] /= units
    path = Path(path).expanduser().resolve()
    write_usd(body, path, units=units, root=root, scan=scan)
    return path


def write_usd(
    body: HumanBody,
    path: str | Path,
    *,
    units: float = 1.0,
    root: ArrayLike | None = None,
    placement: NDArray[np.floating] | None = None,
    scan: ScanVolume | None = None,
    exterior: tuple[ArrayLike, ArrayLike] | None = None,
    skin_opacity: float = SKIN_OPACITY,
) -> dict[str, str]:
    """Write a USD stage and return ``{structure name: prim path}``.

    Args:
        units: Stage meters-per-unit; geometry (in meters) is divided by it.
        root: 4x4 transform on ``/HumanBody`` (in stage units).
        placement: If given, each structure is placed at ``placement @ local_to_body``
            instead of its ``local_to_world``.
        scan: CT to embed under ``/HumanBody/Imaging/CT``.
        exterior: Optional ``(vertices, faces)`` CT envelope in stage-frame meters.
    """
    try:
        from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, Vt
    except ImportError as exc:
        raise ImportError("USD export requires the 'usd' extra of patient-digital-twin") from exc
    path = Path(path)
    if path.suffix.lower() not in USD_SUFFIXES:
        raise ValueError("Use a .usd, .usda, or .usdc output file")
    if not np.isfinite(skin_opacity) or not 0 <= skin_opacity <= 1:
        raise ValueError("skin_opacity must be between 0 and 1")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    xform = UsdGeom.Xform.Define(stage, "/HumanBody")
    stage.SetDefaultPrim(xform.GetPrim())
    xform.AddTransformOp().Set(Gf.Matrix4d((np.eye(4) if root is None else np.asarray(root)).T.tolist()))
    for scope in ("Anatomy", "Looks") + (("Exterior",) if exterior is not None else ()):
        UsdGeom.Scope.Define(stage, f"/HumanBody/{scope}")

    def custom(prim: Any, name: str, kind: str, value: Any) -> None:
        """Author a custom attribute of Sdf type ``kind`` (e.g. ``"FloatArray"``) on ``prim``."""
        prim.CreateAttribute(name, getattr(Sdf.ValueTypeNames, kind), custom=True).Set(value)

    def mesh(prim_path: str, vertices: ArrayLike, faces: ArrayLike, color: Sequence[float], opacity: float,
             matrix: ArrayLike | None = None, enabled: bool = True) -> Any:
        """Define one shaded mesh prim (vertices in meters) and return it."""
        meters, triangles = validate_triangles(vertices, faces, name=prim_path)
        points = (meters / units).astype(np.float32)
        result = UsdGeom.Mesh.Define(stage, prim_path)
        result.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        result.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
        result.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.astype(np.int32).ravel()))
        result.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        result.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        result.CreateDoubleSidedAttr(True)
        result.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)])))
        result.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        result.CreateDisplayOpacityAttr([opacity])
        result.CreateVisibilityAttr(UsdGeom.Tokens.inherited if enabled else UsdGeom.Tokens.invisible)
        if matrix is not None:
            pose = rigid_transform(matrix)
            pose[:3, 3] /= units
            result.AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
        material = UsdShade.Material.Define(stage, "/HumanBody/Looks/" + result.GetPrim().GetName())
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(result.GetPrim()).Bind(material)
        return result.GetPrim()

    if exterior is not None:
        mesh("/HumanBody/Exterior/CT", *exterior, (0.72, 0.77, 0.84), skin_opacity).SetDisplayName("CT exterior")
    prims: dict[str, str] = {}
    for name, structure in body.anatomy.structures.items():
        vertices, faces = structure.mesh.vertices, structure.mesh.faces
        if vertices is None or faces is None:
            continue
        identifier = candidate = Tf.MakeValidIdentifier(name)
        number = 2
        while "/HumanBody/Anatomy/" + candidate in prims.values():
            candidate, number = f"{identifier}_{number}", number + 1
        prims[name] = "/HumanBody/Anatomy/" + candidate
        digest = hashlib.sha256(name.encode()).digest()
        color = (0.88, 0.84, 0.69) if structure.kind.value == "bone" else tuple(0.25 + 0.65 * v / 255 for v in digest[:3])
        matrix = structure.local_to_world if placement is None else placement @ structure.local_to_body
        prim = mesh(prims[name], vertices, faces, color, 1.0, matrix, structure.enabled)
        _surface_material(prim, structure.mesh, path)
        prim.SetDisplayName(name)
        prim.SetCustomDataByKey("anatomy:name", name)
        prim.SetCustomDataByKey("anatomy:kind", structure.kind.value)
        if (graph := structure.centerline) is not None:
            custom(prim, "centerline:points", "Point3fArray", Vt.Vec3fArray.FromNumpy(np.asarray(graph.points / units, np.float32)))
            custom(prim, "centerline:edges", "Int2Array", Vt.Vec2iArray.FromNumpy(np.asarray(graph.edges, np.int32)))
            custom(prim, "centerline:radii", "FloatArray", Vt.FloatArray.FromNumpy(np.asarray(graph.radii / units, np.float32)))
            custom(prim, "centerline:coordinateFrame", "Token", "structure_local")
    if scan is not None:
        UsdGeom.Scope.Define(stage, "/HumanBody/Imaging")
        ct = UsdGeom.Scope.Define(stage, "/HumanBody/Imaging/CT").GetPrim()
        output = scan.metadata["output"]
        custom(ct, "ct:hu", "FloatArray", Vt.FloatArray.FromNumpy(scan.values.ravel(order="C")))
        custom(ct, "ct:shape", "Int3", Gf.Vec3i(*scan.values.shape))
        custom(ct, "ct:arrayOrder", "Token", scan.array_axes)
        custom(ct, "ct:arrayIndexToScan", "Matrix4d", Gf.Matrix4d(np.asarray(output["array_index_to_world"]).T.tolist()))
        custom(ct, "ct:coordinateFrame", "Token", scan.frame)
        custom(ct, "ct:spatialUnit", "Token", output["world_unit"])
        custom(ct, "ct:units", "Token", "HU")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Finish the layer before replacing a previous export.
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        if not stage.GetRootLayer().Export(str(temporary)):
            raise RuntimeError(f"USD export failed: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return prims


def _ct_exterior(scan: ScanVolume, units: float) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Mesh the patient's skin envelope from the CT, in stage-frame meters."""
    from scipy import ndimage

    # Keep only the body: the table and air cavities would otherwise add nested shells.
    body_voxels = scan.values_kji[::EXTERIOR_STRIDE, ::EXTERIOR_STRIDE, ::EXTERIOR_STRIDE] > EXTERIOR_HU
    labels, count = ndimage.label(ndimage.binary_closing(body_voxels, iterations=2))
    if not count:
        raise ValueError(f"CT has no voxels above {EXTERIOR_HU} HU to build an exterior from")
    envelope = ndimage.binary_fill_holes(labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    points, faces = mask_to_mesh(envelope)
    points = transform_points(points * EXTERIOR_STRIDE, scan.ijk_to_world) * units
    return points, faces[:, ::-1] if np.linalg.det(scan.ijk_to_world[:3, :3]) < 0 else faces


def export_patient_twin(
    body: HumanBody,
    output: str | Path,
    *,
    patient_id: str | None = None,
    vessel_names: Sequence[str] = (),
    world_from_patient_m: ArrayLike | None = None,
    ct_exterior: bool = False,
    skin_opacity: float = SKIN_OPACITY,
) -> Path:
    """Write a schema-2 patient bundle directory and return its ``patient_twin.yaml`` path.

    Geometry uses the attached scan's physical frame and units (or the body/RAS frame in
    meters without CT). No CT canonicalization, mask cleanup, resampling, or simulator
    placement is performed. Structure centerlines stay on their USD prims.

    Args:
        output: New directory (must not exist).
        patient_id: Manifest ID; defaults to the CT source folder name or ``"geometry"``.
        vessel_names: Vessels to rasterize on the CT grid, with navigation centerline arrays.
        world_from_patient_m: Optional rigid simulator placement hint stored in the manifest.
        ct_exterior: Add a CT-derived skin envelope at ``/HumanBody/Exterior/CT``.
        skin_opacity: Envelope opacity in [0, 1].

    Example:
        ``export_patient_twin(body, "bundle", vessel_names=["aorta"], ct_exterior=True)``
    """
    if Path(output).expanduser().exists():
        raise FileExistsError(f"Use a new patient-twin output directory: {output}")
    imaging = body.imaging
    scan = None if imaging is None else imaging.scan
    if (ct_exterior or vessel_names) and imaging is None:
        raise ValueError("CT exterior and navigation vessels require attached CT")
    for name in vessel_names:
        if name not in body.anatomy.structures or body.anatomy.structures[name].is_empty:
            raise ValueError(f"Missing or disabled vessel mesh: {name}")
    units = scan.meters_per_unit if scan is not None else 1.0
    frame = scan.frame if scan is not None else "RAS" if body.anatomy.body_to_imaging is not None else "body"
    mask = None
    exterior = None
    if imaging is not None:
        if vessel_names:
            mask = _vessel_mask(body, imaging.scan, imaging.body_to_imaging, vessel_names)
        if ct_exterior:
            exterior = _ct_exterior(imaging.scan, units)
    source_path = None if imaging is None else imaging.source_path

    def write(folder: Path) -> None:
        """Write every bundle file into the staging ``folder``."""
        artifacts = {"anatomy_usd": "patient_anatomy.usdc"}
        if scan is not None:
            artifacts.update(_write_scan(scan, folder, mask))
        else:
            folder.mkdir()
        prims = write_usd(body, folder / "patient_anatomy.usdc", units=units, scan=None,
                          placement=_scan_from_body(body), exterior=exterior, skin_opacity=skin_opacity)
        structures = body.anatomy.structures
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "patient_id": patient_id or (Path(source_path).parent.name if source_path else "geometry"),
            "coordinate_frame": frame,
            "spatial_unit": scan.metadata["output"]["world_unit"] if scan is not None else "m",
            "meters_per_unit": units,
            "anatomy": {
                "structures": {
                    n: {"prim_path": p, "kind": structures[n].kind.value, "enabled": structures[n].enabled}
                    for n, p in prims.items()
                },
                "exterior": None if exterior is None
                else {"source": "CT", "pose": "imaging", "prim_path": "/HumanBody/Exterior/CT"},
                "missing_meshes": [n for n, s in structures.items() if s.mesh.vertices is None],
            },
            "transforms": {} if scan is None else {"voxel_to_scan": scan.ijk_to_world.tolist()},
            "artifacts": artifacts,
        }
        if mask is not None:
            manifest["navigation"] = {"vessel_names": list(vessel_names)}
        if world_from_patient_m is not None:
            manifest["transforms"]["world_from_patient_m"] = rigid_transform(world_from_patient_m).tolist()
        (folder / "patient_twin.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    return publish_directory(output, ".patient-twin-", write) / "patient_twin.yaml"
