# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application session around the existing patient API, shared by both agents."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from .. import (
    HumanBody,
    NVGenerateImporter,
    NVSegmentImporter,
    SegmentationImporter,
    SimpleImporter,
)
from ..geometry import rigid_transform, validate_triangles


class PatientSession:
    """Own a HumanBody and a directory for new exports and Blender exchanges.

    Blender never becomes the source of CT registration or anatomy identity.
    Display poses affect standalone USD; bundles retain the imported placement.
    """

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.body: HumanBody | None = None
        self.revision = 0
        self.token = uuid4().hex

    def require_body(self) -> HumanBody:
        """Return the active patient or fail before executing an edit."""
        if self.body is None:
            raise ValueError("Load a patient first")
        return self.body

    def output_path(self, path: str, *, new: bool = True) -> Path:
        """Resolve an output inside the workspace, rejecting overwrites by default."""
        target = (self.workspace / path).resolve()
        if not target.is_relative_to(self.workspace) or target == self.workspace:
            raise ValueError("Output must be inside the workspace")
        if new and target.exists():
            raise FileExistsError(f"Use a new output path: {target}")
        return target

    def load_patient(
        self, source: str, input_path: str = "", names: list[str] | None = None,
        labelmap: str | None = None, ct_path: str | None = None,
        model_root: str | None = None, python_executable: str | None = None,
    ) -> dict:
        """Load segmentations, STL/OBJ meshes, or run an optional NV backend.

        For meshes input_path is a JSON name-to-file map, relative to that JSON.
        Omit names for all supported anatomy. Inference requires model assets.
        Existing USD/bundles are presentation artifacts, not import sources.
        """
        import json

        from ..anatomy import canonical_name
        from ..scan_volume import from_nifti

        scan = None
        if source == "segmentation":
            importer = SegmentationImporter(input_path, labelmap,
                                            names=names if Path(input_path).is_dir() else None)
            anatomy = importer.to_anatomy_collection()
            if names is not None:
                selected = {canonical_name(name) for name in names}
                if selected - anatomy.structures.keys():
                    raise ValueError(f"Missing anatomy: {sorted(selected - anatomy.structures.keys())}")
                anatomy.structures = {name: structure for name, structure in anatomy.structures.items() if name in selected}
        elif source == "meshes":
            config = Path(input_path).expanduser().resolve()
            mapping = json.loads(config.read_text())
            if names is not None:
                mapping = {name: mapping[name] for name in names}
            anatomy = SimpleImporter({name: config.parent / path for name, path in mapping.items()}).to_anatomy_collection()
        elif source == "nvsegment":
            anatomy = NVSegmentImporter(input_path, bundle_root=model_root,
                                        python_executable=python_executable).to_anatomy_collection(names=names)
            ct_path = input_path
        elif source == "nvgenerate":
            generator = NVGenerateImporter(source_root=model_root, python_executable=python_executable)
            anatomy = generator.to_anatomy_collection(names=names)
            scan = generator.ct_scan
        else:
            raise ValueError("source must be segmentation, meshes, nvsegment, or nvgenerate")
        body = HumanBody(anatomy)
        if scan is not None:
            body.attach_scan(scan)
        elif ct_path is not None:
            body.attach_scan(from_nifti(ct_path), source_path=ct_path)
        self.body = body
        self.token = uuid4().hex
        self.revision += 1
        return self.inspect_patient()

    def inspect_patient(self) -> dict:
        """Summarize structures, visibility, geometry and attached CT without voxel data."""
        body = self.require_body()
        return {
            "revision": self.revision,
            "structures": [
                {"name": s.name, "kind": s.kind.value, "enabled": s.enabled,
                 "vertices": 0 if s.mesh.vertices is None else len(s.mesh.vertices),
                 "centerline_points": 0 if s.centerline is None else len(s.centerline.points)}
                for s in body.anatomy.structures.values()
            ],
            "ct": None if body.imaging is None else {
                "shape": list(body.imaging.scan.values.shape), "frame": body.imaging.scan.frame,
                "array_axes": body.imaging.scan.array_axes, "meters_per_unit": body.imaging.scan.meters_per_unit,
            },
        }

    def set_visibility(self, names: list[str], enabled: bool) -> dict:
        """Hide/show anatomy without removing geometry."""
        body = self.require_body()
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        for name in names:
            if name not in body.anatomy.structures:
                raise KeyError(name)
        for name in names:
            body.anatomy.set_structure_enabled(name, enabled)
        self.revision += 1
        return self.inspect_patient()

    def extract_centerlines(self, names: list[str], spacing_m: float = 0.0015) -> dict:
        """Extract requested vessel/airway graphs through HumanBody; surface errors propagate."""
        result = self.require_body().extract_topology(names=names, spacing_m=spacing_m)
        self.revision += 1
        return {name: {"points": len(graph.points), "edges": len(graph.edges)} for name, graph in result.items()}

    def export_patient(self, path: str, format: str = "usd", vessel_names: list[str] | None = None) -> dict:
        """Export a new standalone USD or patient bundle through the existing exporters."""
        body = self.require_body()
        target = self.output_path(path)
        if format == "usd":
            result = body.export_to_usd(target)
        elif format == "bundle":
            if any(not np.allclose(s.local_to_world, s.local_to_body) for s in body.anatomy.structures.values()):
                raise ValueError("Bundle uses imported poses; reset display poses or export standalone USD")
            result = body.export_patient_twin(target, vessel_names=vessel_names or ())
        else:
            raise ValueError("format must be usd or bundle")
        return {"path": str(result)}

    def apply_snapshot(self, snapshot: dict) -> dict:
        """Validate every Blender edit before committing; reject stale or incomplete snapshots.

        Mesh changes invalidate that structure's centerline. Visibility/rigid pose
        changes retain local graphs. Imported placement and CT buffers never change.
        """
        body = self.require_body()
        if snapshot.get("token") != self.token or snapshot.get("revision") != self.revision:
            raise ValueError("Stale Blender view; show the current patient again before applying edits")
        items = snapshot["structures"]
        names = [item["name"] for item in items]
        expected = {s.name for s in body.anatomy.structures.values() if s.mesh.vertices is not None}
        if len(set(names)) != len(names) or set(names) != expected:
            raise ValueError("Blender snapshot must contain each patient mesh exactly once")
        prepared = []
        for item in items:
            vertices, faces = validate_triangles(item["vertices"], item["faces"])
            pose = rigid_transform(item["matrix"])
            if not isinstance(item["enabled"], bool):
                raise ValueError("enabled must be a boolean")
            prepared.append((body.anatomy.structures[item["name"]], vertices.copy(), faces.copy(), pose, item["enabled"]))
        for structure, vertices, faces, pose, enabled in prepared:
            if (structure.mesh.vertices is None or structure.mesh.faces is None
                    or not np.array_equal(vertices.astype(np.float32), structure.mesh.vertices.astype(np.float32))
                    or not np.array_equal(faces, structure.mesh.faces)):
                structure.vertices, structure.faces = vertices, faces
            structure.local_to_world, structure.enabled = pose, enabled
        self.revision += 1
        return self.inspect_patient()
