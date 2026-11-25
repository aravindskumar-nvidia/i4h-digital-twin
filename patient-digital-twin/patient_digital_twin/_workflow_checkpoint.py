# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal, non-executable pipeline checkpoints. Not a user configuration format."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .anatomy import (
    AnatomicalStructure,
    AnatomyCollection,
    MeshGeometry,
    SurfaceMaterial,
)
from .body import HumanBody
from .geometry import CenterlineGraph
from .scan_volume import ScanVolume


def write_state(path, state):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, default=lambda v: str(v) if isinstance(v, Path) else v.tolist()) + '\n')
    temporary.replace(path)


def save_body(body, folder):
    folder.mkdir(exist_ok=True)
    meta = {'body_to_imaging': body.anatomy.body_to_imaging, 'structures': []}
    for index, (name, structure) in enumerate(body.anatomy.structures.items()):
        arrays = {key: value for key, value in vars(structure.mesh).items()
                  if key != 'material' and value is not None}
        if structure.centerline is not None:
            arrays.update({'graph_' + key: value for key, value in vars(structure.centerline).items()})
        np.savez(folder / f'{index}.npz', **arrays)
        meta['structures'].append(dict(name=name, kind=structure.kind.value, enabled=structure.enabled,
            local_to_body=structure.local_to_body, local_to_world=structure.local_to_world,
            material=asdict(structure.mesh.material) if structure.mesh.material else None))
    if body.imaging:
        np.save(folder / 'ct.npy', body.imaging.scan.values, allow_pickle=False)
        meta['imaging'] = dict(metadata=body.imaging.scan.metadata,
                              body_to_imaging=body.imaging.body_to_imaging, source_path=body.imaging.source_path)
    write_state(folder / 'body.json', meta)


def load_body(folder):
    meta = json.loads((folder / 'body.json').read_text())
    anatomy = AnatomyCollection()
    anatomy.body_to_imaging = meta['body_to_imaging']
    for index, record in enumerate(meta['structures']):
        material = record.pop('material')
        record['local_to_body'] = np.asarray(record['local_to_body'])
        record['local_to_world'] = np.asarray(record['local_to_world'])
        structure = AnatomicalStructure(**record)
        with np.load(folder / f'{index}.npz', allow_pickle=False) as arrays:
            structure.mesh = MeshGeometry(**{key: arrays[key].copy() for key in
                                              ('vertices', 'faces', 'uv', 'normals') if key in arrays},
                                          material=SurfaceMaterial(**material) if material else None)
            if 'graph_points' in arrays:
                structure.centerline = CenterlineGraph(**{key: arrays['graph_' + key].copy()
                                                          for key in ('points', 'edges', 'radii')})
        anatomy.structures[structure.name] = structure
    body = HumanBody(anatomy)
    if 'imaging' in meta:
        imaging = meta['imaging']
        body.attach_scan(ScanVolume(np.load(folder / 'ct.npy', allow_pickle=False), imaging['metadata']),
                         body_to_imaging=imaging['body_to_imaging'], source_path=imaging['source_path'])
    return body
