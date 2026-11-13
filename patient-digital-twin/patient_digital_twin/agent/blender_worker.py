# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executed inside Blender by the bridge; no patient package or USD dependency.

The exchange uses structure-local XYZ meters and a body/world pose, never scan
millimeters. Only objects tagged for this session are replaced or captured.
"""
import json
from pathlib import Path


def show(path):
    """Create a dedicated patient collection, retaining unrelated scene objects."""
    import bpy
    from mathutils import Matrix

    payload = json.loads(Path(path).read_text())
    tag = payload['collection']
    collection = bpy.data.collections.get(tag)
    if collection is None:
        collection = bpy.data.collections.new(tag)
        bpy.context.scene.collection.children.link(collection)
    for obj in list(collection.objects):
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    collection['patient_token'] = payload['token']
    collection['patient_revision'] = payload['revision']
    for item in payload['structures']:
        mesh = bpy.data.meshes.new(item['name'])
        mesh.from_pydata(item['vertices'], [], item['faces'])
        mesh.update()
        obj = bpy.data.objects.new(item['name'], mesh)
        obj['anatomy:name'] = item['name']
        collection.objects.link(obj)
        obj.matrix_world = Matrix(item['matrix'])
        obj.hide_set(not item['enabled'])
        obj.hide_render = not item['enabled']
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in collection.objects:
        if not obj.hide_get():
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.type = 'SOLID'
                area.spaces.active.shading.color_type = 'RANDOM'
                with bpy.context.temp_override(window=window, area=area,
                                               region=next(r for r in area.regions if r.type == 'WINDOW')):
                    bpy.ops.view3d.view_selected()


def snapshot(path, collection_name):
    """Capture evaluated meshes (including modifiers), poses and visibility."""
    import bpy

    collection = bpy.data.collections[collection_name]
    if collection.hide_viewport or collection.hide_render:
        raise ValueError('Use per-object visibility, not collection visibility')
    result = {'token': collection['patient_token'], 'revision': collection['patient_revision'], 'structures': []}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in collection.objects:
        if obj.type != 'MESH' or 'anatomy:name' not in obj:
            raise ValueError('Patient collection must contain only its tagged meshes')
        if obj.mode != 'OBJECT':
            raise ValueError('Leave Edit Mode before applying patient edits')
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            result['structures'].append({
                'name': obj['anatomy:name'], 'vertices': [list(v.co) for v in mesh.vertices],
                'faces': [list(t.vertices) for t in mesh.loop_triangles],
                'matrix': [list(row) for row in evaluated.matrix_world],
                'enabled': not obj.hide_get() and not obj.hide_render,
            })
        finally:
            evaluated.to_mesh_clear()
    Path(path).write_text(json.dumps(result, allow_nan=False))
