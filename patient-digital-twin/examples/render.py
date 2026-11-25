# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save a patient USD preview, or a before/after comparison, with one background Blender.

Run with the active patient Python environment, not Isaac Sim's Python:
    python patient-digital-twin/examples/render.py patient.usdc --output preview.png
    python patient-digital-twin/examples/render.py before.usdc after.usdc --output comparison

Eevee includes materials/textures; --engine workbench gives geometry-only previews.
Only /HumanBody/Anatomy is rendered. Source assets are never saved or exported.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('before', type=Path, help='Patient USD to render')
    cli.add_argument('after', type=Path, nargs='?', help='Second USD for a matched comparison')
    cli.add_argument('--output', type=Path, required=True,
                     help='PNG file for one USD, or a directory containing comparison.png')
    cli.add_argument('--engine', choices=['eevee', 'workbench'], default='eevee',
                     help='eevee: materials and textures; workbench: fast geometry inspection')
    return cli


def render(args: argparse.Namespace) -> None:
    """Validate inputs, launch Blender once, and assemble the comparison in Python."""
    from PIL import Image, ImageDraw
    from pxr import Usd, UsdGeom
    from patient_digital_twin.agent.runtime import ensure_blender

    sources = [p.expanduser().resolve() for p in (args.before, args.after) if p is not None]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        stage = Usd.Stage.Open(str(source))
        if not stage or not stage.GetPrimAtPath('/HumanBody/Anatomy'):
            raise ValueError(f'Expected patient anatomy at /HumanBody/Anatomy: {source}')
        if UsdGeom.GetStageUpAxis(stage) != 'Z':
            raise ValueError(f'Expected a Z-up patient export: {source}')
    output = args.output.expanduser().resolve()
    if len(sources) == 1:
        if output.suffix.lower() != '.png':
            raise ValueError('A single USD requires --output FILE.png')
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)
    executable = ensure_blender()
    with tempfile.TemporaryDirectory(prefix='i4h-render-') as temporary:
        previews = []
        for index, source in enumerate(sources):
            original = Usd.Stage.Open(str(source))
            preview = Usd.Stage.CreateNew(str(Path(temporary) / f'{index}.usda'))
            preview.GetRootLayer().subLayerPaths = [str(source)]
            UsdGeom.SetStageUpAxis(preview, 'Z')
            UsdGeom.SetStageMetersPerUnit(preview, UsdGeom.GetStageMetersPerUnit(original))
            # Retain Looks and its relative texture paths while excluding CT/skin.
            # A path mask restricted to Anatomy drops sibling material definitions.
            for name in ('Exterior', 'Imaging'):
                preview.OverridePrim('/HumanBody/' + name).SetActive(False)
            preview.GetRootLayer().Save()
            previews.append(preview.GetRootLayer().realPath)
        render_output = output if len(sources) == 1 else Path(temporary)
        images = [output] if len(sources) == 1 else [Path(temporary) / 'before.png', Path(temporary) / 'after.png']
        command = [executable, '--background', '--factory-startup', '--python-exit-code', '1',
                   '--python', str(Path(__file__).resolve()), '--',
                   *previews, '--output', str(render_output), '--engine', args.engine]
        subprocess.run(command, check=True)
        for path in images:
            with Image.open(path) as image:
                if image.size != (960, 720) or all(low == high for low, high in image.convert('RGB').getextrema()):
                    raise RuntimeError(f'Render is empty or has unexpected dimensions: {path}')
        if len(images) == 2:
            with Image.open(images[0]) as before, Image.open(images[1]) as after:
                width, height = before.size
                panel = Image.new('RGB', (2 * width, height + 32), (24, 24, 24))
                panel.paste(before, (0, 32))
                panel.paste(after, (width, 32))
                labels = ImageDraw.Draw(panel)
                labels.text((12, 10), 'BEFORE', fill='white')
                labels.text((width + 12, 10), 'AFTER', fill='white')
                panel.save(output / 'comparison.png')
            images = [output / 'comparison.png']
        for path in images:
            print(path, flush=True)


def render_in_blender(args: argparse.Namespace) -> None:
    """Import in separate scenes; use identical physical framing and lighting."""
    import bpy
    from mathutils import Vector

    if bpy.app.version < (5, 0, 0):
        raise RuntimeError('Rendering requires Blender 5.0+. Set BLENDER_EXECUTABLE to a compatible installation.')
    sources = [p for p in (args.before, args.after) if p is not None]
    scenes, corners = [], []
    for index, source in enumerate(sources):
        scene = bpy.data.scenes.new(f'Patient preview {index}')
        bpy.context.window.scene = scene
        scene.unit_settings.system = 'METRIC'
        scene.unit_settings.scale_length = 1.0
        bpy.ops.wm.usd_import(filepath=str(source), prim_path_mask='/HumanBody',
                              import_cameras=False, import_lights=False, import_volumes=False,
                              import_visible_only=True, apply_unit_conversion_scale=True,
                              property_import_mode='NONE')
        bpy.context.view_layer.update()
        meshes = [obj for obj in scene.objects if obj.type == 'MESH' and not obj.hide_render]
        if not meshes:
            raise ValueError(f'No visible anatomy meshes: {source}')
        corners.extend(obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box)
        scenes.append(scene)
    if not all(math.isfinite(value) for point in corners for value in point):
        raise ValueError('Patient bounds contain non-finite coordinates')
    low = Vector(tuple(min(point[i] for point in corners) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in corners) for i in range(3)))
    center, radius = (low + high) / 2, max((high - low).length / 2, 0.00001)
    direction = Vector((1, -1, 0.8)).normalized()
    camera = bpy.data.objects.new('Preview camera', bpy.data.cameras.new('Preview camera'))
    camera.location = center + direction * radius * 3
    camera.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.sensor_fit = 'HORIZONTAL'
    projected = [camera.rotation_euler.to_quaternion().inverted() @ (point - center) for point in corners]
    camera.data.ortho_scale = max(2 * max(abs(p.x) for p in projected),
                                  2 * max(abs(p.y) for p in projected) * 960 / 720, 0.00001) * 1.15
    camera.data.clip_start = radius / 1000
    camera.data.clip_end = radius * 10
    lights = []
    for offset, power in [((1, -2, 3), 160), ((-2, -1, 1), 80), ((1, 2, 2), 120)]:
        light = bpy.data.objects.new('Preview light', bpy.data.lights.new('Preview light', 'AREA'))
        light.location = center + Vector(offset) * radius
        light.rotation_euler = (center - light.location).to_track_quat('-Z', 'Y').to_euler()
        light.data.energy, light.data.size = power * radius ** 2, radius * 2
        lights.append(light)
    for index, scene in enumerate(scenes):
        bpy.context.window.scene = scene
        for obj in [camera, *lights]:
            scene.collection.objects.link(obj)
        scene.camera = camera
        scene.render.engine = 'BLENDER_WORKBENCH' if args.engine == 'workbench' else 'BLENDER_EEVEE'
        if args.engine == 'eevee':
            scene.eevee.taa_render_samples = 16
        scene.render.resolution_x, scene.render.resolution_y = 960, 720
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.view_settings.view_transform = 'AgX' if args.engine == 'eevee' else 'Standard'
        scene.display.shading.light = 'STUDIO'
        scene.display.shading.color_type = 'MATERIAL'
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = True
        scene.display.shading.background_type = 'WORLD'
        scene.world = bpy.data.worlds.new('Preview background')
        scene.world.color = (0.04, 0.04, 0.04)
        target = args.output if len(scenes) == 1 else args.output / ('before.png' if index == 0 else 'after.png')
        scene.render.filepath = str(target)
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    if 'bpy' in sys.modules:
        render_in_blender(parser().parse_args(sys.argv[sys.argv.index('--') + 1:]))
    else:
        cli = parser()
        try:
            render(cli.parse_args())
        except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError,
                subprocess.CalledProcessError) as error:
            cli.exit(1, f'Render failed: {error}\n')
