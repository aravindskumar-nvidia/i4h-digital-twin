# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender-side local bridge launched with --python; no add-on installation needed.

Uses the Blender MCP command shape for execute_code and viewport screenshots.
Network threads only queue work; bpy operations run on Blender's main thread.
"""
import argparse
import contextlib
import io
import json
import queue
import socketserver
import sys
import threading
from pathlib import Path


def main():
    """Start the private command socket and main-thread execution timer."""
    import bpy

    # This is our new private process, never the user's existing Blender scene.
    bpy.context.preferences.view.show_splash = False
    bpy.context.preferences.view.smooth_view = 0
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0

    parser = argparse.ArgumentParser()
    parser.add_argument('--ready', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    pending = queue.Queue()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(60)
            data = bytearray()
            while len(data) < 8 * 1024 * 1024:
                chunk = self.request.recv(65536)
                if not chunk:
                    return
                data.extend(chunk)
                try:
                    command = json.loads(data)
                    break
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
            else:
                return
            done = threading.Event()
            response = {}
            pending.put((command, response, done))
            if not done.wait(60):
                return
            self.request.sendall(json.dumps(response).encode())

    def tick():
        try:
            command, response, done = pending.get_nowait()
        except queue.Empty:
            return 0.05
        try:
            if command['type'] == 'execute_code':
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(command['params']['code'], {'__name__': '__patient_blender__'})
                result = {'executed': True, 'result': output.getvalue()}
            elif command['type'] == 'get_viewport_screenshot':
                path = command['params']['filepath']
                window = bpy.context.window_manager.windows[0]
                area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
                region = next(r for r in area.regions if r.type == 'WINDOW')
                render = bpy.context.scene.render
                previous = (render.filepath, render.resolution_x, render.resolution_y,
                            render.resolution_percentage, render.image_settings.file_format)
                try:
                    scale = min(1., command['params'].get('max_size', 1000) / max(region.width, region.height))
                    render.filepath, render.image_settings.file_format = path, 'PNG'
                    render.resolution_x = max(1, int(region.width * scale))
                    render.resolution_y = max(1, int(region.height * scale))
                    render.resolution_percentage = 100
                    with bpy.context.temp_override(window=window, area=area, region=region):
                        bpy.ops.render.opengl(write_still=True, view_context=True)
                finally:
                    (render.filepath, render.resolution_x, render.resolution_y,
                     render.resolution_percentage, render.image_settings.file_format) = previous
                result = {'filepath': path}
            else:
                raise ValueError('Unsupported Blender command')
            response.update(status='success', result=result)
        except Exception as error:
            response.update(status='error', message=f'{type(error).__name__}: {error}')
        finally:
            done.set()
        return 0.05

    server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    bpy.app.timers.register(tick, first_interval=0.1, persistent=True)
    ready = Path(args.ready)
    temporary = ready.with_suffix('.tmp')
    temporary.write_text(json.dumps({'port': server.server_address[1]}))
    temporary.replace(ready)


if __name__ == '__main__':
    main()
