# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer patient geometry to and from managed Blender."""
from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path
from uuid import uuid4

from .runtime import BlenderRuntime
from .session import PatientSession


class BlenderBridge:
    """Own the Blender view of one patient session."""

    def __init__(self, session: PatientSession, exchange_dir: str | None = None, *, headless: bool = False):
        self.session = session
        self.port: int | None = None
        self.collection = "I4H Patient " + session.token
        self.exchange = (session.workspace / (".blender-" + uuid4().hex)
                         if exchange_dir is None else Path(exchange_dir).resolve())
        if not self.exchange.is_relative_to(session.workspace) or not self.exchange.name.startswith('.blender-'):
            raise ValueError('Blender exchange must be a private .blender- directory inside the workspace')
        self.exchange.mkdir(exist_ok=True)
        self.runtime = BlenderRuntime(self.exchange, headless=headless)

    def close(self) -> None:
        """Close only the Blender instance owned by this bridge."""
        self.runtime.close()
        # Preserve failed-startup logs for diagnosis; successful sessions leave
        # only the outputs the user explicitly exported or saved.
        if self.runtime.process is None or self.port is not None:
            if self.exchange.exists():
                shutil.rmtree(self.exchange)

    def command(self, kind: str, params: dict) -> dict:
        """Exchange one complete JSON command; never retry a possibly executed edit."""
        if self.port is None:
            self.port = self.runtime.start()
        with socket.create_connection(('127.0.0.1', self.port), timeout=60) as connection:
            connection.sendall(json.dumps({"type": kind, "params": params}).encode())
            data = bytearray()
            while True:
                chunk = connection.recv(65536)
                if not chunk:
                    raise ConnectionError("Blender closed the connection before replying")
                data.extend(chunk)
                if len(data) > 8 * 1024 * 1024:
                    raise ValueError("Blender reply exceeds 8 MiB")
                try:
                    response = json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if response.get("status") != "success":
                    raise RuntimeError(response.get("message", "Blender command failed"))
                return response.get("result", {})

    def execute(self, code: str) -> dict:
        """Run Blender Python; changes affect the preview until explicitly applied."""
        return self.command("execute_code", {"code": code})

    def _worker(self, call: str) -> None:
        worker = Path(__file__).with_name("blender_worker.py")
        self.execute(f"exec(compile(open({str(worker)!r}).read(), {str(worker)!r}, 'exec')); {call}")

    def show(self) -> dict:
        """Send the current HumanBody to Blender, in meters, without clearing the scene."""
        body = self.session.require_body()
        payload: dict = {"token": self.session.token, "revision": self.session.revision,
                   "collection": self.collection, "structures": [
                       {"name": s.name, "vertices": s.mesh.vertices.tolist(), "faces": s.mesh.faces.tolist(),
                        "matrix": s.local_to_world.tolist(), "enabled": s.enabled}
                       for s in body.anatomy.structures.values()
                       if s.mesh.vertices is not None and s.mesh.faces is not None
                   ]}
        path = self.exchange / "patient.json"
        path.write_text(json.dumps(payload, allow_nan=False))
        self._worker(f"show({str(path)!r})")
        return {"collection": self.collection, "units": "meters", "structures": len(payload["structures"])}

    def apply_edits(self) -> dict:
        """Commit a validated preview snapshot to HumanBody, leaving source files intact."""
        path = self.exchange / (uuid4().hex + ".json")
        self._worker(f"snapshot({str(path)!r}, {self.collection!r})")
        result = self.session.apply_snapshot(json.loads(path.read_text()))
        # Update revision only after committing all validated edits.
        self.execute(f"import bpy; bpy.data.collections[{self.collection!r}]['patient_revision'] = {self.session.revision}")
        return result

    def screenshot(self, path: str | None = None) -> str:
        """Capture the viewport, optionally retaining it as a named output."""
        target = self.exchange / (uuid4().hex + ".png") if path is None else self.session.output_path(path)
        if target.suffix.lower() != '.png':
            raise ValueError('Screenshot must end in .png')
        target.parent.mkdir(parents=True, exist_ok=True)
        self.command("get_viewport_screenshot", {"filepath": str(target), "max_size": 1000})
        if not target.is_file():
            raise RuntimeError("Blender did not write the screenshot")
        return str(target)

    def open_usd(self, path: str) -> dict:
        """Open an existing USD or bundle in Blender for presentation edits only.

        This does not construct a HumanBody or modify bundle arrays/metadata.
        Use the original patient source to make canonical anatomy edits.
        """
        source = Path(path).expanduser().resolve()
        if source.suffix in {".yaml", ".yml"}:
            import yaml
            manifest = yaml.safe_load(source.read_text())
            source = (source.parent / manifest['artifacts']['anatomy_usd']).resolve()
        if source.suffix.lower() not in {".usd", ".usda", ".usdc"} or not source.is_file():
            raise ValueError("Expected an existing patient USD or patient_twin.yaml")
        self.execute(f"import bpy; bpy.ops.wm.usd_import(filepath={str(source)!r})")
        return {"path": str(source), "presentation_only": True,
                "note": "Save Blender edits as .blend; canonical patient edits require the original import source"}

    def save_scene(self, path: str) -> dict:
        """Save a new .blend presentation file; it is separate from a patient bundle."""
        target = self.session.output_path(path)
        if target.suffix != ".blend":
            raise ValueError("Blender scene must end in .blend")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.execute(f"import bpy; bpy.ops.wm.save_as_mainfile(filepath={str(target)!r})")
        return {"path": str(target)}
