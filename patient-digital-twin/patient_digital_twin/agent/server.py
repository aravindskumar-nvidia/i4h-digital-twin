# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MCP tools for native clients and SDK-launched agent sessions."""
from __future__ import annotations

import argparse
import contextlib
import sys
from functools import wraps
from threading import RLock

from mcp.server.fastmcp import FastMCP, Image

from .blender import BlenderBridge
from .session import PatientSession

INSTRUCTIONS = """You operate an I4H patient digital twin through the patient MCP tools.
HumanBody is authoritative. Use load_patient with segmentation, meshes, nvsegment,
or nvgenerate, then inspect_patient. A missing model is an error, not permission to
switch datasets or inference backends. Request only the anatomy and centerlines
the user asked for. Omit names to import all supported anatomy.
Use blender_show to view the current body, blender_screenshot to inspect it, and
blender_execute for Blender Python edits. Only edit objects in the collection
returned by blender_show; keep unrelated objects intact. Geometry is local XYZ
meters and object matrices are body/world poses in meters. Never rescale the scene
to millimeters. Do not change anatomy:name tags, add/delete patient objects, or
change CT registration. Leave Edit Mode before blender_apply_edits. This validates
edits and clears stale centerlines after geometry changes. Recompute requested
graphs after such edits. Scale/shear matrices must be applied to mesh vertices in
Blender before committing. Materials and lighting are presentation-only: save
them as .blend, not patient metadata. Refresh Blender after any patient-tool edit.
Export to a new workspace path with export_patient. Standalone USD uses current
display poses; bundles retain imported placement and reject differing display
poses. Existing USD is not a patient round-trip format: load the original source
recipe to edit a patient. Do not replace or flatten existing bundles in Blender.
Report paths, structures, CT frame/units, and checks; distinguish static anatomy
from a physics simulation. Do not claim a viewport was inspected without an image.
"""

def build_server(workspace: str, exchange_dir: str | None = None, *, headless: bool = False):
    """Create one serialized patient session; optional MCP imports stay at the boundary."""
    session = PatientSession(workspace)
    blender = BlenderBridge(session, exchange_dir, headless=headless)
    lock = RLock()

    @contextlib.asynccontextmanager
    async def lifespan(server):
        try:
            yield
        finally:
            blender.close()

    server = FastMCP("patient-digital-twin", instructions=INSTRUCTIONS, lifespan=lifespan)

    def register(name, function):
        @wraps(function)
        def call(*args, **kwargs):
            with lock, contextlib.redirect_stdout(sys.stderr):
                return function(*args, **kwargs)
        server.add_tool(call, name=name)

    register("load_patient", session.load_patient)
    register("inspect_patient", session.inspect_patient)
    register("set_visibility", session.set_visibility)
    register("extract_centerlines", session.extract_centerlines)
    register("export_patient", session.export_patient)
    register("blender_show", blender.show)
    register("blender_open_usd", blender.open_usd)
    register("blender_execute", blender.execute)
    register("blender_apply_edits", blender.apply_edits)
    register("blender_save_scene", blender.save_scene)

    @server.tool()
    def blender_screenshot(path: str | None = None) -> Image:
        """Return the viewport image; pass a new workspace .png path to keep it after the session."""
        with lock, contextlib.redirect_stdout(sys.stderr):
            return Image(path=blender.screenshot(path))

    return server


def main(argv: list[str] | None = None) -> int:
    """Serve patient and Blender tools over stdio for either SDK or native clients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--headless", action="store_true", help="Run Blender on a virtual display (Linux; requires Xvfb and xauth)")
    parser.add_argument("--exchange-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Preserve a protocol stream, then direct fd 1 (also inherited by model
    # subprocesses) to stderr. Tool wrappers redirect Python-level prints too.
    import os
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = protocol
    server = build_server(args.workspace, args.exchange_dir, headless=args.headless)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
