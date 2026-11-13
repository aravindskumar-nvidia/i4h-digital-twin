# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate/install Blender and launch a private editing instance for a session."""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

RELEASES = "https://download.blender.org/release/Blender4.5/"
BLENDER_VERSION = '4.5.16'


def ensure_blender() -> str:
    """Find Blender, or install the pinned official 4.5 LTS Linux x64 build.

    Explicit executables win, then PATH, then our managed installation. Blender's
    published SHA-256 manifest is checked before extracting. Other platforms use
    their installed Blender; no system package manager or administrator is needed.
    """
    candidate = os.environ.get("BLENDER_EXECUTABLE", 'blender')
    if found := shutil.which(candidate):
        return found
    if 'BLENDER_EXECUTABLE' in os.environ:
        raise FileNotFoundError(f"Blender executable not found: {candidate}")
    root = Path.home() / '.cache/i4h/blender'
    folder_name = f'blender-{BLENDER_VERSION}-linux-x64'
    destination = root / folder_name
    if (destination / 'blender').is_file():
        return str(destination / 'blender')
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Install Blender 4.5 LTS for this platform and set BLENDER_EXECUTABLE")
    archive = folder_name + '.tar.xz'
    checksums = urllib.request.urlopen(RELEASES + f"blender-{BLENDER_VERSION}.sha256", timeout=30).read().decode()
    expected = next((line.split()[0] for line in checksums.splitlines()
                     if line.split()[-1].lstrip('*') == archive), None)
    if expected is None:
        raise RuntimeError("Blender archive is missing from the official checksum manifest")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".install-") as temp:
        download = Path(temp) / archive
        digest = hashlib.sha256()
        with urllib.request.urlopen(RELEASES + archive, timeout=60) as response, download.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Blender SHA-256 verification failed")
        unpacked = Path(temp) / "unpacked"
        unpacked.mkdir()
        with tarfile.open(download) as package:
            # Explicit validation also supports Python 3.10, without tar filters.
            for member in package.getmembers():
                target = (unpacked / member.name).resolve()
                if not target.is_relative_to(unpacked) or member.isdev() or member.isfifo():
                    raise ValueError("Unsafe path in Blender archive")
                if member.issym() or member.islnk():
                    link = ((target.parent if member.issym() else unpacked) / member.linkname).resolve()
                    if not link.is_relative_to(unpacked):
                        raise ValueError("Unsafe link in Blender archive")
            if hasattr(tarfile, 'data_filter'):
                package.extractall(unpacked, filter='data')
            else:
                package.extractall(unpacked)
        if not destination.exists():
            shutil.move(str(unpacked / folder_name), destination)
    return str(destination / "blender")


class BlenderRuntime:
    """Own a private Blender process and close only that process on server exit."""

    def __init__(self, directory: Path, *, headless: bool = False):
        self.directory = directory
        self.headless = headless
        self.process: subprocess.Popen | None = None

    def start(self) -> int:
        """Launch Blender with the packaged bridge, returning its private port."""
        prefix = []
        if self.headless:
            xvfb = shutil.which('xvfb-run')
            if platform.system() != 'Linux' or xvfb is None or shutil.which('xauth') is None:
                raise RuntimeError('--headless requires Linux with xvfb-run and xauth installed')
            prefix = [xvfb, '-a', '--server-args=-screen 0 1280x960x24 -nolisten tcp']
        elif platform.system() == 'Linux' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('Blender needs a desktop display; forward DISPLAY/WAYLAND_DISPLAY or use --headless with Xvfb')
        executable = ensure_blender()
        ready = self.directory / "ready.json"
        ready.unlink(missing_ok=True)
        script = Path(__file__).with_name("blender_server.py")
        with (self.directory / "blender.log").open('w') as log:
            self.process = subprocess.Popen(
                prefix + [executable, '--factory-startup', '--python', str(script), '--', '--ready', str(ready)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name == 'posix'),
            )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if ready.is_file():
                import json
                return int(json.loads(ready.read_text())['port'])
            if self.process.poll() is not None:
                raise RuntimeError(f"Blender exited; see {self.directory / 'blender.log'}")
            time.sleep(0.1)
        self.close()
        raise TimeoutError(f"Blender startup timed out; see {self.directory / 'blender.log'}")

    def close(self) -> None:
        """Stop our private process without touching any existing Blender session."""
        if self.process is not None and self.process.poll() is None:
            try:
                if os.name == 'posix':
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait(timeout=5)
            except ProcessLookupError:
                self.process.wait(timeout=5)


def main() -> int:
    """Prepare Blender alongside the optional agent integration."""
    print(ensure_blender())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
