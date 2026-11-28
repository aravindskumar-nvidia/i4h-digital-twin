# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-user cache location for downloaded and built dependencies."""
from __future__ import annotations

import os
from pathlib import Path


def cache_directory(name: str) -> Path:
    """Return ``$XDG_CACHE_HOME/i4h/<name>``, using ``~/.cache`` when the variable is unset or empty."""
    return Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'i4h' / name
