# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent editing options; importing this package starts no session and needs no SDK."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..references import resolve_references
from .prompts import build_prompt

BACKENDS = ('codex', 'claude')
BLENDER_MODES = ('auto', 'virtual', 'existing')


@dataclass(frozen=True, kw_only=True)
class SessionOptions:
    """Backend, model, Blender mode and limits for one agent session.

    ``max_turns`` and ``budget_usd`` apply to Claude, ``token_budget`` to Codex; unset
    limits use the backend defaults in ``agent.backends``.
    """
    backend: str = 'codex'
    model: str | None = None
    blender_mode: str = 'auto'
    timeout_s: float = 900
    max_turns: int | None = None
    budget_usd: float | None = None
    token_budget: int | None = None
    max_repairs: int = 2

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f'Unsupported agent backend: {self.backend}')
        if self.blender_mode not in BLENDER_MODES:
            raise ValueError(f'Unsupported Blender mode: {self.blender_mode}')
        if self.backend == 'codex' and (self.max_turns is not None or self.budget_usd is not None):
            raise ValueError('Codex uses token_budget; max_turns and budget_usd apply to Claude')
        if self.backend == 'claude' and self.token_budget is not None:
            raise ValueError('token_budget applies to Codex')
        for value in (self.timeout_s, self.max_turns, self.budget_usd, self.token_budget):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError('Session limits must be positive and finite')
        for value in (self.max_turns, self.token_budget):
            if value is not None and int(value) != value:
                raise ValueError('Turn and token limits must be integers')
        if self.max_repairs < 0 or int(self.max_repairs) != self.max_repairs:
            raise ValueError('max_repairs must be a nonnegative integer')


@dataclass(frozen=True, kw_only=True)
class AgentOptions(SessionOptions):
    """An agent task (a named operation or free prompt) plus its session options."""
    prompt: str = ''
    task: str | None = None
    targets: tuple[str, ...] = ()
    anatomy_references: Mapping[str, str | Path] = field(default_factory=dict)
    include_major_branches: bool = False
    fragment_area_mm2: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        build_prompt(self.task, self.targets, self.prompt, references=self.anatomy_references,
                     major_branching=self.include_major_branches)
        if not np.isfinite(self.fragment_area_mm2) or self.fragment_area_mm2 < 0:
            raise ValueError('Fragment area must be finite and nonnegative')
        resolve_references(self.anatomy_references, self.include_major_branches)
