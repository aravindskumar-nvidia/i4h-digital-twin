# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional SDK editing configuration; importing this module starts no session."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AgentOptions:
    prompt: str = ''
    backend: str = 'codex'
    model: str | None = None
    blender_mode: str = 'auto'
    timeout_s: float = 900
    max_turns: int | None = None
    budget_usd: float | None = None
    token_budget: int | None = None
    max_repairs: int = 2
    anatomy_references: dict[str, Path] = field(default_factory=dict)
    include_major_branches: bool = False
    fragment_area_mm2: float = 1.0
    task: str | None = None
    targets: tuple[str, ...] = ()

    def __post_init__(self):
        from .prompts import build_prompt
        build_prompt(self.task, self.targets, self.prompt, references=self.anatomy_references,
                     major_branching=self.include_major_branches)
        if self.backend not in {'claude', 'codex'}:
            raise ValueError('Agent editing requires a prompt and a supported backend')
        if self.blender_mode not in {'auto', 'virtual', 'existing'}:
            raise ValueError('Unsupported Blender mode')
        if self.backend == 'codex' and (self.max_turns is not None or self.budget_usd is not None):
            raise ValueError('Codex uses token_budget; max_turns and budget_usd apply to Claude')
        if self.backend == 'claude' and self.token_budget is not None:
            raise ValueError('token_budget applies to Codex')
        for value in (self.timeout_s, self.max_turns, self.budget_usd, self.token_budget):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError('Session limits must be positive and finite')
        if self.max_repairs < 0 or int(self.max_repairs) != self.max_repairs:
            raise ValueError('max_repairs must be a nonnegative integer')
        for value in (self.max_turns, self.token_budget):
            if value is not None and int(value) != value:
                raise ValueError('Turn and token limits must be integers')
        from ..references import load_references
        load_references(self.anatomy_references, self.include_major_branches, self.fragment_area_mm2)
