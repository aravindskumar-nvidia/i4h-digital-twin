# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API for patient digital twins."""

from .agent import AgentOptions
from .anatomy import (
    AnatomicalStructure,
    AnatomyCollection,
    Kind,
    MeshGeometry,
    SurfaceMaterial,
    System,
)
from .body import HumanBody, ImagingVolume
from .geometry import CenterlineGraph
from .importers import (
    NVGenerateImporter,
    NVSegmentImporter,
    SegmentationImporter,
    SimpleImporter,
)
from .references import TextureReference
from .surface import CleanupOptions, CleanupResult, prepare_surface
from .texturing import TextureOptions, TextureResult, TextureRuntime, texture_mesh

__all__ = [
    "AnatomicalStructure", "AnatomyCollection", "CenterlineGraph", "HumanBody", "ImagingVolume",
    "Kind", "MeshGeometry", "NVGenerateImporter", "NVSegmentImporter", "SegmentationImporter",
    "SimpleImporter", "System", "SurfaceMaterial", "AgentOptions",
    "CleanupOptions", "CleanupResult", "prepare_surface", "TextureOptions", "TextureRuntime",
    "TextureResult", "texture_mesh", "TextureReference",
]
