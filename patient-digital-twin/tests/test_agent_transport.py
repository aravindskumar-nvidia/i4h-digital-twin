# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real MCP transport checks; model services are never called."""
import asyncio
import sys
from pathlib import Path

import pytest


def test_stdio_mcp_load_edit_export(tmp_path):
    pytest.importorskip('mcp')
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    sample = Path(__file__).resolve().parents[1] / 'examples/data/s0011/segmentations'

    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=[
            '-m', 'patient_digital_twin.agent.server', '--workspace', str(tmp_path),
        ])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = await client.list_tools()
                assert {'load_patient', 'blender_show', 'blender_apply_edits', 'blender_screenshot'} <= {x.name for x in tools.tools}
                result = await client.call_tool('load_patient', {'source': 'segmentation',
                                                'input_path': str(sample), 'names': ['aorta']})
                assert not result.isError, result
                result = await client.call_tool('set_visibility', {'names': ['aorta'], 'enabled': False})
                assert not result.isError
                result = await client.call_tool('export_patient', {'path': 'hidden.usdc'})
                assert not result.isError, result
                result = await client.call_tool('export_patient', {'path': 'hidden.usdc'})
                assert result.isError
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(tmp_path / 'hidden.usdc'))
        assert UsdGeom.Imageable(stage.GetPrimAtPath('/HumanBody/Anatomy/aorta')).ComputeVisibility() == 'invisible'

    asyncio.run(scenario())
