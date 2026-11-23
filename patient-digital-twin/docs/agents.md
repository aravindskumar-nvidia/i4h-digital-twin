# Optional agent editing

Generate a patient USD or bundle normally, then optionally edit it with
Claude/Codex and [Blender MCP](https://github.com/ahujasid/blender-mcp).

From the repository root:

```bash
uv pip install -e './patient-digital-twin[pipeline,codex]'
uvx mcp-for-blender install-addon
```

Open Blender, enable its MCP add-on/server, and authenticate Codex. Then run:

```bash
patient-digital-twin-agent --backend codex \
  --input output/patient_twin.yaml --output edited-patient \
  'Smooth the liver surface and give it a red material. Inspect the result.'
```

Blender must run on the same machine. The command creates a separate editing
scene and writes a new output directory; use its returned USD or manifest in
Isaac. The source and CT data are preserved. For Claude, install the `claude`
extra and use `--backend claude` with its SDK authentication configured.

See [Blender asset editing](blender.md) for supported edits and the output contract.
Navigation uses the vessel selection saved in the bundle. Older bundles select
their sole vessel automatically. For older bundles with several vessels, pass
`--vessel-names aorta ...`. Selection is checked before editing starts.
Use `--help` for session options.

Codex uses `--token-budget` (default 100000) through its native experimental
rollout budget. It uses provider budget units when reported, otherwise uncached
input plus output tokens. Budget checks follow model responses, so usage can
exceed the threshold. `--max-turns` (default 30) and `--budget-usd` (default 5)
apply only to Claude. `--timeout` applies to both backends.
