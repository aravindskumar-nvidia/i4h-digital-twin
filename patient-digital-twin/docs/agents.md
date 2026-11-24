# Optional agent editing

Generate a patient USD or bundle normally, then optionally edit it with
Claude/Codex and [Blender MCP](https://github.com/ahujasid/blender-mcp).

In the active environment from the [patient install](../README.md#install), run from the repository root:

```bash
uv pip install -e './patient-digital-twin[editing]'
patient-digital-twin setup editing
```

Authenticate Codex, then run (Blender starts automatically if needed):

```bash
patient-digital-twin-agent --backend codex \
  --input output/patient_twin.yaml --output edited-patient \
  'Smooth the liver surface and give it a red material. Inspect the result.'
```

The default reuses a compatible local Blender MCP session, or launches Blender
with the upstream add-on. With no display on Linux it uses Xvfb. To always use a
private virtual desktop, add `--blender-mode virtual` (Ubuntu prerequisites:
`sudo apt install xvfb xauth`). This runs a GUI on a virtual display so viewport
screenshots work; it is not Blender's background mode or a VNC server.

Existing Blender must meet the pinned MCP's Blender 3.0+ requirement and support
the USD options used by this workflow. If missing on Linux x86_64, installation of
the verified official 5.1.2 archive requires Y/N confirmation, or
`I4H_INSTALL_BLENDER=1` for non-interactive runs. It is installed under
`$XDG_CACHE_HOME/i4h/blender` (default `~/.cache/i4h/blender`).
Set `BLENDER_EXECUTABLE` to select an installation; other platforms need Blender
on PATH or this override. Managed startup is tested on Linux.
The matching MCP server/add-on is pinned to 2.3.0.
Only processes started by the command are closed on completion or failure;
existing Blender sessions and preferences are left intact. `--blender-mode existing`
disables installation/startup and is the default when using `--mcp-config`.

Scene and output validation failures are returned to the agent for up to two repair attempts.
Use `--max-repairs N` to change this (`0` disables repairs); the conversation, remaining
budget and timeout carry across attempts.

Failed runs retain their workspace (including the pre-export `edited.blend`
checkpoint and import/export diagnostics in `mcp.log`); the error prints its path. Successful fresh edits remove temporary files.

Use a retained workspace to retry publication or continue agent editing.

```bash
patient-digital-twin-agent finalize --workspace output/.asset-edit-xxxxx --output recovered-patient
patient-digital-twin-agent resume --workspace output/.asset-edit-xxxxx \
  --output repaired-patient --backend codex \
  'Repair the open aorta surface while preserving the existing edits.'
```

`finalize` uses the exported USD, re-exporting `edited.blend` if newer or if the
USD is missing. `resume` opens that checkpoint for further agent edits. Both require
a new output directory and the referenced source files, and retain the workspace.

The returned USD or manifest is the new Isaac input; source and CT data are
preserved. For Claude, use `--backend claude`; its SDK is included in `editing`.

See [Blender asset editing](blender.md) for supported edits and the output contract.
Navigation uses the vessel selection saved in the bundle. Older bundles select
their sole vessel automatically. For older bundles with several vessels, pass
`--vessel-names aorta ...`. Selection is checked before Blender starts.
Use `--help` for session options.

Codex uses `--token-budget` (default 100000) through its native experimental
rollout budget. It uses provider budget units when reported, otherwise uncached
input plus output tokens. Budget checks follow model responses, so usage can
exceed the threshold. `--max-turns` (default 30) and `--budget-usd` (default 5)
apply only to Claude. `--timeout` applies to both backends.
