# Anatomy references

Six CC BY 4.0 reference recipes for shape-guided agent editing. `catalog.json` lists
their descriptions and metadata paths. These are separate from tissue texture squares.

| Location | Contents |
| --- | --- |
| `hra/blood-vasculature-{male,female}-v1.3/` | Aorta and named arteries/veins; original HRA scene hierarchy |
| `hra/large-intestine-{male,female}-v1.3/` | Colon regions, caecum, rectum, ileocecal valve and appendix |
| `voxel-man/aorta-male-15882019/` | Label-derived aorta; major branches are not separately labeled |
| `voxel-man/colon-male-15882019/` | Label-derived colon, caecum and rectum |

Each folder contains `metadata.json`, `ATTRIBUTION.txt`, and a
structure-name/label mapping. Model files are prepared on demand in a local cache. GLBs use meters and Y-up; Blender handles glTF axis
conversion on import. Do not apply an additional axis rotation after importing.
Register references to the target patient; their original frames are not patient alignment.

HRA recipes download the original versioned GLBs and verify their SHA-256 checksums.
VOXEL-MAN recipes download the pinned label archive, verify its published checksum,
and extract the selected anatomy into GLB using the recorded smoothing and rotation.
Source records are retained in `provenance/`, and the license in `licenses/`.
Keep this directory together so relative metadata links remain valid.

Use references to guide completion while preserving the target's observed geometry.
The references themselves may have disconnected parts. A node selection is not a
connectivity guarantee, and the branch flag selects only the named major systemic branches.
Record generated completion separately from observed anatomy and carry reference
attribution, license, source, and change notices into derived packages.

Only verified HRA and VOXEL-MAN sources are included. The repository stores the
small manifests, label selections, license evidence, and conversion recipes.

The default cache is `$XDG_CACHE_HOME/i4h/references`, falling back to
`~/.cache/i4h/references`. Set `I4H_REFERENCE_CACHE` to choose another directory.
`load_references(..., cache=path)` also accepts an explicit cache location.
A first VOXEL-MAN request downloads approximately 395 MB. Both organs reuse that
archive. Generated assets are keyed by recipe, selected labels, and converter
versions. Completed cache entries work offline and are checksum-verified on reuse.
Partial downloads and conversions are never treated as complete.
Downloads retry transient failures up to three times and validate byte ranges
and the pinned checksum before caching.

Existing local metadata-plus-GLB assets remain supported. Copied editing workspaces
and derived outputs retain attribution and source provenance.
