# Anatomy references

CC BY 4.0 reference recipes for shape-guided agent editing. These are separate from
tissue texture squares.

## Files

| File | Contents |
| --- | --- |
| `catalog.json` | Every reference definition: shared source metadata and one entry per reference |
| `licenses/CC-BY-4.0.txt` | License text copied into each prepared reference |
| `provenance/hra-library.yaml` | Source record and supplied citations for the HRA library |
| `provenance/voxel-man-record.json` | Source record for the VOXEL-MAN dataset |

No model, attribution, or per-reference metadata file is stored here. They are generated
when a reference is prepared for editing and are carried into exported assets.

## Catalog entries

| ID | Contents |
| --- | --- |
| `hra-blood-vasculature-male-v1.3`, `hra-blood-vasculature-female-v1.3` | Aorta and named arteries/veins; original HRA scene hierarchy |
| `hra-large-intestine-male-v1.3`, `hra-large-intestine-female-v1.3` | Colon regions, caecum, rectum, ileocecal valve and appendix |
| `voxel-man-aorta-male-15882019` | Label-derived aorta; major branches are not separately labeled |
| `voxel-man-colon-male-15882019` | Label-derived colon, caecum and rectum |

Select a reference by ID:

```bash
--anatomy-reference colon=voxel-man-colon-male-15882019
```

`--anatomy-reference` also accepts `CATALOG.json#ID` for another catalog file and the
path of a standalone reference `metadata.json` with its GLB, license, source evidence and
`ATTRIBUTION.txt` beside it. Paths used before consolidation, such as
`assets/anatomy/voxel-man/colon-male-15882019/metadata.json`, still resolve to the same
entry, so saved commands and pipeline checkpoints keep working. IDs are looked up in this
catalog; set `I4H_ANATOMY_CATALOG` to use another catalog for IDs.

## Catalog format

`catalog.json` has three parts. An effective reference is `defaults`, then the shared
source named by `source_ref`, then the entry; later values override earlier ones, nested
objects merge, and `limitations` lists are joined.

- `defaults`: fields common to every reference (units, up axis, license, intended use).
- `sources`: metadata shared by all references from one dataset release: provider,
  version, source evidence, attribution, credits, change notice, shared limitations, and
  a shared preparation recipe when one archive serves several references.
- `references`: one entry per reference, keyed by ID: anatomy, sex, description,
  node `selections`, entry-specific limitations, and either
  - `download`: URL, SHA-256 and size of one original GLB, or
  - `file` and `sha256`: a local GLB relative to the catalog, or
  - a `preparation` recipe (own or shared).

  `structures` holds the node table (`columns` and `rows`) or label mapping (`labels`)
  and the file name it is written to when the reference is prepared.

The format does not restrict organs or datasets. Selections are keyed by target
structure (`<structure>_main`, `<structure>_regions`, `<structure>_major_branches`), and a
node table also selects any structure whose label matches. To add a dataset, add its
license text and source evidence files, one `sources` record, and one entry per
reference. Only geometry verified as CC BY 4.0 is accepted; the loader rejects other
license identifiers.

## Prepared and exported files

Preparing a reference for an editing session writes a self-contained folder:
`reference.glb`, `LICENSE.txt`, the source evidence, a generated `ATTRIBUTION.txt`
(credit, source, license, change notice), the structure listing, and `metadata.json`
with the effective metadata. Exports copy that folder without the GLB into
`refinement/references/<structure>/`, so derived packages carry attribution, license,
source and change notices without this directory.

GLBs use meters and Y-up; Blender handles glTF axis conversion on import. Do not apply an
additional axis rotation after importing. Register references to the target patient;
their original frames are not patient alignment.

HRA entries download the original versioned GLBs and verify their SHA-256 checksums.
VOXEL-MAN entries download the pinned label archive, verify its published checksum,
and extract the selected anatomy into GLB using the recorded smoothing and rotation.

Use references to guide completion while preserving the target's observed geometry.
The references themselves may have disconnected parts. A node selection is not a
connectivity guarantee, and the branch flag selects only the named major systemic branches.
Record generated completion separately from observed anatomy.

## Cache

The default cache is `$XDG_CACHE_HOME/i4h/references`, falling back to
`~/.cache/i4h/references`. Set `I4H_REFERENCE_CACHE` to choose another directory.
`load_references(..., cache=path)` also accepts an explicit cache location.
A first VOXEL-MAN request downloads approximately 395 MB. Both organs reuse that
archive. Generated assets are keyed by recipe, selected labels, and converter
versions; consolidation did not change those keys, so existing caches remain valid.
Completed cache entries work offline and are checksum-verified on reuse.
Partial downloads and conversions are never treated as complete.
Downloads retry transient failures up to three times and validate byte ranges
and the pinned checksum before caching.
