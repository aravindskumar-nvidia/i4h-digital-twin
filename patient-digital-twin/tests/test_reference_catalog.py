# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single anatomy reference catalog, its compatibility forms and generated files."""
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest
from patient_digital_twin import _reference_assets
from patient_digital_twin.references import (
    catalog_reference,
    load_catalog,
    load_references,
    write_reference_files,
)

ASSETS = Path(__file__).parents[1] / 'assets/anatomy'
COLON = 'voxel-man-colon-male-15882019'


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    """Stand in for the downloaded or converted model; no network access."""
    model = tmp_path / 'prepared.glb'
    model.write_bytes(b'prepared model')
    calls = []

    def materialize(metadata, directory, cache=None, *, labels=None):
        calls.append(dict(metadata=metadata, labels=labels))
        return model, dict(metadata, sha256=hashlib.sha256(b'prepared model').hexdigest())
    monkeypatch.setattr(_reference_assets, 'materialize', materialize)
    return calls


def test_repository_keeps_one_catalog_with_license_and_source_evidence_only():
    files = {path.relative_to(ASSETS).as_posix() for path in ASSETS.rglob('*') if path.is_file()}
    assert files == {'.gitignore', 'README.md', 'catalog.json', 'licenses/CC-BY-4.0.txt',
                     'provenance/hra-library.yaml', 'provenance/voxel-man-record.json'}


def test_every_catalog_entry_expands_to_complete_verified_metadata():
    catalog = load_catalog()
    assert len(catalog['references']) == 6
    for reference_id, entry in catalog['references'].items():
        metadata, directory, structures = catalog_reference(reference_id)
        assert metadata['id'] == reference_id and metadata['license']['spdx'] == 'CC-BY-4.0'
        assert metadata['units'] == 'meters' and metadata['up_axis'] == 'Y'
        for key in ('attribution', 'changes', 'description', 'limitations', 'preparation'):
            assert metadata[key], (reference_id, key)
        assert metadata['source']['provider'] == catalog['sources'][entry['source_ref']]['source']['provider']
        assert metadata['source']['url'].startswith('https://')
        assert (directory / metadata['license']['text']).is_file()
        assert (directory / metadata['source']['evidence']).is_file()
        if 'rows' in structures:
            nodes = {row[structures['columns'].index('node_name')] for row in structures['rows']}
            selected = {node for group in metadata['selections'].values() for node in group}
            assert selected <= nodes, reference_id
    hra, _, _ = catalog_reference('hra-large-intestine-male-v1.3')
    assert hra['preparation'] == dict(kind='download-glb', url=hra['source']['download_url'],
                                      checksum='sha256:' + hra['sha256'])
    # Shared and entry-specific limitations both reach the agent.
    assert len(catalog_reference(COLON)[0]['limitations']) == 3


@pytest.mark.parametrize('spec', [
    COLON,
    Path(COLON),
    str(ASSETS / 'catalog.json') + '#' + COLON,
    # Paths documented before consolidation, and saved in existing pipeline checkpoints.
    ASSETS / 'voxel-man/colon-male-15882019/metadata.json',
])
def test_catalog_id_explicit_catalog_and_legacy_path_load_the_same_reference(prepared, spec):
    item = load_references({'colon': spec})['colon']
    assert item['metadata']['id'] == COLON and item['nodes'] == ['colon']
    assert prepared[0]['labels']['rectum'] == [1579, 1582, 1585]
    # The cache identity of prepared models depends on these fields staying unchanged.
    assert prepared[0]['metadata']['extraction']['labels'] == 'structures.json'
    assert prepared[0]['metadata']['anatomy'] == 'colon'


def test_structure_table_selects_nodes_without_an_explicit_selection(prepared):
    item = load_references({'rectum': 'hra-large-intestine-female-v1.3'})['rectum']
    assert item['nodes'] == ['VH_F_rectum']
    rows = list(csv.DictReader(io.StringIO(item['structures_text'])))
    assert rows[0].keys() == {'node_name', 'OntologyID', 'label'}


def test_unknown_reference_lists_available_ids(tmp_path):
    with pytest.raises(ValueError, match='hra-blood-vasculature-female-v1.3'):
        load_references({'colon': 'not-a-reference'})
    with pytest.raises(ValueError, match='Unknown anatomy reference'):
        load_references({'colon': str(ASSETS / 'catalog.json') + '#missing'})
    with pytest.raises(ValueError, match='#REFERENCE_ID'):
        load_references({'colon': ASSETS / 'catalog.json'})


def test_prepared_reference_folder_is_generated_with_complete_provenance(prepared, tmp_path):
    item = load_references({'aorta': 'hra-blood-vasculature-female-v1.3'}, include_major_branches=True)['aorta']
    folder = tmp_path / 'references/aorta'
    metadata = write_reference_files(item, folder)
    assert {path.name for path in folder.iterdir()} == {
        'reference.glb', 'LICENSE.txt', 'source.yaml', 'ATTRIBUTION.txt', 'structures.csv', 'metadata.json'}
    assert json.loads((folder / 'metadata.json').read_text()) == metadata
    assert metadata['license']['text'] == 'LICENSE.txt' and metadata['source']['evidence'] == 'source.yaml'
    assert (folder / 'LICENSE.txt').read_bytes() == (ASSETS / 'licenses/CC-BY-4.0.txt').read_bytes()
    assert (folder / 'source.yaml').read_bytes() == (ASSETS / 'provenance/hra-library.yaml').read_bytes()
    attribution = (folder / 'ATTRIBUTION.txt').read_text(encoding='utf-8')
    for expected in ('Human Reference Atlas', metadata['source']['url'], 'License: CC BY 4.0',
                     'Changes: ' + metadata['changes'], 'Katy Börner', 'Source evidence: source.yaml'):
        assert expected in attribution
    assert 'VH_F_aortic_arch' in (folder / 'structures.csv').read_text()


def test_catalog_accepts_other_organs_and_datasets(tmp_path, monkeypatch):
    model = b'another dataset'
    (tmp_path / 'liver.glb').write_bytes(model)
    (tmp_path / 'LICENSE.txt').write_text('CC BY 4.0')
    (tmp_path / 'record.json').write_text('{}')
    catalog = dict(schema_version=2, defaults=dict(units='meters', up_axis='Y'), sources={'atlas': dict(
        license=dict(spdx='CC-BY-4.0', text='LICENSE.txt'), attribution='Example Atlas authors',
        changes='None', source=dict(provider='Example Atlas', evidence='record.json'),
        limitations=['Shared limitation'])}, references={'atlas-liver': dict(
            source_ref='atlas', anatomy='liver', file='liver.glb', sha256=hashlib.sha256(model).hexdigest(),
            source=dict(url='https://example.org/liver'), limitations=['Liver limitation'],
            selections=dict(liver_main=['liver_surface']))})
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog))
    item = load_references({'liver': f'{path}#atlas-liver'})['liver']
    assert item['model'] == tmp_path / 'liver.glb' and item['nodes'] == ['liver_surface']
    assert item['metadata']['limitations'] == ['Shared limitation', 'Liver limitation']
    assert item['metadata']['source'] == dict(provider='Example Atlas', evidence='record.json',
                                              url='https://example.org/liver')
    # A project catalog can replace the repository one, so its IDs work directly.
    monkeypatch.setenv('I4H_ANATOMY_CATALOG', str(path))
    assert load_references({'liver': 'atlas-liver'})['liver']['metadata']['id'] == 'atlas-liver'
    catalog['sources']['atlas']['license']['spdx'] = 'CC-BY-NC-ND-4.0'
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='CC BY 4.0'):
        load_references({'liver': 'atlas-liver'})
