# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for optional editing, operation order and portable outputs."""
import hashlib
import json

import pytest


def test_packaged_texture_defaults():
    from patient_digital_twin.references import default_texture_reference
    for name in ('aorta', 'colon', 'liver', 'gallbladder', 'bladder',
                 'left kidney', 'right kidney', 'pancreas', 'spleen', 'stomach'):
        reference = default_texture_reference(name)
        reference.validate()
        assert reference.image.is_file()
    assert default_texture_reference('colon').metadata['surface'] == 'mucosa'
    with pytest.raises(ValueError, match='No default texture'):
        default_texture_reference('unknown')



def test_tissue_family_defaults_are_explicit_and_preserve_exact_overrides():
    from patient_digital_twin.anatomy import CATALOG
    from patient_digital_twin.references import default_texture_reference
    from PIL import Image

    examples = {'lung_upper_lobe_right': 'lung', 'atrial_appendage_left': 'heart',
                'iliopsoas_left': 'skeletal_muscle', 'rib_left_3': 'cortical_bone',
                'small_bowel': 'bowel_serosa'}
    for organ, family in examples.items():
        reference = default_texture_reference(organ)
        assert reference.metadata['selection'] == {
            'structure': organ, 'default': family, 'match': 'tissue_family'}
        assert reference.metadata['synthetic']
        with Image.open(reference.image) as image:
            assert image.width == image.height
    assert default_texture_reference('colon').metadata['selection']['default'] == 'colon'
    for unsupported in ('brain', 'costal_cartilages', 'kidney_cyst_left', 'lung_fake'):
        with pytest.raises(ValueError, match='No default texture'):
            default_texture_reference(unsupported)
    directory = default_texture_reference('heart').image.parent
    records = json.loads((directory / 'defaults.json').read_text())
    assigned = []
    for record in records.values():
        names = record.get('applies_to', [])
        assert set(names) <= CATALOG.keys()
        assigned.extend(names)
    assert len(assigned) == len(set(assigned))



def test_non_aortic_major_branches_require_explicit_selection(tmp_path):
    from patient_digital_twin.references import load_references
    (tmp_path / 'reference.glb').write_bytes(b'fixture')
    for name in ('LICENSE.txt', 'source.json', 'ATTRIBUTION.txt'):
        (tmp_path / name).write_text('fixture')
    metadata = dict(file='reference.glb', units='meters', up_axis='Y',
                    sha256=hashlib.sha256(b'fixture').hexdigest(),
                    license={'spdx': 'CC-BY-4.0', 'text': 'LICENSE.txt'},
                    source={'evidence': 'source.json'}, structures='structures.json',
                    selections={'inferior_vena_cava_main': ['ivc'],
                                'inferior_vena_cava_major_branches': ['renal_vein_junction']})
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(metadata))
    assert load_references({'inferior_vena_cava': path})['inferior_vena_cava']['branches'] == []
    result = load_references({'inferior_vena_cava': path}, include_major_branches=True)
    assert result['inferior_vena_cava']['nodes'] == ['ivc', 'renal_vein_junction']
    del metadata['selections']['inferior_vena_cava_major_branches']
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='no named major branches for inferior_vena_cava'):
        load_references({'inferior_vena_cava': path}, include_major_branches=True)
