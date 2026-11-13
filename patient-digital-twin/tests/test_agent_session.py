# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Patient/Blender edits preserve registration, identity and geometry ownership."""
import copy

import nibabel as nib
import numpy as np
import pytest
from patient_digital_twin import CenterlineGraph
from patient_digital_twin.agent.session import PatientSession
from patient_digital_twin.scan_volume import from_array


@pytest.fixture
def patient(tmp_path):
    source = tmp_path / 'labels.nii.gz'
    data = np.zeros((8, 8, 8), np.uint8)
    data[2:6, 2:6, 2:6] = 1
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = [40, -20, 80]
    nib.save(nib.Nifti1Image(data, affine), source)
    mapping = tmp_path / 'labels.json'
    mapping.write_text('{"1": "liver"}')
    session = PatientSession(tmp_path / 'exports')
    session.load_patient('segmentation', str(source), labelmap=str(mapping), names=['liver'])
    session.body.attach_scan(from_array(data, affine, array_axes='ijk', world_unit='mm'))
    return session


def snapshot(session):
    return {'token': session.token, 'revision': session.revision, 'structures': [
        {'name': s.name, 'vertices': s.mesh.vertices.tolist(), 'faces': s.mesh.faces.tolist(),
         'matrix': s.local_to_world.tolist(), 'enabled': s.enabled}
        for s in session.body.anatomy.structures.values()
    ]}


def test_geometry_commit_invalidates_graph_without_changing_ct(patient):
    body = patient.body
    structure = body.anatomy.structures['liver']
    structure.centerline = CenterlineGraph(np.zeros((2, 3)), np.ones(2), np.array([[0, 1]]))
    imaging, registration = body.imaging, body.imaging.body_to_imaging.copy()
    imported_pose = structure.local_to_body.copy()
    edit = snapshot(patient)
    edit['structures'][0]['vertices'][0][0] += .001
    patient.apply_snapshot(edit)
    assert structure.centerline is None
    assert body.imaging is imaging
    np.testing.assert_array_equal(imaging.body_to_imaging, registration)
    np.testing.assert_array_equal(structure.local_to_body, imported_pose)
    path = patient.export_patient('edited.usdc')['path']
    from pxr import Usd
    stage = Usd.Stage.Open(path)
    assert stage.GetPrimAtPath('/HumanBody/Anatomy/liver').IsValid()
    assert not stage.GetPrimAtPath('/HumanBody/Anatomy/liver').GetAttribute('centerline:points')


def test_blender_float32_noop_retains_centerline(patient):
    structure = patient.body.anatomy.structures['liver']
    graph = CenterlineGraph(np.zeros((2, 3)), np.ones(2), np.array([[0, 1]]))
    structure.centerline = graph
    edit = snapshot(patient)
    edit['structures'][0]['vertices'] = np.array(edit['structures'][0]['vertices'], dtype=np.float32).tolist()
    edit['structures'][0]['enabled'] = False
    patient.apply_snapshot(edit)
    assert structure.centerline is graph and not structure.enabled
    assert structure.mesh.vertices is not None


@pytest.mark.parametrize('fault', ['stale', 'missing', 'duplicate', 'scale', 'nan', 'bad_faces'])
def test_invalid_blender_edits_leave_patient_unchanged(patient, fault):
    edit = snapshot(patient)
    before = snapshot(patient)
    if fault == 'stale':
        edit['revision'] -= 1
    elif fault == 'missing':
        edit['structures'].clear()
    elif fault == 'duplicate':
        edit['structures'].append(copy.deepcopy(edit['structures'][0]))
    elif fault == 'scale':
        edit['structures'][0]['matrix'][0][0] = 2
    elif fault == 'nan':
        edit['structures'][0]['vertices'][0][0] = float('nan')
    elif fault == 'bad_faces':
        edit['structures'][0]['faces'][0][0] = 999999
    with pytest.raises(ValueError):
        patient.apply_snapshot(edit)
    assert snapshot(patient) == before


def test_reloading_patient_rejects_old_blender_snapshot(patient, tmp_path):
    edit = snapshot(patient)
    patient.load_patient('segmentation', str(tmp_path / 'labels.nii.gz'), labelmap=str(tmp_path / 'labels.json'))
    edit['revision'] = patient.revision  # Exercise patient identity independently of revision.
    with pytest.raises(ValueError, match='Stale'):
        patient.apply_snapshot(edit)


def test_display_pose_cannot_be_silently_lost_in_bundle(patient):
    pose = patient.body.anatomy.structures['liver'].local_to_world.copy()
    pose[0, 3] += .02
    edit = snapshot(patient)
    edit['structures'][0]['matrix'] = pose.tolist()
    patient.apply_snapshot(edit)
    with pytest.raises(ValueError, match='Bundle uses imported poses'):
        patient.export_patient('bundle', 'bundle')
    patient.export_patient('posed.usdc')


def test_new_paths_and_workspace_boundary(patient, tmp_path):
    patient.export_patient('bundle', 'bundle')
    with pytest.raises(FileExistsError):
        patient.export_patient('bundle', 'bundle')
    with pytest.raises(ValueError):
        patient.export_patient('../outside.usdc')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (patient.workspace / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        patient.output_path('escape/file.usdc')
