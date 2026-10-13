"""Strict profile-only020 predeclaration; no App/configuration or result."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from training import tron1_tight_axle_trial as previous

ROOT=Path(__file__).resolve().parents[1]


def test_only_typed_weight_and_description_change():
    proto=json.loads((ROOT/'config/car_drift_axle_weight_trial.json').read_text())
    path=ROOT/proto['task_profile']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==proto['task_profile_sha256']
    current=json.loads(path.read_text())
    baseline=previous.load_tight_axle_profile(ROOT/proto['baseline_task_profile'])
    assert previous.baseline.axle._dense()._same_typed(current['wheel_axle_orientation_tracking'],{'weight':4.0,'std':.2})
    normalized=copy.deepcopy(current);normalized['wheel_axle_orientation_tracking']['weight']=1.0
    normalized['description']=baseline['description']
    assert previous.baseline.axle._dense()._same_typed(normalized,baseline)
    assert proto['status_at_predeclaration']=='prospective_profile_only_no_trial020_application_or_learning'
    assert proto['held']['all_original_acceptance_gates'] and not proto['held']['contact_gaps_bridged']
    assert proto['training']['additional_environment_steps']==600*2048*24
    with pytest.raises(ValueError):previous.validate_tight_axle_profile(current)
