"""Complete pre-App020 sources/input contracts; not an executed experiment."""
import hashlib
import json
from pathlib import Path

from training import tron1_axle_weight_trial as trial

ROOT = Path(__file__).resolve().parents[1]


def test_complete_sources_and_original_scope_pinned_before_application():
    p = json.loads((ROOT / "config/car_drift_axle_weight_production.json").read_text())
    assert p["trial_id"] == "trial020" and p["baseline_trial"] == "trial019"
    assert p["status_at_predeclaration"] == "complete_public_sources_no_trial020_application_or_learning"
    for name, expected in {**p["expected_historical_source_sha256"], **p["expected_new_source_sha256"]}.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for key in ("runtime_fixture_entrypoint", "training_entrypoint", "evaluation_entrypoint"):
        assert p[key] in p["expected_new_source_sha256"]
    assert hashlib.sha256((ROOT / p["task_profile"]).read_bytes()).hexdigest() == p["task_profile_sha256"]
    checked = trial.load_axle_weight_profile(ROOT / p["task_profile"])
    held = trial.baseline_tight_profile()
    normalized = dict(checked)
    normalized["description"] = held["description"]
    normalized[trial.AXLE_KEY] = dict(checked[trial.AXLE_KEY], weight=1.0)
    assert trial._dense()._same_typed(normalized, held)
    assert p["reward_weight"] == 4.0
    assert p["training"]["resume_checkpoint_sha256"] == "93bebee533bd59b8a7d707a1cd15b83bdff6c02663910afe59f69df91cd71780"
    assert p["training"]["additional_environment_steps"] == 600*2048*24
    assert p["held"]["all_original_behavior_gates"] is True
    assert p["held"]["contact_gaps_bridged"] is False
    assert p["learning_rate_resume"]["scalar_restored"] == 1e-5
    old = p["profile_only_predeclaration_preserved"]
    assert hashlib.sha256((ROOT / old["path"]).read_bytes()).hexdigest() == old["sha256"]
    assert old["application_or_learning_started_at_that_stage"] is False
    assert p["unchanged"]["contact_geometry_margin_or_gap_changed"] is False
    assert p["unchanged"]["new_teacher_or_reference"] is False
    assert p["unchanged"]["new_observation_or_action"] is False
