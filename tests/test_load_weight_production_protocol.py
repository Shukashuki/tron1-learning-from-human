"""Pre-App022 source and single-factor contract; not runtime evidence."""
import hashlib
import json
from pathlib import Path

from training import tron1_load_weight_trial as trial

ROOT = Path(__file__).resolve().parents[1]


def test_complete_public_sources_and_original_scope_pinned():
    protocol = json.loads((ROOT / "config/car_drift_load_weight_production.json").read_text())
    assert protocol["trial_id"] == "trial022"
    assert protocol["baseline_trial"] == "trial021"
    assert protocol["status_at_predeclaration"] == "revised_after_failed_fixture_no_trial022_learning_or_behavior"
    assert protocol["prior_failed_fixture"]["ppo_learning_started"] is False
    assert protocol["prior_failed_fixture"]["behavior_evaluated"] is False
    assert "actual RewardManager term has weight -2" in protocol["sampler_cache_metadata"]
    assert protocol["scientific_change"]["field"] == "wheel_ground_load_reserve.weight"
    assert protocol["scientific_change"]["from"] == -1.0
    assert protocol["scientific_change"]["to"] == -2.0
    for name, expected in {
        **protocol["expected_historical_source_sha256"],
        **protocol["expected_new_source_sha256"],
    }.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for key in ("runtime_fixture_entrypoint", "training_entrypoint", "evaluation_entrypoint"):
        assert protocol[key] in protocol["expected_new_source_sha256"]
    assert hashlib.sha256((ROOT / protocol["task_profile"]).read_bytes()).hexdigest() == protocol["task_profile_sha256"]
    profile = trial.load_load_weight_profile(ROOT / protocol["task_profile"])
    baseline = trial.baseline_axle_mid_profile()
    profile["description"] = baseline["description"]
    profile[trial.LOAD_KEY]["weight"] = -1.0
    assert trial.previous._dense()._same_typed(profile, baseline)
    assert protocol["training"]["additional_environment_steps"] == 600 * 2048 * 24
    assert protocol["training"]["resume_checkpoint_sha256"] == "93bebee533bd59b8a7d707a1cd15b83bdff6c02663910afe59f69df91cd71780"
    assert protocol["held"]["all_original_behavior_gates"] is True
    assert protocol["held"]["contact_gaps_bridged"] is False
    assert protocol["unchanged"]["contact_geometry_margin_or_gap_changed"] is False
    assert protocol["unchanged"]["new_teacher_or_reference"] is False
    assert protocol["unchanged"]["new_observation_or_action"] is False
