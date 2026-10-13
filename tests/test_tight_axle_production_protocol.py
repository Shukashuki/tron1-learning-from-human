"""Pre-App source/data-contract pins, not an executed019 experiment."""
import hashlib
import json
from pathlib import Path

from training import tron1_tight_axle_trial as trial

ROOT = Path(__file__).resolve().parents[1]


def test_all_complete_sources_pinned_before_any_application():
    p = json.loads((ROOT / "config/car_drift_tight_axle_production.json").read_text())
    assert p["trial_id"] == "trial019" and p["baseline_trial"] == "trial018"
    assert p["status_at_predeclaration"] == "complete_public_sources_no_trial019_application_or_learning"
    for name, expected in {**p["expected_historical_source_sha256"], **p["expected_new_source_sha256"]}.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for key in ("runtime_fixture_entrypoint", "training_entrypoint", "evaluation_entrypoint"):
        assert p[key] in p["expected_new_source_sha256"]
    assert hashlib.sha256((ROOT / p["task_profile"]).read_bytes()).hexdigest() == p["task_profile_sha256"]
    checked = trial.load_tight_axle_profile(ROOT / p["task_profile"])
    held = trial.baseline_worst_profile()
    normalized = dict(checked)
    normalized["description"] = held["description"]
    normalized[trial.AXLE_KEY] = dict(checked[trial.AXLE_KEY], std=.4)
    assert trial.baseline.axle._dense()._same_typed(normalized, held)
    assert p["training"]["resume_checkpoint_sha256"] == "93bebee533bd59b8a7d707a1cd15b83bdff6c02663910afe59f69df91cd71780"
    assert p["training"]["additional_environment_steps"] == 600*2048*24
    assert p["held"]["all_original_behavior_gates"] is True
    assert p["held"]["contact_gaps_bridged"] is False
    assert p["learning_rate_resume"]["scalar_restored"] == 1e-5
    assert p["profile_preparation_correction"]["prior_profile_rejected_by_strict_typed_validation"] is True
