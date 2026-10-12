import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("contact_breaks_test", ROOT / "scripts/audit_drift_contact_breaks.py")
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def evidence():
    return [np.arange(5) * .005, np.ones(5), np.deg2rad(np.full(5, 10.)),
        np.ones((5, 2), dtype=int), np.full((5, 2), .05), np.full((5, 2), 50.)]


def test_support_and_slip_only_failures_kept_separate():
    args = evidence(); args[3][1, 1] = 0; args[5][1, 1] = 0
    args[4][1] = 0; args[4][3] = .03
    result = d.decompose(*args)
    assert result["support_failure_samples_while_moving_drift"] == 1
    assert result["slip_only_failure_samples_while_supported_moving_drift"] == 1
    assert [e["sample_index"] for e in result["events"]] == [1, 3]


def test_original_speed_slip_and_sideslip_boundaries():
    args = evidence(); args[1][:] = .4; args[2][:] = np.deg2rad(8.); args[4][:] = .04
    assert d.decompose(*args)["events"] == []
    args[1][0] = .399; args[2][1] = np.deg2rad(7.99); args[2][2] = np.deg2rad(170.)
    args[3][0:3] = 0
    assert d.decompose(*args)["support_failure_samples_while_moving_drift"] == 0


@pytest.mark.parametrize("change", ["bad_time", "nan", "fractional_contact", "negative_contact", "wrong_shape"])
def test_invalid_contact_archive_rejected(change):
    args = evidence()
    if change == "bad_time": args[0][2] += .001
    if change == "nan": args[5][2, 0] = np.nan
    if change == "fractional_contact": args[3] = args[3].astype(float); args[3][0, 0] = .5
    if change == "negative_contact": args[3][0, 0] = -1
    if change == "wrong_shape": args[4] = np.ones((5, 3))
    with pytest.raises(ValueError): d.decompose(*args)
