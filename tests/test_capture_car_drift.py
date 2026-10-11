"""Offline checks for car-demo handedness, reconstructed clocks, and validation."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from capture_car_drift import decode_csv, wrap


HEADERS = [
    "world_x", "world_y", "world_heading", "world_vx", "world_vy",
    "slip_angle", "yaw_rate", "steer", "throttle", "hand_brake", "brake",
    "world_z", "world_vz", "pitch",
]


def moving_rows():
    """Three consistent CARLA +Y samples, separated by 5 ms at 20 m/s."""
    return [
        dict(zip(HEADERS, [
            2.0, y, 90.0, 0.0, 20.0, 0.0, 0.0,
            0.0, 0.0, "False", 0.0, 0.02, 0.0, 0.0,
        ]))
        for y in (5.0, 5.1, 5.2)
    ]


def payload(rows, headers=HEADERS):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def test_constant_forward_motion_reflects_carla_y_and_yaw_to_right_handed_frame():
    clip = decode_csv(payload(moving_rows()))
    np.testing.assert_allclose(clip["position_xy_m"], [[2., -5.], [2., -5.1], [2., -5.2]])
    np.testing.assert_allclose(clip["velocity_xy_m_s"], [[0., -20.]] * 3)
    np.testing.assert_allclose(clip["yaw_rad"], -np.pi / 2.)
    np.testing.assert_allclose(clip["beta_rad"], 0., atol=1e-12)
    np.testing.assert_allclose(clip["time_s_estimated"], [0., .005, .01], atol=1e-12)
    np.testing.assert_array_equal(clip["unique_pose_indices"], [0, 1, 2])


def test_duplicate_pose_rows_preserve_controls_and_have_no_invented_elapsed_time():
    rows = moving_rows()
    duplicate = dict(rows[1], steer=.6, throttle=.4)
    rows.insert(2, duplicate)
    clip = decode_csv(payload(rows))
    assert clip["position_xy_m"].shape == (4, 2)
    np.testing.assert_allclose(clip["time_s_estimated"], [0., .005, .005, .01], atol=1e-12)
    np.testing.assert_array_equal(clip["unique_pose_indices"], [0, 1, 3])
    np.testing.assert_allclose(clip["controls"][2], [.6, .4, 0., 0.])


def test_sideslip_uses_velocity_direction_relative_to_body_heading():
    rows = moving_rows()
    for row in rows:
        row.update(world_heading=80., slip_angle=10., yaw_rate=30.)
    clip = decode_csv(payload(rows))
    np.testing.assert_allclose(clip["beta_rad"], -np.pi / 18., atol=1e-12)
    np.testing.assert_allclose(clip["reported_beta_rad"], -np.pi / 18., atol=1e-12)
    np.testing.assert_allclose(clip["logged_yaw_rate_rad_s"], -np.pi / 6., atol=1e-12)


def test_heading_is_unwrapped_before_yaw_differentiation():
    rows = moving_rows()
    for row, heading in zip(rows, (179., -179., -177.)):
        row["world_heading"] = heading
    clip = decode_csv(payload(rows))
    np.testing.assert_allclose(np.rad2deg(clip["yaw_rad"]), [-179., -181., -183.])


def test_reported_beta_is_reflected_and_wrapped_independently_of_computed_beta():
    rows = moving_rows()
    for row in rows:
        row["slip_angle"] = 350.
    clip = decode_csv(payload(rows))
    np.testing.assert_allclose(np.rad2deg(clip["reported_beta_rad"]), 10., atol=1e-10)
    np.testing.assert_allclose(clip["beta_rad"], 0., atol=1e-12)


def test_controls_keep_carla_semantics_and_the_documented_column_order():
    rows = moving_rows()
    for row in rows:
        row.update(steer=-.7, throttle=.8, brake=.3, hand_brake="True")
    clip = decode_csv(payload(rows))
    np.testing.assert_allclose(clip["controls"], [[-.7, .8, .3, 1.]] * 3)


def test_clock_uses_velocity_projection_and_keeps_nonuniform_observation_intervals():
    rows = moving_rows()
    for row, x, y in zip(rows, (0., .03, .09), (0., .04, .12)):
        row.update(world_x=x, world_y=y, world_vx=3., world_vy=4.)
    clip = decode_csv(payload(rows))
    np.testing.assert_allclose(clip["time_s_estimated"], [0., .01, .03], atol=1e-12)


def test_clock_uses_mean_endpoint_velocity_instead_of_start_velocity():
    rows = moving_rows()
    for row, y, speed in zip(rows, (0., .15, .4), (10., 20., 30.)):
        row.update(world_y=y, world_vy=speed)
    clip = decode_csv(payload(rows))
    np.testing.assert_allclose(clip["time_s_estimated"], [0., .01, .02], atol=1e-12)


@pytest.mark.parametrize("missing", HEADERS)
def test_each_documented_source_column_is_required(missing):
    with pytest.raises(ValueError):
        decode_csv(payload(moving_rows(), [key for key in HEADERS if key != missing]))


@pytest.mark.parametrize("row_count", [0, 1, 2])
def test_rejects_trajectories_too_short_to_check_motion(row_count):
    with pytest.raises(ValueError):
        decode_csv(payload(moving_rows()[:row_count]))


def test_rejects_empty_payload():
    with pytest.raises(ValueError):
        decode_csv(b"")


@pytest.mark.parametrize("column", [key for key in HEADERS if key != "hand_brake"])
@pytest.mark.parametrize("invalid", ["nan", "inf", "-inf"])
def test_rejects_nonfinite_values_in_every_numeric_source_channel(column, invalid):
    rows = moving_rows()
    rows[1][column] = invalid
    with pytest.raises(ValueError):
        decode_csv(payload(rows))


@pytest.mark.parametrize("invalid", ["true", "false", "1", "0", "yes", "", "None"])
def test_hand_brake_requires_the_original_explicit_boolean_encoding(invalid):
    rows = moving_rows()
    rows[1]["hand_brake"] = invalid
    with pytest.raises(ValueError):
        decode_csv(payload(rows))


@pytest.mark.parametrize("column,invalid", [
    ("steer", -1.01), ("steer", 1.01),
    ("throttle", -.01), ("throttle", 1.01),
    ("brake", -.01), ("brake", 1.01),
])
def test_rejects_out_of_range_carla_controls(column, invalid):
    rows = moving_rows()
    rows[1][column] = invalid
    with pytest.raises(ValueError):
        decode_csv(payload(rows))


@pytest.mark.parametrize("speed", [0., .01, .099])
def test_rejects_low_speed_rows_where_sideslip_is_undefined_or_unreliable(speed):
    rows = moving_rows()
    rows[1]["world_vy"] = speed
    with pytest.raises(ValueError):
        decode_csv(payload(rows))


def test_rejects_motion_opposing_reported_velocity_instead_of_clipping_time():
    rows = moving_rows()
    for row in rows:
        row["world_vy"] = -20.
    with pytest.raises(ValueError):
        decode_csv(payload(rows))


def test_rejects_pure_rotation_without_position_based_time_support():
    rows = moving_rows()
    rows[1] = dict(rows[0], world_heading=95., yaw_rate=5.)
    with pytest.raises(ValueError):
        decode_csv(payload(rows))


def test_wrap_uses_the_half_open_signed_angle_interval():
    values = np.array([-3. * np.pi, -np.pi, 0., np.pi, 3. * np.pi, 2. * np.pi + .25])
    np.testing.assert_allclose(wrap(values), [-np.pi, -np.pi, 0., -np.pi, -np.pi, .25])
    assert wrap(np.pi) == pytest.approx(-np.pi)
