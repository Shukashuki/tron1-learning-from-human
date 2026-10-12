"""Synthetic recorded-state/schema tests. No rollout, actor, or actual trial audit.

Base pytest runs the pure tests. Run this file with --exact-mujoco using the
retarget_sim Python for the additional MuJoCo 3.9.0 synthetic position-stage
check; it never calls mj_step/mj_forward or generates a physical episode.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wheel_clearance_audit_under_test", ROOT / "scripts/audit_drift_wheel_clearance.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(n=9):
    p = (n - 1) // 4
    d = {
        "time_s": np.arange(n) * .005, "reference_frame": np.minimum(np.arange(n) // 4, 287),
        "root_pos": np.tile([0., 0., .75], (n, 1)),
        "root_quat_wxyz": np.tile([1., 0., 0., 0.], (n, 1)),
        "root_com_lin_vel_w": np.zeros((n, 3)), "root_ang_vel_w": np.zeros((n, 3)),
        "joint_pos": np.tile(np.arange(8) * .01, (n, 1)), "joint_vel": np.zeros((n, 8)),
        "wheel_pos": np.zeros((n, 2, 3)), "qpos": np.zeros((n, 15)),
        "joint_torque_nm": np.zeros((n, 8)), "wheel_contact_force_w_n": np.zeros((n, 2, 3)),
        "wheel_ground_total_contact_force_w_n": np.zeros((n, 2, 3)),
        "wheel_ground_contact_count": np.ones((n, 2), np.int32),
        "wheel_ground_contact_force_w_n": np.tile([0., 0., 100.], (n, 2, 1)),
        "wheel_lateral_contact_slip_m_s": np.zeros((n, 2)),
        "nonwheel_ground_contact_count": np.zeros(n, int), "nonwheel_ground_contact_force_norm_n": np.zeros(n),
        "policy_time_s": np.arange(p) * .02, "reference_frame_for_action": np.zeros(p, int),
        "action_observation": np.zeros((p, 51)), "actions_raw": np.zeros((p, 8)),
        "actions_clipped": np.zeros((p, 8)), "previous_clipped_actions": np.zeros((p, 8)),
        "joint_names": np.array(audit.JOINT_NAMES), "qpos_joint_names": np.array(audit.NATIVE_JOINT_NAMES),
        "policy_dt_s": np.array(.02), "physics_dt_s": np.array(.005),
        "initial_qpos": np.zeros(15), "initial_qvel": np.zeros(14),
    }
    d["qpos"][:, :3] = d["root_pos"]
    d["qpos"][:, 3:7] = d["root_quat_wxyz"]
    order = [audit.JOINT_NAMES.index(k) for k in audit.NATIVE_JOINT_NAMES]
    d["qpos"][:, 7:] = d["joint_pos"][:, order]
    d["initial_qpos"] = d["qpos"][0].copy()
    return d


def report_fixture():
    profile = {"schema_version": 1, "task": "car_drift"}
    contract = dict(actor_dim=51, critic_dim=120, action_dim=8, decimation=4, physics_dt=.005,
                    policy_fps=50, leg_kp=500., leg_kd=10., task_profile=profile)
    report = dict(engine="MuJoCo", engine_version="3.9.0", contract=copy.deepcopy(contract),
                  terrain=None, physics_stepped=True, hidden_resets=0, root_state_writes=1,
                  completed_full_reference=True, physics_steps=1152, policy_steps=288,
                  root_com_link_offset_m=audit.ROOT_COM_LINK.tolist(), no_visual_mesh=True,
                  removed_visual_geom_count=9, model_sha256=audit.PINS["model"],
                  evaluator_source_sha256=audit.PINS["mujoco_evaluator"],
                  contact_slip_helper_sha256=audit.PINS["contact_slip"], contact={"force_threshold_n": 5.})
    return report, contract, profile


def geometry_fixture(n):
    return {"clearance_m": np.zeros((n, 2)), "clearance_rate_m_s": np.zeros((n, 2)),
            "clearance_rate_defined": np.ones((n, 2), bool),
            "wheel_link_plane_velocity_m_s": np.zeros((n, 2)),
            "manifold_count": np.zeros((n, 2), int), "manifold_min_distance_m": np.zeros((n, 2))}


class GeometryTests(unittest.TestCase):
    def call(self, *, theta=0., center=(0, 0, .2), velocity=(0, 0, 0), omega=(0, 0, 0)):
        return audit.cylinder_plane_geometry(center, [0, np.cos(theta), np.sin(theta)], .127, .025,
                                             [0, 0, 0], [0, 0, 1], velocity, omega)

    def test_zero_camber_cusp_not_zero_rate(self):
        value = self.call(velocity=(0, 0, .03), omega=(2, 0, 0))
        self.assertAlmostEqual(value["clearance_m"], .2 - .127)
        self.assertFalse(value["clearance_rate_defined"])
        self.assertEqual(value["clearance_rate_m_s"], 0.)  # explicitly masked storage sentinel
        self.assertAlmostEqual(value["axis_normal_dot_rate_s"], 2.)

    def test_positive_and_negative_camber_derivative(self):
        for theta in (-.2, .2):
            with self.subTest(theta=theta):
                vz, omega, eps = .03, .8, 1e-7
                value = self.call(theta=theta, velocity=(0, 0, vz), omega=(omega, 0, 0))
                left = self.call(theta=theta - eps * omega, center=(0, 0, .2 - eps * vz))
                right = self.call(theta=theta + eps * omega, center=(0, 0, .2 + eps * vz))
                self.assertTrue(value["clearance_rate_defined"])
                self.assertAlmostEqual(value["clearance_rate_m_s"],
                                       (right["clearance_m"] - left["clearance_m"]) / (2 * eps), places=8)
                self.assertAlmostEqual(value["support_extent_m"], .025 * abs(np.sin(theta)) + .127 * np.cos(theta))

    def test_plane_rotation_translation_and_geom_not_link_center(self):
        theta = .3
        q = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        center, plane = np.array([0., .00707, .2]), np.array([0., 0., .05])
        base = audit.cylinder_plane_geometry(center, [0, np.cos(theta), np.sin(theta)], .127, .025,
                                             plane, [0, 0, 1], [0, 0, .04], [1, 0, 0])
        changed = audit.cylinder_plane_geometry(q @ center + 1, q @ [0, np.cos(theta), np.sin(theta)], .127, .025,
                                                q @ plane + 1, q @ [0, 0, 1], q @ [0, 0, .04], q @ [1, 0, 0])
        for key in base:
            if key != "clearance_rate_defined":
                self.assertAlmostEqual(base[key], changed[key])
        # A tilted plane makes the real local center offset relevant, unlike the link origin.
        normal = np.array([0., .6, .8])
        actual = audit.cylinder_plane_geometry(center, [1, 0, 0], .127, .025, [0, 0, 0], normal, [0, 0, 0], [0, 0, 0])
        link = audit.cylinder_plane_geometry([0, 0, .2], [1, 0, 0], .127, .025, [0, 0, 0], normal, [0, 0, 0], [0, 0, 0])
        self.assertAlmostEqual(actual["clearance_m"] - link["clearance_m"], .6 * .00707)

    def test_axial_cusp_and_invalid_sizes_vectors(self):
        value = audit.cylinder_plane_geometry([0, 0, .2], [0, 0, 1], .127, .025,
                                              [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0])
        self.assertFalse(value["clearance_rate_defined"])
        self.assertAlmostEqual(value["clearance_m"], .175)
        for radius in (False, np.bool_(True), 0, -1, float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                audit.cylinder_plane_geometry([0, 0, 0], [0, 1, 0], radius, .025, [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            audit.cylinder_plane_geometry([0, 0, 0], [0, 2, 0], .127, .025, [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0])

    def test_fd_cusp_crossing_is_an_interval_not_unique_instantaneous_derivative(self):
        angles = np.array([-.01, 0., .01])
        gaps = np.array([self.call(theta=v)["clearance_m"] for v in angles])
        fd = audit.interval_derivatives([0, .005, .010], gaps)
        self.assertGreater(fd["rate"][0], 0)
        self.assertLess(fd["rate"][1], 0)
        np.testing.assert_array_equal(fd["start_time_s"], [0, .005])
        np.testing.assert_array_equal(fd["end_time_s"], [.005, .010])
        self.assertFalse(self.call()["clearance_rate_defined"])

    def test_fd_constant_clearance_and_bad_clocks(self):
        result = audit.interval_derivatives([0, .005, .010], np.full((3, 2), .0002))
        np.testing.assert_array_equal(result["rate"], np.zeros((2, 2)))
        for t in ([0, 0, .01], [0, .01, .005], [0, float("nan"), .01]):
            with self.subTest(t=t), self.assertRaises(ValueError):
                audit.interval_derivatives(t, np.ones(3))


class VelocityTests(unittest.TestCase):
    def test_nonzero_com_offset_world_omega_and_local_freejoint_axes(self):
        # Root yaw90, body-local angular coordinates and world origin translation.
        rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        offset = rot @ audit.ROOT_COM_LINK
        cross = np.array([[0, -offset[2], offset[1]], [offset[2], 0, -offset[0]], [-offset[1], offset[0], 0]])
        jac = np.zeros((6, 14))
        jac[:3, :3] = np.eye(3)
        jac[:3, 3:6] = -cross @ rot
        jac[3:, 3:6] = rot
        # Also prove known joint rates are subtracted before solving the free root.
        jac[:, 6:] = np.arange(48).reshape(6, 8) * .001
        dq = np.arange(8) * .05
        desired = np.array([.4, -.2, .1, .3, .6, -.1])
        qvel = audit.recover_generalized_velocity(jac, dq, np.arange(6, 14), np.arange(6), desired)
        np.testing.assert_allclose(jac @ qvel, desired, atol=1e-14)
        np.testing.assert_array_equal(qvel[6:], dq)
        self.assertFalse(np.allclose(qvel[:3], desired[:3]))
        self.assertFalse(np.allclose(qvel[3:6], desired[3:]))

    def test_native_non_actor_order_dof_mapping(self):
        ids = np.array([6 + audit.NATIVE_JOINT_NAMES.index(k) for k in audit.JOINT_NAMES])
        jac = np.c_[np.eye(6), np.zeros((6, 8))]
        dq = np.arange(8, dtype=float)
        qvel = audit.recover_generalized_velocity(jac, dq, ids, np.arange(6), np.arange(6, dtype=float))
        np.testing.assert_array_equal(qvel[ids], dq)
        self.assertNotEqual(qvel[9], dq[3])

    def test_bad_velocity_layout_and_nonfinite(self):
        good = np.c_[np.eye(6), np.zeros((6, 8))]
        for ids in (np.arange(5, 13), np.array([6] * 8), np.arange(6, 14).astype(float)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                audit.recover_generalized_velocity(good, np.zeros(8), ids, np.arange(6), np.zeros(6))
        good[0, 0] = np.nan
        with self.assertRaises(ValueError):
            audit.recover_generalized_velocity(good, np.zeros(8), np.arange(6, 14), np.arange(6), np.zeros(6))

    def test_position_stage_readonly_and_mutation_refusal(self):
        for mutation in (None, "qpos", "qvel", "time"):
            with self.subTest(mutation=mutation):
                data = SimpleNamespace(qpos=np.arange(15, dtype=float), qvel=np.arange(14, dtype=float), time=.4)
                def position(model, state):
                    if mutation == "time":
                        state.time += .005
                    elif mutation:
                        getattr(state, mutation)[0] += 1
                mj = SimpleNamespace(mj_fwdPosition=position)
                if mutation:
                    with self.assertRaises(ValueError):
                        audit.position_only(mj, object(), data)
                else:
                    audit.position_only(mj, object(), data)
                    self.assertEqual(data.time, .4)


class ArchiveTests(unittest.TestCase):
    def test_valid_named_qpos_is_preserved_exactly(self):
        d = fixture()
        original = {k: v.copy() for k, v in d.items()}
        result = audit.normalize_archive(d, samples=9)
        self.assertEqual(set(d), audit.MJ_KEYS)
        for key in d:
            np.testing.assert_array_equal(d[key], original[key])
        np.testing.assert_array_equal(result["qpos"], d["qpos"])
        self.assertNotIn("normalized_root_quaternion_wxyz", result)
        self.assertNotEqual(result["qpos"][0, 10], result["joint_pos"][0, 3])

    def test_tiny_quaternion_fk_storage_roundoff_accepted_qpos_unchanged(self):
        self.assertEqual(audit.QUATERNION_STORAGE_ATOL, 32 * np.finfo(np.float64).eps)
        for difference in (5 * np.finfo(np.float64).eps, audit.QUATERNION_STORAGE_ATOL):
            with self.subTest(difference=difference):
                d = fixture()
                d["root_quat_wxyz"][2, 0] += difference
                original = {k: v.copy() for k, v in d.items()}
                result = audit.normalize_archive(d, samples=9)
                np.testing.assert_array_equal(result["qpos"], original["qpos"])
                np.testing.assert_array_equal(result["root_quat_wxyz"], original["root_quat_wxyz"])
                for key in original:
                    np.testing.assert_array_equal(d[key], original[key])

    def test_larger_quaternion_mismatch_and_tiny_position_joint_mismatch_rejected(self):
        d = fixture()
        d["root_quat_wxyz"][2, 0] += 2 * audit.QUATERNION_STORAGE_ATOL
        with self.assertRaises(ValueError):
            audit.normalize_archive(d, samples=9)
        for key, index in (("root_pos", (2, 0)), ("joint_pos", (2, 0))):
            with self.subTest(key=key):
                d = fixture()
                d[key][index] += np.finfo(np.float64).eps
                with self.assertRaises(ValueError):
                    audit.normalize_archive(d, samples=9)

    def test_all_numeric_nonfinite_fields_refuse(self):
        for key, value in fixture().items():
            if value.dtype.kind not in "f":
                continue
            with self.subTest(key=key):
                d = fixture()
                d[key].flat[0] = np.nan
                with self.assertRaises(ValueError):
                    audit.normalize_archive(d, samples=9)

    def test_unknown_missing_badshape_names_bool_time_and_negative_counts(self):
        mutations = [lambda d: d.update(schema_version=np.array(2)), lambda d: d.pop("joint_vel"),
                     lambda d: d.update(joint_vel=np.zeros((9, 7))),
                     lambda d: d.update(joint_names=np.array(audit.NATIVE_JOINT_NAMES)),
                     lambda d: d.update(qpos_joint_names=np.array(audit.JOINT_NAMES)),
                     lambda d: d.update(time_s=np.ones(9, bool)),
                     lambda d: d.update(physics_dt_s=np.array(True)),
                     lambda d: d.update(policy_dt_s=np.array(.005)),
                     lambda d: d.update(wheel_ground_contact_count=np.zeros((9, 2), bool)),
                     lambda d: d["wheel_ground_contact_count"].__setitem__((2, 0), -1),
                     lambda d: d["reference_frame"].__setitem__(2, 288)]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                d = fixture()
                mutation(d)
                with self.assertRaises(ValueError):
                    audit.normalize_archive(d, samples=9)

    def test_modified_root_native_qpos_joint_and_initial_state_refuse(self):
        for key, index in (("qpos", (1, 7)), ("root_pos", (2, 0)), ("root_quat_wxyz", (2, 0)),
                           ("initial_qpos", (0,)), ("policy_time_s", (1,)), ("time_s", (2,))):
            with self.subTest(key=key):
                d = fixture()
                d[key][index] += .001
                with self.assertRaises(ValueError):
                    audit.normalize_archive(d, samples=9)

    def test_bad_quaternion_and_action_refuse(self):
        d = fixture()
        d["root_quat_wxyz"][2, 0] = 2
        d["qpos"][2, 3] = 2
        with self.assertRaises(ValueError):
            audit.normalize_archive(d, samples=9)
        d = fixture()
        d["actions_clipped"][1, 0] = 1.00001
        with self.assertRaises(ValueError):
            audit.normalize_archive(d, samples=9)

    def test_report_contract_version_threshold_and_integer_types(self):
        r, c, p = report_fixture()
        audit.check_report(r, c, p)
        for key, value in (("engine_version", "3.8.1"), ("schema_version", 1), ("hidden_resets", False),
                           ("root_state_writes", True), ("physics_steps", 1152.),
                           ("physics_stepped", 1), ("model_sha256", "0" * 64), ("completed_full_reference", False)):
            with self.subTest(key=key):
                mutated = copy.deepcopy(r)
                mutated[key] = value
                with self.assertRaises(ValueError):
                    audit.check_report(mutated, c, p)
        for key, value in (("actor_dim", True), ("physics_dt", .0025), ("leg_kp", 400), ("controller_gain_diagnostic", {})):
            with self.subTest(contract=key):
                mutated, cm = copy.deepcopy(r), copy.deepcopy(c)
                cm[key] = value
                mutated["contract"] = copy.deepcopy(cm)
                with self.assertRaises(ValueError):
                    audit.check_report(mutated, cm, p)

    def test_zero_load_and_present_manifold_kept_separate(self):
        d = audit.normalize_archive(fixture(), samples=9)
        d["wheel_ground_contact_count"][4, 1] = 0
        d["wheel_ground_contact_force_w_n"][4, 1] = 0
        d["wheel_ground_total_contact_force_w_n"][4, 1] = [.01, 0, 0]
        g = geometry_fixture(9)
        g["manifold_count"][4, 1] = 2
        g["manifold_min_distance_m"][4, 1] = -1e-5
        g["clearance_m"][4, 1] = -1e-5
        g["clearance_rate_defined"][4, 1] = False
        event = audit.event_ledgers(d, g)[0]
        self.assertEqual(event["mod4"], 0)
        center = event["context_minus2_plus2"][2]
        self.assertEqual(center["archived_bearing_count_gt5N"], 0)
        self.assertEqual(center["reconstructed_manifold_count"], 2)
        self.assertEqual(center["archived_ground_normal_vector_n"], [0., 0., 0.])
        self.assertEqual(center["archived_ground_total_vector_n"], [.01, 0., 0.])
        self.assertIsNone(center["analytic_clearance_rate_m_s"])

    def test_events_clipped_boundaries_halfopen_source_windows_no_bridge(self):
        d = audit.normalize_archive(fixture(), samples=9)
        d["time_s"] = np.array([0., .775, .78, 1., 1.44, 1.82, 2., 2.32, 5.76])
        d["wheel_ground_contact_count"][:, 1] = 0
        g = geometry_fixture(9)
        events = audit.event_ledgers(d, g)
        self.assertEqual(len(events), 9)
        self.assertEqual([v["raw_timestamp_halfopen_window_ids"] for v in events], [[], [], [1], [1], [], [2], [2], [], []])
        self.assertEqual(len(events[0]["context_minus2_plus2"]), 3)
        self.assertEqual(len(events[-1]["context_minus2_plus2"]), 3)
        self.assertIsNone(events[0]["context_minus2_plus2"][0]["reconstructed_min_contact_distance_m"])
        self.assertEqual([v["mod4"] for v in events], [0, 1, 2, 3, 0, 1, 2, 3, 0])

    def test_accumulated_clock_boundary_both_window_labels_disclosed(self):
        d = audit.normalize_archive(fixture(465), samples=465)
        d["wheel_ground_contact_count"][288, 1] = 0
        d["time_s"][288] = 1.44 - 1e-14
        event = audit.event_ledgers(d, geometry_fixture(465))[0]
        self.assertEqual(event["raw_timestamp_halfopen_window_ids"], [1])
        self.assertEqual(event["nominal5msgrid_halfopen_window_ids"], [])
        self.assertTrue(event["window_assignment_disagrees"])
        self.assertEqual(event["time_s"], 1.44 - 1e-14)

    def test_hash_mutation_and_strict_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "input.json"
            path.write_text('{"value": 1}')
            manifest = {"input": audit.sha256(path)}
            audit.verify_hashes({"input": path}, manifest)
            path.write_text('{"value": 2}')
            with self.assertRaises(ValueError):
                audit.verify_hashes({"input": path}, manifest)
            for invalid in ('{"x":1,"x":2}', '{"x":NaN}', '[]'):
                path.write_text(invalid)
                with self.assertRaises(ValueError):
                    audit.read_json(path)

    def test_existing_and_nonprivate_output_refuse_before_write(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                audit.write_exclusive(Path(td) / "new", {}, {})
        # Repository root exists and is not an allowed new private output.
        with self.assertRaises(ValueError):
            audit.write_exclusive(ROOT, {}, {})

    def test_source_has_no_actor_solver_force_or_integration_calls(self):
        import ast
        tree = ast.parse((ROOT / "scripts/audit_drift_wheel_clearance.py").read_text())
        names = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertTrue({"mj_step", "mj_forward", "mj_contactForce", "load_torchscript", "launch_passive"}.isdisjoint(names))
        self.assertIn("mj_fwdPosition", names)
        self.assertTrue(all(term not in (ROOT / "scripts/audit_drift_wheel_clearance.py").read_text()
                            for term in ("data.efc_force", "data.qfrc_constraint", "data.qacc")))


def exact_mujoco_synthetic_check():
    """A compile/FK-only synthetic model, not a saved or generated rollout."""
    import mujoco as mj
    if mj.__version__ != "3.9.0":
        raise AssertionError("Exact synthetic position-stage check requires MuJoCo 3.9.0")
    # Synthetic topology has the actual free-root/named-joint dimensions and
    # wheel local offsets, but is explicitly NOT the TRON1 experiment model.
    chain = []
    for side in ("L", "R"):
        nodes = [f'<body name="{j}_{side}_Link"><inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>'
                 f'<joint name="{j}_{side}_Joint" axis="0 1 0"/>' for j in ("abad", "hip", "knee")]
        sign = 1 if side == "L" else -1
        nodes += [f'<body name="wheel_{side}_Link" pos="0 {sign * .3} 0">'
                  '<inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>'
                  f'<joint name="wheel_{side}_Joint" axis="0 1 0"/>'
                  f'<geom name="wheel_{side}_collision" type="cylinder" size=".127 .025" pos="0 {sign * .00707} 0" '
                  'quat=".707106781187 .707106781187 0 0"/></body></body></body></body>']
        chain.append("".join(nodes))
    actuators = "".join(f'<motor name="{k}" joint="{k}"/>' for k in audit.JOINT_NAMES)
    xml = '<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>' \
          '<body name="base_Link"><freejoint name="root"/><inertial pos=".04576 .00014 -.16398" mass="1" diaginertia=".1 .1 .1"/>' \
          + "".join(chain) + '</body></worldbody><actuator>' + actuators + '</actuator></mujoco>'
    m = mj.MjModel.from_xml_string(xml)
    d = mj.MjData(m)
    original = fixture(5)
    original["joint_pos"][:] = 0
    original["root_pos"][:] = [0., 0., .12]
    original["root_quat_wxyz"][:] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    original["root_com_lin_vel_w"][:] = [.1, .2, .3]
    original["root_ang_vel_w"][:] = [.4, .5, .6]
    original["qpos"][:, :3] = original["root_pos"]
    original["qpos"][:, 3:7] = original["root_quat_wxyz"]
    original["qpos"][:, 7:] = 0
    d.qpos[:] = original["qpos"][0]
    audit.position_only(mj, m, d)
    base = m.body("base_Link").id
    jp, jr = np.zeros((3, 14)), np.zeros((3, 14))
    point = d.xpos[base] + d.xmat[base].reshape(3, 3) @ audit.ROOT_COM_LINK
    mj.mj_jac(m, d, jp, jr, point, base)
    vids = np.array([int(m.joint(k).dofadr[0]) for k in audit.JOINT_NAMES])
    original["initial_qvel"] = audit.recover_generalized_velocity(np.vstack((jp, jr)), np.zeros(8), vids,
        np.arange(6), np.r_[original["root_com_lin_vel_w"][0], original["root_ang_vel_w"][0]])
    original["initial_qpos"] = original["qpos"][0].copy()
    wheelids = [m.body(k).id for k in audit.WHEEL_NAMES]
    original["wheel_pos"][:] = d.xpos[wheelids]
    original["wheel_ground_contact_count"][:] = 0
    original["wheel_ground_contact_force_w_n"][:] = 0
    saved = {k: v.copy() for k, v in original.items()}
    geometry, report = audit.reconstruct(mj, m, audit.normalize_archive(original, samples=5))
    assert np.all(geometry["manifold_count"] > 0)
    assert np.all(geometry["clearance_m"] < 0)
    assert len(report["zero_bearing_count_events"]) == 10
    assert max(report["fk_verification_max_abs_error"].values()) < 1e-12
    assert not np.allclose(geometry["qvel_reconstructed"][0, :3], original["root_com_lin_vel_w"][0])
    for k, v in saved.items():
        np.testing.assert_array_equal(v, original[k])
    print(json.dumps({"synthetic_exact_mujoco_position_stage": "passed", "version": mj.__version__,
                      "physics_steps": 0, "actor_calls": 0, "experiment_files_read": 0}))


if __name__ == "__main__":
    exact = "--exact-mujoco" in sys.argv
    if exact:
        sys.argv.remove("--exact-mujoco")
    result = unittest.main(exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
    if exact:
        exact_mujoco_synthetic_check()
