"""CPU/config tests only: no Isaac app, CUDA, rollout, actor or physics.

Base-Python pytest runs the actual tensor cases through the installed Torch
CPU interpreter rather than silently skipping essential quaternion tests.
Standalone: retarget_sim/bin/python tests/test_tron1_wheel_axle_tracking.py.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from training import tron1_wheel_axle_tracking as axle


def term(**kwargs):
    return NS(**kwargs)


def configuration():
    return NS(scene=NS(marker=object()), sim=NS(dt=.005), observations=NS(actor_dim=51),
              actions=NS(action_dim=8), commands=NS(motion=NS(body_names=list(axle.BODY_NAMES))),
              events=NS(marker=object()), terminations=NS(marker=object()),
              rewards=NS(motion_body_ori=term(func=lambda *args: None, weight=1., params={
                  "command_name": "motion", "std": .4, "body_names": list(axle.NON_WHEEL_BODY_NAMES)}),
                  drift_wheel_vertical_velocity=term(weight=-5., params={"std": .02}),
                  drift_wheel_ground_load_reserve=term(weight=-1., params={"scale_fraction": .2})))


def snapshot(value):
    if isinstance(value, NS): return {k: snapshot(v) for k, v in vars(value).items()}
    if type(value) is dict: return {k: snapshot(v) for k, v in value.items()}
    if type(value) in (list, tuple): return (type(value).__name__, [snapshot(v) for v in value])
    if isinstance(value, np.ndarray): return (str(value.dtype), value.shape, value.tolist())
    return value


def fake_managers():
    module = types.ModuleType("isaaclab.managers")
    module.RewardTermCfg = term
    return module


class TestConfiguration(unittest.TestCase):
    def test_import_is_lazy_in_fresh_process(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from training import tron1_wheel_axle_tracking; "
                                 "assert 'torch' not in sys.modules; assert 'isaaclab' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_settings_copied_and_no_change_before_bad_settings_fail(self):
        original = {"weight": 1., "std": .4}
        actual = axle.validate_wheel_axle_settings(original)
        self.assertEqual(actual, original); self.assertIsNot(actual, original)
        invalid = [None, [], {}, {"weight": 1.}, {**original, "hidden": True}]
        for key in original:
            invalid += [{**original, key: v} for v in (True, False, None, "1", float("nan"),
                                                       float("inf"), -1., 0., .3, 2.)]
        for value in invalid:
            with self.subTest(value=value):
                cfg = configuration(); before = snapshot(cfg)
                with self.assertRaises(ValueError): axle.configure_wheel_axle_tracking(cfg, value)
                self.assertEqual(snapshot(cfg), before)

    def test_add_exactly_one_term_keep_every_old_object(self):
        cfg = configuration(); before = snapshot(cfg); orientation = cfg.rewards.motion_body_ori
        settings = {"weight": 1., "std": .4}; copy_settings = settings.copy()
        with patch.dict(sys.modules, {"isaaclab.managers": fake_managers()}):
            audit = axle.configure_wheel_axle_tracking(cfg, settings)
        new = getattr(cfg.rewards, axle.TERM_NAME)
        self.assertIs(new.func, axle.wheel_axle_orientation_tracking)
        self.assertEqual(new.params, {"command_name": "motion", "std": .4})
        self.assertEqual(new.weight, 1.); self.assertIs(cfg.rewards.motion_body_ori, orientation)
        self.assertEqual(settings, copy_settings)
        self.assertFalse(audit["wheel_axial_phase_tracked"])
        self.assertFalse(audit["continuous_contact_certified"])
        delattr(cfg.rewards, axle.TERM_NAME)
        self.assertEqual(snapshot(cfg), before)

    def test_refuse_overwrite_and_changed_original_orientation(self):
        mutations = [lambda cfg: setattr(cfg.rewards, axle.TERM_NAME, None),
                     lambda cfg: setattr(cfg.rewards.motion_body_ori, "weight", 2.),
                     lambda cfg: setattr(cfg.rewards.motion_body_ori, "weight", True),
                     lambda cfg: cfg.rewards.motion_body_ori.params.update(std=.5),
                     lambda cfg: cfg.rewards.motion_body_ori.params.update(command_name="new_teacher"),
                     lambda cfg: cfg.rewards.motion_body_ori.params.update(body_names=list(axle.BODY_NAMES)),
                     lambda cfg: cfg.rewards.motion_body_ori.params.update(body_names=list(reversed(axle.NON_WHEEL_BODY_NAMES)))]
        for change in mutations:
            with self.subTest(change=change):
                cfg = configuration(); change(cfg); before = snapshot(cfg)
                with self.assertRaises(ValueError): axle.configure_wheel_axle_tracking(cfg, axle.SETTINGS)
                self.assertEqual(snapshot(cfg), before)

    def test_actual_isolated_dense_factory_diff_only_new_reward(self):
        # Reuse the original factory AST fixture, without importing pytest/Isaac.
        source = ast.parse((ROOT / "tests/test_car_drift_task.py").read_text())
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in {
            "fake_configuration", "config_constructor", "extract_configuration_factory"}]
        from eval_tracking_mujoco import read_contract
        namespace = {"NS": NS, "ROOT": ROOT, "ast": ast, "Path": Path, "np": np,
                     "math": math, "read_contract": read_contract}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "original_isolated_factory_helpers", "exec"), namespace)
        original_fake = namespace["fake_configuration"]
        def upstream_config():
            cfg = original_fake()
            # Actual upstream motion_body_ori defaults, absent from older stub.
            cfg.rewards.motion_body_ori.weight = 1.
            cfg.rewards.motion_body_ori.params.update(command_name="motion", std=.4)
            return cfg
        namespace["fake_configuration"] = upstream_config
        managers = fake_managers(); sensors = types.ModuleType("isaaclab.sensors"); sensors.ContactSensorCfg = term
        from training.tron1_ground_load_substeps_trial import configure_dense_ground_load
        # patch.dict restores the WHOLE registry. Load real configuration
        # dependencies first, so restoring fake Isaac modules cannot remove a
        # newly loaded real module while leaving its training-package attribute.
        dependencies = {name: importlib.import_module("training." + name) for name in (
            "tron1_drift", "tron1_ground_load_reserve", "tron1_ground_load_substeps")}
        profile = json.loads((ROOT / "config/car_drift_dense_ground_load_task.json").read_text())
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
                "isaaclab.managers": managers, "isaaclab.sensors": sensors}):
            motion, asset = Path(directory) / "fixture.npz", Path(directory) / "fixture.usd"
            contract = read_contract()
            np.savez(motion, fps=np.array([50.]), joint_names=np.array(contract["observed_joint_velocity_order"]),
                     joint_pos=np.zeros((3, 8)), body_pos_w=np.zeros((3, len(contract["body_names"]), 3)))
            asset.write_text("# Offline existence-only asset fixture\n")
            cfg = namespace["extract_configuration_factory"]()(motion, asset, num_envs=2048,
                                                                device="cuda:5", task_profile=profile)
            configure_dense_ground_load(cfg, profile)
            before = snapshot(cfg); old_terms = dict(vars(cfg.rewards))
            axle.configure_wheel_axle_tracking(cfg, axle.SETTINGS)
            self.assertEqual(set(vars(cfg.rewards)) - set(old_terms), {axle.TERM_NAME})
            for key, value in old_terms.items(): self.assertIs(getattr(cfg.rewards, key), value)
            delattr(cfg.rewards, axle.TERM_NAME)
            self.assertEqual(snapshot(cfg), before)
            self.assertEqual(cfg.sim.dt, .005); self.assertEqual(cfg.decimation, 4)
            self.assertEqual(cfg.rewards.motion_body_ori.params["body_names"], list(axle.NON_WHEEL_BODY_NAMES))
        training_package = importlib.import_module("training")
        for name, module in dependencies.items():
            self.assertIs(sys.modules["training." + name], module)
            self.assertIs(getattr(training_package, name), module)


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    class TestTorchCPU(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            import torch
            cls.torch = torch

        def setUp(self):
            # This readback does not initialize CUDA (unlike probing devices).
            self.assertFalse(self.torch.cuda.is_initialized())

        def tearDown(self):
            self.assertFalse(self.torch.cuda.is_initialized(), "CPU tests must never initialize Torch CUDA")

        def identity(self, count=3, dtype=None):
            return self.torch.tensor([1., 0., 0., 0.], dtype=dtype or self.torch.float64).repeat(count, 2, 1)

        def axis_rotation(self, axis, angles):
            angles = self.torch.as_tensor(angles, dtype=self.torch.float64)
            result = self.torch.zeros((*angles.shape, 4), dtype=self.torch.float64)
            result[..., 0] = (angles / 2).cos(); result[..., axis + 1] = (angles / 2).sin()
            return result

        def multiply(self, q, p):
            w, x, y, z = q.unbind(-1); a, b, c, d = p.unbind(-1)
            return self.torch.stack((w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                                     w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a), -1)

        def assert_close(self, a, b, tol=2e-14):
            self.torch.testing.assert_close(a, b, rtol=tol, atol=tol)

        def command_env(self, names=None, count=3):
            names = list(axle.BODY_NAMES) if names is None else names
            q = self.identity(count)[:, :1].repeat(1, len(names), 1)
            command = NS(cfg=NS(body_names=names), body_quat_relative_w=q.clone(), robot_body_quat_w=q.clone(),
                         body_quat_w=self.torch.full_like(q, float("nan")))
            calls = []
            def get_term(name): calls.append(name); return command
            def forbidden(*args): raise AssertionError("No step/reset/compute/inference is allowed")
            return NS(num_envs=count, command_manager=NS(get_term=get_term), step=forbidden, reset=forbidden), command, calls

        def test_identity_sign_and_nonunit_quaternion_invariance(self):
            ref, actual = self.identity(), self.identity()
            for ref_scale in (1., -1., 3., -2.):
                for actual_scale in (1., -1., 7., -.5):
                    with self.subTest(ref=ref_scale, actual=actual_scale):
                        self.assert_close(axle.wheel_axle_orientation_reward(ref*ref_scale, actual*actual_scale),
                                          self.torch.ones(3, dtype=ref.dtype))

        def test_directed_angles_mean_two_including_pi(self):
            angles = self.torch.tensor([[0., 0.], [.2, .4], [math.pi, math.pi]], dtype=self.torch.float64)
            actual = self.axis_rotation(0, angles); ref = self.identity()
            self.assert_close(axle.wheel_axle_angle_error(ref, actual), angles)
            expected = (-angles.square().mean(-1)/.4**2).exp()
            self.assert_close(axle.wheel_axle_orientation_reward(ref, actual), expected)
            self.assertLess(float(expected[-1]), 1e-20)

        def test_independent_arbitrary_local_y_spin_and_quaternion_phase(self):
            ref = self.multiply(self.axis_rotation(2, [[.2,.1],[-.4,.6],[1.,-.7]]),
                                self.axis_rotation(0, [[.1,.2],[.3,-.2],[.4,.5]]))
            actual = self.axis_rotation(0, [[.2,.1],[-.3,.5],[.6,-.7]])
            baseline = axle.wheel_axle_orientation_reward(ref, actual)
            for phase in (-23*math.pi, -1.7, 0., .3, 18*math.pi):
                with self.subTest(phase=phase):
                    spin_ref = self.axis_rotation(1, [[phase,-phase],[phase+.4,phase-1.],[phase*2,phase/2]])
                    spin_actual = self.axis_rotation(1, [[phase+2.,phase-.5],[-phase,phase*3],[phase-.8,phase+1.]])
                    self.assert_close(axle.wheel_axle_orientation_reward(self.multiply(ref, spin_ref),
                                      -self.multiply(actual, spin_actual)), baseline)

        def test_common_world_yaw_transform_invariance(self):
            ref = self.axis_rotation(0, [[.1,.2],[.3,-.4],[.5,.6]])
            actual = self.axis_rotation(2, [[.3,.6],[.8,-1.],[2.,-2.]])
            yaw = self.axis_rotation(2, [[.8,.8],[-1.,-1.],[2.,2.]])
            self.assert_close(axle.wheel_axle_orientation_reward(self.multiply(yaw, ref), self.multiply(yaw, actual)),
                              axle.wheel_axle_orientation_reward(ref, actual))

        def test_float32_and_no_tensor_or_rng_mutation(self):
            ref = self.identity(dtype=self.torch.float32)
            actual = self.axis_rotation(0, [[.1,.2],[.3,-.4],[.5,.6]]).float().requires_grad_()
            ref_before, actual_before = ref.clone(), actual.detach().clone()
            rng = self.torch.random.get_rng_state().clone()
            reward = axle.wheel_axle_orientation_reward(ref, actual)
            self.assertEqual(tuple(reward.shape), (3,)); self.assertEqual(str(reward.device), "cpu")
            self.assertEqual(reward.dtype, self.torch.float32)
            self.assertTrue(self.torch.equal(ref, ref_before)); self.assertTrue(self.torch.equal(actual, actual_before))
            self.assertTrue(self.torch.equal(rng, self.torch.random.get_rng_state()))
            reward.sum().backward(); self.assertTrue(self.torch.isfinite(actual.grad).all().item())

        def test_reject_malformed_tensor_type_shape_dtype_and_values(self):
            invalid = [None, [], np.ones((3,2,4)), self.torch.zeros((3,2,4), dtype=self.torch.int64),
                       self.torch.ones((3,2,4), dtype=self.torch.complex128), self.torch.zeros((3,2,4)),
                       self.torch.ones((0,2,4)), self.torch.ones((3,3,4)), self.torch.ones((3,2,3)),
                       self.torch.ones((3,2,4,1)), self.torch.full((3,2,4), float("nan")),
                       self.torch.full((3,2,4), float("inf"))]
            for value in invalid:
                with self.subTest(value=str(type(value)), shape=getattr(value,"shape",None)):
                    with self.assertRaises(ValueError): axle.wheel_axle_orientation_reward(value, self.identity())
            for other in (self.identity(2), self.identity(dtype=self.torch.float32)):
                with self.subTest(shape=other.shape, dtype=other.dtype):
                    with self.assertRaises(ValueError): axle.wheel_axle_orientation_reward(self.identity(), other)

        def test_bad_std_fails_closed(self):
            for value in (True, False, None, "0.4", float("nan"), float("inf"), 0., -.4, .3):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError): axle.wheel_axle_orientation_reward(self.identity(), self.identity(), value)

        def test_named_ordered_wheels_relative_target_and_nonwheels_ignored(self):
            names = list(reversed(axle.BODY_NAMES))
            env, command, calls = self.command_env(names)
            angles = [[.2,.4],[.3,.6],[.5,.1]]; wheel_quat = self.axis_rotation(0, angles)
            for i, name in enumerate(axle.WHEEL_BODY_NAMES):
                command.robot_body_quat_w[:, names.index(name)] = wheel_quat[:, i]
            for name in axle.NON_WHEEL_BODY_NAMES:
                command.robot_body_quat_w[:, names.index(name)] = self.axis_rotation(0, [1.,2.,3.])
            before = command.robot_body_quat_w.clone()
            expected = axle.wheel_axle_orientation_reward(self.identity(), wheel_quat)
            self.assert_close(axle.wheel_axle_orientation_tracking(env), expected)
            self.assertEqual(calls, ["motion"])
            self.assertTrue(self.torch.equal(before, command.robot_body_quat_w))

        def test_hostile_missing_duplicate_extra_names_and_dimensions(self):
            names_cases = [list(axle.BODY_NAMES[:-1]), list(axle.BODY_NAMES)+["imu"],
                           list(axle.BODY_NAMES[:-1])+["wheel_L_Link"],
                           list(axle.BODY_NAMES[:-1])+["other_wheel"],
                           list(axle.BODY_NAMES[:-1])+[True], "wheel_L_Link"]
            for names in names_cases:
                with self.subTest(names=names):
                    env, _, _ = self.command_env(names)
                    with self.assertRaises(ValueError): axle.wheel_axle_orientation_tracking(env)
            for shape in ((3,10,4),(3,8,4),(2,9,4),(3,9,3)):
                with self.subTest(shape=shape):
                    env, command, _ = self.command_env()
                    command.robot_body_quat_w = self.torch.ones(shape, dtype=self.torch.float64)
                    with self.assertRaises(ValueError): axle.wheel_axle_orientation_tracking(env)
            for value in (True, 0, -1, 3.0):
                env, _, _ = self.command_env(); env.num_envs = value
                with self.subTest(num_envs=value):
                    with self.assertRaises(ValueError): axle.wheel_axle_orientation_tracking(env)
            env, _, _ = self.command_env()
            with self.assertRaises(ValueError): axle.wheel_axle_orientation_tracking(env, command_name="new_teacher")
else:
    class TestActualTorchCPUBridge(unittest.TestCase):
        def test_all_actual_torch_cpu_cases_without_silent_skip(self):
            interpreter = Path("/home/roy/miniconda3/envs/retarget_sim/bin/python")
            self.assertTrue(interpreter.is_file(), "Essential actual Torch CPU tests require the declared installed interpreter")
            result = subprocess.run([str(interpreter), str(Path(__file__).resolve()), "--tensor-only"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            marker = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(marker, {"actual_torch_cpu_tests": 9, "simulator_apps": 0, "cuda_allocations": 0})


if __name__ == "__main__":
    tensor_only = "--tensor-only" in sys.argv
    if tensor_only:
        if not HAS_TORCH: raise RuntimeError("Actual Torch CPU interpreter required")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTorchCPU)
    else:
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if tensor_only:
        print(json.dumps({"actual_torch_cpu_tests": result.testsRun, "simulator_apps": 0, "cuda_allocations": 0}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
