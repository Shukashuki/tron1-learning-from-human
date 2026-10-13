"""Actual Torch CPU geometry tests; no Isaac/actor/physics integration.

Run directly with the existing retarget Python, where Torch is installed.
The main pytest environment may skip them if Torch is unavailable.
"""
import math
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from training import tron1_tight_axle_tracking as new
from training import tron1_wheel_axle_tracking as old


@unittest.skipIf(torch is None, "Existing environment has no Torch")
class GeometryTests(unittest.TestCase):
    def quats(self):
        return torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]], dtype=torch.float64)

    def test_exp_scale_only(self):
        ref, actual = self.quats(), self.quats()
        angle = .3
        actual[0, 0] = torch.tensor([math.cos(angle/2), math.sin(angle/2), 0., 0.])
        before = actual.clone()
        theta = old.wheel_axle_angle_error(ref, actual)
        expected = torch.exp(-theta.square().mean(-1) / .04)
        torch.testing.assert_close(new.wheel_axle_orientation_reward(ref, actual), expected)
        torch.testing.assert_close(new.wheel_axle_orientation_reward(ref, actual),
                                   old.wheel_axle_orientation_reward(ref, actual).pow(4))
        torch.testing.assert_close(actual, before)

    def test_spin_sign_and_directed_reversal(self):
        ref, actual = self.quats(), self.quats()
        actual[0, 0] = torch.tensor([0., 0., 1., 0.])  # pi spin around local Y
        actual[0, 1] = -ref[0, 1]
        torch.testing.assert_close(new.wheel_axle_orientation_reward(ref, actual), torch.ones(1, dtype=torch.float64))
        actual[0, 0] = torch.tensor([0., 1., 0., 0.])  # reverses directed axle
        self.assertLess(float(new.wheel_axle_orientation_reward(ref, actual)[0]), 1e-40)

    def test_name_mapping_reads_last_tensors(self):
        names = list(reversed(old.BODY_NAMES))
        ref = self.quats()[:, :1].repeat(1, len(names), 1)
        actual = ref.clone()
        actual[:, names.index(old.WHEEL_BODY_NAMES[0])] = torch.tensor([0., 1., 0., 0.])
        command = SimpleNamespace(cfg=SimpleNamespace(body_names=names), body_quat_relative_w=ref,
                                  robot_body_quat_w=actual)
        env = SimpleNamespace(num_envs=1, command_manager=SimpleNamespace(get_term=lambda name: command))
        indices = [names.index(name) for name in old.WHEEL_BODY_NAMES]
        torch.testing.assert_close(new.wheel_axle_orientation_tracking(env),
                                   new.wheel_axle_orientation_reward(ref[:, indices], actual[:, indices]))

    def test_wrong_std_and_nonfinite_rejected(self):
        for std in (.4, .1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError): new.wheel_axle_orientation_reward(self.quats(), self.quats(), std)
        actual = self.quats(); actual[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError): new.wheel_axle_orientation_reward(self.quats(), actual)


if __name__ == "__main__":
    unittest.main()
