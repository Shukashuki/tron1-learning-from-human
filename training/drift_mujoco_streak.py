"""Exact native-clock qualifying-window tracker for residual reward shaping.

It mirrors the fixed turn assessor's physical thresholds and core interval.
The assessor itself is never edited or replaced. A completed contiguous
window earns a record-improvement bonus; contact gaps are never bridged.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class DriftStreakTracker:
    def __init__(self, *, start_s=1.44, end_s=4.0):
        self.start_s, self.end_s = float(start_s), float(end_s)
        self.reset()

    def reset(self):
        self.previous_wrapped_course = None
        self.unwrapped_course = 0.
        self.active_start_course = None
        self.active_start_time = None
        self.current_course = None
        self.current_time = None
        self.best_weighted_turn_deg = 0.
        self.largest_qualifying_turn_deg = 0.
        self.longest_qualifying_duration_s = 0.

    def _close_window(self):
        if self.active_start_course is None:
            return 0.
        turn = abs(np.rad2deg(self.current_course - self.active_start_course))
        duration = self.current_time - self.active_start_time
        self.largest_qualifying_turn_deg = max(self.largest_qualifying_turn_deg, float(turn))
        self.longest_qualifying_duration_s = max(self.longest_qualifying_duration_s, float(duration))
        weighted = float(turn * min(duration / .2, 1.))
        improvement = max(0., weighted - self.best_weighted_turn_deg)
        self.best_weighted_turn_deg = max(self.best_weighted_turn_deg, weighted)
        self.active_start_course = self.active_start_time = None
        self.current_course = self.current_time = None
        return float(improvement)

    def observe(self, sample):
        velocity = np.asarray(sample["root_com_lin_vel_w"], dtype=float)[:2]
        speed = float(np.linalg.norm(velocity))
        wrapped_course = float(np.arctan2(velocity[1], velocity[0]))
        if self.previous_wrapped_course is None:
            self.unwrapped_course = wrapped_course
        else:
            delta = wrapped_course - self.previous_wrapped_course
            self.unwrapped_course += float(np.arctan2(np.sin(delta), np.cos(delta)))
        self.previous_wrapped_course = wrapped_course
        quat = np.asarray(sample["root_quat_wxyz"], dtype=float)
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        beta = abs((wrapped_course - yaw + np.pi) % (2*np.pi) - np.pi) * 180. / np.pi
        force = np.asarray(sample["wheel_ground_contact_force_w_n"], dtype=float)
        count = np.asarray(sample["wheel_ground_contact_count"])
        slip = np.asarray(sample["wheel_lateral_contact_slip_m_s"], dtype=float)
        time = float(sample["time_s"])
        qualifies = (self.start_s - 1e-8 <= time <= self.end_s + 1e-8
                     and speed >= .8 and beta >= 8.
                     and np.all((count > 0) & (np.linalg.norm(force, axis=1) >= 5.))
                     and np.mean(np.abs(slip)) >= .04)
        if not qualifies:
            return self._close_window()
        if self.active_start_course is None:
            self.active_start_course = self.unwrapped_course
            self.active_start_time = time
        self.current_course = self.unwrapped_course
        self.current_time = time
        return 0.

    def finish(self):
        return self._close_window()
