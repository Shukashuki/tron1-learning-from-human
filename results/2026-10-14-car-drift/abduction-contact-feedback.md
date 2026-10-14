# Drift-turn abduction contact feedback: local rescue, full-episode FAIL

The objective remains one **same-policy** high-speed sliding ≥45° turn in Isaac and MuJoCo, with ≥0.8 m/s speed, ≥0.2 s and ≥45° in the **same uninterrupted** native-clock interval, and the original path/completion gates. This experiment only tests the frozen Isaac-trained actor plus a MuJoCo-only physical-rate inner correction. **No candidate is a sim2sim PASS.** The original jump demonstration is unchanged.

## Why abduction?

The [selected calibrated MuJoCo episode](turn-worst-load/mujoco-margin1000-solimp050-090-002-assessment.json) has [41 physical samples](abduction-feedback/contact-break-decomposition.json) that lose support while the root is moving sideways at speed. The [exact saved-state 20 ms action branches](abduction-feedback/one-window-action-sensitivity.json) replay each historical decision window with the original action, then change one bounded action coordinate at a time by at most 0.08. The original branch reproduces saved positions, torques, forces and contact counts to its strict replay tolerances. In **19/41 failing samples**, at least one single-coordinate branch retains both wheels' ≥5 N load through all four physical substeps. Of the smallest such changes, 18 are **positive same-side hip abduction**, one is left knee; the abduction branches lower failed-wheel clearance by a median 0.308 mm and have ≥77.9 N minimum wheel normal force across the full window. This is local action authority, **not** a deployable prediction or a proof that the other 22 samples are impossible to rescue by coordinated actions.

[`eval_drift_predictive_abduction_mujoco.py`](../../scripts/eval_drift_predictive_abduction_mujoco.py) therefore probes a causal 200 Hz correction before the original PD/DC motor law. It reads current finite-cylinder/ground signed clearance and its backward-difference upward rate, predicts a short forward clearance, and adds a bounded **positive same-side abduction target** when that signal exceeds a threshold. Its 50 Hz parent actor, 51D observation, reference, friction, model, actuator envelope and [frozen assessor](../../scripts/assess_drift_turn.py) are unchanged. The [zero-gain parity audit](abduction-feedback/zero-parity.json) reproduces all 349 original decisions and 1,396 physical steps exactly, including generalized positions, joint torques, normal forces, slip, observations, actions and contact counts.

The [predeclared six full-episode runs](abduction-feedback/sweep.json) use thresholds 1.2/1.5/1.8 mm, gains 0.5/1.0, 5 ms lookahead and ≤0.08 rad bias. All six **FAIL**; two also terminate early. Representative results:

| Frozen gate, actual 200 Hz core | No feedback | 1.5 mm / gain 0.5 | 1.5 mm / gain 1.0 | 1.8 mm / gain 0.5 |
|---|---:|---:|---:|---:|
| Full reference completed | Yes | Yes | Yes | Yes |
| Minimum speed | 0.945 m/s | 0.816 m/s | **0.745 m/s** | 0.906 m/s |
| Path RMSE | 0.038 m | 0.075 m | 0.091 m | 0.034 m |
| Both wheels ≥5 N | 91.62% | 84.60% | 82.26% | 88.89% |
| Longest jointly qualifying interval | 0.185 s | 0.215 s | 0.325 s | 0.175 s |
| Largest jointly qualifying turn | 3.33° | 7.26° | 11.48° | 3.86° |
| Full gate | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

Individual [reports and assessments](abduction-feedback/) preserve exact actor, motion, model, contract, source and rollout hashes; representative complete physical rollouts are [gain 0.5](abduction-feedback/t1500-g050-rollout.npz) and [gain 1.0](abduction-feedback/t1500-g100-rollout.npz). The 0.325 s interval is not a near PASS: the same interval turns only 11.48°, and its episode drops below the 0.8 m/s speed floor elsewhere. A positive local action perturbation is not a suitable global rule: repeated correction changes subsequent state, speed, path and contact timing, and in these trials **reduces** overall wheel support. No further gain sweep is justified by these results. The next policy experiment should expose filtered wheel-ground load and wheel vertical motion in a *matched observation* and train for sustained contact jointly with speed/path/sliding objectives, then independently verify the same actor and controller in Isaac and MuJoCo.
