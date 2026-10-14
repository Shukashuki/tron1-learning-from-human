# High-speed WF drift: predictive 200 Hz knee feedback, sim2sim still FAIL

The requested outcome remains the **same policy** performing one uninterrupted high-speed sliding ≥45° turn in both Isaac and MuJoCo under the [unchanged gate](turn-worst-load-pair.md). These are MuJoCo-only, evaluation-time inner-controller diagnostics around the frozen normalized Isaac-trained actor; **none passed** even in MuJoCo, so there is no new paired sim2sim claim. The minimum-speed requirement remains 0.8 m/s; zero deceleration is not required.

## Why test physical-rate feedback?

The calibrated model's core has 24 left and 20 right wheel-contact loss onsets. The preceding 5 ms sample still carries median ground-normal force 113/129 N, then bearing contact disappears. A [saved-state finite-cylinder geometry audit](predictive-contact/baseline-clearance-precursors.json) shows median signed clearance **1.863/1.877 mm immediately before** left/right gaps and **2.086/2.228 mm at** the gaps. The compiled floor and wheel geoms each have 1 mm `margin`. [MuJoCo's contact documentation](https://mujoco.readthedocs.io/en/latest/computation.html#margin-and-gap) says pair margins add and active forces require distance below the summed margin, so this is consistent with crossing an approximately 2 mm contact-activation boundary. That geometric interpretation is an inference; it does not by itself identify every force or closed-loop cause.

The [root-fixed kinematic Jacobian audit](predictive-contact/wheel-height-jacobian.json) finds median own-knee wheel-link-height derivatives +0.249 m/rad on the left and −0.242 m/rad on the right. This fixes the action signs: negative left / positive right knee-target corrections lower the corresponding wheel at a fixed root pose. The hip vertical effect is much smaller in the sampled turn core. These are kinematics, not evidence that a controller can preserve contact, speed and slide simultaneously.

## Frozen-actor experiment and controls

The inner loop applies a bounded correction before the original 200 Hz DC-motor/PD torque computation. The outer actor remains at 50 Hz with its original 51D observation and raw 8D action. The first seven runs use backward-difference wheel-link height and upward speed; their exact [v1 source](predictive-contact/height-feedback-v1.py) is archived. The next six use backward-difference **finite-cylinder clearance** and its upward rate in [`eval_drift_predictive_clearance_mujoco.py`](../../scripts/eval_drift_predictive_clearance_mujoco.py). Neither signal looks into a future state. Both keep the same actor, reference, friction, contact response, 50/200 Hz clocks and [frozen assessor](../../scripts/assess_drift_turn.py).

A zero-gain full-episode control for each evaluator reproduces the calibrated parent rollout's **349 actor decisions and 1,396 physics steps** exactly: qpos, root state, applied torques, wheel ground forces, slip, actor observations and actions all have zero maximum error; every contact count matches ([height control](predictive-contact/height-zero-parity.json), [clearance control](predictive-contact/clearance-zero-parity.json)). This isolates the new low-level correction as the only dynamic intervention.

| Actual 200 Hz core, same original gate | Frozen actor / no feedback | Best height signal (`p04`) | Best clearance turn (`c05`) | Strongest clearance support (`c02`) |
|---|---:|---:|---:|---:|
| Minimum COM speed | 0.967 m/s | 0.946 m/s | 0.917 m/s | 0.964 m/s |
| Root path RMSE | 0.041 m | 0.039 m | 0.042 m | 0.040 m |
| Both wheels bearing ≥5 N | 91.23% | 91.42% | 90.84% | 91.81% |
| Longest jointly fast, sliding, supported interval | 0.140 s | 0.145 s | 0.180 s | 0.190 s |
| Largest turn in a qualifying interval | 4.54° | 6.86° | **8.40°** | 6.10° |
| Full-clip gate | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

All [13 trials in the verified sweep](predictive-contact/sweep.json) completed the full reference, kept actor/model/motion/contract hashes identical, and **all FAIL**. The sweep ranges to 0.04 rad bounded knee correction. Even where path speed and slip remain adequate, support stays around 91–92%, far from the uninterrupted ≈2 s needed for a 45° turn. This is a one-reference exploratory parameter sweep, not a statistically independent robustness benchmark. Individual machine-readable reports and assessments are under [predictive-contact](predictive-contact/); representative complete native-clock rollouts are [height `p04`](predictive-contact/p04-rollout.npz), [clearance `c04`](predictive-contact/c04-rollout.npz), and [clearance `c05`](predictive-contact/c05-rollout.npz).

Clearance feedback did move ordinary support states deeper inside the contact shell: in the baseline, the left/right median supported clearance is +0.756/+0.275 mm; in the stronger `c05` run it is +0.531/−0.246 mm. But the [same diagnostic on `c05`](predictive-contact/c05-clearance-precursors.json) still finds pre-loss medians +1.884/+1.847 mm and loss medians +2.118/+2.131 mm, with no meaningful reduction in contact gaps. Thus this particular proportional predictive knee-target family does **not** suppress the boundary-crossing oscillation. It is not evidence that contact feedback in general is impossible. A subsequent policy should explicitly model native-rate contact/clearance state and validate the identical observation, actuator and feedback path in Isaac before any transfer claim; merely enlarging this correction or smoothing the scorer would not satisfy the task.
