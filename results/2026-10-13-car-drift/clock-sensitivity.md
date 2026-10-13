# Frozen-policy MuJoCo contact-clock sensitivity (2026-10-13)

This is an **evaluation-only MuJoCo diagnostic**, not a paired sim2sim experiment or a new trained policy. The historical 200 Hz trial021/trial022 final actors, motion, physical MJCF, motor/domain draw, contact material, 50 Hz action period and unchanged drift thresholds are reused. Only the MuJoCo physics clock changes from 5 ms × 4 to 2.5 ms × 8 or 1.25 ms × 16. Each new run completes the full 5.76 s reference with 288 policy calls and native-rate contact recording. [`scripts/audit_mujoco_clock_sensitivity.py`](../../scripts/audit_mujoco_clock_sensitivity.py) checks the historical input hashes, original contract equality and exact contract-difference set before running; the source rollouts are never edited.

| Actor | MuJoCo clock | Native longest qualifying grounded drift | Moving-drift duration | Native moving-drift wheel-support failures | Failed-support sample time | MuJoCo-only original checks |
|---|---:|---:|---:|---:|---:|---|
| trial021 | 200 Hz, original | 0.165 s | 0.675 s | 5 | 0.0250 s | continuous drift FAIL only |
| trial021 | 400 Hz | 0.0875 s | 0.710 s | 11 | 0.0275 s | continuous drift FAIL only |
| trial021 | 800 Hz | 0.05125 s | 0.704 s | 20 | 0.0250 s | continuous drift FAIL only |
| trial022 | 200 Hz, original | 0.155 s | 0.730 s | 5 | 0.0250 s | continuous drift FAIL only |
| trial022 | 400 Hz | 0.1075 s | 0.718 s | 10 | 0.0250 s | continuous drift FAIL only |
| trial022 | 800 Hz | 0.05125 s | 0.724 s | 20 | 0.0250 s | continuous drift FAIL only |

The minimum unchanged contiguous requirement is **0.200 s**. Every recorded moving-drift loss at 200, 400 and 800 Hz is an isolated **one-native-step** interruption; as the step shrinks, more such interruptions are resolved. Both actors therefore remain below the original native-clock gate, with shorter longest intervals at finer clocks. Similar aggregate failed-support time (~25 ms) is not evidence that the individual gaps are identical continuous-time events: their occurrence times shift, and each finer-clock run is a new closed-loop physical trajectory.

The same actor does **not** mean identical torque or position trajectories after changing the integrator clock. Relative to the aligned 200 Hz baseline, the 400/800 Hz root-position trajectory RMSE is 10.36/10.46 mm for trial021 and 8.60/11.06 mm for trial022; mean absolute 50 Hz clipped-action differences are ~0.0060/0.0065 and ~0.0066/0.0071, respectively. No physical-clock convergence or unique MuJoCo contact-law cause is established by this sweep. It does show that simply reducing MuJoCo's step size does not make these frozen policies satisfy the original drift requirement.

**Why native contact sampling matters.** If the 400 Hz episodes were *counterfactually* inspected only every second step (a 200 Hz observation grid), their apparent longest drift intervals would become 0.210 s (trial021) and 0.320 s (trial022), crossing the threshold even though the actual 400 Hz episodes fail at 0.0875 and 0.1075 s. The coarse view sees only 4/5 moving-drift support-failure samples instead of 11/10. This is observation aliasing, **not a valid PASS**. At 800 Hz, inspecting only every fourth sample similarly misses many losses and yields apparent 0.160/0.175 s intervals instead of the native 0.05125 s. The official native-clock assessor and all original thresholds are preserved.

These MuJoCo-only results cannot certify sim2sim: no matching refined-clock Isaac rollouts were made, and no 400/800 Hz paired-gate claim is made. The original nominal 200 Hz paired trial021/trial022 outcomes remain FAIL. A next causal experiment must keep the two engines on the same declared clock and isolate a physical or training change—such as wheel-ground geometry/contact modeling or a stronger contact-margin objective—while preserving the original behavior gates and measuring raw point-level contact evidence.

Reproduce with the existing Torch + MuJoCo environment; use a fresh output directory for each actor:

```bash
python scripts/audit_mujoco_clock_sensitivity.py \
  --historical-report outputs/drift-sim2sim-20261013/trial022/mujoco/report.json \
  --output-dir outputs/my-new-trial022-clock-study
```

The private output holds each derived clock contract, full rollout/report and a hash-linked summary. The summary's coarse-grid numbers are explicitly labeled *diagnostic only*; they do not replace native contact evidence or a paired assessment.
