# Paired saved-pose wheel geometry at MuJoCo drift dropouts (2026-10-13)

This is a read-only, common-geometry comparison of the **original 200 Hz Isaac and MuJoCo physical rollouts** for trial021 and trial022. [`scripts/audit_paired_drift_wheel_geometry.py`](../../scripts/audit_paired_drift_wheel_geometry.py) maps each simulator's named generalized coordinates into the same MuJoCo finite-cylinder model and calls `mj_fwdPosition` only. No actor, force solve, integration, contact-gap interpolation, controller or behavior gate is used. The common MJCF wheel-link height agrees with archived Isaac wheel-link heights within 0.295 µm over all samples and with MuJoCo heights exactly. Isaac force/count columns below are **actual native PhysX telemetry**; Isaac *clearance* is a counterfactual MJCF projection and is **not** native PhysX signed separation.

| Actor | MuJoCo loss time (s) | Isaac actual right-wheel normal (N) | Isaac common-MJCF clearance (µm) | MuJoCo common-MJCF clearance (µm) | Wheel-link height MuJoCo−Isaac (mm) |
|---|---:|---:|---:|---:|---:|
| trial021 | 0.970 | 91.1 | −3.1 | +19.0 | +0.036 |
| trial021 | 1.115 | 84.3 | −1.3 | +91.2 | +0.118 |
| trial021 | 1.285 | 72.8 | +2.5 | +2.7 | +0.055 |
| trial021 | 1.905 | 90.8 | +1.5 | +73.5 | +0.177 |
| trial021 | 2.010 | 100.4 | +0.8 | +28.5 | +0.157 |
| trial022 | 0.975 | 92.7 | −2.9 | +79.9 | +0.257 |
| trial022 | 1.120 | 81.9 | −1.1 | +62.4 | +0.136 |
| trial022 | 1.280 | 80.5 | +2.4 | +3.9 | +0.013 |
| trial022 | 1.885 | 82.4 | +1.8 | +138.6 | +0.261 |
| trial022 | 1.980 | 88.6 | +0.8 | +142.4 | +0.271 |

At all ten times, Isaac still has a bearing right-wheel contact; MuJoCo's right-wheel manifold/normal force is zero (verified separately by the [exact-rollout contact-point replay](contact-point-audit.md)). The Isaac poses projected into the common cylinder model are within approximately ±3 µm of the geometric support boundary, while MuJoCo poses have a positive 2.7–142.4 µm gap. The wheel-link height difference at those times is only 0.013–0.271 mm. Root-height differences can be much larger and change sign (for trial022, −0.99 to +7.01 mm at these events), because leg configuration and body attitude also change; a simple root-height offset is not a justified correction.

Over *other supported moving-drift samples* for the right wheel, the common MJCF clearance's 10th/median/90th percentiles are −2.178/−1.395/−0.311 mm in trial021 MuJoCo and −2.208/−1.287/−0.236 mm in trial022 MuJoCo. The Isaac-pose projection stays near the boundary (−3.1/+0.5/+2.0 µm and −3.1/+0.4/+1.9 µm, respectively). These very different penetration distributions make contact compliance/rebound and collision conventions **plausible factors to investigate**, not established causes: MuJoCo uses explicit `solref=[0.04,2]`, while the PhysX collision geometry, contact offsets and force solver are not recreated by MJCF position projection. Earlier [normal-contact calibration](../2026-10-11-car-drift/normal_contact_diagnostic.json) did **not** establish a reliable PhysX-to-MuJoCo parameter mapping. Arbitrarily changing MuJoCo contact stiffness or counting a projected positive-gap PhysX contact as MuJoCo support would not be a valid policy-transfer success.

The original paired trial021/trial022 drift verdicts remain FAIL. The common-model projection narrows a useful next validation target—measure wheel-ground load and geometry over a matched physical clock in both engines, then test a *predeclared* contact-model or training intervention without relaxing any gate—but does not certify sim2sim or hardware behavior.

Reproduce with the existing Torch + MuJoCo environment and a fresh output directory:

```bash
python scripts/audit_paired_drift_wheel_geometry.py \
  --isaac-dir outputs/drift-sim2sim-20261013/trial022/isaac \
  --mujoco-dir outputs/drift-sim2sim-20261013/trial022/mujoco \
  --event-decomposition outputs/drift-sim2sim-20261013/trial022/diagnostics/mujoco_contact_breaks.json \
  --output-dir outputs/my-new-paired-wheel-geometry-audit
```
