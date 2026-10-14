# High-speed turn: load-reserve retry, paired sim2sim FAIL

The second `drift_drl/traj_4` wheel-foot turn candidate adds **one reward term** to the generic first-trained baseline: a smooth symmetric wheel ground-normal-load reserve cost, weight −2 and scale 0.20 × the live robot weight, evaluated at the last 5 ms physics substep. The [task profile](../../config/drift_turn_load_reserve_task.json) changes no motion, controller, friction, motor-domain randomization, PPO budget, or acceptance threshold. An eight-environment/one-update native fixture passed before the single designated 2,048-environment, 600-update training run. That run completed naturally after 29,491,200 more environment steps; final checkpoint SHA-256 is `282dc0a6493c4fd5caed8fd42c2607ff3688b6d520ad716c7f91b1af81c29d29`.

One final-checkpoint Isaac evaluation exported normalized actor SHA-256 `8ad06855fe18549e091aaec848b5cf67d50efb501b6b6ae62f99320d77e919cb`. The *same actor* ran one zero-shot MuJoCo episode on the same pinned motion SHA-256 `da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254`. MuJoCo's deployment observation reconstruction of all 350 recorded Isaac observations differed by at most 4.77e−7. Both engines completed the 6.96 s reference. These are nominal one-seed simulations, not robustness or hardware results.

| Actual 200 Hz physical measurement, source turn core | Isaac | MuJoCo default |
|---|---:|---:|
| Whole-core travel-course turn | 61.42° | 63.37° |
| Minimum whole-core COM speed | 0.920 m/s | 0.920 m/s |
| Root path RMSE | 0.0380 m | 0.0226 m |
| Core samples with both wheels bearing ≥5 N | 100.0% | 51.9% |
| Longest uninterrupted *jointly fast, sliding, supported* window | 2.260 s | 0.025 s |
| Largest actual course turn within a qualifying window | 54.64° | 1.16° |
| [Frozen new-task gate](../../scripts/assess_drift_turn.py) | **PASS** | **FAIL** |

The gate accepts speed ≥0.8 m/s, body sideslip ≥8°, mean absolute wheel-ground patch slip ≥0.04 m/s, both ground-filtered wheels bearing ≥5 N, ≥0.20 s uninterrupted, and ≥45° actual travel-course change **in that same interval**. It additionally requires full-reference completion and core path RMSE/maximum error ≤0.15/0.30 m. It neither demands zero deceleration nor bridges single 5 ms support gaps. The large whole-core MuJoCo turn and sustained speed therefore do not erase its lack of continuous two-wheel-supported sliding.

[Four-panel source → kinematics → Isaac → MuJoCo video](turn-load-reserve/motion_pipeline.mp4) · [Overview](turn-load-reserve/overview.png) · [paired machine-readable assessment](turn-load-reserve/assessment.json) · [render provenance](turn-load-reserve/render_report.json). The car panel depicts the *simulated CARLA source*; the robot panels replay saved poses without new physics. Gate verdicts come only from native physics arrays, not the video. The original jump animation and historical drift trials remain untouched.

## Evaluation-only contact-response diagnostic

The generic paired MuJoCo model compiles the default `solref=[0.004, 1.0]`. A second MuJoCo rollout of the **unchanged actor and reference** changes only the saved contract's `mujoco_contact_solref` to `[0.04, 2.0]`, the response used in earlier DOA experiments. Both-wheel bearing rises from 51.9% to 88.7% and longest jointly qualified interval from 0.025 to 0.115 s, but the maximum qualifying turn is still only 3.44°: **FAIL** under the same gate. Core minimum speed is 0.926 m/s and path RMSE is 0.0168 m. This sensitivity is not an independently trained candidate, contact-parameter calibration, or sim2sim success; it supports investigating the wheel–ground solver/contact law before another reward-only training run. Exact diagnostic hashes are in the [assessment](turn-load-reserve/assessment.json).

Relative to the [generic baseline](turn-baseline-pair.md), the new load reserve improves the Isaac continuous turn from 1.905 s / 46.72° to 2.260 s / 54.64°. Its MuJoCo default counterpart remains effectively broken (0.020 s / 1.07° → 0.025 s / 1.16°). One paired episode per actor does **not** establish a reliable effect size. The requested same-policy sim2sim drift remains unresolved.
