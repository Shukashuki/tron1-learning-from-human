# High-speed turn: first trained baseline, paired sim2sim FAIL

The separate `drift_drl/traj_4` turn reference was trained in Isaac with the existing generic BeyondMimic-style PPO tracker and the original `car_drift_task.json` reward profile (not the later DOA ground-load/axle refinements). The supported original004 checkpoint initialized one fixed 2048-environment, 600-update run with motor-domain randomization; it exited naturally after 29,491,200 environment steps. The final checkpoint SHA-256 is `bcd7dee5ac6049c437cb62571de6a94c320a9797f6094e13be853593caf50373`. A previous 8-environment/one-update native fixture completed. An attempted trial022-parent fixture stopped **before PPO** because that checkpoint's loaded optimizer LR, 3.375e−5, did not match the old LR-sync entrypoint's explicitly required 1e−5; it was not continued or used for this training.

One deterministic final-checkpoint Isaac/PhysX evaluation exported normalized actor SHA-256 `12a5643add301a22e626883fe45955a63ef454898a3733d2f5f4702157f54e97` and saved every 5 ms physical state. The **same actor file** then ran one local MuJoCo episode on reference SHA-256 `da9945c25e6a9afc8ed27c6a3af117b7c5f091a5a8e98834457964a5640b9254`; reconstructing all 350 recorded Isaac policy observations for deployment differed by at most 4.77e−7. Both engines completed the full reference; neither is a hardware or multi-seed robustness result.

| New-task physical measurement, source turn core | Isaac | MuJoCo |
|---|---:|---:|
| Whole-core travel-course turn | 57.10° | 61.61° |
| Minimum whole-core COM speed | 0.860 m/s | 0.905 m/s |
| Root path RMSE | 0.0616 m | 0.0272 m |
| Core samples with both wheels bearing ≥5 N | 96.3% | 49.5% |
| Longest uninterrupted **jointly fast, sliding, supported** window | 1.905 s | 0.020 s |
| Actual turn within a qualifying window | 46.72° | 1.07° maximum |
| [Frozen new-task gate](../../scripts/assess_drift_turn.py) | **PASS** | **FAIL** |

The gate requires ≥0.8 m/s COM speed, ≥8° body sideslip, ≥0.04 m/s mean absolute wheel-ground patch slip, two ground-filtered wheels bearing ≥5 N, ≥0.20 s uninterrupted, and ≥45° actual course change **in the same physical window**. It also requires full-reference completion, ≤0.15 m core path RMSE and ≤0.30 m maximum error. It does not bridge one-step support gaps or substitute body heading for travel direction. This gate is for the new turn task and does not regrade the historical DOA results.

[Four-panel source → kinematics → Isaac → MuJoCo video](turn-baseline/motion_pipeline.mp4) · [Overview](turn-baseline/overview.png) · [paired assessment](turn-baseline/assessment.json) · [render provenance](turn-baseline/render_report.json). The 1600×900 video contains 176 visual frames at 12.5 fps (0.5× playback), was fully decoded after writing, and integrates zero new physics; the contact gate was computed from the original 200 Hz arrays, not from rendered pixels. Its source panel is a schematic view of a *simulated CARLA car*, not a physical captured vehicle.

MuJoCo's left/right wheel contact fractions in the core are 68.0%/75.8%; neither wheel is persistently unsupported, but only 49.5% of samples have both. Saved-pose FK against the declared 0.127 m radius/0.005 m half-width cylinders puts the missing-wheel analytic clearance positive in 98.8% of left and 100% of right unloaded samples. Missing-contact median analytic clearance is approximately 0.675/0.991 mm left/right; this is *geometry at recorded states*, not the actual solver's original manifold, a PhysX separation measurement, or proof of a unique cause. The nominal MuJoCo model has 1 mm contact margins on the floor and wheel geoms, so positive analytic clearance alone cannot identify its contact threshold. There are zero recorded non-wheel ground contacts in the core.

The result suggests prioritizing continuous wheel load/contact during this **turn**, while preserving speed, sideslip and path tracking. It does not support a claim that the actor can perform the requested drift in sim2sim yet. Previous DOA trial outcomes and the initial jump animation remain unchanged.

## Predeclared one-factor retry (no result yet)

The next separate candidate uses [the explicit load-reserve profile](../../config/drift_turn_load_reserve_task.json). Against this new-task baseline, its only physical/reward difference is adding the existing last-5-ms-substep smooth, symmetric wheel ground-normal-load cost with weight −2 and exponential scale 0.20 × live robot weight; the JSON description also changes. The source reference, original004 parent, 2048 environments, 600 updates, seed 42, motor-v1 randomization, friction, controller, clocks, final-only selection and the above two-engine gate remain fixed. A fresh native small fixture and one final-only paired rollout are required; improvement and sim2sim success are unknown.
