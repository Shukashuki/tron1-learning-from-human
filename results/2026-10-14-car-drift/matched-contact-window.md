# High-speed turn: 20 ms matched-state contact window

This is a **physics diagnostic, not a policy-transfer PASS**. It extends the [5 ms audit](matched-contact-step.md) to one complete 20 ms action window (four native 5 ms steps). The [audit code](../../scripts/audit_drift_matched_contact_window.py) starts MuJoCo independently from each of 128 policy-aligned Isaac states between 1.44 and 4.00 s, with matching root pose, COM twist, and eight joint positions/velocities. Within each window it injects the next four *recorded Isaac joint torques*, without calling the actor or resetting between steps. Both use the same nominal 0.15 contact friction and 200 Hz physics. Initial wheel-center mismatch is at most 0.72 µm. Isaac has two-wheel ≥5 N support on all 512 recorded next steps. A matching MuJoCo-own-rollout four-step replay reproduces all 512 bearing contact counts and generalized positions to ≤1.42e−14, checking that the reset procedure itself is not the source of the discrepancy.

| Same frozen actor's Isaac torque windows replayed in MuJoCo | Original model | 1 mm collision margin + `solref=[0.05, 2]` |
|---|---:|---:|
| Native steps with both wheels ≥5 N | 45.70% | 99.80% |
| Complete 20 ms windows with both wheels ≥5 N throughout | 1.56% | 99.22% |
| Two-wheel support at step 1 / 2 / 3 / 4 | 13.28 / 56.25 / 57.81 / 55.47% | 100 / 100 / 100 / 99.22% |
| Left/right wheel-height RMSE after 5 ms | 0.513 / 0.930 mm | 0.114 / 0.102 mm |
| Left/right wheel-height RMSE after 20 ms | 0.529 / 0.400 mm | 0.835 / 0.776 mm |

The calibrated variant uses the [frozen 1 mm-margin MJCF](turn-worst-load/margin1000-model.xml) (SHA-256 `d557bf853c74b0e80c6f5ea124c427986fe395d95cf1b37c7817e8d5a5f28b6d`) and [contact contract](turn-worst-load/contact-tc005-ratio2-contract.json) (SHA-256 `f37a7d74938f43aff2754c80da00cdd769f0ec700b7a69d3a9fe9b1f51db8073`). The MJCF has an absolute mesh directory specific to this workspace; its hash identifies the audited file, not a portable model path. Its [separate 512-step matched-state 5 ms ledger](turn-worst-load/matched-contact-step-margin1000-tc005.json) has 100% initial bearing manifold and 100% next-step support; left/right next-wheel-height RMSE is 0.113/0.094 mm, with normal-force p95 144/152 N versus Isaac 142/149 N. The higher 20 ms height error shows that short-step agreement does not imply trajectory equivalence.

Crucially, an independent **full 7 s closed-loop** MuJoCo run of the **unchanged Isaac-trained actor** under the calibrated variant still **FAILS** the [unchanged gate](turn-worst-load-pair.md): minimum core speed 0.967 m/s, path RMSE 0.041 m, two-wheel support 91.23%, and longest uninterrupted jointly fast/sliding/supported window only **0.140 s / 4.54°** (required ≥0.20 s / ≥45°). Of the 513 core samples, 45 lack simultaneous ≥5 N support. There is no contact-gap interpolation. Thus strong one-window physical matching is insufficient for this same-policy sim2sim task; the independent windows do not reveal whether accumulated state error, subsequent actor feedback, or both create the long-horizon gaps. This calibration is exploratory and does **not** establish a unique PhysX-equivalent MuJoCo contact law or hardware robustness.

Raw four-step ledgers: [original](turn-worst-load/matched-contact-window-default.json), [calibrated](turn-worst-load/matched-contact-window-margin1000-tc005.json). Independent closed-loop [assessment](turn-worst-load/mujoco-margin1000-tc005-assessment.json) and [report](turn-worst-load/mujoco-margin1000-tc005-report.json) preserve exact model, contract, actor, motion and rollout hashes. The original [paired Isaac/MuJoCo video](turn-worst-load/motion_pipeline.mp4) remains the nominal uncalibrated comparison; it is not footage of this calibrated diagnostic.

### Frozen-actor contact-margin sensitivity

To test whether the remaining single-step support gaps disappear with a slightly larger collision margin, three more **evaluation-only MuJoCo** full-clip episodes held the actor, `solref=[0.05, 2]`, contact friction, controller, reference, 200 Hz clock and gate fixed. Only the robot collision margin changed. Every run completed the reference but **none passed**:

| Robot collision margin | Minimum core speed | Core path RMSE | Core two-wheel ≥5 N support | Longest qualifying interval | Largest qualifying turn |
|---|---:|---:|---:|---:|---:|
| 1.0 mm | 0.967 m/s | 0.041 m | 91.23% | 0.140 s | 4.54° |
| 1.5 mm | 0.931 m/s | 0.030 m | 90.06% | 0.145 s | 5.89° |
| 2.0 mm | 0.937 m/s | 0.039 m | 90.45% | 0.105 s | 3.44° |
| 3.0 mm | 0.948 m/s | 0.038 m | 91.03% | 0.135 s | 4.91° |

The [1.5](turn-worst-load/mujoco-margin1500-tc005-assessment.json), [2.0](turn-worst-load/mujoco-margin2000-tc005-assessment.json), and [3.0 mm](turn-worst-load/mujoco-margin3000-tc005-assessment.json) assessments each link to a corresponding `report.json`; their exact [1.5](turn-worst-load/margin1500-model.xml), [2.0](turn-worst-load/margin2000-model.xml), and [3.0 mm](turn-worst-load/margin3000-model.xml) MJCFs are archived alongside. The margins do not monotonically improve sustained contact. This sweep does not show that further contact-law adjustment is impossible, but offers no reason to call margin increase a transfer fix. The evaluator rejects `solref` time constants above 0.05 s by its pre-existing validation bound, so no additional softer-contact episodes were run.

### Frozen-actor leg residual feasibility check

As a policy-level probe, [`search_drift_leg_residual.py`](../../scripts/search_drift_leg_residual.py) adds constant, mirrored raw-action offsets to the frozen actor's hip and knee outputs, leaving abduction/wheel actions, original clipping, all physics and the assessor unchanged. Two small grids produced **18 separate full MuJoCo closed-loop episodes** at 1 mm margin / `solref=[0.05, 2]`. The zero-residual episode exactly reproduces the earlier calibrated rollout SHA-256 `d28199b462896493fd7f980ddf38da53061b9cf770e396ae5b934f52a78db345`. All nonzero candidates **FAIL** the original gate; these are not Isaac-tested residual actors.

| Hip / knee raw-action offset | Minimum core speed | Path RMSE | Two-wheel support | Longest qualifying interval / turn | Verdict |
|---|---:|---:|---:|---:|---:|
| 0 / 0 (frozen actor) | 0.967 m/s | 0.041 m | 91.23% | 0.140 s / 4.54° | FAIL |
| +0.01 / −0.01 | 0.924 m/s | 0.047 m | 90.45% | 0.235 s / 8.48° | FAIL |
| +0.04 / −0.04 | 0.323 m/s | 0.378 m | 99.22% | 0 / 0° | FAIL |

The +0.01 / −0.01 candidate exceeds the 0.20 s *duration* threshold but not the simultaneous 45° turn threshold. Stronger joint preload can almost eliminate wheel-support gaps but destroys the required high-speed path; a static action bias is therefore not adopted. Complete candidate summaries with exact rollout hashes: [±0.04 grid](turn-worst-load/leg-residual-grid-004.json), [small-offset grid](turn-worst-load/leg-residual-grid-002.json). This does not rule out a state-dependent, trained residual policy, which would need independent paired Isaac and MuJoCo evaluation.

### Contact-impedance physics sweep and closed-loop check

A [predeclared 21-setting `solimp` sweep](turn-worst-load/solimp-matched-window-sweep.json), implemented by [`sweep_drift_matched_contact_solimp.py`](../../scripts/sweep_drift_matched_contact_solimp.py), retained the 1 mm margin, `solref=[0.05, 2]`, friction 0.15, Isaac state/torque windows, and 200/50 Hz clocks. Candidates had to retain both-wheel ≥5 N support throughout at least 99% of the 128 independent 20 ms windows and match each side's Isaac 95th-percentile normal force within 20%; the predeclared tie-breaker minimized final-step wheel-height RMSE. The selected `solimp=[0.5, 0.9, 0.002, 0.5, 2]` met 100% complete-window support and 1.15% maximum force-p95 relative error, but only reduced the combined 20 ms wheel-height RMSE from approximately 0.806 to 0.794 mm. The [full selected matched-window ledger](turn-worst-load/matched-contact-window-solimp050-090-002.json) includes a 512-step MuJoCo self-replay with exact contact counts and ≤7.11e−15 generalized-position error. The [selected MJCF](turn-worst-load/margin1000-solimp050-090-002-model.xml) has SHA-256 `953f6a1cd1bffe2bb34a8aff02b1aa674ddc889aa570bcb30eaf3284192d08f0` and the same workspace-specific absolute mesh path limitation as the earlier variant.

The **unchanged actor** then completed a separate 7 s MuJoCo closed-loop episode on that model, but the [unchanged gate still FAILS](turn-worst-load/mujoco-margin1000-solimp050-090-002-assessment.json): minimum core speed 0.945 m/s (above the accepted 0.8 m/s floor), path RMSE 0.038 m, two-wheel support 91.62%, longest fully qualifying interval 0.185 s, and largest qualifying turn 3.33° (45° required). The [engine report](turn-worst-load/mujoco-margin1000-solimp050-090-002-report.json) records the exact actor, reference, model and rollout hashes. The entire core turns 62.65°, but contact gaps split it into short segments; summing them would change the frozen assay. Better short-horizon contact matching is thus **not** a demonstrated sim2sim drift solution. The next experiment should target accumulated closed-loop contact-state divergence and/or a controller trained with a sustained bearing-contact objective in both engines, not declare this calibration a transferred policy.

### Turn-core closed-loop reseed: pre-turn history is not the main cause

The [reseed audit](../../scripts/audit_drift_turn_core_reseed.py) reconstructs the recorded Isaac pose, COM twist, joints, and previous action at 0, 1.44, 2.00, or 3.00 s; the **same frozen actor** then controls MuJoCo from that state to 4.00 s. At each reseed, the initial wheel-position error is ≤0.31 µm and the first 51D observation / 8D raw action match Isaac to ≤1.20e−7 / ≤9.84e−7. The 0 s control reproduces the original MuJoCo episode **exactly** for all first 800 native steps: zero generalized-position, actor-action, torque, and wheel-normal-force error, with 100% matching contact counts. These controls validate the state/action alignment of the later counterfactuals.

| MuJoCo contact model; reseed time | First two-wheel support loss | Two-wheel support until 4 s | Contact-loss onsets | Longest supported stretch |
|---|---:|---:|---:|---:|
| 1 mm margin / original impedance; 1.44 s | 1.510 s (70 ms later) | 91.21% | 45 | 0.160 s |
| 1 mm margin / selected impedance; 1.44 s | 1.510 s (70 ms later) | 91.80% | 42 | 0.215 s |
| 1 mm margin / selected impedance; 2.00 s | 2.030 s (30 ms later) | 90.75% | 36 | 0.210 s |
| 1 mm margin / selected impedance; 3.00 s | 3.030 s (30 ms later) | 93.50% | 13 | 0.180 s |

The full [original-impedance](turn-worst-load/turn-core-reseed-margin1000-tc005.json) and [selected-impedance](turn-worst-load/turn-core-reseed-solimp050-090-002.json) ledgers retain per-native-step support, force, state error, and torque error. This **rules out pre-turn accumulated drift as a sufficient explanation**: even when it is erased at the exact Isaac turn state, MuJoCo regains contact gaps within 30–70 ms. It does not prove a unique physical or control cause; the actor and contact dynamics immediately close a different loop after reseeding. The longest *supported* stretch in this table ignores speed, sideslip, slip, and turn angle; these partial, artificially reseeded continuations are **not** full-reference sim2sim PASS attempts. The existing all-substep worst-load Isaac reward and 50 Hz contact-blind actor also explain why merely scoring a short 20 ms physics window is a weak surrogate for uninterrupted multisecond contact; a shared contact-aware controller or longer-horizon cross-engine training experiment is the next substantive test.
