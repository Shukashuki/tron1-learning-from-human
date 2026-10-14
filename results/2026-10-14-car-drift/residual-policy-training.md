# High-speed turn: MuJoCo residual-policy training, sim2sim still FAIL

The objective and [unchanged physical gate](turn-worst-load-pair.md) remain one **same-policy** high-speed, sliding, both-wheel-supported ≥45° course turn in Isaac and MuJoCo. The user accepted a sufficiently high speed rather than zero deceleration; the fixed minimum remains 0.8 m/s. This experiment does **not** satisfy the objective. The frozen Isaac-trained parent actor already passes nominal Isaac but fails MuJoCo; every new residual candidate here still fails MuJoCo, so no residual actor is promoted as a paired result.

## Method and validity controls

[`drift_mujoco_training_env.py`](../../scripts/drift_mujoco_training_env.py) reuses the deployment evaluator's exact 51D observation, frozen normalized parent actor (SHA-256 `20bcb91bc8a9200d5dfbfb4a1f322140f8f88725697d0771ea60c5266355ba31`), 8D action order/clipping, DC motor/PD law, 50/200 Hz clocks, reference and termination. Its [full-episode parity control](residual-ppo/training-env-parity.json) matches all **349 actor decisions and 1,396 physical steps** of the frozen 1 mm-margin / `solref=[0.05, 2]` MuJoCo evaluator: maximum saved actor-observation, generalized-position, applied-torque and wheel-ground-force errors are all **zero**, with all contact counts identical. This proves the training scaffold starts from deployment-identical dynamics, not sim2sim success.

[`train_drift_mujoco_residual.py`](../../scripts/train_drift_mujoco_residual.py) freezes the parent actor and trains a 51D→8D bounded residual and critic with PPO. The [dense reward](../../training/drift_mujoco_residual_reward.py) retains path/velocity/heading tracking and uses all four native contact/slip samples per action. The [streak reward](../../training/drift_mujoco_streak.py) pays for *record improvements in one uninterrupted physically qualifying travel-course turn*, without bridging gaps. Its largest angle and duration match the original assessor exactly for the three archived test trajectories in this [read-only reward audit](residual-ppo/reward-alignment.json). With the streak term, the promising 0.235 s / 8.48° static-residual trajectory scores 414.8 versus the parent actor's 381.7; the 99.2%-supported but slow/non-sliding trajectory scores only 175.6. The streak reward therefore fixes the earlier ordering error, but remains **training shaping**, never the acceptance test.

Exploration standard deviation about 0.020 caused only 6/8 zero-mean noisy episodes to finish, versus 8/8 at about 0.005 with the same seed. The latter was used for the complete predeclared 50-update trial. The three earlier 10-update runs were engineering pilots during source revisions; their [manifests and per-update metrics](residual-ppo/) are preserved but are not a controlled generalization comparison. A [TorchScript export control](residual-ppo/export-parity-trial03.json) on an earlier checkpoint confirms max action error 3.05e−7 on actual saved observations, zero full-rollout generalized-position error and identical contact counts when the exported actor is replayed by the original MuJoCo deployment evaluator. It is **not** an Isaac transfer check.

## Bounded 200-episode PPO result

The audited [trial04 manifest](residual-ppo/trial04-manifest.json) fixes seed 42, 50 updates × 4 full-reference attempts, the 1 mm-margin calibrated model, and the original parent actor. All 200 exploratory training episodes completed the full reference at the lower noise level. A deterministic full physical evaluation and checkpoint were saved at update 1, every five updates, and final update 50. [Every scheduled assessment](residual-ppo/) is **FAIL**.

| Frozen gate, actual 200 Hz core | Parent actor | Best observed checkpoint, update 5 | Final checkpoint, update 50 |
|---|---:|---:|---:|
| Minimum core speed | 0.967 m/s | 0.905 m/s | 0.947 m/s |
| Core path RMSE | 0.041 m | 0.026 m | 0.036 m |
| Two-wheel ≥5 N support | 91.23% | 90.06% | 91.03% |
| Longest uninterrupted jointly qualifying interval | 0.140 s | 0.125 s | 0.110 s |
| Largest course turn in a qualifying interval | 4.54° | **5.71°** | 4.51° |
| Required gate | ≥0.20 s and ≥45° in the same interval | **FAIL** | **FAIL** |

Even the best angle is over 39° short; the progression is not monotonic. The [per-update ledger](residual-ppo/trial04-metrics.jsonl), [best](residual-ppo/trial04-eval-0005-assessment.json) and [final](residual-ppo/trial04-eval-0050-assessment.json) physical assessments, [checkpoint 5](residual-ppo/trial04-checkpoint-0005.pt) and [checkpoint 50](residual-ppo/trial04-checkpoint-0050.pt) preserve negative results. Torch checkpoints are executable pickle-format artifacts; load only from this trusted experiment source. The original jump animation and earlier drift comparisons remain untouched.

## Contact-loss precursor

The [read-only 200 Hz precursor audit](residual-ppo/contact-precursors.json) of the calibrated parent rollout finds 24 left and 20 right loss onsets in the 1.44–4.00 s turn core. Immediately *before* an onset, median normal force is still 113 N left / 129 N right, then the next sample has no bearing contact. This is an abrupt 5 ms loss, not a gradual decline through 5 N. Previous-sample wheel-link height exceeds the stable-sample 90th percentile for 62.5% left / 85% right onsets; backward-difference upward speed does so for 54% / 65%. These are imperfect retrospective associations, not a validated predictor or causal feedback test. Wheel-link height is not exact cylinder-plane clearance. A future physical-rate contact controller or augmented shared observation would need its own paired physics verification; simply extending this low-dimensional PPO run is unsupported by the observed trend.
