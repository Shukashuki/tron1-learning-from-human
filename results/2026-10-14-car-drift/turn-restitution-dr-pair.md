# High-speed turn: training-time restitution randomization, paired sim2sim FAIL

This [pre-result one-factor trial](turn-restitution-dr-protocol.json) kept the original004 parent checkpoint, pinned `drift_drl/traj_4` WF turn, 2,048 environments, 600 PPO updates, seed 42, motor-v1 randomization, rewards, 50/200 Hz clocks, friction 0.15 and [unchanged high-speed turn gate](../../scripts/assess_drift_turn.py) from the [worst-substep candidate](turn-worst-load-pair.md). The only task change beyond descriptive metadata was robot restitution sampled in `[0, 0.3]` during training. A native eight-environment fixture confirmed actual PhysX material draws. Deterministic Isaac evaluation restored robot restitution to zero; MuJoCo used the unmodified audited MJCF, not any soft-contact diagnostic.

Training completed naturally after 29,491,200 environment steps in 1,192 s. The final checkpoint SHA-256 is `70fbab0156c0bd82cf866e73fb437688fc7d2adc25daea6eebf5d578c33d5e70`. Its final native reward audit recorded 57,616 ground-force samples, 14,404 reward reads and four physical samples in the last control window. The exported normalized actor SHA-256 is `804e6cb9bddba4215879921426a63fd77d45a36653f445cf808aabcac18a332b`. The **same actor** completed one deterministic full-reference episode in each engine. Reconstructing all 350 Isaac action observations in the MuJoCo evaluator had maximum absolute error 4.77e−7. This is one-seed evidence, not a robustness estimate or hardware validation.

| Actual 200 Hz turn-core measurement | Isaac | Nominal MuJoCo |
|---|---:|---:|
| Whole-core travel-course turn; not a pass gate | 59.31° | 58.08° |
| Minimum whole-core COM speed | 0.791 m/s | 0.815 m/s |
| Root path RMSE | 0.164 m | 0.170 m |
| Core samples with both wheels bearing ≥5 N | 92.6% | 41.5% |
| Longest uninterrupted *jointly fast, sliding, supported* window | 1.105 s | 0.020 s |
| Largest actual course turn within a qualifying window | 31.54° | 1.43° |
| Frozen new-task gate | **FAIL** | **FAIL** |

The gate requires speed ≥0.8 m/s, body sideslip ≥8°, mean absolute wheel–ground patch slip ≥0.04 m/s, both wheels bearing ≥5 N, ≥0.20 s uninterrupted, and ≥45° actual travel-course change in the **same** physical window, plus full-clip completion and ≤0.15/0.30 m core path RMSE/maximum error. No 5 ms contact gap is bridged. In this run, the whole-core turns near 60° cannot rescue the short qualifying windows or excess path RMSE. Unlike the preceding actor, this candidate also loses Isaac's PASS: its wheel normal-force 95th percentiles rise to approximately 217/173 N from 142/149 N, and two-wheel support falls from 100% to 92.6%. MuJoCo support falls from 54.0% to 41.5%. These paired single-episode differences are evidence against adopting this restitution intervention, not a causal estimate of all possible restitution ranges.

[Source → robot kinematics → Isaac → MuJoCo video](turn-restitution-dr/motion_pipeline.mp4) · [Overview](turn-restitution-dr/overview.png) · [machine-readable paired assessment](turn-restitution-dr/assessment.json) · [render provenance](turn-restitution-dr/render_report.json). The source is a simulated CARLA vehicle, not physical-car motion capture. Robot panels only replay saved poses; native physical arrays determine the gate. The original jump animation and older trials remain intact. Same-policy sim2sim high-speed drift is still unresolved.
