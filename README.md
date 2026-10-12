# TRON1 Learning from Human

Early jump demonstration — CMU `16_03`, DC1600 policy: takeoff, landing, and recovery in Isaac and MuJoCo. Click the preview to watch.

[![TRON1 jump demonstration: Isaac and MuJoCo](results/2026-10-03-sim2sim/overview.png)](results/2026-10-03-sim2sim/tracking_comparison.mp4)

[Jump video](results/2026-10-03-sim2sim/tracking_comparison.mp4) · [Assessment](results/2026-10-03-sim2sim/assessment.json) · [Original pilot video (MuJoCo ended early)](results/2026-10-02-tron1-jump/tracking_comparison.mp4)

## 1. Motivation

Can a wheel-legged robot reproduce selected human-inspired motions and transfer the same learned controller between physics simulators?

This project combines the TRON1 WF robot model with publicly available CMU motion-capture data. We build robot motion references, learn feedback control in Isaac Sim, and evaluate the same policy in MuJoCo without retraining. The goal is a small, reproducible sim2sim study—not full human-motion imitation or hardware deployment.

## 2. Method

### Human motion to robot reference

We decode CMU ASF/AMC recordings and extract three positional landmarks: the pelvis and both ankles. Pelvis orientation is also used. After heading alignment, uniform scaling, and morphology-dependent offsets, the pelvis maps to the robot base and the ankles map to wheel-center targets.

A TRON1-specific adapter uses GMR's two-stage inverse kinematics to generate the floating-base pose and six leg-joint trajectories. Wheel-center motion is included; wheel spin is not a human-derived tracking target. These are kinematic references, not evidence that the motion is dynamically feasible. The training export adds velocities, resamples to 50 Hz, and appends a terminal hold.

Forward jump, turning jump, sideways jump, and step-up use this GMR pipeline. Rolling stop and crouch instead use explicit wheel-task adaptations: duration is doubled, and displacement or crouch depth is adjusted. They are not unmodified human imitation or direct GMR outputs. See the [reference recipes](config/suite/motion_suite.json).

The car-drift extension uses a captured DOA/CARLA vehicle demonstration, not human mocap or physical-car capture. It shows steering-induced inertia drift without throttle or hand-brake input, starting already moving. Vehicle XY is scaled by 0.08 and time by 2; heading is retained and the differentiated COM sideslip profile is approximately retained (maximum discrepancy 0.396°). Bounded leg IK supplies an upright, grounded robot reference; wheel spin cancels rolling velocity but allows lateral slip. A disclosed 1.5 s synthetic stop and 0.5 s stationary tail complete the reference. This is a separate task adapter, not GMR applied to a car.

### Reference to learned control

We use BeyondMimic-style motion tracking with PPO, reusing its motion commands, tracking rewards, and adaptive reference-state initialization through a custom TRON1 configuration. The policy learns feedback control in physics; the base trajectory is not prescribed during rollout.

The eight policy outputs are clipped elementwise to `[-1, 1]` and split into six leg actions and two wheel actions:

$$
a = \operatorname{clip}(\pi_\theta(o), -1, 1), \qquad a \in [-1,1]^8
$$

$$
q_{\mathrm{des,leg}} = q_0 + (1.0\;\mathrm{rad})\,a_{\mathrm{leg}},
\qquad
\tau_{\mathrm{cmd,wheel}} = (12\;\mathrm{N\,m})\,a_{\mathrm{wheel}}.
$$

Here, `q₀` is the fixed leg pose at reference frame zero—not the time-varying reference. Leg targets use PD control (default `Kp = 500`, `Kd = 10`; overrides are recorded per trial); wheels use direct torque commands. Applied torques pass through the DC-motor torque–speed envelope and shaft-resistance model, with nominal output limits of 80 N·m for legs and 12 N·m for wheels. These are simulation settings, not verified hardware ratings.

There is no reference wheel torque: PPO learns wheel actions from tracking rewards and interaction with the simulator. Wheel-center position/velocity remain tracking objectives, but wheel axial rotation is excluded. The [control contract](training/tron1_tracking.py) defines the exact observations and actions.

## 3. Experiments and Training Environment

| Item | Setup |
|---|---|
| Training | RTX 4090 D; Isaac Sim 5.1 / PhysX; Isaac Lab 2.3; Torch 2.7; RSL-RL 3.0.1 |
| Cross-simulator evaluation | MuJoCo 3.9.0; Torch 2.5.1; same actor and observation normalization |
| Control | 50 Hz policy, 200 Hz physics; 51 actor observations, 8 actions |
| Budget per motion | 2,048 environments × 600 PPO updates; approximately 29.49 million environment steps; seed 42 |

Each CMU motion has a separate policy fine-tuned from the same previous jump checkpoint, not trained from scratch. Car-drift retries use the declared resume checkpoint in the experiment log. We evaluate the predesignated final checkpoint. Training randomizes wheel shaft resistance (0–0.3 N·m), motor no-load speed (0.85–1.00× nominal), and torque capacity (0.85–1.00× nominal). Parameters stay fixed within each episode; 20% of resets use nominal parameters.

Human-motion demonstrations use **human keypoints → kinematic reference → Isaac policy → MuJoCo policy**, with synchronized task time and height curves. The car task substitutes captured vehicle motion in the first panel and plots sideslip, speed and wheel-ground slip. Policy panels replay recorded physics, not reference animation. Adaptations and terminal holds are labeled; all videos use 0.5× playback.

| Motion / CMU clip | Result in Isaac / MuJoCo | Four-panel video |
|---|---|---|
| Forward jump / `16_05` | Pass; base rise 13.45 / 14.77 cm | [Watch](results/2026-10-03-motion-pipeline-suite/forward_jump/motion_pipeline.mp4) |
| Turning jump / `83_51` | Pass under the defined gates, but only 33.32° / 38.90° of airborne turn—not 90° | [Watch](results/2026-10-03-motion-pipeline-suite/turn_jump/motion_pipeline.mp4) |
| Sideways jump / `141_05` | Fail; insufficient rise and sideways excursion; reference defects retained | [Watch](results/2026-10-03-motion-pipeline-suite/side_jump/motion_pipeline.mp4) |
| Rolling stop / `16_08`, adapted | Pass; starts already rolling, not from standstill | [Watch](results/2026-10-03-motion-pipeline-suite/rolling_stop/motion_pipeline.mp4) |
| Crouch / `134_01`, adapted | Pass; depth 16.36 / 16.42 cm against an 18 cm reference | [Watch](results/2026-10-03-motion-pipeline-suite/crouch/motion_pipeline.mp4) |
| Step-up / `83_03` | Fail; neither rollout finishes with both wheels on the 11.64 cm ledge | [Watch](results/2026-10-03-motion-pipeline-suite/step_up/motion_pipeline.mp4) |

Four of six tasks pass the predefined gates in both engines, with the turning-jump caveat above. Each task has only one nominal-condition evaluation per engine, starting from reference frame zero and its initial velocities. This is not a statistical success rate or proof of robustness for all six motions. Contact evidence is incomplete, and no hardware validation is claimed. The ledge geometry is estimated from mocap, not measured from the original CMU scene.

[Video gallery](results/2026-10-03-motion-pipeline-suite/index.html) · [Metrics and assessments](results/2026-10-03-motion-suite/summary.json) · [Detailed experiment log and reproduction instructions](EXPERIMENT_LOG.md)

Car drift uses actual ground-contact measurements at **200 Hz in both engines**. The same-actor baseline fails: Isaac sustains 0.28 s of qualifying drift, but MuJoCo reaches only 0.03 s because of real contact micro-gaps; both fail stable-exit sideslip. A higher-damping retry also fails, with MuJoCo terminating at 2.22 s. Historical snapshots remain available; the original 50 Hz Isaac baseline is not overwritten.

| Car-drift trial | Result | Four-panel video / assessment |
|---|---|---|
| Baseline, common PD 500/10 | Fail; continuous drift and exit do not pass in both engines | [Watch](results/2026-10-11-car-drift/trial001-physics/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial001-physics/assessment.json) |
| Stop retry, common PD 500/30 | Fail; unstable wheel support, early MuJoCo termination | [Watch](results/2026-10-11-car-drift/trial002/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial002/assessment.json) |
| Stable retry, common PD 500/10 | Isaac passes (0.385 s drift); MuJoCo recovers but reaches only 0.11 s continuous drift, so sim2sim fails | [Watch](results/2026-10-11-car-drift/trial003/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial003/assessment.json) |
| Contact-stability retry, same PD | Isaac passes (0.45 s drift); MuJoCo reaches 0.15 s, so sim2sim still fails | [Watch](results/2026-10-11-car-drift/trial004/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial004/assessment.json) |
| All-substep reward retry, same PD | Fail in both; zero qualifying drift, with tracking and stopping regressions | [Watch](results/2026-10-11-car-drift/trial005/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial005/assessment.json) |
| Lower-stiffness retry, common PD 250/10 | Fail; 0.46 / 0.12 s drift, with worse path, height and stopping | [Watch](results/2026-10-11-car-drift/trial006/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial006/assessment.json) |
| Frozen-actor diagnostic, common deployment PD 400/10 | Isaac passes (0.52 s); MuJoCo reaches only 0.13 s; zero additional training | [Watch](results/2026-10-11-car-drift/trial007/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial007/assessment.json) |
| Frozen-actor damping diagnostic, common deployment PD 500/12 | Isaac passes (0.43 s); MuJoCo reaches only 0.15 s; zero additional training | [Watch](results/2026-10-11-car-drift/trial008/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial008/assessment.json) |
| Opposite damping diagnostic, common deployment PD 500/8 | Isaac passes (0.46 s); MuJoCo reaches only 0.145 s; zero additional training | [Watch](results/2026-10-11-car-drift/trial009/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial009/assessment.json) |
| Ground-load reserve retry, original PD 500/10 | Fail in both; zero qualifying drift and worse reference tracking | [Watch](results/2026-10-11-car-drift/trial010/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial010/assessment.json) |
| Unchanged-objective continuation, original PD 500/10 | Fail; 0.405 / 0.140 s continuous drift, with tracking and stopping regressions | [Watch](results/2026-10-11-car-drift/trial011/motion_pipeline.mp4) · [Assessment](results/2026-10-11-car-drift/trial011/assessment.json) |
| LR-synchronized continuation, original PD 500/10 | Isaac passes (0.495 s); MuJoCo reaches 0.165 s; tracking and stopping pass, but sim2sim still fails | [Watch](results/2026-10-12-car-drift/trial012/motion_pipeline.mp4) · [Assessment](results/2026-10-12-car-drift/trial012/assessment.json) |
| LR-synchronized load-reserve retry, original PD 500/10 | Isaac passes (0.490 s); MuJoCo reaches 0.170 s; tracking and stopping pass, but sim2sim still fails | [Watch](results/2026-10-12-car-drift/trial013/motion_pipeline.mp4) · [Assessment](results/2026-10-12-car-drift/trial013/assessment.json) |
| LR-synchronized all-substep retry, original PD 500/10 | Isaac passes (0.435 s); MuJoCo reaches 0.175 s; all other behavior gates pass, but sim2sim still fails | [Watch](results/2026-10-12-car-drift/trial014/motion_pipeline.mp4) · [Assessment](results/2026-10-12-car-drift/trial014/assessment.json) |
| Stronger signed-sideslip retry, original PD 500/10 | Isaac passes (0.550 s); MuJoCo reaches 0.120 s; all other behavior gates pass, but sim2sim still fails | [Watch](results/2026-10-12-car-drift/trial015/motion_pipeline.mp4) · [Assessment](results/2026-10-12-car-drift/trial015/assessment.json) |
| Physics-rate ground-load cost retry, original PD 500/10 | Isaac passes (0.495 s); MuJoCo reaches 0.135 s; all other behavior gates pass, but sim2sim still fails | [Watch](results/2026-10-12-car-drift/trial016/motion_pipeline.mp4) · [Assessment](results/2026-10-12-car-drift/trial016/assessment.json) |

The explicit MuJoCo contact-compliance calibration is softer than the baseline, not hardware-identified or claimed equivalent to PhysX. Contact-stability shaping reduces leg-command changes and improves cross-engine root RMSE to 1.18 cm, but six genuine 5 ms right-wheel contact gaps still split the drift. Fixed-policy 400/800 Hz MuJoCo diagnostics do not resolve the contact loss and are not paired successes. See the [experiment log](EXPERIMENT_LOG.md) for profiles and limitations.

Trial005 tests verified all-substep reward sampling but loses the target behavior: sideslip at speeds ≥0.4 m/s peaks at only 5.59°/5.30°, and stopping also fails. Both engines agree closely but are wrong relative to the reference. Its actor is not adopted; trial004 remains the better, still-failing candidate.

Trial006 resumes trial004 and halves common leg stiffness, retaining its rewards and sampling. It also fails: path RMSE grows to 0.68 / 0.80 m, with more MuJoCo wheel unloading and delayed drift. Its actor is not adopted; neither lower stiffness nor the sampling retry establishes successful transfer.

Trial010 adds a verified ground-normal load-reserve reward, but the final actor loses the target drift: moving sideslip peaks at 6.71° / 7.32°, below the unchanged 8° gate, and path RMSE grows to 0.28 / 0.26 m. Both finish and stop with full tail support, but that is not motion imitation. This actor is not adopted; trial004 remains the better, still-failing baseline. Continued PPO and the new reward are not causally separated by this single run.

Trial011 continues trial004 for the same 600 updates without changing its objective or controller. Both complete the motion, but path RMSE grows to 0.272 / 0.275 m and tail maximum speed reaches 0.438 / 0.440 m/s, exceeding the fixed 0.15 m and 0.20 m/s gates. MuJoCo's 0.140 s uninterrupted drift also falls short of 0.20 s. Unchanged-objective continuation can therefore regress too; this single-seed control does not identify the cause. Its actor is not adopted, and the original jump animation and previous trials remain unchanged.

Trial012 synchronizes PPO's learning-rate scalar with the restored optimizer immediately after loading trial004, then runs the same 600 updates; adaptive scheduling and observation-normalizer updates continue normally. Both recover reference tracking and stopping, with root RMSE of 0.036 / 0.037 m and tail maximum speed of 0.036 / 0.073 m/s. However, MuJoCo's 0.165 s uninterrupted drift remains below 0.20 s, so this actor is not adopted as a successful sim2sim policy. This single-seed comparison does not establish causality or statistical robustness.

Trial013 holds that initial LR synchronization fixed and adds the original wheel-ground load-reserve penalty, again starting from trial004 for 600 updates. MuJoCo's longest drift increases by only one 5 ms sample to 0.170 s; genuine wheel-support gaps remain. Root RMSE is 0.040 / 0.045 m, and total qualifying drift decreases in both engines. All other original behavior gates pass, but this final actor is not a successful transfer. Contact gaps are not bridged and the 0.20 s gate is unchanged.

Trial014 instead keeps LR synchronization and averages the original squared wheel-link vertical-velocity cost over all four physical substeps, without a load-reserve reward. After the same 600 updates, uninterrupted drift is 0.435 / 0.175 s and root RMSE is 0.047 / 0.050 m. Tracking and stopping pass, but genuine MuJoCo wheel-unloading gaps remain; averaging this velocity cost does not guarantee continuous support. The same final actor is verified in both engines, and the unchanged 0.20 s gate still fails.

Trial015 holds that sampling fixed and increases signed-sideslip reward weight from 2 to 4, again resuming trial004 for 600 updates. Drift begins earlier, but MuJoCo's contact interruptions become more frequent: uninterrupted drift is 0.550 / 0.120 s, despite more total qualifying drift. Root RMSE is 0.044 / 0.041 m and stopping passes. The same final actor is verified in both engines; gaps are not bridged, and this single-seed result is not a successful transfer or proof of robustness.

Trial016 instead compares with trial013: only the load-reserve cost changes to the mean of four instantaneous physics-substep costs, retaining the original last-substep wheel-vertical-velocity reward. After 600 updates from trial004, uninterrupted drift is 0.495 / 0.135 s and root RMSE is 0.047 / 0.044 m. Tracking and stopping pass, but MuJoCo bearing interruptions remain. Averaging the soft load cost does not enforce continuous support; the unchanged 0.20 s gate fails. Both engines use the verified final actor, and no new wheel teacher is introduced.

Trial007 freezes trial004's actor byte-for-byte and changes only deployment Kp to 400. Tracking and stopping pass in both engines, but MuJoCo's 0.13 s continuous drift still fails. This is a zero-shot controller diagnostic—not a policy trained at Kp 400—and the gain setting is not adopted.

Trial008 retains the same actor weights and normalizer, changing only deployment Kd to 12. Tracking and stopping pass, but MuJoCo still reaches only 0.15 s uninterrupted drift. Its exported file has different serialization bytes; all tensors and all 288 old/new observation outputs match exactly, and both engines use the same actual export. The damping change is not adopted.

Trial009 tests the opposite damping direction, Kd 8, without new training. It also preserves tracking and stopping but fails MuJoCo's continuous-drift gate at 0.145 s. No tested gain change is adopted, and this bounded scalar-gain diagnostic sequence ends here.

A separate [wheel-contact diagnostic](results/2026-10-11-car-drift/normal_contact_diagnostic.json) rejects a proposed MuJoCo-to-PhysX spring mapping. A [native-plane check](results/2026-10-11-car-drift/native_plane_diagnostic.json) reduces contact oscillation in the nominal surrogate but still fails identification; neither yields adopted parameters.

Drift validation requires planar speed at least 0.4 m/s with a positive forward component, sideslip at least 8°, both wheels bearing ground contact and measured contact-patch lateral slip at least 0.04 m/s, concurrently for 0.20 s, followed by stable stopping. Spins, flight and reference playback do not pass. The source starts already rolling; its approximate clock is reconstructed from a CSV without timestamps. [Captured car preview](results/2026-10-11-car-drift-source/source_preview.mp4) · [Pinned source configuration](config/drift_sources.json) · [DOA license](notices/DOA-LICENSE.txt).

Credits: [CMU data sources and acknowledgment](config/mocap_sources.json), [pinned GMR adapter](config/gmr_retarget_wf.json), [BeyondMimic license](notices/BeyondMimic-LICENCE.txt), and [LimX TRON1 model license](notices/LimX-TRON1-LICENSE.txt). Third-party code, models, and data retain their original terms; raw mocap, robot assets, and checkpoints are not redistributed here.
