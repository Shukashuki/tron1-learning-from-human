# MuJoCo wheel–ground contact-point audit (2026-10-13)

This is a diagnostic of the **existing failed sim2sim drift episodes**, not a new policy, corrected gate, or claim of success. The original rollouts stored per-wheel bearing-point counts, normal force and contact-patch slip, but not each `data.contact` point. [`scripts/audit_mujoco_contact_points.py`](../../scripts/audit_mujoco_contact_points.py) replays each episode with its recorded actor, motion, model and contract and captures the native contact manifold at every 5 ms sample. For trial021, trial022 and the 1.25× zero-shot run, **every array in the replayed rollout is bit-for-bit equal** to its historical rollout (1,153, 1,153 and 921 physical samples, respectively). The new point data are still replay measurements, not originally archived raw points. All original policies, physical rollouts, assessor thresholds and FAIL verdicts remain unchanged.

## Point-level finding

Every moving-drift support failure in these three episodes (5 + 5 + 4 samples) is a **truly empty MuJoCo wheel–ground contact manifold**, not a contact that merely falls below the evaluator's 5 N bearing-point threshold. Each gap lasts one 5 ms sample. At that sample the wheel has a positive finite-cylinder/plane clearance of 1.1–142.4 µm; the raw manifold and normal force are zero. At the prior sample its manifold contains one or two points and bears 83.1–134.8 N; at the next sample the wheel has recontacted with signed geometric penetration of 0.694–1.312 mm and bears 98.2–161.1 N. Contact sets follow `1 or 2 → 0 → 1 or 2`; there is **no contact-point position to track inside the gap**. The analytic clearance is a geometric proxy, not a solved contact point or force.

| Episode | Failed wheel, time (s) | Point count: before → gap → after | Gap clearance (µm) | Force: before → gap → after (N) | Recontact signed clearance (mm) |
|---|---|---|---:|---:|---:|
| trial021 | R 0.970 | 2→0→2 | 19.0 | 87.9→0→110.5 | −0.795 |
| trial021 | R 1.115 | 1→0→2 | 91.2 | 86.6→0→109.2 | −0.720 |
| trial021 | R 1.285 | 1→0→2 | 2.7 | 83.5→0→103.6 | −0.802 |
| trial021 | R 1.905 | 1→0→1 | 73.5 | 101.8→0→132.3 | −0.808 |
| trial021 | R 2.010 | 1→0→1 | 28.5 | 111.9→0→142.7 | −0.982 |
| trial022 | R 0.975 | 1→0→2 | 79.9 | 87.5→0→108.8 | −0.768 |
| trial022 | R 1.120 | 1→0→2 | 62.4 | 84.8→0→101.1 | −0.723 |
| trial022 | R 1.280 | 1→0→2 | 3.9 | 83.1→0→98.2 | −0.763 |
| trial022 | R 1.885 | 1→0→1 | 138.6 | 98.5→0→128.6 | −0.694 |
| trial022 | R 1.980 | 1→0→1 | 142.4 | 108.5→0→139.7 | −0.800 |
| fast125 | R 0.875 | 1→0→2 | 34.8 | 88.4→0→109.4 | −0.820 |
| fast125 | R 1.580 | 1→0→1 | 111.4 | 130.2→0→161.1 | −0.948 |
| fast125 | L 1.660 | 1→0→1 | 1.1 | 91.3→0→113.6 | −0.948 |
| fast125 | R 3.890 | 1→0→1 | 11.9 | 134.8→0→156.5 | −1.312 |

For all 6,454 wheel-side/time samples (two wheels × 3,227 physical samples), **zero bearing points coincides with zero raw manifold points**; there are no raw contacts hidden below 5 N. Summed replay point-normal forces match the archived normal-force channels exactly, and point counts match exactly. The initial rollout itself is not modified. Near trial022's 0.975 s event, for example, the right wheel goes from one point at the +25 mm cylinder edge, −9.1 µm signed distance, 87.5 N and +0.172 m/s point lateral slip at 0.970 s; to no point, +79.9 µm clearance and 0 N at 0.975 s; to two edge points, minimum −0.768 mm signed distance, 108.8 N total and about +0.243 m/s slip at 0.980 s. The same edge before/after moves ~11.2 mm in world XY over the 10 ms interval. That displacement is ordinary wheel travel, **not** an instantaneous point teleport; no persistent point ID exists across the empty manifold.

The finite wheel cylinders have radius 0.127 m and half-width 0.025 m. Across all three full replays, 93.2–97.9% of raw wheel–ground contact points lie within 0.1 mm of a cylinder axial edge. For a plane with normal `n` and wheel axis `a`, the ideal finite-cylinder support is `r sqrt(1−(a·n)^2) + b |a·n|`, so its geometric derivative has a cusp at exactly zero camber. **That cusp is not an established cause of these losses**: gap camber spans approximately −0.6° to +6.2°, with several losses far from zero camber. What is directly observed is a discrete contact-set/normal-force change as the rolling wheel makes a small positive separation and then re-enters with finite penetration at the 5 ms sampling clock. The compliant MuJoCo solver and discrete collision detection can make such an impact look abrupt; the recorded data do not isolate solver artifacts from genuine micro-lift, wheel-edge geometry, or controller/physics-clock interaction. The exact within-step crossing time is unobserved.

The fast125 run has another failure mode: **eight supported moving-drift samples** at 3.875–3.920 s (apart from the 3.890 s empty-manifold sample) keep both wheels loaded but have mean absolute contact-patch lateral slip only 0.024–0.040 m/s, below the unchanged 0.05 m/s gate. They must not be counted as contact loss. Faster reference execution reduced moving-drift samples from 146 to 85; fewer raw support gaps do not imply improved robustness. The original paired verdict remains FAIL (trial022 MuJoCo 0.155 s, fast125 MuJoCo 0.145 s versus the 0.200 s continuous-drift requirement).

## Reproduction and limits

The script writes a private complete point ledger (`all_contact_points.json`), ±2-sample event windows and a separate replay rollout under a **new** output directory. It rejects an output directory that already exists and aborts if any historical rollout array differs. It never shifts, fills, or bridges a gap. To reproduce the representative trial022 audit:

```bash
python scripts/audit_mujoco_contact_points.py \
  --historical-report outputs/drift-sim2sim-20261013/trial022/mujoco/report.json \
  --event-decomposition outputs/drift-sim2sim-20261013/trial022/diagnostics/mujoco_contact_breaks.json \
  --output-dir outputs/my-new-trial022-contact-point-audit
```

Run in the existing Torch + MuJoCo 3.9 environment; no package install is needed. The point ledger records geom pair, world and wheel-local point position, signed contact distance, wheel-facing normal, normal/tangent force, point velocity and point lateral slip. MuJoCo contact indices are ephemeral and must not be interpreted as IDs. Repeating this at 400/800 Hz with a held 50 Hz actor could test time-step sensitivity, but it would be a **different physical episode** requiring its own paired-engine contract and cannot replace this 200 Hz result. These measurements alone do not identify a unique causal remedy or validate real-world drift.
