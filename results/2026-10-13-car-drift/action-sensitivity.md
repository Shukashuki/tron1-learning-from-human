# Saved-state MuJoCo action sensitivity at drift contact gaps (2026-10-13)

## Outcome

The original trial021 and trial022 policies still **FAIL** the unchanged 0.20 s continuous grounded contact-patch drift gate in MuJoCo: their longest intervals are 0.165 s and 0.155 s, respectively. This audit is a local diagnostic of the ten isolated right-wheel contact gaps, not a retrained policy, a new full episode, or a sim2sim pass.

| Exact local counterfactual | Trial021 gaps rescued | Trial022 gaps rescued | What it establishes |
| --- | ---: | ---: | --- |
| Perturb one clipped action coordinate for only the immediately preceding 5 ms physics step | 3 / 5 | 1 / 5 | Local next-step contact sensitivity; the 50 Hz actor cannot command this arbitrary substep |
| Perturb one clipped action coordinate at the actual policy decision, then hold it across its four 5 ms steps | 1 / 5 | 3 / 5 | Both wheels bear at **every** step of that isolated 20 ms window, under nominal MuJoCo physics |

The complete unmodified four-step baseline replays the archived position, torque, force, contact count and physical timestamps at every native sample (maximum position error 2.22e-16). The original model, motor/PD law, 50 Hz action hold, 200 Hz integration, rollout hashes, contact thresholds, and behavior verdicts are untouched. Each branch clips one action dimension within its original bounds, with signed perturbations of 0.002, 0.005, 0.01, 0.02, 0.04 or 0.08. The 20 ms success criterion is both wheels' existing force-filtered bearing-contact count >0 at all four physical steps; it does **not** test sideslip, exit stability or subsequent contact.

The four individual full-window rescues were:

| Trial | Original gap time | Original right-wheel clearance | Smallest tested full-window rescue | Right-wheel force at the former gap |
| --- | ---: | ---: | --- | ---: |
| 021 | 1.115 s | +91.2 µm | right knee action +0.04 | 143.7 N |
| 022 | 1.120 s | +62.4 µm | left abduction action +0.02 | 72.4 N |
| 022 | 1.280 s | +3.9 µm | right abduction action −0.002 | 83.3 N |
| 022 | 1.980 s | +142.4 µm | right abduction action +0.01 | 105.6 N |

The other six windows have **no** single-coordinate full-window rescue among the tested perturbations; this does not rule out coordinated actions, earlier anticipation, or other amplitudes. The one-step and whole-window sets differ because the latter action acts on the preceding physical steps and changes the state arriving at the gap. For example, trial022's 1.980 s gap can be prevented by the whole-window branch but not by a last-step single-coordinate perturbation. Do not add the one-step and window counts or interpret either as a success rate over independent rollouts.

## Interpretation and next experiment

The discontinuity is sensitive to small action changes in some saved states, so the fixed nominal MuJoCo geometry does not make *every* gap unavoidable. But six of ten windows are not rescued by this restricted control family, and no branch has been run through the rest of the episode or paired with Isaac. A better-defined next training hypothesis is anticipatory two-wheel support over multiple policy windows, while retaining an explicit drift objective and the original gates. It must be evaluated with a new frozen actor in complete paired 200 Hz episodes, then across seeds and randomized actuator/contact parameters. This audit alone does not validate that intervention or identify a unique contact-law cause.

Reproduction code: [`audit_drift_one_step_action_sensitivity.py`](../../scripts/audit_drift_one_step_action_sensitivity.py) and [`audit_drift_one_window_action_sensitivity.py`](../../scripts/audit_drift_one_window_action_sensitivity.py). Both require the original private `mujoco/report.json`, adjacent `rollout.npz`, and `diagnostics/mujoco_contact_breaks.json` for each trial; they refuse changed input hashes and existing output directories. Their raw branch arrays stay in ignored private `outputs/` directories. Public [trial021](trial021/assessment.json) and [trial022](trial022/assessment.json) assessments retain the original FAIL verdict.
