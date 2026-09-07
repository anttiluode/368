# Review of 368: fast adaptation, retained knowledge, and structural admission

Reviewed 7 September 2026. Source: [anttiluode/368, commit 543a022](https://github.com/anttiluode/368/tree/543a0225e796a7d8b9075fa911e42f3c5316635e). This is a read-only review with additional experiments, not a change to the repository or an independent external replication.

## Main finding

The implementation makes a useful, causally valid distinction between proposing a predictor and admitting it after subsequent evidence. Its published numerical results reproduce exactly. However, the main single-model baseline does not forget old observations. Adding ordinary exponential forgetting produces a much stronger nonsplitting comparison, and changes the interpretation of the growth result.

On the same 100 switching worlds, the new single model has mean MSE **0.02508**, compared with **0.12206** for the prospective grower. The grower is better during the first 16 decisions after familiar contexts return, and under noise bursts. The result is a measurable adaptation/retention trade-off. It does not yet establish an overall advantage for structural growth.

## What was verified

- Read the implementation, documentation, tests, committed receipts, branch list, and workflow status at the pinned commit.
- Executed all five existing test functions directly; all pass. The published GitHub CI run also passed.
- Reran 800 Gate 0 evaluations covering the switching, noise-burst, and quadratic controls. Every compared value in the compact 100-world receipt matches exactly.
- Reran all 400 Gate 1 evaluations, covering two learners with and without probes on 100 worlds. All reported aggregate MSE, return MSE, probe rates, and birth rates match exactly.
- Perturbed current and future outcomes at four cuts in a stream. The growing learner's predictions already due remained unchanged.
- Checked the new discounted RLS update against an independent, batch exponentially weighted least-squares solution at all six development settings.

The first Gate 0 test is named as a causality test but originally checks determinism only. The additional perturbation check tests the information boundary directly. In the upstream runner an outcome is sampled before `predict`, but it is not passed to the learner until `observe`; code inspection and the perturbation check do not indicate outcome leakage.

## The missing adaptive baseline

In [benchmark.py](https://github.com/anttiluode/368/blob/543a0225e796a7d8b9075fa911e42f3c5316635e/benchmark.py), `WeightedRLS.update` contains no forgetting factor, window eviction, or reset. For `SingleLearner`, all completed observations retain equal weight in the accumulated least-squares fit. As multiple maps are encountered, this becomes a compromise over their history. Calling it continual overwriting obscures that behavior.

By comparison, each growth candidate is freshly trained on the latest 40 observations. Thus the single-versus-growing comparison changes both recency handling and structural capacity.

The review baseline keeps the same four coefficients and 4-by-4 RLS covariance matrix, with one added scalar discount. Immediately before each update on a completed observation it divides the covariance by the discount. This discounts old evidence; no context labels, change times, diagnostic observations, or extra experts are supplied.

The discount was selected by lowest mean switching-world MSE on development seeds 0-19 from the fixed grid `[0.5, 0.7, 0.85, 0.95, 0.98, 1.0]`. The selected value was **0.5**. Only afterward was it evaluated on seeds 1000-1099. These are reused evaluation worlds from the repository, not a newly collected blind test set. Development selection optimized the switching condition, not a joint objective over all controls.

| Method | Switching MSE | First 16 decisions after a familiar context returns | New experts |
|---|---:|---:|---:|
| Original single model | 0.21142 | 0.21709 | 0 |
| Fixed bank, three supplied experts | 0.09940 | 0.08040 | 0 |
| Prospective grower | 0.12206 | 0.11254 | 2.07 |
| Error-only grower | 0.07693 | 0.05886 | 3.98 |
| **Single model with forgetting, new review baseline** | **0.02508** | **0.16503** | **0** |

The paired difference, forgetting minus growing, is **-0.09698 MSE**, with bootstrap 95% interval **[-0.10304, -0.09103]** across the 100 world pairs. On the early-return metric, the difference instead favors the grower: **+0.05249**, interval **[+0.03661, +0.06858]**.

The forgetting model also has higher noise-burst MSE: **0.03911 versus 0.02147**, a paired difference of **+0.01764**, interval **[+0.01685, +0.01846]**. Its quadratic-control MSE is **0.07126 versus 0.03751**. Fast adaptation is therefore not a universal replacement for memory or restraint. Early-return performance is consistent with useful retention in the bank; it does not by itself identify what each anonymous expert stores.

An additional control retains the same nomination, 40-event training, 32-decision audit, and admission threshold, but replaces the old model whenever a candidate is accepted. It keeps one persistent expert. This reaches **0.18915 MSE**, with **3.51 replacements**, and early-return MSE **0.48030**. Simply turning the current delayed admission procedure into a reset rule is insufficient. Its poor result does not negate the stronger continuously adapting baseline.

## Claims that need narrowing

### Rejecting a candidate is not identifying the cause of error

Noise bursts and the missing quadratic feature both lead to zero admissions. The learner does not output different diagnoses for them. The richer feature is supplied manually in a separate comparator; the growing learner does not discover or choose it.

Furthermore, the admission criterion is a fixed **0.15 reduction in mean squared error**. This is a large absolute threshold relative to these controls. Zero births therefore establish restraint at the tested scale, not a general ability to distinguish irreducible noise from a wrong representation.

A useful hard positive control follows directly from the action set. For two true response maps separated by Frobenius distance `d`, uniform legal actions and readouts give expected squared disagreement `d^2 / 4`. Two different maps at the protocol's allowed minimum separation `d = 0.6` differ by only **0.09 expected MSE**. Even a perfect new predictor compared with a still-perfect predictor of the previous map would then have less expected advantage than the 0.15 admission threshold. This calculation assumes independent mean-zero observation noise and the original uniform scored-query distribution. Finite-window sampling can differ. The important test is whether the rule rejects small but real new contexts as well as noise.

### Future candidate accuracy does not yet price retained structure

The candidate is scored alone against the existing mixture. That shows whether a recent predictor is useful, but does not compare the actual deployed enlarged mixture with the best alternative use of the same evidence.

`complexity_penalty` is an MSE threshold; it does not vary with measured memory, extra model evaluations, or interference. `conflict` and `protect_threshold` are present but unused. State accounting is approximate and excludes, among other things, the growing admission log. The six-expert cap is a safeguard, not evidence that growth remains well regulated over an unbounded stream.

These are extensions still required by the [Paper protocol](https://github.com/anttiluode/Paper/blob/006930e6ead02be8ba9c4cd759e3cd56aa54de18/PROTOCOL.md), rather than a reason to discard this first implementation.

### Gate 1 establishes an extra-observation benefit, with attribution still open

The fixed bank's reported MSE improvement **0.10082 to 0.07461**, at **0.03390 probes per decision**, reproduces exactly. Diagnostic outcomes update responsibility, and the scored query is chosen independently. Those are useful design choices.

However, there is no random-probe or periodic-probe comparison at a matched observation budget. Thus the experiment does not yet isolate how much of the benefit comes from choosing a high-disagreement query, from choosing when to ask, or simply from receiving extra evidence.

Diagnostic and scored observation noise also consume the same world random-number generator. Buying a probe advances that generator and changes subsequent scored noise. Maps, schedules, and scored actions remain matched across the comparison, but scored noise samples do not. This is not outcome leakage and does not automatically invalidate the aggregate improvement. Separate streams would make the paired comparison cleaner.

### Gate 2 is not auditable from this snapshot

The attachment describes active candidate admission and explicitly says it was left uncommitted. At the reviewed commit, the repository has no `active_admission.py`, Gate 2 receipt, or Gate 2 implementation. Its claimed 32-to-16-decision admission improvement cannot be verified here. A future comparison should report decisions and total paid observations separately.

## Recommended next experiment

The next comparison should include a fast current predictor and a bank of retained predictors. When a new candidate succeeds, evaluate the competing actions using the same subsequent evidence:

1. Update or replace the fast current predictor.
2. Retrieve an already retained predictor.
3. Preserve the candidate as an additional reusable specialization.

Compare the full prediction policies, rather than only a candidate alone versus a blended incumbent. Score overall error, early return to familiar conditions, noise robustness, retained conditional capabilities, probe counts, and memory. Give random and disagreement-based probes comparable budgets and share an independent scored-noise stream. Include close but genuinely different maps as well as the existing easy negative controls.

The most useful question exposed by this review is: **when does preserving another model improve the trade-off beyond what fast adaptation and retrieval can already achieve?** That question builds directly on the paper and on the specific strength observed in 368.

## Files and reproduction

`audit.py` reproduces the experiments. `development.json` records every discount candidate and development seed result. `alternative_baselines.json` contains all new per-world evaluations. `gate0_reproduction.json` and `gate1_reproduction.json` retain the fresh upstream runs. `summary.json` contains the comparisons and intervals.

From this directory:

```bash
git clone https://github.com/anttiluode/368.git 368
git -C 368 checkout 543a0225e796a7d8b9075fa911e42f3c5316635e
python -m pip install numpy==2.3.5
python audit.py --repo 368 --out rerun
```

AI-assisted analysis and code by ChatGPT at Antti Luode's request. All numerical review results were executed; proposed follow-up experiments remain proposals.
