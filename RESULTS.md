# Gate 0 — prospective structural admission

Status: **executed**. Hyperparameters were fixed after development seeds 0–19; this file reports held-out seeds 1000–1099.

## Main switching world

Hidden sequence: `A → B → A → C → B → A`. Block lengths are independently sampled from 80–160 decisions. Each context is a randomly generated 2×2 linear response map; maps are separated by a minimum Frobenius distance of 0.6. Observation noise σ=0.05. Context labels are unavailable to learners.

| learner | mean MSE | median MSE | mean births | final experts | approximate persistent scalar slots |
|---|---:|---:|---:|---:|---:|
| single | 0.211418 | 0.195783 | 0 | 1.00 | 20 |
| fixed bank | 0.099400 | 0.078810 | 0 | 3.00 supplied | 63 |
| prospective grower | 0.122057 | 0.120288 | **2.07** | 3.07 | 267.67 |
| error-only grower | **0.076929** | 0.068449 | 3.98 | 4.98 | 304.58 |

Paired over the same 100 worlds:

- prospective grower − single: **−0.089361 MSE**, bootstrap 95% interval `[-0.100726, -0.077924]`;
- prospective grower − fixed bank: **+0.022657**, interval `[+0.009084, +0.035236]`;
- error-only grower − prospective grower: **−0.045128**, interval `[-0.049167, -0.041110]`.

The prospective rule therefore buys restraint, not the best raw score. A fixed three-expert architecture performs better while using less tracked state because it is handed the correct capacity class from the beginning. The error-only grower adapts fastest but over-allocates.

Returned-context early-16 MSE is 0.11254 for the prospective grower versus 0.21709 for the single model. This is evidence of retained specialization, but it is not yet the rapid clean retrieval target of the full protocol.

## Attacker 1 — noise is not a new world

The response map remains A throughout. Two intervals increase observation noise from σ=0.05 to σ=0.30. The conditional mean never changes.

| learner | mean MSE | mean births | worlds with any birth |
|---|---:|---:|---:|
| single | 0.021468 | 0 | 0% |
| prospective grower | 0.021468 | **0** | **0%** |
| error-only grower | 0.022630 | 2.04 | **100%** |

This is the cleanest Gate 0 result. Persistent error is useful for **nominating** a candidate, but it is poor evidence for **admitting** one. The later causal evaluation window rejects the false structure.

## Attacker 2 — a missing feature is not a missing context

One stationary world adds the interaction `0.5 * u0 * u1`. More linear experts do not solve that representational error.

| learner | mean MSE | mean births |
|---|---:|---:|
| single linear | 0.037514 | 0 |
| prospective linear grower | 0.037514 | **0** |
| error-only grower | 0.038751 | 0.36 |
| single + quadratic feature | **0.006407** | 0 |

The richer single model beats the prospective grower by **0.031106 MSE**, paired bootstrap 95% interval `[0.030838, 0.031381]` improvement.

A good structural learner must therefore distinguish at least three sources of surprise:

```text
new context        → specialization may help
noise              → better uncertainty model / do not grow
missing feature    → change representation, not expert count
```

Gate 0 only implements one crude prospective test for that distinction. It is not yet a complete solution.

## Information boundary

For a scored trial, the sequence is always:

1. choose action/readout;
2. compute current mixture prior;
3. **log prediction**;
4. reveal consequence;
5. compute posterior responsibility;
6. update existing experts;
7. score/evolve a provisional candidate for future decisions.

No current consequence may choose the forecast that is being scored against that same consequence.

## Receipts

- `results/gate0_summary.json` — compact aggregate and paired comparisons.
- `results/gate0_heldout.json` — slim per-world held-out receipt.

The benchmark is deterministic under its seeds. The receipt is a software experiment, not independent replication of the upstream Paper/JelloBrain lineage.
