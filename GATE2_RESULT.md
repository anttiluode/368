# Gate 2 — fast current + retained memory

Status: **executed on the `gate2-fast-memory` branch** after importing Astra's review of commit `543a022`.

The review changed the benchmark. A four-coefficient exponentially forgetting RLS learner (`discount=0.5`) gets about **0.0251 MSE** on the switching world, far better than the original growing learner's `0.1221`. But the old bank retained an advantage immediately after familiar worlds returned. Gate 2 asks whether those strengths can coexist.

## Architecture

Gate 2 separates two jobs that Gate 0 had mixed:

```text
FAST CURRENT MODEL
    adapts aggressively to what works now

FROZEN RETAINED MODELS
    remember maps that earned a prospective adequacy audit
```

A provisional retained model is copied only after the fast current learner has settled. It is then scored on 16 later outcomes **before seeing each outcome** and must remain both accurate and functionally distinct from existing memories. Retrieval uses only three already-completed observations. If a retained model fits those observations much better than the current fast model, its state is copied back into the fast model before the next scored prediction.

No hidden context label reaches the learner.

Three methods isolate the effect:

- `fast_only`: Astra's strong forgetting baseline.
- `cache_only`: creates exactly the same type of retained memories but never retrieves them.
- `hybrid`: creates and causally retrieves retained memories.

The evaluator generates one scored action/noise stream and reuses it exactly across all methods. This also fixes the shared-RNG weakness Astra found in the earlier PING comparison.

## Main recurring world — 100 reused evaluation seeds

Sequence: `A -> B -> A -> C -> B -> A`.

| method | overall MSE ↓ | first 16 after familiar return ↓ | retained memories | retrievals |
|---|---:|---:|---:|---:|
| fast only | 0.025071 | 0.165754 | 0 | 0 |
| cache only | 0.025071 | 0.165754 | 2.98 | 0 |
| **fast + retained retrieval** | **0.022754** | **0.130669** | 2.99 | 3.49 |

Paired hybrid-minus-fast MSE is **-0.00231684**, bootstrap 95% interval `[-0.00299103, -0.00172017]` across the same 100 worlds.

The cache-only arm is important. It stores roughly three snapshots yet its predictions are exactly the fast-only predictions. The gain therefore comes from **reuse**, not from merely allocating extra state.

The effect is modest: about 9.2% relative reduction from the already-strong fast baseline, not a new-order-of-magnitude result.

## Attackers

### Close but real recurring contexts

Astra noted that Gate 0's fixed `0.15` admission threshold could reject legitimate contexts at the Paper protocol's minimum Frobenius separation `0.6`. Gate 2 therefore constructs B and C each at exactly distance `0.6` from A, with orthogonal perturbation directions.

| method | overall MSE ↓ | early return ↓ | memories |
|---|---:|---:|---:|
| fast only | 0.008274 | 0.033449 | 0 |
| cache only | 0.008274 | 0.033449 | 2.89 |
| hybrid | **0.008194** | **0.031900** | 2.89 |

Hybrid-minus-fast paired difference is `-7.96e-5`, bootstrap 95% interval `[-1.24e-4, -3.40e-5]`. So the new adequacy/distinctness rule can retain close genuine maps. The improvement is again small.

### Distinct worlds that never return

Sequence: `A -> B -> C`, once each.

| method | MSE ↓ | memories |
|---|---:|---:|
| fast only | 0.023446 | 0 |
| cache only | 0.023446 | **2.78** |
| hybrid | 0.023440 | **2.78** |

The paired hybrid-fast interval crosses zero. The system stores almost three models and gets no measurable benefit because there is no reuse opportunity.

This is the clearest Gate 2 failure.

> **Successful initial fit is evidence that a memory can exist. It is not evidence that the memory deserves to survive.**

### Stationary world

Hybrid is indistinguishable from fast-only and retains about one baseline snapshot. The extra state does not help.

### Wrong representation

In the stationary quadratic world, the linear retention mechanism creates **zero** memories. All three linear conditions remain at MSE `0.070817`. This still does not mean the system diagnoses “missing feature”; it only means the proposed retained linear snapshot fails its later adequacy audit.

### Smooth drift

Fast-only and hybrid are effectively tied (`0.008541` versus `0.008544`); the paired interval spans zero. This is desirable insofar as discrete memories do not dominate a problem that the fast tracker already handles well, but the hybrid still keeps about one memory and therefore has an unpriced cost.

### Noise bursts

Hybrid improves the fast-only MSE by only `8.45e-5`; the paired interval is narrowly below zero. It retains about one baseline snapshot. This is much weaker than the original grower's noise advantage and should not be oversold.

## What changed after the Astra review

Gate 0 asked:

> When should a learner split?

Gate 2 makes that two questions:

```text
When should a model be BORN?
        !=
When should that model be KEPT?
```

The current admission rule answers the first moderately well. The no-return attacker says it does not answer the second at all.

A retained model has value only if future evidence makes it reusable enough to repay:

```text
memory cost
+ routing cost
+ diagnostic cost
+ interference / complexity cost
```

That value cannot be inferred from how well the model explained the block in which it was born.

## Next gate: survival must be earned

Gate 3 should keep birth cheap/provisional but attach an explicit ledger to every memory.

When a retained model is selected, score both predictions **before** seeing the same next outcome:

```text
what FAST would have predicted
versus
what RETAINED MEMORY predicts
```

After the outcome arrives, credit the memory with the error it actually saved. Charge carrying cost while it persists. Memories that never return should lose their provisional status and be recycled; memories that repeatedly rescue prediction should survive.

The decisive attacker is now obvious:

```text
A -> B -> C               no recurrence
```

versus

```text
A -> B -> A -> C -> B -> A
```

The same birth machinery should initially behave similarly in both. Only the second stream supplies evidence that old models deserve long-term survival.

That is a more precise form of the original question:

> **When is preserving another model worth more than simply becoming very good at forgetting it?**

## Evidence status

The code and workflows were run successfully in GitHub Actions on 100 seeds. The compact receipts are:

- `results/gate2_fast_memory_summary.json`
- `results/gate2_attacks_summary.json`

These seeds (`1000–1099`) have already been used throughout the 368 lineage and should not be described as a fresh blind test set. Hyperparameters in this Gate 2 implementation were hand-set before the 100-world run, not selected on those held-out outcomes. The result remains a toy software experiment, not independent validation or a general continual-learning algorithm.
