# 368 — When Should a Learner Split?

**Fast adaptation is cheap. Memory is useful. The question is when memory earns the cost of continuing to exist.**

This repository follows the research note [When Should a Learner Split?](https://github.com/anttiluode/Paper). It began by asking whether contradiction should cause ordinary updating or admission of another expert. An independent-in-spirit audit by Astra then found the baseline we had failed to include: a tiny single RLS model with ordinary exponential forgetting beats every original Gate 0 learner on overall switching error.

That correction changed the project.

The strongest current question is no longer simply “when should a learner split?” It is:

> **When is preserving an old model worth more than simply becoming very good at forgetting it?**

The uploaded review and reproduction files are committed in this repository (`REPORT.md`, `audit.py`, `summary.json`, and related receipts). See [GATE2_RESULT.md](GATE2_RESULT.md) for the newest result.

## Gate 0 — prospective birth

The hidden world changes without labels through:

```text
A → B → A → C → B → A
```

The original grower begins with one causal linear predictor. Persistent prediction failure may nominate another predictor, but that provisional expert is admitted only if it predicts a later pre-outcome evaluation window better than the incumbent.

Original 100-world result:

| learner | MSE ↓ | new experts |
|---|---:|---:|
| original non-forgetting single | 0.21142 | 0 |
| fixed bank of 3 | 0.09940 | 0 (supplied) |
| prospective grower | 0.12206 | 2.07 |
| error-only grower | 0.07693 | 3.98 |

The result established that future evidence can veto some false growth, but not that growth is necessary.

## Astra audit — the missing baseline

Astra reran Gate 0 and Gate 1 exactly and checked the causal prediction boundary. It then added the comparison we should have had from the start: the same four-coefficient current model with exponential forgetting, choosing `discount=0.5` on development seeds 0–19.

On the same reused 100 evaluation worlds:

| learner | switching MSE ↓ | first 16 after familiar return ↓ |
|---|---:|---:|
| prospective grower | 0.12206 | **0.11254** |
| **fast forgetting single** | **0.02508** | 0.16503 |

So fast adaptation dominates overall, while retained specialization helps immediately when old conditions return. The review also found that the original admission threshold was not a measured resource price, that rejecting a candidate does not diagnose why prediction failed, and that Gate 1 needed matched-cost random/periodic PING controls.

See [REPORT.md](REPORT.md).

## Gate 1R — PING survives the matched-cost audit

Gate 1 uses a diagnostic intervention to decide which existing model is currently plausible. Gate 1R reruns that idea with identical scored actions/noise and exactly matched paid-observation budgets.

| fixed-bank policy | MSE ↓ | probes / decision |
|---|---:|---:|
| passive | 0.10700 | 0 |
| random probe, active timing | 0.09518 | 0.03454 |
| disagreement probe, periodic timing | 0.09415 | 0.03454 |
| random probe, periodic timing | 0.10447 | 0.03454 |
| **disagreement probe, adaptive timing** | **0.07022** | 0.03454 |

Both pieces matter: disagreement queries beat random queries at the same times, and adaptive timing beats periodic timing using disagreement queries. The result is documented in [GATE1_REAUDIT.md](GATE1_REAUDIT.md).

This does **not** solve structural learning; it says that when several explanations already exist, controlling **what evidence to request and when to request it** can materially improve causal retrieval.

## Gate 2 — fast current + retained memory

Gate 2 keeps Astra's strong fast learner and adds frozen memories behind it.

```text
FAST CURRENT
    aggressively tracks now
        │
        ├── surprise can nominate a snapshot
        │
        ▼
PROVISIONAL MEMORY
    must predict later outcomes while frozen
    and be functionally distinct
        │
        ▼
RETAINED MEMORY
    can be retrieved from completed evidence
    before the next scored prediction
```

Three arms isolate the mechanism:

- `fast_only`: forgetting RLS;
- `cache_only`: stores the same memories but never retrieves them;
- `hybrid`: stores and causally retrieves them.

### Main recurring world, 100 reused evaluation seeds

| method | MSE ↓ | early familiar-return MSE ↓ | memories |
|---|---:|---:|---:|
| fast only | 0.025071 | 0.165754 | 0 |
| cache only | 0.025071 | 0.165754 | 2.98 |
| **hybrid** | **0.022754** | **0.130669** | 2.99 |

Hybrid-minus-fast paired MSE is `-0.00231684`, bootstrap 95% interval `[-0.00299103, -0.00172017]`.

The cache-only equality matters: **storage itself does nothing to prediction. Reuse does.**

### The decisive negative

In `A → B → C`, where distinct worlds never recur, the retention system still stores about **2.78 memories** yet gains no measurable prediction advantage over fast-only.

So Gate 2 finds the next wall:

> **Successful initial fit can justify a provisional memory. It cannot justify keeping that memory forever.**

A close-context attacker at exactly the Paper protocol's minimum distance `0.6` still produces a small but repeatable hybrid gain, so the new rule is not merely rejecting subtle real contexts. Smooth drift is essentially tied with fast-only; a stationary quadratic mismatch creates no linear retained models.

See [GATE2_RESULT.md](GATE2_RESULT.md) and the compact receipts in [`results/`](results/).

## Current architecture

The experiment now separates four operations that earlier repos repeatedly collapsed:

```text
ADAPT
    change the fast current model

RETAIN
    preserve a candidate after future adequacy evidence

RETRIEVE
    reuse old knowledge when completed evidence makes it plausible

INVESTIGATE
    buy a PING when existing explanations remain ambiguous
```

The missing fifth operation is **FORGET / RECYCLE MEMORY ITSELF**.

A retained model should acquire value only when its reuse actually saves future prediction error. Carrying it should cost memory. A model that never becomes useful again should eventually lose the right to occupy structure.

That is Gate 3.

## Run

```bash
python -m pip install -r requirements.txt
pytest -q
python gate1_probe_controls.py --start 1000 --n 100
python gate2_fast_memory.py --start 1000 --n 100
python gate2_attacks.py --start 1000 --n 100
```

The `1000–1099` worlds are reused throughout this repository lineage and must not be described as a fresh blind test set.

## What this does not claim

Mixtures of experts, latent-cause inference, dynamic expansion, fast/slow learning, episodic retrieval, active sensing and continual-learning methods all predate this repository. This work does not establish consciousness, a biological growth law, or a new state-of-the-art continual learner.

Its useful contribution is narrower: keep exposing the decision boundaries with attackers until “grow” stops being a synonym for “prediction was bad.”
