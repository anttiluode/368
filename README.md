# 368 — When Should a Learner Split?

**One learner. Several hidden worlds. Growth is allowed, but it has to earn its existence.**

This repository implements the first executable step after the research note [When Should a Learner Split?](https://github.com/anttiluode/Paper). It starts from one small causal predictor and asks whether contradiction should cause ordinary updating, retrieval of existing knowledge, or admission of a new expert.

The rule is intentionally stricter than “large error → grow”. A candidate expert is only admitted if, after a bounded training window, it predicts a **subsequent** evaluation window better than the incumbent by a fixed margin. Every scored forecast is logged conceptually before its outcome is used for learning.

## Gate 0 result

Held out: **100 independently generated worlds, seeds 1000–1099**. The main world changes without labels through

```text
A → B → A → C → B → A
```

Each hidden context is a different 2×2 action→consequence map. The learner sees the action, selected scalar readout and consequence, never the context label.

| Learner | Base MSE ↓ | New experts | What it means |
|---|---:|---:|---|
| single linear | 0.21142 | 0 | continual overwriting |
| fixed bank of 3 | **0.09940** | 0 (3 supplied) | capacity is preallocated |
| **prospective grower** | 0.12206 | **2.07** | starts with 1, admits structure only after future evidence |
| error-only grower | **0.07693** | 3.98 | fastest here, but grows too easily |

The prospective grower improves over the single learner by **0.08936 MSE** (paired bootstrap 95% interval 0.07792–0.10073 improvement), but it does **not** beat the fixed bank or the aggressive error-only grower on prediction error. That is an important negative: conservative structural admission has a real adaptation cost.

The attackers explain why the cost may be worthwhile:

- **Noise burst, no world change:** prospective grower creates **0 experts in 100/100 worlds**. Error-only growth creates at least one false expert in **100/100**, averaging **2.04 births**.
- **Wrong representation, no world change:** the environment needs one quadratic interaction. The prospective grower creates **0 experts** because more linear experts do not prospectively help. A single model given the missing quadratic feature reaches **0.00641 MSE**, versus **0.03751** for the linear grower.

So Gate 0 does not answer “growth wins.” It establishes something narrower and more useful:

> **Prediction failure is not enough evidence for specialization. A split should survive a future-prediction test, and sometimes the correct response to failure is neither another expert nor more memory.**

See [RESULTS.md](RESULTS.md) and the frozen receipts in [`results/`](results/).

## The mechanism

Every expert is a tiny recursive least-squares forward model. Existing experts form a causal sticky mixture. After an outcome arrives, likelihood updates responsibility and gates learning for the *next* decision.

The growing learner additionally tracks persistent inadequacy:

```text
prediction fails repeatedly
        ↓
train provisional expert on bounded recent evidence
        ↓
DO NOT admit it yet
        ↓
score incumbent and candidate on later outcomes
before either sees each outcome
        ↓
future advantage > fixed structural penalty ?
        │
      yes ──► admit anonymous expert
       no ──► discard candidate
```

The `error_only` condition removes that prospective admission test. It exists specifically as an attacker.

## Run

```bash
python -m pip install -r requirements.txt
python benchmark.py --start 0 --n 20 --out results/dev_run.json
pytest -q
```

The full frozen held-out receipt used 100 worlds. CI runs deterministic smoke/causality tests rather than recomputing the full receipt on every commit.

## What this does **not** claim

Mixtures of experts, dynamic network expansion, latent-cause inference, Bayesian responsibility and continual-learning methods all predate this repository. This is not evidence for consciousness, a biological growth rule, or a novel general-purpose continual-learning algorithm.

The useful question is operational: **can structure be admitted only when future evidence shows that the extra structure earns its memory and interference cost?** Gate 0 says that criterion prevents two obvious kinds of false growth, but is still too conservative to win the main prediction benchmark.

## Next gate

The paper's harder protocol remains open: paid diagnostic PINGs, smooth drift, fixed total memory, recurrent baselines, return-to-context reacquisition, and eventually a substrate in which the *partition itself* is embodied rather than represented by a Python list of experts.

That is where 368 should go next—not back to another biological analogy.
