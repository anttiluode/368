# Gate 1R — what to ask and when to ask

Status: **executed after Astra's audit**.

Astra reproduced the original Gate 1 improvement but identified two unresolved confounds:

1. the active PING arm had no random or periodic probe comparison at a matched observation budget;
2. diagnostic probes and scored observations consumed the same world RNG, so purchasing a probe changed which noise sample appeared in the later scored observation.

Gate 1R fixes both.

## Design

The fixed three-model bank is unchanged. For each of 100 evaluation seeds, the adaptive disagreement policy runs first and determines its own probe count and timing. Every comparison then receives exactly the same total number of paid observations.

The arms are:

| arm | when to ask | what to ask |
|---|---|---|
| passive | never | — |
| active disagreement | uncertainty-triggered | query where experts disagree most |
| random same timing | **exact same times as active** | random query |
| disagreement periodic | same total budget, evenly spaced | disagreement query |
| random periodic | same total budget, evenly spaced | random query |

Scored actions and scored-noise samples are identical across all arms. Probe noise comes from a separate deterministic stream. Diagnostic outcomes update responsibility only; they do not train expert parameters directly.

## Result — 100 worlds

| arm | MSE ↓ | first 16 after familiar return ↓ | probes / decision |
|---|---:|---:|---:|
| passive | 0.106998 | 0.096747 | 0 |
| **active disagreement** | **0.070221** | **0.063595** | 0.034543 |
| random same timing | 0.095177 | 0.089319 | 0.034543 |
| disagreement periodic | 0.094152 | 0.081941 | 0.034543 |
| random periodic | 0.104472 | 0.093355 | 0.034543 |

Paired comparisons:

```text
active disagreement - random same timing
    -0.024956 MSE
    bootstrap 95% [-0.037878, -0.012803]

active disagreement - disagreement periodic
    -0.023931 MSE
    bootstrap 95% [-0.036527, -0.011974]

disagreement periodic - random periodic
    -0.010320 MSE
    bootstrap 95% [-0.017651, -0.003584]
```

So the original PING effect is not merely “more observations help.” In this fixed-bank toy, **what to ask matters**, and **when to ask matters**.

The random-same-timing arm improves over passive by `-0.01182` mean MSE, but its bootstrap interval crosses zero. Extra evidence alone is therefore not a secure explanation of the active result at this budget.

The early-return comparison also favors active disagreement over random probes at the same times by `-0.02572`, interval `[-0.04216, -0.01054]`.

## Connection to the older lineage

This is intentionally not advertised as a new principle. `AlgoSchalgo` had already found that active measurement timing and measurement choice were separable contributors under bounded sensing. Gate 1R matters because the same distinction now appears inside the **model-retrieval** problem, with the observation budget and noise stream controlled.

The operational loop is:

```text
several retained explanations remain plausible
        ↓
should I spend an observation now?
        ↓
if yes: which intervention makes them disagree most?
        ↓
PING
        ↓
update responsibility
        ↓
make the independently scored prediction
```

That is narrower than “scientific method,” but it is a clean piece of the larger architecture.

## What it does not solve

Gate 1R assumes a supplied fixed bank. It does not decide which memories deserve to exist, whether a new memory should survive, or whether a feature representation should change. It also does not yet optimize a joint prediction/probe-cost objective; all comparisons are displayed against the same paid-observation count instead.

The compact frozen receipt is `results/gate1_matched_controls_summary.json`.
