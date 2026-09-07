# Model and information boundary

## Environment

A trial exposes a two-vector action `u`, an integer readout `s ∈ {0,1}`, and only after prediction a scalar consequence

`y = (A[z] @ u)[s] + noise`.

The hidden mode `z` is unavailable. Base worlds use `A,B,A,C,B,A` with random block lengths. Maps are freshly sampled per seed.

The legal action set is the four signed coordinate probes and four normalized diagonal probes from the Paper protocol.

## Expert

For each readout, the two action components occupy a separate pair of coefficients, giving a four-dimensional feature vector. Each expert is a recursive least-squares linear predictor. This class can represent one base response map exactly but cannot represent the quadratic control without an added interaction feature.

## Responsibility

Before outcome `y_t`, expert weights are propagated with a sticky prior. Their weighted mean is the scored prediction. After `y_t` arrives, Gaussian residual likelihoods update posterior responsibility. Posterior responsibility gates each expert's RLS update.

This is a small causal mixture, not a full reproduction of Wolpert & Kawato or a modern Bayesian nonparametric model.

## Structural admission

The prospective grower begins with one expert. A large normalized residual must persist before a candidate is *nominated*. The candidate is fitted on the most recent 40 completed examples. It is then evaluated for 32 subsequent decisions.

During evaluation both candidate and incumbent issue their prediction before each outcome. The candidate may adapt only after each scored outcome, exactly as the incumbent does. Admission uses

`mean(incumbent squared error - candidate squared error) > 0.15`.

At most one nomination can begin per 64 decisions; at most six experts may exist. The threshold was selected on development seeds, not the frozen 100-world held-out set.

## Error-only attacker

The error-only condition uses the same nomination signal but immediately creates the candidate, without prospective evidence. It is intentionally not an eligible scientific winner. Its purpose is to test whether “surprise → grow” over-allocates under noise.

## Resource accounting

`state_slots_approx` counts expert RLS parameters/covariances, responsibility weights, the bounded recent-data buffer, and a provisional expert when present. It is an approximate scalar-slot audit, not byte-exact accounting. The current implementation stays below the Paper protocol's proposed 1,024 float64-equivalent persistent-slot ceiling, but the full byte accounting and paid-compute accounting remain future work.

## Scope

This model contains explicit Python expert objects. It therefore demonstrates *algorithmic structural admission*, not self-organized physical compartment growth. Translating a successful rule back into JelloBrain-like material should happen only after the abstract benchmark earns it.
