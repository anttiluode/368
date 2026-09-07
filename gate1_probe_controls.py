#!/usr/bin/env python3
"""Gate 1R: matched-cost controls for diagnostic PINGs.

Astra's review reproduced Gate 1 but identified two attribution problems:

1. active PINGs were not compared with random/periodic extra observations at a
   matched budget;
2. probe and scored observations consumed the same world RNG, so buying a probe
   changed later scored noise.

This audit fixes both without changing the fixed-bank model.  For every seed it
first runs the original-style adaptive disagreement policy and records exactly
how many probes it bought at each decision.  Then it runs three controls on the
same scored actions and exact same scored-noise sequence:

- random_same_timing: random query, same probe times/counts as active;
- disagreement_periodic: disagreement query, same total budget spread evenly;
- random_periodic: random query, same total budget spread evenly.

Probe noise has its own deterministic stream and never advances scored noise.
Diagnostic outcomes update responsibility only, exactly as in Gate 1.

This isolates three questions more cleanly:
- does *any* extra observation help?
- does choosing a disagreement query help beyond random query?
- does choosing when to ask help beyond spending the same budget periodically?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmark import ACTIONS, FixedBank, SwitchingWorld, WorldConfig, phi


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def predictions(learner, u, s):
    x = phi(u, s)
    return x, np.asarray([m.predict(x) for m in learner.models], dtype=float)


def posterior(p, preds, y, sigma):
    z = -0.5 * ((y - preds) / sigma) ** 2
    z -= np.max(z)
    q = p * np.exp(z)
    q /= max(float(np.sum(q)), 1e-300)
    return q


def disagreement_probe(learner, p):
    best = None
    for ai, u in enumerate(ACTIONS):
        for s in (0, 1):
            _, preds = predictions(learner, u, s)
            mu = float(p @ preds)
            var = float(p @ ((preds - mu) ** 2))
            if best is None or var > best[0]:
                best = (var, ai, s, preds)
    return best


def world_and_scored_stream(seed):
    world = SwitchingWorld(seed, WorldConfig(noise=0.05), "base")
    action_rng = np.random.default_rng(seed + 8_000_001)
    noise_rng = np.random.default_rng(seed + 8_100_001)
    score = []
    for t in range(world.horizon()):
        ai = int(action_rng.integers(len(ACTIONS)))
        s = int(action_rng.integers(2))
        u = ACTIONS[ai]
        y = world.target(t, u, s) + float(noise_rng.normal(0.0, 0.05))
        score.append((ai, s, y, world.latent_label(t)))
    return world, score


def probe_return(world, seed, t, slot, ai, s):
    # Noise is keyed by decision/slot, not by policy or chosen query.  Different
    # arms therefore see the same noise realization for corresponding paid
    # observations while the underlying target may differ with the query.
    rng = np.random.default_rng(seed * 1_000_003 + 8_200_001 + 17 * t + slot)
    return world.target(t, ACTIONS[ai], s) + float(rng.normal(0.0, 0.05))


def periodic_schedule(T, total):
    counts = np.zeros(T, dtype=int)
    if total <= 0:
        return counts
    # Midpoint placement avoids systematically spending at t=0 and t=T-1.
    pos = np.floor((np.arange(total) + 0.5) * T / total).astype(int)
    pos = np.clip(pos, 0, T - 1)
    for p in pos:
        counts[p] += 1
    return counts


def run_policy(seed, mode, schedule=None, max_probes=2, entropy_threshold=0.25):
    world, scored = world_and_scored_stream(seed)
    learner = FixedBank(k=3, noise=0.05)
    random_rng = np.random.default_rng(seed + 8_300_001)
    sq = []
    labels = []
    probe_counts = np.zeros(len(scored), dtype=int)

    for t, (score_ai, score_s, y, label) in enumerate(scored):
        p = learner.prior()

        if mode == "active_disagreement":
            bought = 0
            for slot in range(max_probes):
                h = entropy(p) / max(math.log(len(p)), 1e-12)
                if h < entropy_threshold:
                    break
                var, ai, s, preds = disagreement_probe(learner, p)
                if var < 1e-6:
                    break
                yq = probe_return(world, seed, t, slot, ai, s)
                p = posterior(p, preds, yq, learner.noise)
                bought += 1
            probe_counts[t] = bought
        else:
            n = int(schedule[t]) if schedule is not None else 0
            for slot in range(n):
                if "disagreement" in mode:
                    _, ai, s, preds = disagreement_probe(learner, p)
                elif "random" in mode:
                    ai = int(random_rng.integers(len(ACTIONS)))
                    s = int(random_rng.integers(2))
                    _, preds = predictions(learner, ACTIONS[ai], s)
                else:
                    raise ValueError(mode)
                yq = probe_return(world, seed, t, slot, ai, s)
                p = posterior(p, preds, yq, learner.noise)
            probe_counts[t] = n

        u = ACTIONS[score_ai]
        x, preds = predictions(learner, u, score_s)
        pred = float(p @ preds)
        sq.append(float((y - pred) ** 2))
        learner.observe(u, score_s, y, {"x": x, "prior": p, "preds": preds})
        labels.append(label)

    seen = set()
    early = []
    i = 0
    while i < len(labels):
        label = labels[i]
        j = i + 1
        while j < len(labels) and labels[j] == label:
            j += 1
        if label in seen:
            early.extend(sq[i:min(i + 16, j)])
        seen.add(label)
        i = j

    return {
        "seed": int(seed),
        "mode": mode,
        "mse": float(np.mean(sq)),
        "returned_early16_mse": float(np.mean(early)),
        "probes": int(np.sum(probe_counts)),
        "probes_per_decision": float(np.mean(probe_counts)),
        "probe_counts": probe_counts.tolist(),
    }


def evaluate_seed(seed):
    active = run_policy(seed, "active_disagreement")
    timing = np.asarray(active["probe_counts"], dtype=int)
    periodic = periodic_schedule(len(timing), int(np.sum(timing)))
    passive = run_policy(seed, "random_same_timing", np.zeros_like(timing))
    random_same = run_policy(seed, "random_same_timing", timing)
    disagreement_periodic = run_policy(seed, "disagreement_periodic", periodic)
    random_periodic = run_policy(seed, "random_periodic", periodic)
    passive["mode"] = "passive"
    return [passive, active, random_same, disagreement_periodic, random_periodic]


def aggregate(rows):
    out = []
    for mode in ("passive", "active_disagreement", "random_same_timing",
                 "disagreement_periodic", "random_periodic"):
        q = [r for r in rows if r["mode"] == mode]
        out.append({
            "mode": mode,
            "n": len(q),
            "mse": float(np.mean([r["mse"] for r in q])),
            "returned_early16_mse": float(np.mean([r["returned_early16_mse"] for r in q])),
            "probes_per_decision": float(np.mean([r["probes_per_decision"] for r in q])),
        })
    return out


def paired(rows, a, b, key="mse"):
    aa = {r["seed"]: r for r in rows if r["mode"] == a}
    bb = {r["seed"]: r for r in rows if r["mode"] == b}
    seeds = sorted(set(aa) & set(bb))
    d = np.asarray([aa[s][key] - bb[s][key] for s in seeds])
    rng = np.random.default_rng(368_1001)
    boots = d[rng.integers(len(d), size=(10_000, len(d)))].mean(axis=1)
    return {"a": a, "b": b, "metric": key, "mean_difference": float(np.mean(d)),
            "bootstrap95": [float(x) for x in np.quantile(boots, [0.025, 0.975])]}


def run(seeds):
    rows = [r for seed in seeds for r in evaluate_seed(seed)]
    return {
        "schema": "repo368/gate1-matched-probe-controls-v1",
        "aggregate": aggregate(rows),
        "paired": [
            paired(rows, "active_disagreement", "passive"),
            paired(rows, "random_same_timing", "passive"),
            paired(rows, "active_disagreement", "random_same_timing"),
            paired(rows, "active_disagreement", "disagreement_periodic"),
            paired(rows, "disagreement_periodic", "random_periodic"),
            paired(rows, "active_disagreement", "random_periodic"),
            paired(rows, "active_disagreement", "random_same_timing", "returned_early16_mse"),
        ],
        "claim_boundary": (
            "All paid-probe arms use exactly the active policy's per-seed total probe budget; random-same-timing additionally uses the exact active probe times. "
            "Scored actions and scored noise are identical across arms. This tests query/timing attribution for the fixed bank only."
        ),
        "runs": rows,
    }


def slim(r):
    return {k: r[k] for k in ("schema", "aggregate", "paired", "claim_boundary")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/gate1_matched_controls.json")
    args = ap.parse_args()
    r = run(range(args.start, args.start + args.n))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(slim(r), indent=2))
