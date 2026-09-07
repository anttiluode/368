#!/usr/bin/env python3
"""Additional Gate 2 attackers.

The first fast-memory run shows a small but repeatable gain on the recurring
A->B->A->C->B->A world.  These controls ask whether that gain is actually tied
to reuse rather than merely carrying extra state.

- stationary: one map only; one retained snapshot is a cache with no retrieval
  value beyond stabilizing a fast tracker.
- no_return: A->B->C once each; retaining old maps cannot earn reuse in-stream.
- close: the usual recurring sequence, but B and C are each exactly Frobenius
  distance 0.6 from A (the Paper protocol's minimum allowed separation).  This
  attacks the earlier 0.15-admission-threshold problem with genuinely distinct
  but relatively close maps.

All methods receive exactly the same scored actions and noise samples.  Hidden
labels are evaluator-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark import ACTIONS, SwitchingWorld, WorldConfig
from gate2_fast_memory import HybridConfig, aggregate, evaluate_stream, paired_bootstrap, sigma_for


def build_stream(seed: int, family: str):
    if family == "stationary":
        cfg = WorldConfig(noise=0.05, sequence=("A", "A", "A", "A", "A", "A"))
        world = SwitchingWorld(seed, cfg, family="stationary_005")
    elif family == "no_return":
        cfg = WorldConfig(noise=0.05, sequence=("A", "B", "C"))
        world = SwitchingWorld(seed, cfg, family="base")
        world.family = "no_return"
    elif family == "close":
        cfg = WorldConfig(noise=0.05)
        world = SwitchingWorld(seed, cfg, family="base")
        rng = np.random.default_rng(seed + 3_680_017)
        A = rng.uniform(-0.45, 0.45, size=(2, 2))
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        B = A + 0.6 * q[:, 0].reshape(2, 2)
        C = A + 0.6 * q[:, 1].reshape(2, 2)
        world.maps = [A, B, C]
        world.family = "close"
        # Exact construction checks: A-B and A-C are 0.6; B-C is sqrt(.72).
        assert abs(np.linalg.norm(A - B) - 0.6) < 1e-12
        assert abs(np.linalg.norm(A - C) - 0.6) < 1e-12
    else:
        raise ValueError(family)

    action_rng = np.random.default_rng(seed + 1_000_003)
    noise_rng = np.random.default_rng(seed + 2_000_003)
    rows = []
    for t in range(world.horizon()):
        u = ACTIONS[int(action_rng.integers(len(ACTIONS)))].copy()
        s = int(action_rng.integers(2))
        y = world.target(t, u, s) + float(noise_rng.normal(0.0, sigma_for(world, t)))
        rows.append((u, s, y, world.latent_label(t)))
    return rows


def run(seeds):
    cfg = HybridConfig()
    rows = []
    for seed in seeds:
        for family in ("stationary", "no_return", "close"):
            events = build_stream(seed, family)
            for method in ("fast_only", "cache_only", "hybrid"):
                r = evaluate_stream(events, method, cfg)
                r.update({"seed": int(seed), "family": family})
                rows.append(r)
    return {
        "schema": "repo368/fast-retained-memory-attacks-v1",
        "config": cfg.__dict__,
        "aggregate": aggregate(rows),
        "paired": {
            "stationary_hybrid_minus_fast": paired_bootstrap(rows, "stationary", "hybrid", "fast_only"),
            "no_return_hybrid_minus_fast": paired_bootstrap(rows, "no_return", "hybrid", "fast_only"),
            "close_hybrid_minus_fast": paired_bootstrap(rows, "close", "hybrid", "fast_only"),
        },
        "claim_boundary": (
            "These controls test whether retained snapshots help specifically when contexts recur and whether the same rule can retain close real contexts. "
            "They do not price memory into one universal scalar objective."
        ),
        "runs": rows,
    }


def slim(r):
    return {k: r[k] for k in ("schema", "config", "aggregate", "paired", "claim_boundary")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/gate2_attacks.json")
    a = ap.parse_args()
    r = run(range(a.start, a.start + a.n))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(slim(r), indent=2))
