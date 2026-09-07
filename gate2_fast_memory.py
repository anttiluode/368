#!/usr/bin/env python3
"""Gate 2: fast adaptation versus retained reusable models.

Astra's review found that an exponentially forgetting single RLS model dominates
Gate 0 on overall switching MSE, while the retained/growing bank is better just
after familiar contexts return and under noise bursts.  This gate tests the
missing hybrid directly.

The learner has two qualitatively different stores:

FAST CURRENT
    one ordinary four-coefficient RLS model with exponential forgetting.

RETAINED MEMORY
    frozen snapshots that are admitted only after a *future*, pre-outcome
    adequacy audit.  A snapshot must also be functionally distinct from all
    already-retained snapshots.

Retention and retrieval are separate.  `cache_only` is allowed to create the
same memories as the hybrid but never use them; it should therefore match the
fast-only predictor.  `hybrid` may retrieve an old frozen snapshot only from
completed evidence and copy it back into the fast current model for the next
prediction.

This is deliberately not a claim that the retention rule is optimal.  The gate
asks a narrower question exposed by the review: can a fast learner preserve its
adaptation advantage while retained models buy measurable reuse/noise
robustness?  Hidden world labels are used only by the evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from benchmark import ACTIONS, SwitchingWorld, WeightedRLS, WorldConfig, phi


@dataclass(frozen=True)
class HybridConfig:
    discount: float = 0.5
    # Surprise nominates the *possibility* that the current map changed.
    bad_ewma_decay: float = 0.80
    bad_ewma_threshold: float = 0.030
    # A provisional memory is copied only after the fast model has settled.
    stable_window: int = 8
    stable_mse: float = 0.012
    min_candidate_gap: int = 32
    # The frozen copy must remain adequate on later outcomes.
    eval_window: int = 16
    accept_mse: float = 0.012
    # Mean squared prediction difference over every legal action/readout.
    distinct_mse: float = 0.020
    max_memories: int = 6
    # Retrieval uses only completed observations in this short window.
    router_window: int = 3
    retrieve_ratio: float = 0.70
    retrieve_abs_mse: float = 0.030
    retrieve_cooldown: int = 4


def sigma_for(world: SwitchingWorld, t: int) -> float:
    sigma = world.cfg.noise
    if world.family == "noise_burst":
        frac = t / max(1, world.horizon() - 1)
        if 0.30 <= frac < 0.40 or 0.68 <= frac < 0.76:
            sigma = 0.30
    return float(sigma)


def make_stream(seed: int, family: str = "base", noise: float = 0.05):
    """Create one evaluator stream shared exactly across all methods.

    The world object supplies maps and schedules, but scored observation noise
    comes from a separate RNG.  Thus a learner cannot change another method's
    future scored-noise sequence by doing extra internal computation.
    """
    world = SwitchingWorld(seed, WorldConfig(noise=noise), family=family)
    action_rng = np.random.default_rng(seed + 1_000_003)
    noise_rng = np.random.default_rng(seed + 2_000_003)
    rows = []
    for t in range(world.horizon()):
        u = ACTIONS[int(action_rng.integers(len(ACTIONS)))].copy()
        s = int(action_rng.integers(2))
        target = world.target(t, u, s)
        y = target + float(noise_rng.normal(0.0, sigma_for(world, t)))
        rows.append((u, s, y, world.latent_label(t)))
    return rows


def prediction_grid(model: WeightedRLS) -> np.ndarray:
    out = []
    for u in ACTIONS:
        for s in (0, 1):
            out.append(model.predict(phi(u, s)))
    return np.asarray(out, dtype=float)


def functional_distance(a: WeightedRLS, b: WeightedRLS) -> float:
    d = prediction_grid(a) - prediction_grid(b)
    return float(np.mean(d * d))


def copy_model(src: WeightedRLS) -> WeightedRLS:
    dst = WeightedRLS(dim=src.dim, ridge=1.0, noise=src.noise)
    dst.theta = src.theta.copy()
    dst.P = src.P.copy()
    dst.n_eff = src.n_eff
    return dst


class FastOnly:
    name = "fast_only"

    def __init__(self, cfg: HybridConfig):
        self.cfg = cfg
        self.fast = WeightedRLS(noise=0.05)

    def predict(self, u, s):
        x = phi(u, s)
        p = self.fast.predict(x)
        return p, {"x": x, "fast_pred": p, "pred": p}

    def observe(self, u, s, y, meta):
        # Discount old precision immediately before incorporating the completed
        # observation.  This is the same convention as Astra's review baseline.
        self.fast.P /= self.cfg.discount
        self.fast.update(meta["x"], y)

    @property
    def births(self):
        return 0

    @property
    def memories(self):
        return []

    @property
    def retrievals(self):
        return 0


class FastRetained:
    """Fast current model plus prospectively validated frozen snapshots."""

    name = "fast_retained"

    def __init__(self, cfg: HybridConfig, retrieval: bool):
        self.cfg = cfg
        self.retrieval_enabled = bool(retrieval)
        self.fast = WeightedRLS(noise=0.05)
        self._memories: list[WeightedRLS] = []
        self._births = 0
        self.duplicate_rejections = 0
        self.adequacy_rejections = 0
        self._retrievals = 0
        self.t = 0
        self.bad_ewma = 0.0
        self.armed = True  # the first stable world may earn the first memory
        self.last_candidate = -10**9
        self.recent_fast_errors = deque(maxlen=cfg.stable_window)
        self.router_history = deque(maxlen=cfg.router_window)
        self.candidate = None
        self.last_retrieval_t = -10**9
        self.log = []

    @property
    def births(self):
        return self._births

    @property
    def memories(self):
        return self._memories

    @property
    def retrievals(self):
        return self._retrievals

    def _recent_mse(self, model: WeightedRLS) -> float:
        if not self.router_history:
            return math.inf
        return float(np.mean([(y - model.predict(x)) ** 2 for x, y in self.router_history]))

    def _maybe_retrieve(self):
        if not self.retrieval_enabled or not self._memories:
            return
        if self.t - self.last_retrieval_t < self.cfg.retrieve_cooldown:
            return
        if len(self.router_history) < self.cfg.router_window:
            return
        fast_loss = self._recent_mse(self.fast)
        mem_losses = np.asarray([self._recent_mse(m) for m in self._memories])
        j = int(np.argmin(mem_losses))
        best = float(mem_losses[j])
        # The retained model must be both absolutely plausible and materially
        # better than the current fast model on evidence that has already arrived.
        if best < self.cfg.retrieve_abs_mse and best < self.cfg.retrieve_ratio * max(fast_loss, 1e-12):
            self.fast = copy_model(self._memories[j])
            self._retrievals += 1
            self.last_retrieval_t = self.t
            self.log.append({"t": self.t, "event": "retrieve", "memory": j,
                             "memory_mse": best, "fast_mse": fast_loss})

    def predict(self, u, s):
        # Retrieval is based only on completed history and happens before the
        # scored outcome of this decision exists.
        self._maybe_retrieve()
        x = phi(u, s)
        fast_pred = self.fast.predict(x)
        meta = {"x": x, "fast_pred": fast_pred, "pred": fast_pred}
        if self.candidate is not None:
            meta["candidate_pred"] = self.candidate["model"].predict(x)
        return fast_pred, meta

    def _distinct_from_memory(self, model: WeightedRLS) -> tuple[bool, float]:
        if not self._memories:
            return True, math.inf
        ds = [functional_distance(model, m) for m in self._memories]
        return min(ds) >= self.cfg.distinct_mse, float(min(ds))

    def _start_candidate_if_ready(self):
        if self.candidate is not None or not self.armed:
            return
        if len(self._memories) >= self.cfg.max_memories:
            return
        if self.t - self.last_candidate < self.cfg.min_candidate_gap:
            return
        if len(self.recent_fast_errors) < self.cfg.stable_window:
            return
        if float(np.mean(self.recent_fast_errors)) >= self.cfg.stable_mse:
            return
        self.candidate = {"model": copy_model(self.fast), "n": 0, "loss": 0.0, "started": self.t}
        self.last_candidate = self.t
        self.log.append({"t": self.t, "event": "candidate_start"})

    def _finish_candidate_if_ready(self):
        c = self.candidate
        if c is None or c["n"] < self.cfg.eval_window:
            return
        mse = float(c["loss"] / c["n"])
        distinct, nearest = self._distinct_from_memory(c["model"])
        accepted = mse < self.cfg.accept_mse and distinct
        if accepted:
            self._memories.append(c["model"])
            self._births += 1
            self.log.append({"t": self.t, "event": "retain", "mse": mse,
                             "nearest_memory_mse": nearest, "memory": len(self._memories) - 1})
        elif not distinct:
            self.duplicate_rejections += 1
            self.log.append({"t": self.t, "event": "reject_duplicate", "mse": mse,
                             "nearest_memory_mse": nearest})
        else:
            self.adequacy_rejections += 1
            self.log.append({"t": self.t, "event": "reject_inadequate", "mse": mse,
                             "nearest_memory_mse": nearest})
        self.candidate = None
        # One surprise episode nominates one provisional memory.  Another burst
        # of inadequacy must re-arm structural consideration.
        self.armed = False
        self.bad_ewma = 0.0

    def observe(self, u, s, y, meta):
        x = meta["x"]
        fast_err = float((y - meta["fast_pred"]) ** 2)

        # Score a frozen candidate on this outcome before either the candidate
        # or fast model sees the outcome.  The candidate never learns in audit.
        if self.candidate is not None:
            cp = float(meta.get("candidate_pred", self.candidate["model"].predict(x)))
            self.candidate["loss"] += (y - cp) ** 2
            self.candidate["n"] += 1

        self.bad_ewma = (self.cfg.bad_ewma_decay * self.bad_ewma +
                         (1.0 - self.cfg.bad_ewma_decay) * min(fast_err, 1.0))
        if self.bad_ewma > self.cfg.bad_ewma_threshold:
            self.armed = True

        self.recent_fast_errors.append(fast_err)
        self.router_history.append((x.copy(), float(y)))

        self.fast.P /= self.cfg.discount
        self.fast.update(x, y)
        self.t += 1

        self._finish_candidate_if_ready()
        self._start_candidate_if_ready()


def make_learner(kind: str, cfg: HybridConfig):
    if kind == "fast_only":
        return FastOnly(cfg)
    if kind == "cache_only":
        return FastRetained(cfg, retrieval=False)
    if kind == "hybrid":
        return FastRetained(cfg, retrieval=True)
    raise ValueError(kind)


def evaluate_stream(events, kind: str, cfg: HybridConfig):
    learner = make_learner(kind, cfg)
    errors = []
    labels = []
    predictions = []
    for u, s, y, label in events:
        pred, meta = learner.predict(u, s)
        predictions.append(float(pred))
        errors.append(float((y - pred) ** 2))
        labels.append(label)
        learner.observe(u, s, y, meta)

    # Evaluation metadata: labels never reach learner.
    seen = set()
    return_errors = []
    i = 0
    while i < len(labels):
        label = labels[i]
        j = i + 1
        while j < len(labels) and labels[j] == label:
            j += 1
        if "->" not in label:
            if label in seen:
                return_errors.extend(errors[i:min(i + 16, j)])
            seen.add(label)
        i = j

    memory_slots = len(learner.memories) * 20
    return {
        "method": kind,
        "mse": float(np.mean(errors)),
        "return_early16_mse": float(np.mean(return_errors)) if return_errors else None,
        "births": int(learner.births),
        "final_memories": int(len(learner.memories)),
        "retrievals": int(learner.retrievals),
        "memory_slots": int(memory_slots),
        "duplicate_rejections": int(getattr(learner, "duplicate_rejections", 0)),
        "adequacy_rejections": int(getattr(learner, "adequacy_rejections", 0)),
        "predictions": predictions,
        "log": getattr(learner, "log", []),
    }


def evaluate_one(seed: int, family: str, kind: str, cfg: HybridConfig):
    events = make_stream(seed, family)
    r = evaluate_stream(events, kind, cfg)
    r.update({"seed": int(seed), "family": family})
    return r


def aggregate(rows):
    out = []
    for family in sorted(set(r["family"] for r in rows)):
        for method in ("fast_only", "cache_only", "hybrid"):
            q = [r for r in rows if r["family"] == family and r["method"] == method]
            if not q:
                continue
            out.append({
                "family": family,
                "method": method,
                "n": len(q),
                "mse": float(np.mean([r["mse"] for r in q])),
                "return_early16_mse": (float(np.mean([r["return_early16_mse"] for r in q]))
                                        if q[0]["return_early16_mse"] is not None else None),
                "births": float(np.mean([r["births"] for r in q])),
                "retrievals": float(np.mean([r["retrievals"] for r in q])),
                "final_memories": float(np.mean([r["final_memories"] for r in q])),
                "memory_slots": float(np.mean([r["memory_slots"] for r in q])),
                "duplicate_rejections": float(np.mean([r["duplicate_rejections"] for r in q])),
                "adequacy_rejections": float(np.mean([r["adequacy_rejections"] for r in q])),
            })
    return out


def paired_bootstrap(rows, family: str, a: str, b: str):
    aa = {r["seed"]: r for r in rows if r["family"] == family and r["method"] == a}
    bb = {r["seed"]: r for r in rows if r["family"] == family and r["method"] == b}
    seeds = sorted(set(aa) & set(bb))
    if not seeds:
        return None
    d = np.asarray([aa[s]["mse"] - bb[s]["mse"] for s in seeds])
    rng = np.random.default_rng(368_2002)
    boots = d[rng.integers(len(d), size=(10_000, len(d)))].mean(axis=1)
    return {"a_minus_b": float(np.mean(d)),
            "bootstrap95": [float(z) for z in np.quantile(boots, [0.025, 0.975])]}


def causal_prefix_check(seed: int = 1000):
    """Changing the current/future outcomes cannot alter predictions already due."""
    cfg = HybridConfig()
    events = make_stream(seed, "base")
    base = evaluate_stream(events, "hybrid", cfg)["predictions"]
    for cut in (0, 40, 120, 300):
        changed = [(u, s, y if i < cut else 100.0 - y, label)
                   for i, (u, s, y, label) in enumerate(events)]
        altered = evaluate_stream(changed, "hybrid", cfg)["predictions"]
        if not np.array_equal(np.asarray(base[:cut + 1]), np.asarray(altered[:cut + 1])):
            raise AssertionError(("outcome leakage", cut))
    return True


def run(seeds, cfg: HybridConfig | None = None):
    cfg = cfg or HybridConfig()
    rows = []
    for seed in seeds:
        for family in ("base", "noise_burst", "quadratic_stationary", "smooth"):
            events = make_stream(seed, family)
            # Same immutable event list for every method.
            for method in ("fast_only", "cache_only", "hybrid"):
                r = evaluate_stream(events, method, cfg)
                r.update({"seed": int(seed), "family": family})
                rows.append(r)
    receipt = {
        "schema": "repo368/fast-retained-memory-v1",
        "config": asdict(cfg),
        "claim_boundary": (
            "Gate 2 tests a toy fast-current/frozen-memory hybrid. Hidden labels are evaluation-only. "
            "Memory admission uses later pre-outcome adequacy and functional distinctness; retrieval uses completed evidence. "
            "It does not establish an optimal continual-learning or memory-consolidation algorithm."
        ),
        "aggregate": aggregate(rows),
        "paired": {
            "base_hybrid_minus_fast": paired_bootstrap(rows, "base", "hybrid", "fast_only"),
            "noise_hybrid_minus_fast": paired_bootstrap(rows, "noise_burst", "hybrid", "fast_only"),
            "smooth_hybrid_minus_fast": paired_bootstrap(rows, "smooth", "hybrid", "fast_only"),
        },
        "runs": rows,
    }
    return receipt


def slim(receipt):
    return {"schema": receipt["schema"], "config": receipt["config"],
            "claim_boundary": receipt["claim_boundary"],
            "aggregate": receipt["aggregate"], "paired": receipt["paired"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/gate2_fast_memory.json")
    args = ap.parse_args()
    causal_prefix_check(1000)
    receipt = run(range(args.start, args.start + args.n))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(slim(receipt), indent=2))
