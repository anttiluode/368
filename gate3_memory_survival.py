#!/usr/bin/env python3
"""Gate 3: a retained model must earn survival through useful reuse.

Gate 2 showed that a fast forgetting model plus frozen retrieval beats fast-only
on recurring contexts, but it also stores almost three models in A->B->C where
nothing ever returns.  Gate 3 separates *birth* from *survival*.

Birth is unchanged from Gate 2: a snapshot is retained provisionally only after
later pre-outcome adequacy and a functional-distinctness check.

Survival adds a causal reuse ledger.  When a memory is retrieved, the learner
keeps a shadow copy of the fast model that would have continued without that
retrieval.  For the next four scored decisions, both the deployed and shadow
predictions are made before each outcome.  After outcomes arrive, the memory is
credited by

    saved error = error(shadow fast) - error(retrieved trajectory)

using exactly the same observations.  A positive four-step saving above a fixed
threshold renews that memory's lease.  An unproven/unused memory expires after a
fixed lease interval and its model state is recycled.

No hidden context labels participate in birth, retrieval, credit, renewal, or
eviction.  The labels below exist only for evaluator metrics.

The key matched worlds use the same generated maps, block lengths, actions and
noise:

    recurrent: A -> B -> A -> C -> B -> A
    novel:     A -> B -> C -> D -> E -> F

If memory has value because of reuse, recurrence should earn persistent state;
novel one-shot contexts should shed old snapshots rather than accumulate them.

This is a toy lease rule, not an optimal memory pricing algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from benchmark import ACTIONS, WeightedRLS, phi
from gate2_fast_memory import (
    FastOnly,
    FastRetained,
    HybridConfig,
    copy_model,
    functional_distance,
)


@dataclass(frozen=True)
class SurvivalConfig:
    lease_steps: int = 360
    credit_window: int = 4
    renew_min_saved_error: float = 0.005


def draw_maps(rng, n=6, min_distance=0.6):
    mats = []
    while len(mats) < n:
        m = rng.uniform(-1.0, 1.0, size=(2, 2))
        if all(np.linalg.norm(m - z) >= min_distance for z in mats):
            mats.append(m)
    return mats


def make_six_block_stream(seed: int, pattern: tuple[int, ...]):
    """Same exogenous random objects for any six-block pattern at a seed."""
    if len(pattern) != 6:
        raise ValueError("six blocks required")
    world_rng = np.random.default_rng(seed)
    maps = draw_maps(world_rng, 6)
    lengths = [int(world_rng.integers(80, 161)) for _ in range(6)]
    action_rng = np.random.default_rng(seed + 1_000_003)
    noise_rng = np.random.default_rng(seed + 2_000_003)
    labels = "ABCDEF"
    events = []
    t = 0
    for bi, mi in enumerate(pattern):
        for _ in range(lengths[bi]):
            u = ACTIONS[int(action_rng.integers(len(ACTIONS)))].copy()
            s = int(action_rng.integers(2))
            target = float((maps[mi] @ u)[s])
            y = target + float(noise_rng.normal(0.0, 0.05))
            events.append((u, s, y, labels[mi]))
            t += 1
    return events


class EarnedSurvival(FastRetained):
    """Gate-2 hybrid with provisional leases and causal reuse credit."""

    def __init__(self, cfg: HybridConfig, survival: SurvivalConfig):
        super().__init__(cfg, retrieval=True)
        self.survival = survival
        self.ledger = []
        self.next_uid = 0
        self.evictions = 0
        self.positive_reuses = 0
        self.negative_reuses = 0
        self.total_saved_error = 0.0
        self.active_credit = None
        self.model_slot_steps = 0

    def _finish_candidate_if_ready(self):
        c = self.candidate
        if c is None or c["n"] < self.cfg.eval_window:
            return
        mse = float(c["loss"] / c["n"])
        distinct, nearest = self._distinct_from_memory(c["model"])
        accepted = mse < self.cfg.accept_mse and distinct
        if accepted:
            self._memories.append(c["model"])
            uid = self.next_uid
            self.next_uid += 1
            self.ledger.append({
                "uid": uid,
                "born_t": self.t,
                "lease_until": self.t + self.survival.lease_steps,
                "saved_error": 0.0,
                "positive_reuses": 0,
                "retrievals": 0,
                "renewals": 0,
            })
            self._births += 1
            self.log.append({"t": self.t, "event": "retain_provisional", "mse": mse,
                             "nearest_memory_mse": nearest, "memory_uid": uid})
        elif not distinct:
            self.duplicate_rejections += 1
            self.log.append({"t": self.t, "event": "reject_duplicate", "mse": mse,
                             "nearest_memory_mse": nearest})
        else:
            self.adequacy_rejections += 1
            self.log.append({"t": self.t, "event": "reject_inadequate", "mse": mse,
                             "nearest_memory_mse": nearest})
        self.candidate = None
        self.armed = False
        self.bad_ewma = 0.0

    def _find_uid(self, uid):
        for i, rec in enumerate(self.ledger):
            if rec["uid"] == uid:
                return i
        return None

    def _evict_expired(self):
        active_uid = None if self.active_credit is None else self.active_credit["uid"]
        i = len(self.ledger) - 1
        while i >= 0:
            rec = self.ledger[i]
            if rec["uid"] != active_uid and self.t > rec["lease_until"]:
                self.log.append({"t": self.t, "event": "evict", "memory_uid": rec["uid"],
                                 "saved_error": rec["saved_error"],
                                 "positive_reuses": rec["positive_reuses"]})
                self.ledger.pop(i)
                self._memories.pop(i)
                self.evictions += 1
            i -= 1

    def _maybe_retrieve_survival(self):
        self._evict_expired()
        if not self._memories:
            return
        if self.t - self.last_retrieval_t < self.cfg.retrieve_cooldown:
            return
        if len(self.router_history) < self.cfg.router_window:
            return
        # Do not begin a second counterfactual until the current one closes.
        if self.active_credit is not None:
            return
        fast_loss = self._recent_mse(self.fast)
        mem_losses = np.asarray([self._recent_mse(m) for m in self._memories])
        j = int(np.argmin(mem_losses))
        best = float(mem_losses[j])
        if best < self.cfg.retrieve_abs_mse and best < self.cfg.retrieve_ratio * max(fast_loss, 1e-12):
            rec = self.ledger[j]
            rec["retrievals"] += 1
            shadow = copy_model(self.fast)
            self.fast = copy_model(self._memories[j])
            self._retrievals += 1
            self.last_retrieval_t = self.t
            self.active_credit = {
                "uid": rec["uid"],
                "shadow": shadow,
                "remaining": self.survival.credit_window,
                "saved": 0.0,
            }
            self.log.append({"t": self.t, "event": "retrieve", "memory_uid": rec["uid"],
                             "memory_mse": best, "fast_mse": fast_loss})

    def predict(self, u, s):
        self._maybe_retrieve_survival()
        x = phi(u, s)
        fast_pred = self.fast.predict(x)
        meta = {"x": x, "fast_pred": fast_pred, "pred": fast_pred}
        if self.candidate is not None:
            meta["candidate_pred"] = self.candidate["model"].predict(x)
        if self.active_credit is not None:
            meta["shadow_pred"] = self.active_credit["shadow"].predict(x)
        return fast_pred, meta

    def _finish_credit(self):
        c = self.active_credit
        if c is None:
            return
        idx = self._find_uid(c["uid"])
        if idx is not None:
            rec = self.ledger[idx]
            benefit = float(c["saved"])
            rec["saved_error"] += benefit
            self.total_saved_error += benefit
            if benefit >= self.survival.renew_min_saved_error:
                rec["positive_reuses"] += 1
                rec["renewals"] += 1
                rec["lease_until"] = max(rec["lease_until"], self.t + self.survival.lease_steps)
                self.positive_reuses += 1
                verdict = "renew"
            else:
                self.negative_reuses += 1
                verdict = "no_renew"
            self.log.append({"t": self.t, "event": verdict, "memory_uid": rec["uid"],
                             "saved_error_window": benefit,
                             "lease_until": rec["lease_until"]})
        self.active_credit = None

    def observe(self, u, s, y, meta):
        # Credit both pre-outcome predictions on the exact same outcome.
        if self.active_credit is not None:
            shadow_pred = float(meta["shadow_pred"])
            deployed_pred = float(meta["fast_pred"])
            self.active_credit["saved"] += ((y - shadow_pred) ** 2 -
                                             (y - deployed_pred) ** 2)
            shadow = self.active_credit["shadow"]
            shadow.P /= self.cfg.discount
            shadow.update(meta["x"], y)
            self.active_credit["remaining"] -= 1

        super().observe(u, s, y, meta)
        self.model_slot_steps += 20 * len(self._memories)

        if self.active_credit is not None and self.active_credit["remaining"] <= 0:
            self._finish_credit()
        self._evict_expired()


class ForeverHybrid(FastRetained):
    """Gate 2 hybrid plus resource-time accounting; never evicts."""

    def __init__(self, cfg):
        super().__init__(cfg, retrieval=True)
        self.model_slot_steps = 0
        self.evictions = 0

    def observe(self, u, s, y, meta):
        super().observe(u, s, y, meta)
        self.model_slot_steps += 20 * len(self._memories)


def learner_for(kind, cfg, survival):
    if kind == "fast_only":
        return FastOnly(cfg)
    if kind == "retain_forever":
        return ForeverHybrid(cfg)
    if kind == "earned_survival":
        return EarnedSurvival(cfg, survival)
    raise ValueError(kind)


def evaluate(events, kind, cfg, survival):
    learner = learner_for(kind, cfg, survival)
    errors = []
    labels = []
    for u, s, y, label in events:
        pred, meta = learner.predict(u, s)
        errors.append(float((y - pred) ** 2))
        labels.append(label)
        learner.observe(u, s, y, meta)

    seen = set()
    early = []
    i = 0
    while i < len(labels):
        label = labels[i]
        j = i + 1
        while j < len(labels) and labels[j] == label:
            j += 1
        if label in seen:
            early.extend(errors[i:min(i + 16, j)])
        seen.add(label)
        i = j

    memories = len(getattr(learner, "memories", []))
    return {
        "method": kind,
        "mse": float(np.mean(errors)),
        "return_early16_mse": float(np.mean(early)) if early else None,
        "births": int(getattr(learner, "births", 0)),
        "retrievals": int(getattr(learner, "retrievals", 0)),
        "final_memories": int(memories),
        "evictions": int(getattr(learner, "evictions", 0)),
        "model_slot_steps": int(getattr(learner, "model_slot_steps", 0)),
        "total_saved_error": float(getattr(learner, "total_saved_error", 0.0)),
        "positive_reuses": int(getattr(learner, "positive_reuses", 0)),
        "negative_reuses": int(getattr(learner, "negative_reuses", 0)),
    }


def aggregate(rows):
    out = []
    for family in ("recurrent", "novel"):
        for method in ("fast_only", "retain_forever", "earned_survival"):
            q = [r for r in rows if r["family"] == family and r["method"] == method]
            out.append({
                "family": family,
                "method": method,
                "n": len(q),
                "mse": float(np.mean([r["mse"] for r in q])),
                "return_early16_mse": (float(np.mean([r["return_early16_mse"] for r in q]))
                                        if q and q[0]["return_early16_mse"] is not None else None),
                "births": float(np.mean([r["births"] for r in q])),
                "retrievals": float(np.mean([r["retrievals"] for r in q])),
                "final_memories": float(np.mean([r["final_memories"] for r in q])),
                "evictions": float(np.mean([r["evictions"] for r in q])),
                "model_slot_steps": float(np.mean([r["model_slot_steps"] for r in q])),
                "total_saved_error": float(np.mean([r["total_saved_error"] for r in q])),
                "positive_reuses": float(np.mean([r["positive_reuses"] for r in q])),
                "negative_reuses": float(np.mean([r["negative_reuses"] for r in q])),
            })
    return out


def paired(rows, family, a, b, key):
    aa = {r["seed"]: r for r in rows if r["family"] == family and r["method"] == a}
    bb = {r["seed"]: r for r in rows if r["family"] == family and r["method"] == b}
    seeds = sorted(set(aa) & set(bb))
    d = np.asarray([aa[s][key] - bb[s][key] for s in seeds], dtype=float)
    rng = np.random.default_rng(368_3003)
    boots = d[rng.integers(len(d), size=(10_000, len(d)))].mean(axis=1)
    return {"a": a, "b": b, "metric": key, "mean_difference": float(np.mean(d)),
            "bootstrap95": [float(x) for x in np.quantile(boots, [0.025, 0.975])]}


def run(seeds):
    cfg = HybridConfig()
    survival = SurvivalConfig()
    rows = []
    patterns = {
        "recurrent": (0, 1, 0, 2, 1, 0),
        "novel": (0, 1, 2, 3, 4, 5),
    }
    for seed in seeds:
        for family, pattern in patterns.items():
            events = make_six_block_stream(seed, pattern)
            for method in ("fast_only", "retain_forever", "earned_survival"):
                r = evaluate(events, method, cfg, survival)
                r.update({"seed": int(seed), "family": family})
                rows.append(r)
    return {
        "schema": "repo368/earned-memory-survival-v1",
        "gate2_config": asdict(cfg),
        "survival_config": asdict(survival),
        "aggregate": aggregate(rows),
        "paired": [
            paired(rows, "recurrent", "earned_survival", "fast_only", "mse"),
            paired(rows, "recurrent", "earned_survival", "retain_forever", "mse"),
            paired(rows, "recurrent", "earned_survival", "retain_forever", "model_slot_steps"),
            paired(rows, "novel", "earned_survival", "retain_forever", "mse"),
            paired(rows, "novel", "earned_survival", "retain_forever", "model_slot_steps"),
            paired(rows, "novel", "earned_survival", "retain_forever", "final_memories"),
        ],
        "claim_boundary": (
            "Gate 3 tests a fixed lease plus pre-outcome counterfactual reuse credit. It demonstrates only whether this toy rule can trade retained-state cost against recurrent prediction benefit. "
            "The lease and credit thresholds are not claimed optimal or biological."
        ),
        "runs": rows,
    }


def slim(r):
    return {k: r[k] for k in ("schema", "gate2_config", "survival_config", "aggregate", "paired", "claim_boundary")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/gate3_survival.json")
    args = ap.parse_args()
    r = run(range(args.start, args.start + args.n))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(slim(r), indent=2))
