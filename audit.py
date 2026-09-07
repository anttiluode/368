#!/usr/bin/env python3
"""Read-only review of repo 368; upstream files are never changed.

The discount grid is selected using seeds 0--19 only. The original 100-world
evaluation set is then reused for this review, not described as a new blind test.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--out', type=Path, default=Path(__file__).parent)
args = ap.parse_args()
sys.path.insert(0, str(args.repo.resolve()))
import benchmark as b
import active_probe as a

args.out.mkdir(parents=True, exist_ok=True)


def save(name, obj):
    (args.out/name).write_text(json.dumps(obj, indent=2)+'\n')


def announce(message, **data):
    print(json.dumps({'phase': message, **data}), flush=True)


class ForgettingSingle(b.SingleLearner):
    def __init__(self, discount):
        super().__init__()
        self.discount = discount

    def observe(self, u, s, y, meta):
        # Standard exponentially discounted RLS: forget old precision before
        # incorporating this completed observation. Prediction precedes this.
        self.model.P /= self.discount
        super().observe(u, s, y, meta)


class ReplacingBank(b.GrowingBank):
    def __init__(self):
        super().__init__(b.GrowConfig())

    def observe(self, u, s, y, meta):
        previous = self.births
        super().observe(u, s, y, meta)
        if self.births > previous:
            # Same nomination, 40-event training, 32-future-event audit and
            # threshold. Accepted candidates replace the single old model.
            self.models = [self.models[-1]]
            self.r = np.ones(1)


def stream(seed, family='base'):
    world = b.SwitchingWorld(seed, b.WorldConfig(), family)
    rng = np.random.default_rng(seed+1000003)
    events = []
    for t in range(world.horizon()):
        u = b.ACTIONS[int(rng.integers(len(b.ACTIONS)))].copy()
        s = int(rng.integers(2))
        events.append((u, s, world.sample(t, u, s), world.latent_label(t)))
    return events


def run_stream(learner, events, keep_predictions=False):
    errors, labels, predictions = [], [], []
    for u, s, y, label in events:
        pred, meta = learner.predict(u, s)
        errors.append((y-pred)**2)
        predictions.append(pred)
        learner.observe(u, s, y, meta)
        labels.append(label)  # Evaluation metadata never reaches the learner.
    seen, return_errors = set(), []
    i = 0
    while i < len(labels):
        label = labels[i]
        j = i+1
        while j < len(labels) and labels[j] == label:
            j += 1
        if label in seen:
            return_errors.extend(errors[i:min(i+16, j)])
        seen.add(label)
        i = j
    result = {'mse': float(np.mean(errors)),
              'return_early16_mse': float(np.mean(return_errors)) if return_errors else None,
              'accepted_candidates': int(getattr(learner, 'births', 0)),
              'final_models': len(getattr(learner, 'models', [None]))}
    if keep_predictions:
        result['predictions'] = predictions
    return result


def paired_interval(x, y):
    d = np.asarray(x)-np.asarray(y)
    rng = np.random.default_rng(3682026)
    boots = d[rng.integers(len(d), size=(10000, len(d)))].mean(axis=1)
    return {'mean_difference': float(d.mean()),
            'bootstrap95': [float(z) for z in np.quantile(boots, [.025, .975])]}


upstream_sha = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
frozen = json.loads((args.repo/'results/gate0_heldout.json').read_text())
heldout = [dict(zip(frozen['columns'], row)) for row in frozen['rows']]

# Check discounted RLS against a separate, batch weighted least-squares solve.
rng = np.random.default_rng(44)
xs, ys = rng.normal(size=(30,4)), rng.normal(size=30)
for discount in (.5, .7, .85, .95, .98, 1.0):
    model = b.WeightedRLS()
    for x, y in zip(xs, ys):
        model.P /= discount
        model.update(x, y)
    weights = discount**np.arange(29,-1,-1)
    exact = np.linalg.solve(discount**30*np.eye(4)+xs.T@(weights[:,None]*xs), xs.T@(weights*ys))
    np.testing.assert_allclose(model.theta, exact, rtol=1e-9, atol=1e-9)

# One explicit causal perturbation check, in addition to upstream assertions.
events = stream(1000)
base = run_stream(b.GrowingBank(b.GrowConfig()), events, True)['predictions']
for cut in (0, 40, 120, 300):
    changed = [(u, s, y if i < cut else 100-y, label)
               for i, (u, s, y, label) in enumerate(events)]
    altered = run_stream(b.GrowingBank(b.GrowConfig()), changed, True)['predictions']
    np.testing.assert_array_equal(base[:cut+1], altered[:cut+1])
announce('causal perturbation checks passed')

# Development tuning precedes any evaluation of the new baseline on 1000--1099.
discounts = (.5, .7, .85, .95, .98, 1.0)
dev = []
dev_streams = [stream(seed) for seed in range(20)]
for discount in discounts:
    scores = [run_stream(ForgettingSingle(discount), ev)['mse'] for ev in dev_streams]
    dev.append({'discount': discount, 'mean_mse': float(np.mean(scores)), 'seed_mse': scores})
selected = min(dev, key=lambda x: x['mean_mse'])['discount']
save('development.json', {'seeds': list(range(20)), 'grid': dev, 'selected_discount': selected})
announce('development complete', selected_discount=selected,
         grid=[{'discount': r['discount'], 'mean_mse': r['mean_mse']} for r in dev])

review = []
for seed in range(1000, 1100):
    for family in ('base', 'noise_burst', 'quadratic_stationary'):
        ev = stream(seed, family)
        review.append({'seed': seed, 'family': family, 'method': 'forgetting_single',
                       **run_stream(ForgettingSingle(selected), ev)})
        if family == 'base':
            review.append({'seed': seed, 'family': family, 'method': 'prospective_replace',
                           **run_stream(ReplacingBank(), ev)})
save('alternative_baselines.json', review)
announce('new baseline evaluation complete')

reproduced = []
max_difference = 0.0
for record in heldout:
    seed = record['seed']
    for family, method in [('base','single'),('base','fixed'),('base','growing'),('base','error_only'),
                           ('noise_burst','growing'),('noise_burst','error_only'),
                           ('quadratic_stationary','growing'),('quadratic_stationary','single_quadratic')]:
        r = b.evaluate_one(seed, family, method)
        reproduced.append(r)
        prefix = {'base':'base','noise_burst':'noise','quadratic_stationary':'quadratic'}[family]+'_'+method
        for key in ('mse', 'births'):
            frozen_key = prefix+'_'+key
            if frozen_key in record:
                delta = abs(r[key]-record[frozen_key])
                max_difference = max(max_difference, delta)
                assert delta < 1e-12, (seed, frozen_key, delta)
save('gate0_reproduction.json', {'runs': reproduced, 'aggregate': b.aggregate(reproduced),
                                'max_frozen_difference': max_difference})
announce('Gate 0 reproduced', max_frozen_difference=max_difference)

gate1 = a.run(range(1000,1100), out=str(args.out/'gate1_reproduction.json'))
gate1_frozen = json.loads((args.repo/'results/gate1_summary.json').read_text())
gate1_delta = 0.0
for r in gate1['aggregate']:
    ref = next(z for z in gate1_frozen['aggregate'] if z['learner']==r['learner'] and z['active']==r['active'])
    for key in ('mse','returned_early16_mse','probes_per_decision','births'):
        gate1_delta = max(gate1_delta, abs(r[key]-ref[key]))
assert gate1_delta < 1e-12, gate1_delta
announce('Gate 1 reproduced', max_frozen_difference=gate1_delta)

summary = {'upstream_commit': upstream_sha, 'selected_discount': selected,
           'gate0_max_frozen_difference': max_difference, 'gate1_max_frozen_difference': gate1_delta,
           'evaluation_seeds': [1000,1099], 'aggregate': [], 'paired_base': {}}
for family, method in sorted(set((r['family'],r['method']) for r in review)):
    rows = [r for r in review if r['family']==family and r['method']==method]
    summary['aggregate'].append({'family': family, 'method': method,
        'mean_mse': float(np.mean([r['mse'] for r in rows])),
        'return_early16_mse': float(np.mean([r['return_early16_mse'] for r in rows])) if family=='base' else None,
        'mean_accepted_candidates': float(np.mean([r['accepted_candidates'] for r in rows])),
        'final_models': sorted(set(r['final_models'] for r in rows))})
for method in ('forgetting_single', 'prospective_replace'):
    scores = [next(r['mse'] for r in review if r['seed']==h['seed'] and r['family']=='base' and r['method']==method) for h in heldout]
    summary['paired_base'][method+'_minus_growing'] = paired_interval(scores, [h['base_growing_mse'] for h in heldout])
summary['paired_tradeoffs'] = {}
for family, metric in [('base','return_early16_mse'), ('noise_burst','mse')]:
    new_values, old_values = [], []
    for seed in range(1000,1100):
        new = next(r for r in review if r['seed']==seed and r['family']==family and r['method']=='forgetting_single')
        old = next(r for r in reproduced if r['seed']==seed and r['family']==family and r['learner']=='growing')
        new_values.append(new[metric])
        old_values.append(old['mse'] if metric=='mse' else float(np.mean([z['early'] for z in old['recovery'] if z['returned']])))
    summary['paired_tradeoffs'][family+'_'+metric+'_forgetting_minus_growing'] = paired_interval(new_values, old_values)
summary['runtime'] = {'numpy': np.__version__}
save('summary.json', summary)
announce('review complete', summary=summary)
