#!/usr/bin/env python3
"""Gate 1: paid diagnostic PINGs for causal module selection.

The scored action is drawn independently by the evaluator. Before it is revealed,
a learner may buy up to two diagnostic action/readout outcomes from the same hidden
context. Diagnostics update only responsibility (which expert seems applicable), not
expert parameters. Every diagnostic return is counted.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from benchmark import ACTIONS, GrowConfig, FixedBank, GrowingBank, SwitchingWorld, WorldConfig, phi


def entropy(p):
    p=np.asarray(p,float); p=p[p>0]
    return float(-np.sum(p*np.log(p)))


def predict_experts(learner,u,s):
    x=phi(u,s)
    return x,np.asarray([m.predict(x) for m in learner.models],float)


def posterior(p,preds,y,sigma):
    z=-0.5*((y-preds)/sigma)**2
    z-=np.max(z); q=p*np.exp(z); q/=max(np.sum(q),1e-300)
    return q


def choose_probe(learner,p):
    best=None
    for ai,u in enumerate(ACTIONS):
        for s in (0,1):
            _,preds=predict_experts(learner,u,s)
            mu=float(p@preds)
            var=float(p@((preds-mu)**2))
            if best is None or var>best[0]: best=(var,ai,s,preds)
    return best


def evaluate(seed:int, learner_kind='growing', active=True, max_probes=2, entropy_threshold=0.25):
    world=SwitchingWorld(seed,WorldConfig(noise=0.05),'base')
    if learner_kind=='fixed': learner=FixedBank(k=3,noise=0.05)
    elif learner_kind=='growing': learner=GrowingBank(GrowConfig())
    else: raise ValueError(learner_kind)
    score_rng=np.random.default_rng(seed+8000001)
    probe_count=0; sq=[]; rows=[]; seen=set()
    last_label=None
    for t in range(world.horizon()):
        label=world.latent_label(t)
        if label!=last_label:
            seen.add(label); last_label=label
        p=learner.prior()
        if active and len(learner.models)>1:
            for _ in range(max_probes):
                h=entropy(p)/max(math.log(len(p)),1e-12)
                if h < entropy_threshold: break
                var,ai,sq_read,preds=choose_probe(learner,p)
                if var < 1e-6: break
                uq=ACTIONS[ai]
                yq=world.sample(t,uq,sq_read)
                p=posterior(p,preds,yq,learner.noise)
                probe_count += 1
        u=ACTIONS[int(score_rng.integers(len(ACTIONS)))]; s=int(score_rng.integers(2))
        x,preds=predict_experts(learner,u,s)
        pred=float(p@preds)
        meta={'x':x,'prior':p,'preds':preds}
        if isinstance(learner,GrowingBank) and learner.candidate is not None and learner.candidate['phase']=='eval':
            meta['candidate_pred']=learner.candidate['model'].predict(x); meta['incumbent_pred']=pred
        y=world.sample(t,u,s)
        e=(y-pred)**2; sq.append(e)
        learner.observe(u,s,y,meta)
        rows.append((label,e))
    seen=set(); i=0; return_early=[]
    while i<len(rows):
        label=rows[i][0]; j=i+1
        while j<len(rows) and rows[j][0]==label: j+=1
        if label in seen: return_early.extend(e for _,e in rows[i:min(i+16,j)])
        seen.add(label); i=j
    return {'seed':seed,'learner':learner_kind,'active':active,'mse':float(np.mean(sq)),
            'returned_early16_mse':float(np.mean(return_early)),'probes':probe_count,
            'probes_per_decision':probe_count/len(sq),'births':getattr(learner,'births',0),
            'experts_final':len(learner.models)}


def run(seeds=range(1000,1020),out=None):
    rows=[]
    for seed in seeds:
        for learner in ('fixed','growing'):
            for active in (False,True): rows.append(evaluate(seed,learner,active))
    agg=[]
    for learner in ('fixed','growing'):
        for active in (False,True):
            q=[r for r in rows if r['learner']==learner and r['active']==active]
            agg.append({'learner':learner,'active':active,'n':len(q),'mse':float(np.mean([r['mse'] for r in q])),
                        'returned_early16_mse':float(np.mean([r['returned_early16_mse'] for r in q])),
                        'probes_per_decision':float(np.mean([r['probes_per_decision'] for r in q])),
                        'births':float(np.mean([r['births'] for r in q]))})
    receipt={'schema':'repo368/paid-ping-v1','aggregate':agg,'runs':rows,
             'claim_boundary':'Diagnostic probes update responsibility only and are charged. The evaluator chooses the scored query independently. This tests active module selection, not active structural birth.'}
    if out: Path(out).write_text(json.dumps(receipt,indent=2))
    return receipt

if __name__=='__main__':
    r=run(out='results/gate1_probe.json'); print(json.dumps(r['aggregate'],indent=2))
