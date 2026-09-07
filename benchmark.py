#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

ACTIONS = np.asarray([
    [1.0,0.0],[-1.0,0.0],[0.0,1.0],[0.0,-1.0],
    [1/math.sqrt(2),1/math.sqrt(2)], [1/math.sqrt(2),-1/math.sqrt(2)],
    [-1/math.sqrt(2),1/math.sqrt(2)], [-1/math.sqrt(2),-1/math.sqrt(2)],
], dtype=float)


def phi(u: np.ndarray, s: int) -> np.ndarray:
    x = np.zeros(4, dtype=float)
    x[2*s:2*s+2] = u
    return x

@dataclass
class WorldConfig:
    noise: float = 0.05
    block_min: int = 80
    block_max: int = 160
    min_map_distance: float = 0.6
    sequence: tuple[str,...] = ("A","B","A","C","B","A")

class SwitchingWorld:
    def __init__(self, seed: int, cfg: WorldConfig, family: str="base"):
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.family = family
        self.maps = self._draw_maps(3)
        self.labels = ["A","B","C"]
        self.mode_lookup = {k:i for i,k in enumerate(self.labels)}
        self.steps = []
        self._build_schedule()
    def _draw_maps(self,n):
        mats=[]
        while len(mats)<n:
            m=self.rng.uniform(-1,1,size=(2,2))
            if all(np.linalg.norm(m-z) >= self.cfg.min_map_distance for z in mats): mats.append(m)
        return mats
    def _build_schedule(self):
        seq = self.cfg.sequence
        if self.family.startswith("stationary") or self.family == "quadratic_stationary" or self.family == "noise_burst": seq=("A",)*6
        for bi,name in enumerate(seq):
            L=int(self.rng.integers(self.cfg.block_min,self.cfg.block_max+1))
            if self.family == "smooth" and bi>0:
                prev=self.mode_lookup[seq[bi-1]]; cur=self.mode_lookup[name]
                drift=min(60,L)
                for j in range(drift):
                    a=(j+1)/drift
                    self.steps.append(("blend", prev, cur, a))
                for _ in range(L-drift): self.steps.append(("mode",cur))
            else:
                mi=self.mode_lookup[name]
                for _ in range(L): self.steps.append(("mode",mi))
    def horizon(self): return len(self.steps)
    def target(self,t,u,s):
        spec=self.steps[t]
        if spec[0]=="mode": M=self.maps[spec[1]]
        else:
            _,i,j,a=spec; M=(1-a)*self.maps[i]+a*self.maps[j]
        val=float((M@u)[s])
        if self.family == "quadratic_stationary": val += 0.5*u[0]*u[1]
        return val
    def sample(self,t,u,s):
        sigma=self.cfg.noise
        if self.family=="noise_burst":
            frac=t/max(1,self.horizon()-1)
            if 0.30 <= frac < 0.40 or 0.68 <= frac < 0.76:
                sigma=0.30
        return self.target(t,u,s)+float(self.rng.normal(0,sigma))
    def latent_label(self,t):
        spec=self.steps[t]
        if spec[0]=="mode": return self.labels[spec[1]]
        return f"{self.labels[spec[1]]}->{self.labels[spec[2]]}"

class WeightedRLS:
    def __init__(self, dim=4, ridge=1.0, noise=0.05):
        self.dim=dim; self.theta=np.zeros(dim); self.P=np.eye(dim)/ridge
        self.noise=noise; self.n_eff=0.0
    def predict(self,x): return float(self.theta@x)
    def update(self,x,y,weight=1.0):
        w=float(max(0.0,weight))
        if w<1e-9: return
        xs=x*math.sqrt(w); ys=y*math.sqrt(w)
        Px=self.P@xs; den=1.0+float(xs@Px)
        K=Px/den; err=ys-float(xs@self.theta)
        self.theta += K*err
        self.P -= np.outer(K,xs)@self.P
        self.n_eff += w
    def clone(self):
        q=WeightedRLS(self.dim,1.0,self.noise); q.theta=self.theta.copy(); q.P=self.P.copy(); q.n_eff=self.n_eff; return q
    def nll(self,x,y,sigma=None):
        s=float(sigma or self.noise); e=y-self.predict(x)
        return 0.5*((e/s)**2)+math.log(s*math.sqrt(2*math.pi))

class SingleLearner:
    name="single"
    def __init__(self, noise=0.05, quadratic=False):
        self.quadratic=quadratic; self.model=WeightedRLS(5 if quadratic else 4, noise=noise)
        self.births=0
    def _x(self,u,s):
        x=phi(u,s)
        return np.r_[x,u[0]*u[1]] if self.quadratic else x
    def predict(self,u,s): return self.model.predict(self._x(u,s)), {"expert":0}
    def observe(self,u,s,y,meta): self.model.update(self._x(u,s),y)
    def count_params(self): return self.model.dim + self.model.dim*self.model.dim

class FixedBank:
    name="fixed_bank"
    def __init__(self,k=3,noise=0.05,stick=0.985,temp=1.0):
        self.models=[WeightedRLS(noise=noise) for _ in range(k)]
        for i,m in enumerate(self.models): m.theta += (i-(k-1)/2)*1e-6*np.array([1,-1,1,-1])
        self.r=np.ones(k)/k; self.noise=noise; self.stick=stick; self.temp=temp; self.births=0
    def prior(self):
        k=len(self.models); return self.stick*self.r+(1-self.stick)/k
    def predict(self,u,s):
        x=phi(u,s); p=self.prior(); preds=np.array([m.predict(x) for m in self.models])
        return float(p@preds), {"x":x,"prior":p,"preds":preds}
    def observe(self,u,s,y,meta):
        p=meta["prior"]; preds=meta["preds"]
        z=-0.5*((y-preds)/self.noise)**2
        z=z-np.max(z); post=p*np.exp(z); post=post/max(np.sum(post),1e-300); self.r=post
        for m,w in zip(self.models,post): m.update(meta["x"],y,float(w))
    def count_params(self): return len(self.models)*(4+16)+len(self.models)

@dataclass
class GrowConfig:
    noise: float=0.05
    stick: float=0.985
    bad_z: float=3.0
    bad_ewma_threshold: float=1.8
    bad_ewma_decay: float=0.96
    min_gap: int=64
    train_window: int=40
    eval_window: int=32
    complexity_penalty: float=0.15
    max_experts: int=6
    protect_threshold: float=0.0

class GrowingBank(FixedBank):
    name="growing_bank"
    def __init__(self,cfg:GrowConfig, conflict=False):
        self.gcfg=cfg; self.conflict=conflict
        super().__init__(k=1,noise=cfg.noise,stick=cfg.stick)
        self.births=0; self.rejections=0; self.bad_ewma=0.0; self.t=0; self.last_nom=-10**9
        self.buffer=[]; self.candidate=None; self.admission_log=[]
    def predict(self,u,s):
        pred,meta=super().predict(u,s)
        if self.candidate is not None and self.candidate["phase"]=="eval":
            x=phi(u,s); meta["candidate_pred"]=self.candidate["model"].predict(x)
            meta["incumbent_pred"]=pred
        return pred,meta
    def _maybe_nominate(self):
        if self.candidate is not None or len(self.models)>=self.gcfg.max_experts: return
        if self.t-self.last_nom<self.gcfg.min_gap: return
        if self.bad_ewma<self.gcfg.bad_ewma_threshold: return
        if len(self.buffer)<self.gcfg.train_window: return
        c=WeightedRLS(noise=self.noise)
        for x,y in self.buffer[-self.gcfg.train_window:]: c.update(x,y)
        self.candidate={"model":c,"phase":"eval","n":0,"candidate_loss":0.0,"incumbent_loss":0.0,
                        "started":self.t,"train_n":self.gcfg.train_window}
        self.last_nom=self.t
    def observe(self,u,s,y,meta):
        x=meta["x"]
        if self.candidate is not None and self.candidate["phase"]=="eval":
            cp=meta.get("candidate_pred", self.candidate["model"].predict(x))
            ip=meta.get("incumbent_pred", float(meta["prior"]@meta["preds"]))
            self.candidate["candidate_loss"] += (y-cp)**2
            self.candidate["incumbent_loss"] += (y-ip)**2
            self.candidate["n"] += 1
            self.candidate["model"].update(x,y)
        pred=float(meta["prior"]@meta["preds"])
        z=abs(y-pred)/self.noise
        excess=max(0.0,z-self.gcfg.bad_z)
        self.bad_ewma=self.gcfg.bad_ewma_decay*self.bad_ewma+(1-self.gcfg.bad_ewma_decay)*excess
        self.buffer.append((x.copy(),float(y)))
        if len(self.buffer)>self.gcfg.train_window: self.buffer=self.buffer[-self.gcfg.train_window:]
        super().observe(u,s,y,meta)
        self.t += 1
        if self.candidate is not None and self.candidate["phase"]=="eval" and self.candidate["n"]>=self.gcfg.eval_window:
            c=self.candidate
            adv=(c["incumbent_loss"]-c["candidate_loss"])/c["n"]
            accept=adv > self.gcfg.complexity_penalty
            self.admission_log.append({"t":self.t,"advantage":float(adv),"accepted":bool(accept),"experts_before":len(self.models)})
            if accept:
                self.models.append(c["model"])
                old=self.r*(1-0.25); self.r=np.r_[old,0.25]; self.r/=self.r.sum(); self.births+=1
            else: self.rejections+=1
            self.candidate=None; self.bad_ewma=0.0
        self._maybe_nominate()
    def count_params(self):
        extra=0 if self.candidate is None else 20
        return len(self.models)*20+len(self.models)+extra+len(self.buffer)*5

class ErrorOnlyGrowingBank(GrowingBank):
    name="error_only_growth"
    def _maybe_nominate(self):
        if self.candidate is not None or len(self.models)>=self.gcfg.max_experts: return
        if self.t-self.last_nom<self.gcfg.min_gap: return
        if self.bad_ewma<self.gcfg.bad_ewma_threshold: return
        if len(self.buffer)<self.gcfg.train_window: return
        c=WeightedRLS(noise=self.noise)
        for x,y in self.buffer[-self.gcfg.train_window:]: c.update(x,y)
        self.models.append(c)
        old=self.r*(1-0.25); self.r=np.r_[old,0.25]; self.r/=self.r.sum()
        self.births += 1
        self.admission_log.append({"t":self.t,"advantage":None,"accepted":True,"reason":"error_only"})
        self.last_nom=self.t; self.bad_ewma=0.0

def evaluate_one(seed:int, family:str, learner_kind:str, noise=0.05, grow_cfg:GrowConfig|None=None):
    wcfg=WorldConfig(noise=noise)
    world=SwitchingWorld(seed,wcfg,family=family)
    rng=np.random.default_rng(seed+1000003)
    if learner_kind=="single": learner=SingleLearner(noise=noise)
    elif learner_kind=="single_quadratic": learner=SingleLearner(noise=noise,quadratic=True)
    elif learner_kind=="fixed": learner=FixedBank(k=3,noise=noise)
    elif learner_kind=="growing":
        cfg=grow_cfg or GrowConfig(noise=noise); cfg=GrowConfig(**{**asdict(cfg),"noise":noise}); learner=GrowingBank(cfg)
    elif learner_kind=="error_only":
        cfg=grow_cfg or GrowConfig(noise=noise); cfg=GrowConfig(**{**asdict(cfg),"noise":noise}); learner=ErrorOnlyGrowingBank(cfg)
    else: raise ValueError(learner_kind)
    rows=[]; se=[]
    last_label=None; switch_points=[]
    for t in range(world.horizon()):
        u=ACTIONS[int(rng.integers(len(ACTIONS)))]; s=int(rng.integers(2)); y=world.sample(t,u,s)
        pred,meta=learner.predict(u,s)
        err=(y-pred)**2; learner.observe(u,s,y,meta)
        label=world.latent_label(t)
        if label!=last_label: switch_points.append(t); last_label=label
        rows.append({"t":t,"label":label,"pred":pred,"y":y,"se":err,"experts":len(getattr(learner,"models",[0]))})
        se.append(err)
    recovery=[]
    seen_labels=set([rows[0]["label"]]) if rows else set()
    for sp in switch_points[1:]:
        if "->" in rows[sp]["label"]: continue
        label_here=rows[sp]["label"]
        returned=label_here in seen_labels
        seen_labels.add(label_here)
        a=[r["se"] for r in rows[sp:min(sp+16,len(rows))]]
        b=[r["se"] for r in rows[min(sp+48,len(rows)):min(sp+64,len(rows))]]
        if a and b: recovery.append({"t":sp,"label":rows[sp]["label"],"returned":bool(returned),"early":float(np.mean(a)),"late":float(np.mean(b))})
    return {
        "seed":seed,"family":family,"learner":learner_kind,"mse":float(np.mean(se)),
        "tail_mse":float(np.mean(se[-100:])),"births":int(getattr(learner,"births",0)),
        "rejections":int(getattr(learner,"rejections",0)),"experts_final":len(getattr(learner,"models",[0])),
        "state_slots_approx":int(learner.count_params()),"recovery":recovery,
        "admissions":getattr(learner,"admission_log",[]),
    }


def aggregate(rows):
    keys=sorted(set((r["family"],r["learner"]) for r in rows))
    out=[]
    for fam,lrn in keys:
        q=[r for r in rows if r["family"]==fam and r["learner"]==lrn]
        ret=[z for r in q for z in r["recovery"] if z.get("returned")]
        out.append({"family":fam,"learner":lrn,"n":len(q),"mse_mean":float(np.mean([r["mse"] for r in q])),
                    "mse_median":float(np.median([r["mse"] for r in q])),"births_mean":float(np.mean([r["births"] for r in q])),
                    "births_max":int(max(r["births"] for r in q)),"experts_final_mean":float(np.mean([r["experts_final"] for r in q])),
                    "state_slots_mean":float(np.mean([r["state_slots_approx"] for r in q])),
                    "returned_context_early16_mse":float(np.mean([z["early"] for z in ret])) if ret else None,
                    "returned_context_late16_mse":float(np.mean([z["late"] for z in ret])) if ret else None})
    return out


def run(seeds, out_path=None):
    gcfg=GrowConfig()
    rows=[]
    families=[("base",0.05),("stationary_005",0.05),("stationary_015",0.15),("stationary_030",0.30),("noise_burst",0.05),("smooth",0.05),("quadratic_stationary",0.05)]
    methods=["single","fixed","growing","error_only"]
    for fam,noise in families:
        for seed in seeds:
            for m in methods:
                rows.append(evaluate_one(seed,fam,m,noise=noise,grow_cfg=gcfg))
            if fam=="quadratic_stationary": rows.append(evaluate_one(seed,fam,"single_quadratic",noise=noise,grow_cfg=gcfg))
    receipt={"schema":"repo368/structural-specialization-benchmark-v1","seeds":list(seeds),"grow_config":asdict(gcfg),
             "claim_boundary":"Causal toy benchmark. Module birth is admitted only by prospective pre-outcome prediction advantage. It does not establish a novel continual-learning algorithm or biological growth mechanism.",
             "aggregate":aggregate(rows),"runs":rows}
    if out_path:
        Path(out_path).parent.mkdir(parents=True,exist_ok=True); Path(out_path).write_text(json.dumps(receipt,indent=2))
    return receipt

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--start",type=int,default=1000); ap.add_argument("--n",type=int,default=20); ap.add_argument("--out",default="results/benchmark.json")
    args=ap.parse_args(); r=run(range(args.start,args.start+args.n),args.out)
    print(json.dumps(r["aggregate"],indent=2))
