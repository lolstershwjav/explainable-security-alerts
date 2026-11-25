"""Evaluate classification, calibration, ranking decisions and explanation quality."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
import torch

from alert_triage.data import FEATURES,LABELS,PHASES,SHA
from alert_triage.explain import POSITIVE,compare
from alert_triage.models import Predictor


def ece(outcome,confidence,bins=12):
    k=np.minimum(np.floor(confidence*bins).astype(int),bins-1)
    observed=np.bincount(k,weights=outcome,minlength=bins)
    predicted=np.bincount(k,weights=confidence,minlength=bins)
    return float(np.abs(observed-predicted).sum()/len(outcome))


def metrics(y,p):
    label=p.argmax(1)
    target=y==POSITIVE
    return dict(accuracy=float(accuracy_score(y,label)),
                macro_f1=float(f1_score(y,label,average="macro")),
                attack_ap=float(average_precision_score(target,p[:,POSITIVE])),
                log_loss=float(log_loss(y,p,labels=[0,1,2])),
                brier=float(((np.eye(3)[y]-p)**2).sum(1).mean()),
                ece=ece(label==y,p.max(1)),attack_ece=ece(target,p[:,POSITIVE]))


def queue(frame,y,prob,budget):
    outputs={k:[] for k in ("fifo","attack_risk","classifier_confidence")}
    for _,indices in frame.groupby(frame.start.dt.floor("D")).groups.items():
        idx=np.asarray(indices)
        n=max(1,int(np.ceil(len(idx)*budget)))
        orders={"fifo":idx,"attack_risk":idx[np.argsort(-prob[idx,POSITIVE],kind="stable")],
                "classifier_confidence":idx[np.argsort(-prob[idx].max(1),kind="stable")]}
        for key,rank in orders.items():
            outputs[key].extend(rank[:n])
    res={}
    for key,idx in outputs.items():
        sampled=y[idx]
        res[key]=dict(reviewed=len(idx),attack_recall=float((sampled==POSITIVE).sum()/(y==POSITIVE).sum()),
                      attack_precision=float((sampled==POSITIVE).mean()),
                      false_positive_fraction=float((sampled==1).mean()))
    return res


def load(directory):
    parts={k:pd.read_csv(directory/f"{k}.csv.gz",parse_dates=["start"]) for k in PHASES}
    orgs=[set(p.OrgId) for p in parts.values()]
    for i,part in enumerate(parts.values()):
        if any(orgs[i]&orgs[j] for j in range(i)):
            raise ValueError("Organization leakage across dataset splits")
        if part.duplicated(["OrgId","IncidentId"]).any():
            raise ValueError("Repeated incident key in one partition")
    return parts


def sample(parts,cfg,seed):
    out={}
    for i,(name,frame) in enumerate(parts.items()):
        limit=cfg["max_"+("training" if name=="train" else name)]
        if len(frame)>limit:
            rng=np.random.default_rng([seed,i])
            idx=np.sort(rng.choice(len(frame),limit,replace=False))
            frame=frame.iloc[idx].reset_index(drop=True)
        out[name]=frame
    return out


def uncertainty(frame,y,predictions,seed=894):
    """Bootstrap complete organizations, preserving paired model comparisons."""
    rng=np.random.default_rng(seed)
    blocks=[np.asarray(ix) for ix in frame.groupby("OrgId").indices.values()]
    combined={name:np.mean(v,axis=0) for name,v in predictions.items()}
    retained={name:[] for name in (*combined,"tree_minus_mlp")}
    truth=y==POSITIVE
    for _ in range(600):
        idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        if not truth[idx].any():
            continue
        scores={name:float(average_precision_score(truth[idx],p[idx,POSITIVE]))
                for name,p in combined.items()}
        for name,entry in scores.items():
            retained[name].append(entry)
        retained["tree_minus_mlp"].append(scores["tree"]-scores["mlp"])
    result={}
    for name,values in retained.items():
        lo,hi=np.quantile(values,[.025,.975])
        result[name]=dict(low=float(lo),high=float(hi),n=len(values))
    return result


def run(data,cfg):
    torch.set_num_threads(1)
    parts=load(data)
    runs=[]
    dashboard=[]
    fixed_test={name:[] for name in ("logistic","tree","mlp")}
    for seed in cfg["seeds"]:
        selected=sample(parts,cfg,seed)
        scaler=StandardScaler().fit(selected["train"][FEATURES].to_numpy())
        matrices={name:scaler.transform(frame[FEATURES].to_numpy()).clip(-8,8)
                  for name,frame in selected.items()}
        classes={name:frame.label.map({x:i for i,x in enumerate(LABELS)}).to_numpy(dtype=int)
                 for name,frame in selected.items()}
        models={}
        results={}
        for name in ("logistic","tree","mlp"):
            m=Predictor(name,seed)
            elapsed=m.fit(matrices["train"],classes["train"],matrices["validation"],
                          classes["validation"],cfg["mlp_epochs"])
            m.calibrate(matrices["calibration"],classes["calibration"])
            test,p=matrices["test"],m.probs(matrices["test"])
            results[name]=dict(raw=metrics(classes["test"],m.probs(test,False)),
                               calibrated=metrics(classes["test"],p),
                               queue=queue(selected["test"],classes["test"],p,cfg["queue_budget"]),
                               temperature=m.temperature,fit_seconds=elapsed)
            models[name]=m
            if len(selected["test"])==len(parts["test"]):
                fixed_test[name].append(p)
        rng=np.random.default_rng(seed+900)
        choices=np.sort(rng.choice(len(test),cfg["explain_rows"],replace=False))
        observed=test[choices]
        attrib,prototypes=compare(models,matrices["train"],observed,FEATURES,seed,cfg)
        if not dashboard:
            probabilities={name:m.probs(observed) for name,m in models.items() if name!="logistic"}
            for j,idx in enumerate(choices[:24]):
                dashboard.append(dict(
                    case=j+1,observed_grade=LABELS[classes["test"][idx]],
                    scores={k:v[j].tolist() for k,v in probabilities.items()},
                    explanations={name:rows[j] for name,rows in prototypes.items()}))
        runs.append(dict(seed=seed,models=results,explanations=attrib,
                         sizes={name:len(frame) for name,frame in selected.items()}))
        print(f"Finished seed {seed}",flush=True)
    clusters=uncertainty(selected["test"],classes["test"],fixed_test) if fixed_test["tree"] else {}
    return dict(config=cfg,source_sha256=SHA,classes=LABELS,runs=runs,dashboard=dashboard,
                organization_bootstrap_ap=clusters,
                full_cohorts={k:dict(n=len(v),organizations=v.OrgId.nunique(),
                               class_counts=v.label.value_counts().to_dict()) for k,v in parts.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("data/derived"))
    p.add_argument("--config",type=Path,default=Path("configs/study.json"))
    p.add_argument("--output",type=Path,default=Path("results/study.json"))
    args=p.parse_args()
    result=run(args.data,json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result,separators=(",",":"),allow_nan=False))
    temp.replace(args.output)


if __name__=="__main__":
    main()
