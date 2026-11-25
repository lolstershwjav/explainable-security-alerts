"""Explain true-positive risk and compare local attribution diagnostics."""
from time import perf_counter

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import torch
from xgboost import DMatrix

POSITIVE=2


def integrated_gradients(neural,rows,reference,steps=24):
    x=torch.tensor(rows,dtype=torch.float32)
    base=torch.tensor(reference,dtype=torch.float32)
    points=torch.linspace(0,1,steps+1).view(-1,1,1)
    path=(base+points*(x-base)).requires_grad_()
    out=neural(path.reshape(-1,x.shape[1]))[:,POSITIVE]
    grad=torch.autograd.grad(out.sum(),path)[0]
    ave=((grad[:-1]+grad[1:])*.5).mean(0)
    return ((x-base)*ave).detach().numpy()


def tree_shap(tree,rows):
    """Use the gradient-boosting backend's exact TreeSHAP contributions."""
    result=tree.get_booster().predict(DMatrix(rows),pred_contribs=True,strict_shape=True)
    return result[:,POSITIVE,:-1]


def lime_matrix(predictor,reference,rows,seed,samples):
    local=LimeTabularExplainer(reference,mode="classification",discretize_continuous=False,
                               feature_selection="none",random_state=seed,verbose=False)
    values=np.zeros_like(rows,dtype=float)
    for i,x in enumerate(rows):
        # Repeat seeds so the paired stability test uses the same neighborhood draws.
        generated=local.explain_instance(x,predictor.probs,labels=(POSITIVE,),
                                         num_features=rows.shape[1],num_samples=samples)
        for field,weight in generated.local_exp[POSITIVE]:
            values[i,field]=weight
    return values


def explainers(models,reference,seed,samples):
    median=np.median(reference,axis=0)
    return {
        "tree_shap": (models["tree"],lambda x:tree_shap(models["tree"].model,x)),
        "tree_lime": (models["tree"],lambda x:lime_matrix(models["tree"],reference,x,seed,samples)),
        "mlp_ig": (models["mlp"],lambda x:integrated_gradients(models["mlp"].model,x,median)),
        "mlp_lime": (models["mlp"],lambda x:lime_matrix(models["mlp"],reference,x,seed,samples)),
    }


def deletion(predictor,rows,top,reference,rng):
    y=predictor.probs(rows)[:,POSITIVE]
    nominal=rows.copy()
    source=np.arange(len(rows))[:,None]
    nominal[source,top]=reference[top]
    drop=np.abs(y-predictor.probs(nominal)[:,POSITIVE])
    alternatives=[]
    for _ in range(15):
        indices=np.stack([rng.choice(rows.shape[1],top.shape[1],replace=False) for _ in rows])
        other=rows.copy()
        other[source,indices]=reference[indices]
        alternatives.append(np.abs(y-predictor.probs(other)[:,POSITIVE]))
    return drop,np.mean(alternatives,axis=0)


def compare(models,reference,rows,features,seed,cfg):
    rng=np.random.default_rng(seed)
    k=cfg["top_k"]
    jitter=np.clip(rows+rng.normal(0,.05,rows.shape),reference.min(0),reference.max(0))
    median=np.median(reference,axis=0)
    profiles,examples={},{}
    for name,(predictor,fn) in explainers(models,reference,seed,cfg["lime_samples"]).items():
        start=perf_counter()
        values=fn(rows)
        milliseconds=(perf_counter()-start)*1000/len(rows)
        alternate=fn(jitter)
        top=np.argsort(-np.abs(values),axis=1,kind="stable")[:,:k]
        top_alt=np.argsort(-np.abs(alternate),axis=1,kind="stable")[:,:k]
        overlap=np.asarray([len(set(a)&set(b))/len(set(a)|set(b))
                            for a,b in zip(top,top_alt,strict=True)])
        stable=np.abs(predictor.probs(rows)[:,POSITIVE]-predictor.probs(jitter)[:,POSITIVE])<=.05
        removed,chance=deletion(predictor,rows,top,median,np.random.default_rng(seed+90))
        profiles[name]=dict(
            top_k_jaccard=float(overlap.mean()),stable_prediction_fraction=float(stable.mean()),
            stable_jaccard=float(overlap[stable].mean()) if stable.any() else None,
            top_deletion=float(removed.mean()),random_deletion=float(chance.mean()),
            deletion_delta=float((removed-chance).mean()),milliseconds=milliseconds)
        examples[name]=[[dict(feature=features[int(idx)],value=float(values[i,idx]))
                         for idx in inds] for i,inds in enumerate(top)]
    return profiles,examples
