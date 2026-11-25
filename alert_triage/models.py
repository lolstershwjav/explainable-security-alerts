"""Matched predictors and multiclass temperature scaling."""
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
import torch
from torch import nn
from xgboost import XGBClassifier


class Predictor:
    def __init__(self,name,seed):
        self.name=name
        self.seed=seed
        self.temperature=1.

    def fit(self,x,y,val,vy,epochs):
        start=perf_counter()
        if self.name=="tree":
            self.model=XGBClassifier(n_estimators=125,max_depth=4,learning_rate=.07,
                                     subsample=.85,colsample_bytree=.85,
                                     objective="multi:softprob",num_class=3,tree_method="hist",
                                     n_jobs=1,random_state=self.seed).fit(x,y)
        elif self.name=="logistic":
            self.model=LogisticRegression(C=.75,max_iter=400,random_state=self.seed).fit(x,y)
        else:
            torch.manual_seed(self.seed)
            rng=torch.Generator().manual_seed(self.seed)
            self.model=nn.Sequential(nn.Linear(x.shape[1],48),nn.SiLU(),
                                     nn.Linear(48,24),nn.SiLU(),nn.Linear(24,3))
            optimizer=torch.optim.AdamW(self.model.parameters(),lr=.004,weight_decay=.0005)
            tx, labels = torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.long)
            best,best_state=-1,None
            for _ in range(epochs):
                for idx in torch.randperm(len(y),generator=rng).split(256):
                    loss=nn.functional.cross_entropy(self.model(tx[idx]),labels[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                metric=f1_score(vy,self.logits(val).argmax(axis=1),average="macro")
                if metric>best:
                    best=metric
                    best_state={k:v.detach().clone() for k,v in self.model.state_dict().items()}
            self.model.load_state_dict(best_state)
        return perf_counter()-start

    def logits(self,x):
        if self.name=="tree":
            return self.model.predict(x,output_margin=True)
        if self.name=="logistic":
            return self.model.decision_function(x)
        with torch.inference_mode():
            return self.model(torch.as_tensor(x,dtype=torch.float32)).numpy()

    def probs(self,x,calibrated=True):
        return softmax(self.logits(x)/(self.temperature if calibrated else 1.),axis=1)

    def calibrate(self,x,y):
        outputs=self.logits(x)
        fit=minimize_scalar(lambda t:log_loss(y,softmax(outputs/np.exp(t),axis=1),labels=[0,1,2]),
                            method="bounded",bounds=(-2.,2.))
        self.temperature=float(np.exp(fit.x))
