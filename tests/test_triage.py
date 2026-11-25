"""Verify temporal aggregation, calibration and numerical explanation contracts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch import nn
from xgboost import DMatrix

from alert_triage.data import FEATURES, LABELS, PHASES, snapshots
from alert_triage.explain import integrated_gradients, tree_shap
from alert_triage.models import Predictor
from alert_triage.study import ece, queue, run


class TriageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_snapshot_removes_late_and_duplicate_rows(self):
        a=dict(OrgId=1,IncidentId=2,AlertId=3,DetectorId=4,AlertTitle=9,
               Timestamp="2024-05-23T00:00:00Z",Category="InitialAccess",
               IncidentGrade="TruePositive",EntityType="Ip",EvidenceRole="Impacted")
        late={**a,"Timestamp":"2024-05-23T01:00:00Z","EntityType":"MailMessage"}
        b={**a,"IncidentId":3,"IncidentGrade":"FalsePositive"}
        conflict={**b,"Timestamp":"2024-05-23T00:03:00Z","IncidentGrade":"BenignPositive"}
        result=snapshots(pd.DataFrame([a,a,late,b,conflict]))
        self.assertEqual(len(result),1)
        self.assertEqual(result.iloc[0].entity_mailmessage,0)
        self.assertEqual(result.iloc[0].entity_ip,1)
        self.assertAlmostEqual(result.iloc[0].evidence_log,np.log1p(1))

    def test_integrated_gradients_is_complete_for_linear_model(self):
        m=nn.Linear(4,3)
        x=np.array([[.5,-2,3,1.],[-1,2,4,1]],dtype=np.float32)
        ref=np.zeros(4,dtype=np.float32)
        attr=integrated_gradients(m,x,ref)
        with torch.inference_mode():
            diff=m(torch.tensor(x))[:,2].numpy()-m(torch.tensor(ref))[2].item()
        np.testing.assert_allclose(attr.sum(1),diff,rtol=1e-6,atol=1e-6)

    def test_tree_contributions_sum_to_target_margin(self):
        rng=np.random.default_rng(81)
        x=rng.normal(size=(160,5))
        labels=(np.arange(len(x))%3)
        p=Predictor("tree",14)
        p.fit(x,labels,x,labels,1)
        contributions=tree_shap(p.model,x[:12])
        values=p.model.get_booster().predict(DMatrix(x[:12]),pred_contribs=True,strict_shape=True)
        np.testing.assert_allclose(contributions.sum(1)+values[:,2,-1],
                                   p.logits(x[:12])[:,2],atol=2e-5)

    def test_calibration_changes_confidence_without_changing_argmax(self):
        raw=np.array([[2.,0,-2.],[-3.,1,0.],[-2.,-1,3.]])
        p=Predictor("mlp",7)
        p.logits=lambda x:raw
        y=np.array([0,1,0])
        p.calibrate(np.zeros_like(raw),y)
        np.testing.assert_array_equal(raw.argmax(1),softmax(raw/p.temperature,axis=1).argmax(1))
        self.assertEqual(ece(np.array([0,1]),np.array([0,1.])),0.)

    def test_review_budget_is_constant_and_risk_is_priority(self):
        times=pd.date_range("2024-06-12", periods=6, freq="h",tz="UTC")
        frame=pd.DataFrame({"start":times})
        y=np.array([1,1,0,2,2,2])
        p=np.array([[0,.9,.1],[0,.8,.2],[.8,.1,.1],[.1,.1,.8],[.1,.2,.7],[.1,.3,.6]])
        out=queue(frame,y,p,1/3)
        self.assertEqual({x["reviewed"] for x in out.values()},{2})
        self.assertGreater(out["attack_risk"]["attack_recall"],out["fifo"]["attack_recall"])

    def test_benchmark_path_and_artifact_privacy(self):
        rng=np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,phase in enumerate(PHASES):
                rows=120
                table=pd.DataFrame(rng.normal(size=(rows,len(FEATURES))),columns=FEATURES)
                table["OrgId"]=i*1000+np.arange(rows)//6
                table["IncidentId"]=np.arange(rows)
                table["start"]=pd.date_range(f"2024-06-{i+10:02}",periods=rows,freq="h",tz="UTC")
                table["label"]=np.resize(LABELS,rows)
                table.to_csv(root/f"{phase}.csv.gz",index=False)
            cfg=dict(seeds=[7],max_training=200,max_validation=200,max_calibration=200,
                     max_test=200,mlp_epochs=2,explain_rows=4,top_k=3,lime_samples=90,
                     queue_budget=.1)
            output=run(root,cfg)
        self.assertEqual(len(output["dashboard"]),4)
        self.assertEqual(len(output["runs"][0]["explanations"]),4)
        public=json.dumps(output,allow_nan=False)
        for forbidden in ("OrgId","IncidentId","2024-06-"):
            self.assertNotIn(forbidden,public)


if __name__=="__main__":
    unittest.main()
