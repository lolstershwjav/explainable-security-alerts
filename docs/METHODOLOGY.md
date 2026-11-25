# Methodology

## Questions

Can coarse incident observations support accurate triage on new organizations and later dates? Does calibration improve probability estimates after this distribution shift? Do feature attributions remain stable under small changes and identify features that influence the model? Can risk scores help a fixed-budget reviewer find true positives sooner in a simulated queue?

## Data and disjoint cohorts

The [Microsoft GUIDE archive](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction), release 2, contains anonymized evidence, alerts, and incidents with recorded analyst triage decisions. Microsoft distributes it under [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/). The official paper describes the collection and its labels [1]. The pinned SHA-256 identifies the exact 538 MB archive used. No source records or source IDs are bundled.

Read a strict column allowlist from the two official evidence files. Use deterministic organization buckets and retain roughly three of eleven incident hash buckets so all selected evidence for an incident is included. Discard rows without recognized grades and incidents with inconsistent grades, then compute the earliest observed evidence timestamp. Deduplicate selected early evidence and aggregate the first 30 minutes of recorded evidence per incident. Only repeated counts, coarse categorical proportions and cyclical timing features reach the models. Exclude raw organization, incident, detector, alert and entity identifiers from predictors. Detector and title identifiers contribute only as distinct counts. Exclude device and user fields, grade-related metadata such as verdict, and any response action fields.

| Role | Organization bucket | Incident start dates in UTC | Incidents |
|---|---:|---|---:|
| Training | 0–59 | May 20–31, 2024 | 2,103 |
| Validation | 60–69 | June 1–4, 2024 | 4,753 |
| Calibration | 70–79 | June 5–8, 2024 | 4,623 |
| Test | 80–99 | June 9–17, 2024 | 8,051 |

The test role uses the official test archive. Other roles use the official train archive. Roles share no organization identifiers. All row-derived scaling is fit on training data. Test organizations and dates differ simultaneously, so their separate causal effects are unidentifiable in this design. GUIDE timestamps are privacy-transformed, and the 30-minute cut reflects the released observations.

## Predictors and calibration

Compare logistic regression, histogram gradient boosting and a two-layer MLP on the same incident snapshots. Choose the neural checkpoint by validation macro-F1. Fit a positive temperature on the dedicated calibration cohort by minimizing multiclass log loss. Temperature scaling preserves the within-row argmax. It can reorder one class's probabilities across examples. Evaluate accuracy, macro-F1, binary AP for the TruePositive class, Brier score, log loss, top-label expected calibration error (ECE), and TruePositive-specific ECE. An organization bootstrap resamples 600 groups-with-replacement draws on seed-averaged test probabilities. Model means across seeds use the original test cohort.

## Explanation experiment

Explain the TruePositive class irrespective of the selected class. TreeSHAP [2] reports aggregate contributions in raw gradient-boosting logit units. Integrated gradients [4] approximate the neural logit difference from the training median with 24 straight-path integration intervals. LIME [3] fits a weighted local probability surrogate using 480 neighborhood draws and preserves the random seed in paired checks. Compare LIME within both predictor families. Its units differ from TreeSHAP and integrated gradients.

The faithfulness diagnostic measures absolute change in calibrated TruePositive probability after replacing the three highest-magnitude attributed features with training medians, minus the mean change for 15 random sets of three. Replacement can create unrealistic combinations in correlated features. It is a model sensitivity check. Stability measures top-three Jaccard overlap after Gaussian jitter with 0.05 standard-deviation in training-normalized units and clipping to training bounds. Report overlap both overall and conditional on TruePositive probability changing by at most five points. Average 60 test incidents per explanation run over three seeds. This study includes no analyst experiment.

## Queue replay

Simulate one daily queue with approximately 10% of incidents reviewed. Compare first-in-first-out by evidence timestamp, highest calibrated TruePositive risk, and highest confidence in the selected class. Measure true-positive recall and precision among reviewed cases, plus the reviewed fraction that analysts graded false positive. Labels exist only in the evaluation. The replay assumes each daily set is available before ranking and is a fixed-budget simulation.

## References

1. Freitas et al. [AI-Driven Guided Response for Security Operation Centers with Microsoft Copilot for Security](https://arxiv.org/abs/2407.09017). 2024.
2. Lundberg and Lee. [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). NeurIPS, 2017.
3. Ribeiro, Singh and Guestrin. [Why Should I Trust You?](https://arxiv.org/abs/1602.04938). KDD, 2016.
4. Sundararajan, Taly and Yan. [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html). ICML, 2017.
