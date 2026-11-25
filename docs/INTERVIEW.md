# Technical interview preparation

1. **Why use both organization and temporal holdouts?** Incidents can share organization-specific detectors and historical conditions. Disjoint organizations and time windows test a stronger transfer setting than random rows.
2. **Why aggregate early evidence?** Final incidents can accumulate evidence and responses after triage. A bounded 30-minute snapshot better approximates information available for an early decision.
3. **Why report macro-F1 and true-positive AP?** Macro-F1 weighs classes equally at a fixed classification rule. AP tests ranking quality for the class an incident reviewer may prioritize.
4. **What changes with temperature scaling?** Dividing logits by one positive value preserves their order within an incident, while changing confidence and potentially the ranking of a single class across incidents.
5. **What does TreeSHAP decompose?** The tree model's class logit relative to its background reference. The additivity test checks that contributions plus the reference recover the score.
6. **What does integrated gradients integrate?** Partial derivatives along the path from a chosen baseline to the case, weighted by each input change.
7. **Why compare both explanation stability and removal sensitivity?** Similar top features under slight input changes can help comprehension. However, stable attributions can still identify features with little effect on the underlying model output.
8. **What is a limitation of feature removal?** Replacing one feature at a time can create combinations outside the training distribution. The score response is an experimental sensitivity diagnostic.
9. **Can the review-queue results establish analyst benefit?** The queue simulation shows how retrospective labels move under a fixed review budget. It does not measure analyst speed, accuracy, or experience.
10. **Why do organization-bootstrap intervals matter?** Examples from the same organization are correlated. Resampling organizations as blocks preserves within-organization dependence while estimating sensitivity to the tested organizations.

## Evidence-based résumé drafts

- Benchmarked security incident classification on 8,051 analyst-labeled Microsoft GUIDE incidents across 486 held-out organizations using time-separated model, validation and calibration cohorts.
- Evaluated TreeSHAP, LIME and integrated gradients with paired feature-removal and stability diagnostics, finding higher tree-model score sensitivity for SHAP and greater top-three stability for LIME.
- In a fixed-budget retrospective triage simulation, increased true-positive recall from 10.71% to 21.29% with calibrated neural risk ranking and quantified the accompanying increase in reviewed false positives.

The project fits CrowdStrike's emphasis on sound experimental design and clear interpretation. Reproduce the calibration and explanation checks before using these draft bullets.
