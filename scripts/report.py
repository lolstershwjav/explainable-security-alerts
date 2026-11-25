"""Generate measured triage figures and technical report."""
import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT=Path("results")
METHODS={"tree_shap":"TreeSHAP","tree_lime":"Tree LIME","mlp_ig":"MLP integrated gradients",
         "mlp_lime":"MLP LIME"}
MODELS={"logistic":"Logistic regression","tree":"Gradient boosting","mlp":"Neural classifier"}


def figures(data):
    runs=data["runs"]
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.4))
    for i,(name,title) in enumerate(METHODS.items()):
        vals=[r["explanations"][name] for r in runs]
        axes[0].scatter(mean(t["top_k_jaccard"] for t in vals),
                        100*mean(t["deletion_delta"] for t in vals),s=58,label=title,
                        c=["#516f9b","#9c6c97","#258b75","#b47c4a"][i])
    axes[0].set(xlabel="Top-three Jaccard overlap",ylabel="Deletion sensitivity vs. random (pp)",
                xlim=(.61,1.01),ylim=(0,8))
    axes[0].legend(frameon=False,fontsize=7.4,loc="upper left")
    options=["fifo","attack_risk","classifier_confidence"]
    positions=np.arange(len(options))
    for dx,name,color,label in [(-.17,"attack_recall","#258b75","True-positive recall"),
                                (.17,"false_positive_fraction","#b47c4a","Reviewed false-positive share")]:
        vals=[100*mean(r["models"]["mlp"]["queue"][policy][name] for r in runs)
              for policy in options]
        axes[1].bar(positions+dx,vals,.29,label=label,color=color)
    axes[1].set(xticks=positions,xticklabels=["FIFO","True-positive risk","Confidence"],
                ylabel="Percent",title="Neural classifier · 10% review budget",ylim=(0,47))
    axes[1].legend(frameon=False,fontsize=7.3,loc="upper left")
    for ax in axes:
        ax.grid(alpha=.21)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(w_pad=2)
    fig.savefig(ROOT/"triage-metrics.png",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def report(data):
    dark=colors.HexColor("#17384e")
    root=Path(matplotlib.get_data_path())/"fonts"/"ttf"
    pdfmetrics.registerFont(TTFont("StudySans",str(root/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("StudySans-Bold",str(root/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("StudySans",normal="StudySans",bold="StudySans-Bold")
    styles=getSampleStyleSheet()
    for k in ("Normal","BodyText"):
        styles[k].fontName="StudySans"
    for k in ("Title","Heading2"):
        styles[k].fontName="StudySans-Bold"
        styles[k].textColor=dark
    styles["Title"].fontSize=22
    styles["Title"].leading=26
    styles["Title"].alignment=0
    styles["BodyText"].leading=13.6
    styles["BodyText"].spaceAfter=7
    body=[]
    runs=data["runs"]

    def p(s,k="BodyText"):
        body.append(Paragraph(s,styles[k]))
    def section(t,s):
        p(t,"Heading2")
        p(s)
    def table(rows,widths):
        tab=Table(rows,colWidths=widths,hAlign="LEFT")
        tab.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"StudySans"),
          ("FONTNAME",(0,0),(-1,0),"StudySans-Bold"),("FONTSIZE",(0,0),(-1,-1),8.4),
          ("BACKGROUND",(0,0),(-1,0),dark),("TEXTCOLOR",(0,0),(-1,0),colors.white),
          ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eff5f8")]),
          ("ALIGN",(1,0),(-1,-1),"RIGHT"),("TOPPADDING",(0,0),(-1,-1),5),
          ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        body.append(tab)

    def avg(model,method,metric):
        return mean(r["models"][model][method][metric] for r in runs)

    p("Explainable Security Alert Classification","Title")
    p("Classification, attribution and triage on organization- and time-held-out incidents")
    p("<b>Abstract.</b> On 8,051 analyst-graded security incidents, the neural model achieved "
      "0.545 macro-F1. A fixed-budget queue sorted by neural true-positive risk raised recall "
      "from 10.71% to 21.29%, along with a rise in reviewed false positives. TreeSHAP "
      "attributions produced larger deletion sensitivity than tree LIME, which produced "
      "more stable top-three explanations under the tested perturbation.")
    section("1. Questions and related work",
            "Can local explanations remain stable and reflect influential features under label and "
            "organization shift? GUIDE supplies incident-level triage grades [1]. TreeSHAP assigns "
            "additive attributions [2]. LIME estimates a local surrogate [3], and integrated gradients "
            "trace the neural logit along a fixed input path [4].")
    section("2. Cohort construction and models",
            "Build one early observation snapshot per incident from the first 30 minutes of "
            "recorded evidence. Use disjoint organization cohorts and incident start dates for "
            "fitting, model selection, calibration and evaluation. Exclude identifiers and response "
            "fields from model inputs. Fit a logistic baseline, gradient-boosted trees and a "
            "two-layer MLP. Fit positive temperature scaling on a separate calibration cohort.")
    section("3. Calibration and prediction",
            "Temperature scaling preserves the most likely label within each incident. "
            "It may reorder probabilities for a single grade across incidents. Table 1 compares "
            "measured ranking and probability quality.")
    rows=[["Model","Macro-F1","Test TP AP","Raw loss","Calibrated loss"]]
    for m,title in MODELS.items():
        rows.append([title,*(f"{avg(m,method,metric):.3f}" for method,metric in
                    [("calibrated","macro_f1"),("calibrated","attack_ap"),
                     ("raw","log_loss"),("calibrated","log_loss")])])
    table(rows,[140,78,80,83,105])
    ci=data["organization_bootstrap_ap"]["tree_minus_mlp"]
    p(f"The seed-averaged organization bootstrap places the paired tree-minus-neural "
      f"true-positive AP difference between {100*ci['low']:+.2f} and {100*ci['high']:+.2f} "
      "percentage points. Neural top-label ECE rose from 21.09% to 23.08% despite lower log loss.")
    body.append(PageBreak())

    section("4. Attribution experiment",
            "Explain the true-positive score for 60 sampled test incidents per seed. The "
            "paired diagnostics average three seeds. Apply small feature jitter to measure "
            "top-three Jaccard overlap and compare median replacement against random feature sets.")
    rows=[["Method","Jaccard","Sensitivity vs. random (pp)","Time per case (ms)"]]
    for m,title in METHODS.items():
        a=[r["explanations"][m] for r in runs]
        rows.append([title,f"{mean(x['top_k_jaccard'] for x in a):.3f}",
                     f"{100*mean(x['deletion_delta'] for x in a):.2f}",
                     f"{mean(x['milliseconds'] for x in a):.3f}"])
    table(rows,[151,64,145,126])
    body.append(Image(str(ROOT/"triage-metrics.png"),width=486,height=180,kind="proportional"))
    p("<b>Figure 1.</b> Model sensitivity and stability, and measured outcomes in the simulated queue.")
    section("5. Queue simulation",
            "Review approximately 10% of each daily queue. Ordering by the neural true-positive "
            "probability captures more true positives than FIFO. It also raises the fraction "
            "graded false positive among reviewed cases, from 23.21% to 36.79%. Ranking by "
            "confidence in any predicted grade answers a different question. It reduced true-positive "
            "recall to 5.76% on the neural model.")
    section("6. Limitations and conclusion",
            "The split combines organization and calendar shift. Sparse early exposure of "
            "evidence limits model discrimination and attributions can evaluate unrealistic "
            "feature combinations. Queue experiments assume a complete daily batch and "
            "do not measure human response quality. The useful next comparison would "
            "combine an independent cohort with analyst evaluation and realistic incident costs.")
    p("References","Heading2")
    p("[1] Freitas et al., GUIDE (2024). [2] Lundberg and Lee, NeurIPS (2017). "
      "[3] Ribeiro et al., KDD (2016). [4] Sundararajan et al., ICML (2017). "
      "Persistent source and license links appear in docs/METHODOLOGY.md.")
    def footer(c,doc):
        c.setFont("StudySans",8)
        c.setFillColor(dark)
        c.drawString(52,26,"Explainable Security Alert Classification")
        c.drawRightString(553,26,str(doc.page))
    pdf=SimpleDocTemplate(str(ROOT/"explainable-alert-triage-report.pdf"),
                          title="Explainable Security Alert Classification",
                          leftMargin=52,rightMargin=52,topMargin=42,bottomMargin=42)
    pdf.build(body,onFirstPage=footer,onLaterPages=footer)


if __name__=="__main__":
    with gzip.open(ROOT/"study.json.gz","rt",encoding="utf8") as source:
        measurements=json.load(source)
    figures(measurements)
    report(measurements)
