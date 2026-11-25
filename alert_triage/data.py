"""Stream GUIDE evidence and build organization-disjoint incident snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

URL="https://www.kaggle.com/api/v1/datasets/download/microsoft/microsoft-security-incident-prediction"
SHA="23ea64ee8f950c77142935064b3e515f3be1e17120bc229233d932ed4a079ed3"
LABELS=("BenignPositive","FalsePositive","TruePositive")
KEYS=["OrgId","IncidentId"]
COLUMNS=[*KEYS,"AlertId","DetectorId","AlertTitle","Timestamp","Category",
         "IncidentGrade","EntityType","EvidenceRole"]
ENTITIES=("ip","user","mailmessage","machine","file","process","url")
CATEGORIES=("initialaccess","exfiltration","suspiciousactivity","impact","commandandcontrol",
            "execution","discovery","credentialaccess")
PHASES={
    "train": (0,60,"2024-05-20","2024-06-01"),
    "validation": (60,70,"2024-06-01","2024-06-05"),
    "calibration": (70,80,"2024-06-05","2024-06-09"),
    "test": (80,100,"2024-06-09","2024-06-18"),
}
FEATURES=["evidence_log","distinct_alerts_log","distinct_detectors_log","distinct_titles_log",
          "distinct_entity_types_log","distinct_categories_log","impacted_fraction","related_fraction",
          "hour_sin","hour_cos","weekday_sin","weekday_cos",
          *("entity_"+s for s in ENTITIES),"entity_other",
          *("category_"+s for s in CATEGORIES),"category_other"]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def download(path):
    if path.is_file() and sha(path)==SHA:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(".tmp")
    try:
        with urlopen(URL,timeout=120) as stream, partial.open("wb") as sink:
            shutil.copyfileobj(stream,sink)
        if sha(partial)!=SHA:
            raise ValueError("GUIDE archive hash does not match the pinned release")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def bucket(ids):
    v=ids.to_numpy(dtype=np.int64)
    return ((v * 109297) ^ (v // 7)) % 100


def select_evidence(archive,member,official_test):
    frames=[]
    with archive.open(member) as source:
        for chunk in pd.read_csv(source,usecols=COLUMNS,chunksize=170_000,low_memory=False):
            org=chunk.OrgId.to_numpy(dtype=np.int64)
            incident=chunk.IncidentId.to_numpy(dtype=np.int64)
            scope=(bucket(chunk.OrgId)>=80)==official_test
            # Retain all evidence for the same sampled organization/incident key.
            sample=((incident ^ (org*7129)) % 11) < 3
            selection=scope & sample & chunk.IncidentGrade.isin(LABELS).to_numpy()
            frames.append(chunk.loc[selection])
    return pd.concat(frames,ignore_index=True)


def snapshots(frame,horizon_minutes=30):
    frame=frame.loc[frame.IncidentGrade.isin(LABELS)].copy()
    frame["time"]=pd.to_datetime(frame.Timestamp,format="mixed",utc=True,errors="coerce")
    frame=frame.loc[frame.time.notna()]
    grouped=frame.groupby(KEYS,sort=False)
    consistent=grouped.IncidentGrade.nunique()
    valid=consistent[consistent.eq(1)].index
    frame=frame.loc[pd.MultiIndex.from_frame(frame[KEYS]).isin(valid)].copy()
    frame["start"]=frame.groupby(KEYS,sort=False).time.transform("min")
    frame=frame.loc[(frame.time-frame.start).dt.total_seconds()<=60*horizon_minutes]
    frame=frame.drop_duplicates([*KEYS,"AlertId","DetectorId","AlertTitle","Timestamp",
                                 "Category","EntityType","EvidenceRole"])
    frame["impacted_fraction"]=frame.EvidenceRole.str.lower().eq("impacted")
    frame["related_fraction"]=frame.EvidenceRole.str.lower().eq("related")
    for prefix,values,col in [("entity",ENTITIES,"EntityType"),("category",CATEGORIES,"Category")]:
        lowered=frame[col].str.lower()
        for value in values:
            frame[f"{prefix}_{value}"]=lowered.eq(value)
        frame[f"{prefix}_other"]=~lowered.isin(values)
    grouped=frame.groupby(KEYS,sort=False)
    bundle=grouped.agg(start=("start","first"),label=("IncidentGrade","first"),
        evidence_log=("AlertId","size"),distinct_alerts_log=("AlertId","nunique"),
        distinct_detectors_log=("DetectorId","nunique"),distinct_titles_log=("AlertTitle","nunique"),
        distinct_entity_types_log=("EntityType","nunique"),distinct_categories_log=("Category","nunique"))
    log_fields=[k for k in bundle if k.endswith("_log")]
    bundle[log_fields]=np.log1p(bundle[log_fields])
    fraction_fields=[k for k in FEATURES if "_fraction" in k or k.startswith(("entity_","category_"))]
    bundle=bundle.join(grouped[fraction_fields].mean())
    hour=bundle.start.dt.hour/24*2*np.pi
    day=bundle.start.dt.dayofweek/7*2*np.pi
    bundle["hour_sin"],bundle["hour_cos"]=np.sin(hour),np.cos(hour)
    bundle["weekday_sin"],bundle["weekday_cos"]=np.sin(day),np.cos(day)
    return bundle.reset_index()


def split_snapshots(train,test):
    data={}
    for phase,(lo,hi,left,right) in PHASES.items():
        frame=test if phase=="test" else train
        org=bucket(frame.OrgId)
        mask=(org>=lo)&(org<hi)&(frame.start>=pd.Timestamp(left,tz="UTC"))&(
             frame.start<pd.Timestamp(right,tz="UTC"))
        data[phase]=frame.loc[mask,[*KEYS,"start","label",*FEATURES]].sort_values(
            ["start",*KEYS],kind="stable").reset_index(drop=True)
        if set(data[phase].label)!=set(LABELS):
            raise ValueError(f"Class support insufficient in {phase}")
    return data


def prepare(archive,output):
    if sha(archive)!=SHA:
        raise ValueError("GUIDE archive hash differs from pinned data")
    with ZipFile(archive) as z:
        train=snapshots(select_evidence(z,"GUIDE_Train.csv",False))
        test=snapshots(select_evidence(z,"GUIDE_Test.csv",True))
    out=split_snapshots(train,test)
    output.mkdir(parents=True,exist_ok=True)
    summary={}
    for phase,table in out.items():
        destination=output/f"{phase}.csv.gz"
        table.to_csv(destination,index=False,compression={"method":"gzip","mtime":0})
        summary[phase]=dict(incidents=len(table),orgs=int(table.OrgId.nunique()),
                            class_counts=table.label.value_counts().to_dict())
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,default=Path("data/guide.zip"))
    p.add_argument("--output",type=Path,default=Path("data/derived"))
    args=p.parse_args()
    download(args.archive)
    prepare(args.archive,args.output)


if __name__=="__main__":
    main()
