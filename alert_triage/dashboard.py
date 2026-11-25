"""Build an offline viewer from anonymized incident prediction evidence."""
import gzip
import json
from pathlib import Path


def main():
    root=Path("results")
    with gzip.open(root/"study.json.gz","rt",encoding="utf8") as f:
        data=json.load(f)["dashboard"]
    encoded=json.dumps(data,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    html=Path("dashboard/template.html").read_text()
    (root/"alert-triage.html").write_text(html.replace("__TRIAGE_DATA__",encoded),encoding="utf8")


if __name__=="__main__":
    main()
