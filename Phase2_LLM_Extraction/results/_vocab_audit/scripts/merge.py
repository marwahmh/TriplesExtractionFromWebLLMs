import csv, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import OUT, MODELS
J = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judged")

def cell(t, i): return t[i].strip() if i < len(t) else ""

for m in MODELS:
    # --- schema.org sheet
    p = os.path.join(OUT, m, "judgments_schemaorg.csv")
    rows = list(csv.DictReader(open(p)))
    tsv = [l.rstrip("\n").split("\t") for l in open(os.path.join(J, f"{m}.schemaorg.tsv")) if l.strip()]
    assert len(rows) == len(tsv), (m, "schemaorg", len(rows), len(tsv))
    for r, t in zip(rows, tsv):
        assert r["element"] == t[1]
        r["fits_meaning"], r["is_best_fit"], r["better_fit"] = cell(t,2), cell(t,3), cell(t,4)
        r["issue_code"], r["judge_notes"] = cell(t,5) or "NONE", cell(t,6)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # --- minted sheet
    p = os.path.join(OUT, m, "judgments_minted.csv")
    rows = list(csv.DictReader(open(p)))
    tsv = [l.rstrip("\n").split("\t") for l in open(os.path.join(J, f"{m}.minted.tsv")) if l.strip()]
    assert len(rows) == len(tsv), (m, "minted", len(rows), len(tsv))
    cols = list(rows[0].keys())
    if "label_has_whitespace" not in cols:
        cols.insert(cols.index("exists_verbatim_in_schema_org"), "label_has_whitespace")
    for r, t in zip(rows, tsv):
        assert r["element"] == t[1]
        r["fits_meaning"], r["schemaorg_alternative"], r["suggested_schemaorg"] = cell(t,2), cell(t,3), cell(t,4)
        r["issue_code"], r["judge_notes"] = cell(t,5) or "NONE", cell(t,6)
        r["label_has_whitespace"] = "yes" if re.search(r"[\s\-]", r["element"]) else "no"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"merged {m}")
