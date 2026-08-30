"""Objective domain/range compliance for schema.org properties in each sample."""
import csv, os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import CLASSES, PROPS, OUT, MODELS

DATATYPES = {"Text","Number","Integer","Float","Boolean","Date","DateTime","Time","URL","Duration","Distance","Mass","Energy","Quantity"}

def ancestors(label):
    c = CLASSES.get(label)
    if not c: return None
    path = [p.strip() for p in (c.get("path") or "").split(">") if p.strip()]
    return set(path) | {label}

def ok_against(type_label, allowed):
    """None = not checkable (type not a schema.org class)."""
    anc = ancestors(type_label)
    if anc is None: return None
    if not allowed: return None
    allowed = set(allowed)
    if allowed & {"Thing"}: return True
    return bool(anc & allowed)

def compute(model):
    rows = list(csv.DictReader(open(os.path.join(OUT, model, "sample_relations.csv"))))
    per_el = defaultdict(lambda: {"d_ok":0,"d_bad":0,"d_na":0,"r_ok":0,"r_bad":0,"r_na":0,"bad_ex":[]})
    tot = {"d_ok":0,"d_bad":0,"d_na":0,"r_ok":0,"r_bad":0,"r_na":0}
    for r in rows:
        if r["property_source"] != "schema.org": continue
        p = PROPS.get(r["property"])
        st = per_el[("property", r["property"])]
        if not p:
            st["d_na"] += 1; st["r_na"] += 1; tot["d_na"]+=1; tot["r_na"]+=1; continue
        d = ok_against(r["subject_type"], p.get("domain"))
        rg = ok_against(r["object_type"], p.get("range"))
        for val, dk in ((d,"d"), (rg,"r")):
            k = f"{dk}_na" if val is None else (f"{dk}_ok" if val else f"{dk}_bad")
            st[k] += 1; tot[k] += 1
        if d is False or rg is False:
            ex = f'{r["subject_type"]} --{r["property"]}--> {r["object_type"]}'
            if ex not in st["bad_ex"]: st["bad_ex"].append(ex)
    return per_el, tot, rows

if __name__ == "__main__":
    for m in MODELS:
        per_el, tot, rows = compute(m)
        dc = tot["d_ok"]+tot["d_bad"]; rc = tot["r_ok"]+tot["r_bad"]
        print(f'{m:20s} domain checkable={dc:3d} ok={tot["d_ok"]:3d} ({100*tot["d_ok"]/max(dc,1):.0f}%)  '
              f'range checkable={rc:3d} ok={tot["r_ok"]:3d} ({100*tot["r_ok"]/max(rc,1):.0f}%)  '
              f'(not checkable: dom {tot["d_na"]}, rng {tot["r_na"]})')
        # merge into the schemaorg sheet
        p = os.path.join(OUT, m, "judgments_schemaorg.csv")
        sr = list(csv.DictReader(open(p)))
        cols = list(sr[0].keys())
        for c in ("domain_ok","domain_bad","range_ok","range_bad","domain_range_violations"):
            if c not in cols: cols.insert(cols.index("example_usages"), c)
        for row in sr:
            st = per_el.get((row["element_kind"], row["element"]))
            if row["element_kind"] != "property" or not st:
                for c in ("domain_ok","domain_bad","range_ok","range_bad","domain_range_violations"): row[c] = ""
                continue
            row["domain_ok"], row["domain_bad"] = st["d_ok"], st["d_bad"]
            row["range_ok"], row["range_bad"] = st["r_ok"], st["r_bad"]
            row["domain_range_violations"] = " || ".join(st["bad_ex"][:3])
        with open(p,"w",newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(sr)
