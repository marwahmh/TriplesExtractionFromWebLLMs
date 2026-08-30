"""What this sample can and cannot resolve.

Prints the judged and computed columns side by side with 95% Wilson intervals.
The judged columns do not separate the backends at this sample size; the computed
ones do. Section 5's two tables are this script's output.
"""
import csv, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import OUT, MODELS

def wilson(k, n, z=1.96):
    if not n: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-h), 100*(c+h))

def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:3.0f}% [{lo:.0f},{hi:.0f}] n={n}"

print("JUDGED (does not separate the backends)")
for m in MODELS:
    rows = list(csv.DictReader(open(os.path.join(OUT, m, "judgments_schemaorg.csv"))))
    P = [r for r in rows if r["element_kind"] == "property"]
    T = [r for r in rows if r["element_kind"] == "type"]
    mi = [r for r in csv.DictReader(open(os.path.join(OUT, m, "judgments_minted.csv"))) if r["source"] == "minted"]
    y = lambda rs: sum(1 for r in rs if r["fits_meaning"] == "yes")
    print(f"  {m:20s} property {fmt(y(P),len(P)):22s} type {fmt(y(T),len(T)):22s} minted {fmt(y(mi),len(mi))}")

print("\nCOMPUTED (separates them sharply, and carries no annotator error)")
for m in MODELS:
    P = [r for r in csv.DictReader(open(os.path.join(OUT, m, "judgments_schemaorg.csv"))) if r["element_kind"] == "property"]
    do = sum(int(r["domain_ok"] or 0) for r in P); db = sum(int(r["domain_bad"] or 0) for r in P)
    ro = sum(int(r["range_ok"] or 0) for r in P);  rb = sum(int(r["range_bad"] or 0) for r in P)
    fb = sum(1 for r in P if r["justification_is_fallback"] == "yes")
    print(f"  {m:20s} domain {fmt(do,do+db):22s} range {fmt(ro,ro+rb):22s} fallback {fmt(fb,len(P))}")

print("\nPER-RELATION (n=250 each, the best power in this sample)")
for m in MODELS:
    sc = {(r["element_kind"], r["element"]): r for r in csv.DictReader(open(os.path.join(OUT, m, "judgments_schemaorg.csv")))}
    mi = {}
    for r in csv.DictReader(open(os.path.join(OUT, m, "judgments_minted.csv"))):
        mi.setdefault((r["element_kind"], r["element"]), r)
    rel = list(csv.DictReader(open(os.path.join(OUT, m, "sample_relations.csv"))))
    bad = 0
    for r in rel:
        src = "schema.org" if r["property_source"] == "schema.org" else "minted"
        j = (sc if src == "schema.org" else mi).get(("property", r["property"]))
        if j and j["issue_code"] not in ("NONE", "NOT_CANONICALISED"): bad += 1
    print(f"  {m:20s} property carries any defect {fmt(bad,len(rel))}")
