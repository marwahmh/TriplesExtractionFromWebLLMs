"""Carry judgments across a redraw.

An element judged in an earlier draw keeps its verdict, because the verdict is about
the term. Two lists are printed for attention: elements new to this draw, and carried
elements whose verdict was not NONE and whose usages have changed, since those are the
verdicts that could have turned on a usage the redraw dropped.
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import OUT, MODELS
J = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judged")

def load(m, sheet):
    d = {}
    for l in open(os.path.join(J, f"{m}.{sheet}.tsv")):
        if not l.strip(): continue
        t = l.rstrip("\n").split("\t")
        d.setdefault((t[0], t[1]), t)
    return d

new_rows, recheck = [], []
for m in MODELS:
    for sheet in ("schemaorg", "minted"):
        old = load(m, sheet)
        p = os.path.join(OUT, m, f"judgments_{sheet}.csv")
        rows = list(csv.DictReader(open(p)))
        out = []
        for r in rows:
            k = (r["element_kind"], r["element"])
            t = old.get(k)
            if not t:
                new_rows.append((m, sheet, r)); out.append(None); continue
            out.append(t)
            if len(t) > 5 and t[5] not in ("", "NONE"):
                recheck.append((m, sheet, r, t))
        # stash the carried verdicts next to the sheet for merge.py to pick up
        with open(os.path.join(J, f"{m}.{sheet}.carried.tsv"), "w") as fh:
            for r, t in zip(rows, out):
                fh.write("\t".join(t) if t else f"{r['element_kind']}\t{r['element']}\t\t\t\t\t")
                fh.write("\n")

print(f"NEW elements to judge: {len(new_rows)}")
print(f"CARRIED with a defect code, worth re-checking against the new usages: {len(recheck)}")
if __name__ == "__main__" and "--list" in sys.argv:
    which = sys.argv[sys.argv.index("--list") + 1]
    src = new_rows if which == "new" else recheck
    for item in src:
        m, sheet, r = item[0], item[1], item[2]
        if sheet == "schemaorg":
            print(f'{m}|{sheet}|{r["element_kind"]}|{r["element"]}|n={r["n_occurrences"]}|DEF:{r["schema_definition"][:95]}|DR:{r["schema_domain"][:45]}>{r["schema_range"][:30]}|EX:{r["example_usages"][:130]}')
        else:
            print(f'{m}|{sheet}|{r["element_kind"]}|{r["element"]}|{r["source"][:4]}|n={r["n_occurrences"]}|D:{r["vocab_definition"][:85]}|P:{r["vocab_parent"][:22]}|C:{r["schema_org_candidates"][:50]}|EX:{r["example_usages"][:110]}')
