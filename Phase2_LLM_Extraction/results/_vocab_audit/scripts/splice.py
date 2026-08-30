"""Assemble this draw's judged TSVs: carried verdicts where the element was judged in
an earlier draw, newly written verdicts otherwise. Fails loudly on any gap."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import OUT, MODELS
HERE = os.path.dirname(os.path.abspath(__file__))
J, NEW = os.path.join(HERE, "judged"), os.path.join(HERE, "judged_new")

def load_new(sheet):
    d = {}
    for l in open(os.path.join(NEW, f"{sheet}.tsv")):
        if not l.strip(): continue
        t = l.rstrip("\n").split("\t")
        d[(t[0], t[1], t[2])] = t[3:]          # model, kind, element -> verdict cols
    return d

gaps = 0
for sheet in ("schemaorg", "minted"):
    new = load_new(sheet)
    for m in MODELS:
        rows = list(csv.DictReader(open(os.path.join(OUT, m, f"judgments_{sheet}.csv"))))
        carried = [l.rstrip("\n").split("\t") for l in open(os.path.join(J, f"{m}.{sheet}.carried.tsv"))]
        out = []
        for r, c in zip(rows, carried):
            assert c[1] == r["element"], (m, sheet, c[1], r["element"])
            verdict = c[2:]
            if not any(x.strip() for x in verdict):          # blank -> must be a new element
                v = new.get((m, r["element_kind"], r["element"]))
                if v is None:
                    print(f"  GAP {m} {sheet} {r['element_kind']} {r['element']}"); gaps += 1
                    v = ["", "", "", "", ""]
                verdict = v
            out.append([r["element_kind"], r["element"]] + list(verdict))
        with open(os.path.join(J, f"{m}.{sheet}.tsv"), "w") as fh:
            for o in out:
                fh.write("\t".join(x.rstrip("\n") for x in o) + "\n")
print("gaps:", gaps)
