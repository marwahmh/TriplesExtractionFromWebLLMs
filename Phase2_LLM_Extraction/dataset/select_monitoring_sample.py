"""Pick (and re-resolve) the 10-article monitoring sample.

The sample is one article per DOMAIN — all ten domains in articles.csv — chosen so that no
outlet appears more than twice and every article is long enough to yield a usable graph.
Selection is an exact optimisation, not a hand-pick: a DP over (domain, source) assignments
maximising  sum(min(chars, LENGTH_CAP)) + DIVERSITY_BONUS per *distinct* outlet, subject to
MIN_CHARS, MAX_PER_SOURCE and EXCLUDE_SOURCES.  Re-running reproduces the same 10.

Written to  dataset/monitoring_sample.json, keyed by URL — NOT by artN, which is only the row
position in articles.csv and silently renumbers when the CSV is edited.  `resolve()` re-derives
the current artN for each URL from the `URL:` header of articles/art*.txt and shouts if a pinned
article has vanished.

    python dataset/select_monitoring_sample.py            # re-select + write
    python dataset/select_monitoring_sample.py --check    # only verify the existing manifest
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "dataset" / "articles.csv"
ARTICLES = ROOT / "dataset" / "articles"
MANIFEST = ROOT / "dataset" / "monitoring_sample.json"

# --- selection knobs ---------------------------------------------------------------------
MIN_CHARS = 2400          # below this an article yields too few assertions to monitor
LENGTH_CAP = 6000         # extra length past this is not worth trading diversity for
MAX_PER_SOURCE = 2
DIVERSITY_BONUS = 3000    # value of bringing in an outlet not yet used
EXCLUDE_SOURCES = {"Yahoo! News"}
PIN = {"Climate": "https://www.bbc.com/news/articles/c3ry307rxqro"}   # art1: the walkthrough
# ------------------------------------------------------------------------------------------


def load_rows():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    for i, r in enumerate(rows, 1):
        p = ARTICLES / f"art{i}.txt"
        r["art_id"] = f"art{i}"
        r["chars"] = len(p.read_text(encoding="utf-8")) if p.exists() else 0
    return [r for r in rows if r["source"] not in EXCLUDE_SOURCES]


def select(rows):
    domains = sorted({r["domain"] for r in rows})
    sources = sorted({r["source"] for r in rows})
    si = {s: i for i, s in enumerate(sources)}

    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["chars"] >= MIN_CHARS:
            cells.setdefault((r["domain"], r["source"]), []).append(r)
    for v in cells.values():                      # longest first within a cell
        v.sort(key=lambda r: -r["chars"])
    for domain, url in PIN.items():               # a pinned URL is the only option for its domain
        r = next(x for x in rows if x["url"] == url)
        cells[(domain, r["source"])] = [r]
        for k in [k for k in cells if k[0] == domain and k[1] != r["source"]]:
            del cells[k]

    NEG = -(10 ** 9)

    @functools.lru_cache(maxsize=None)
    def best_from(d: int, counts: tuple[int, ...]):
        if d == len(domains):
            return 0, ()
        best = (NEG, ())
        for s in sources:
            used = counts[si[s]]
            cand = cells.get((domains[d], s))
            if not cand or used >= MAX_PER_SOURCE or len(cand) <= used:
                continue
            r = cand[used]                        # 2nd pick from an outlet = its 2nd longest here
            nxt = list(counts)
            nxt[si[s]] += 1
            score, path = best_from(d + 1, tuple(nxt))
            if score == NEG:
                continue
            total = score + min(r["chars"], LENGTH_CAP) + (DIVERSITY_BONUS if used == 0 else 0)
            if total > best[0]:
                best = (total, (r,) + path)
        return best

    score, picked = best_from(0, tuple([0] * len(sources)))
    if score == NEG:
        raise SystemExit("no assignment satisfies the constraints — relax MIN_CHARS or EXCLUDE_SOURCES")
    return list(picked)


def sha(art_id: str) -> str:
    return hashlib.sha256((ARTICLES / f"{art_id}.txt").read_bytes()).hexdigest()[:16]


def write_manifest(picked):
    payload = {
        "purpose": "10-article monitoring sample: watched through the whole pipeline, and the "
                   "articles the 200 manually-checked grounding facts are drawn from.",
        "selection": {"min_chars": MIN_CHARS, "length_cap": LENGTH_CAP,
                      "max_per_source": MAX_PER_SOURCE, "diversity_bonus": DIVERSITY_BONUS,
                      "exclude_sources": sorted(EXCLUDE_SOURCES), "pinned": PIN},
        "articles": [{"domain": r["domain"], "source": r["source"], "art_id": r["art_id"],
                      "title": r["title"], "url": r["url"], "date": r.get("Date", ""),
                      "chars": r["chars"], "sha256_16": sha(r["art_id"])}
                     for r in sorted(picked, key=lambda r: r["domain"])],
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _url_of(art_id: str) -> str | None:
    p = ARTICLES / f"{art_id}.txt"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines()[:5]:
        if line.startswith("URL:"):
            return line[4:].strip()
    return None


def resolve(manifest_path: Path = MANIFEST, strict: bool = True) -> list[str]:
    """Current artN ids for the pinned URLs. Use this instead of hard-coding the ids anywhere —
    it survives an articles.csv edit, and fails loudly when an article is gone."""
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_url = {}
    for p in ARTICLES.glob("art*.txt"):
        u = _url_of(p.stem)
        if u:
            by_url[u] = p.stem
    ids, moved, lost = [], [], []
    for a in man["articles"]:
        cur = by_url.get(a["url"])
        if cur is None:
            lost.append(a["title"])
            continue
        if cur != a["art_id"]:
            moved.append(f'{a["art_id"]} -> {cur}  {a["title"][:50]}')
        ids.append(cur)
    if moved:
        print("RENUMBERED since the manifest was written:")
        for m in moved:
            print("  ", m)
        print("   results/<model>/artN/ still points at the OLD text — re-check before reusing them.")
    if lost:
        msg = "articles missing from dataset/articles/: " + "; ".join(lost)
        if strict:
            raise SystemExit(msg)
        print("WARNING:", msg)
    return ids


def main():
    if "--check" in sys.argv:
        ids = resolve()
        print("monitoring sample resolves to:", " ".join(ids))
        return
    picked = select(load_rows())
    payload = write_manifest(picked)
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    for a in payload["articles"]:
        print(f'  {a["domain"]:22} {a["source"]:18} {a["art_id"]:6} {a["chars"]:6}  {a["title"][:58]}')
    srcs = {a["source"] for a in payload["articles"]}
    print(f'  -> {len(payload["articles"])} articles · {len(srcs)} outlets · '
          f'{sum(a["chars"] for a in payload["articles"]):,} chars')


if __name__ == "__main__":
    main()
