#!/usr/bin/env python
"""Does an open-weight verbaliser produce hypotheses the entailment model can actually judge?

Runs the gate before committing the whole pipeline to a verbalisation scheme:
  1. rewrite each edge as a sentence with a small open instruct model (greedy),
  2. build object-swapped corruptions of the same edges,
  3. score both against the article and report the separation AUC.

The mechanical `subject property object` template fails this at AUC ~0.70 on art1 — the encoder
scores a corrupted triple as high as the true one. Anything we ship must beat it clearly.

    python verbalise_check.py --articles art1 art5 --n 60
"""
import argparse
import random
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--results-model", default="gemini-2.5-flash")
    ap.add_argument("--track", default="e2e")
    ap.add_argument("--articles", nargs="*", default=["art1", "art5"])
    ap.add_argument("--verbaliser", default=None)
    ap.add_argument("--n", type=int, default=60, help="relations sampled per article")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-auc", type=float, default=0.75)
    args = ap.parse_args()

    sys.path.insert(0, str(args.root / "notebooks"))
    import nli_core

    rd = args.root / "results" / args.results_model
    vname = args.verbaliser or nli_core.DEFAULT_VERBALISER

    print(f"loading verbaliser {vname} ...", flush=True)
    t0 = time.perf_counter()
    verb = nli_core.Verbaliser(vname, batch_size=args.batch_size)
    print(f"  on {verb.device} in {time.perf_counter()-t0:.1f}s", flush=True)

    print("loading verifier ...", flush=True)
    ver = nli_core.Verifier(batch_size=args.batch_size)
    ver.probe(verbose=False)
    print(f"  on {ver.device}, probe AUC {ver.probe_auc:.3f}", flush=True)

    overall = {}
    for aid in args.articles:
        article = (args.root / "dataset/articles" / f"{aid}.txt").read_text(encoding="utf-8")
        rows = [r for r in nli_core.load_claims(rd, aid, args.track)
                if r["kind"] == "relation" and r.get("subject") and r.get("object")]
        # sample at RANDOM: the logs are in graph order, so the first N over-represent whatever
        # property the extractor emitted first (on art1 that is 21/60 `about` edges, the least
        # verifiable kind, which would make the gate look worse than the corpus really is)
        random.Random(0).shuffle(rows)
        rows = rows[:args.n]
        if not rows:
            print(f"{aid}: no relations found"); continue
        corrupt = nli_core.corrupt_rows(rows, seed=0)

        print(f"\n===== {aid}: {len(rows)} relations =====", flush=True)
        t0 = time.perf_counter()
        hyp = verb.verbalise(rows)
        hyp_c = verb.verbalise(corrupt)
        print(f"  verbalised {2*len(rows)} in {time.perf_counter()-t0:.0f}s", flush=True)

        print("\n  --- LLM-free template baseline (what fails today) ---")
        try:
            base = nli_core.discrimination_check(
                ver, article, rows,
                [nli_core.template_hypothesis(r) for r in rows],
                [nli_core.template_hypothesis(r) for r in corrupt],
                min_auc=0.0)
        except AssertionError as e:
            print("   ", e); base = {"auc": float("nan")}

        print("\n  --- verbalised by " + vname + " ---")
        try:
            got = nli_core.discrimination_check(ver, article, rows, hyp, hyp_c,
                                                min_auc=args.min_auc)
            ok = True
        except AssertionError as e:
            print("    GATE FAILED:", e); got = {"auc": float("nan")}; ok = False
        overall[aid] = (base["auc"], got["auc"], ok)

        # how often did the verbaliser just echo the relation name instead of writing a verb?
        echo = sum(1 for r, h in zip(rows, hyp)
                   if nli_core.spaced_property(r.get("property")) in h.lower())
        print(f"\n  echoed the raw relation name: {echo}/{len(rows)} ({echo/len(rows):.0%})")
        print("\n  examples (true vs object-swapped):")
        for i in range(min(4, len(rows))):
            print(f"    + {hyp[i][:92]}")
            print(f"    - {hyp_c[i][:92]}")

    print("\n" + "=" * 70)
    print(f"{'article':10} {'template AUC':>14} {'verbalised AUC':>16} {'gate':>6}")
    for aid, (b, g, ok) in overall.items():
        print(f"{aid:10} {b:>14.3f} {g:>16.3f} {'PASS' if ok else 'FAIL':>6}")


if __name__ == "__main__":
    main()
