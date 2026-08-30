# MINE judge audit: does `llama33-70b` judge a fact the way a reader would?

Closes the gap named in Chapter 5, Section "Threats" (a 50-item MINE validation
sample was drawn but never labelled). This directory holds a fresh, larger sample
from the headline MINE run and an independent verdict for every item.

## What was sampled

- Run: `main_test_r1`, judge `llama33-70b`, the 80 test-split articles, the run
  behind Table `tab:mine`.
- 250 items: 62 or 63 per backend (`gemini-2.5-flash`, `deepseek-v4-flash`,
  `llama33-70b`, `qwen3-32b-think`), each backend split ~half `found` / half not
  found by the judge, drawn from both `e2e_consolidated` and `e2e_grounded`
  (133 / 117). Drawn in two batches with the same design: 100 items
  (`random.seed(0)`, indices 0-99) and 150 more (`random.seed(1)`, indices
  100-249, excluding rows already drawn). The two batches agree with the judge at
  the same rate (88.0% and 87.3%), so they are pooled.
- Source rows: `results/<model>/art*/e2e/mine_judgments_e2e_*.json` with
  `judge == "llama33-70b"` and `article_id` in `dataset/splits.json["test"]`.

## Who judged, and from what

Every item was labelled by hand and read in full: for rows the judge marked `found`, the support lines the judge cited (no
context is stored for hits, `STORE_CONTEXT = "misses"`); for `partial` / `absent`
rows, the complete stored retrieval context (up to 150 lines). The verdict scale is
the judge's own: `found` / `partial` / `absent`. Rule applied: a missing quantity,
date, party, or qualifier that changes the claim makes a fact `partial`; a missing
descriptor ("Italian", "nationalist", "near Manchester", "fourth") does not.

The annotator never saw the judge's verdict while labelling. The key
(`mine_judge_key_*.json`) was joined only at scoring time.

## Result

| | value |
|---|---|
| Binary agreement (found vs not) | **219 / 250 = 87.6%** |
| Cohen's kappa (binary) | 0.75 |
| Three-way agreement (found / partial / absent) | 196 / 250 = 78.4% |
| Judge too lenient (judge found, annotator not) | 13 |
| Judge too strict (judge not found, annotator found) | 18 |
| Precision of the judge's `found` label | 109 / 122 = 0.893 |
| Precision of the judge's not-found label | 110 / 128 = 0.859 |

Per backend (agreement; lenient; strict): gemini 51/63 = 81.0% (1; 11),
deepseek 56/62 = 90.3% (2; 4), llama 57/63 = 90.5% (5; 1), qwen 55/62 = 88.7%
(5; 2). Per variant: consolidated 89.5% (9 lenient, 5 strict), grounded 85.5%
(4 lenient, 13 strict).

Reading of the errors:

- The 13 lenient calls are all **qualifier or content drops**, never fabricated
  facts: "world's busiest" without "by international passenger traffic"; "expected
  to hold rates" for "held"; "Trump warns Macron" without the tariff; the reason
  Motsepe gave without his conclusion; a death without the age. Ten of the 13 fall
  on the open-weight backends (llama 5, qwen 5).
- **11 of the 18 strict calls are on gemini graphs**, and the strict errors
  cluster on long contexts (median 105 lines; 5 of 18 at the 150-line cap). In
  every one of them the supporting edges are present in the stored context
  (`Bernie Sanders sponsor Abdul El-Sayed`, the full Louvre item list, `UK event
  2026 Solar Eclipse ... 2026-08-12`, the GEMA demonstration with both songs) and
  the judge either did not use them or read a hub label without its edges. This is
  the same effect Table `tab:mine` shows as gemini's ceiling falling below its
  recall: the judge loses facts in long contexts. On gemini, 11 of the 32 rows the
  judge marked not-found were found on hand labelling, so gemini's reported 0.823 is
  an underestimate; the open-weight recalls carry the opposite, smaller, bias.
- The net effect over the sample is small: the judge marks 122 facts found, the
  annotator 127.

## Files

- `mine_judge_sample_llama33-70b_main_test_r1.json`: the 250 items with `human`
  (y/n), `human_verdict`, and `note` filled in; `support`, `context`, `top_nodes`
  kept so the call can be re-checked.
- `mine_judge_key_llama33-70b_main_test_r1.json`: the judge's verdict per index.
- `mine_judge_agreement_llama33-70b_main_test_r1.json`: the numbers above plus the
  judge-vs-annotator confusion matrix and the indices of every disagreement.
- `mine_judge_audit_llama33-70b_main_test_r1.csv`: flat one-row-per-item view.

## Caveats

- One annotator; a second pass over the 31 disagreements would settle the
  borderline ones (indices 17, 53, 96, 111, 160, 207, 228).
- Hits were judged from the cited support only. A judge that cites the right lines
  for the wrong reason would not be caught; a judge that cites nothing relevant
  would be, and none did.
