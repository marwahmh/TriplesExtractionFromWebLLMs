# Final week — what runs, what gets written, in what order

Submission ~2026-08-30. Written 2026-08-23. Scope decision: **test set on gemini-2.5-flash,
llama33-70b, qwen3-32b-think only.** Everything below is sequenced so that the writing is never
blocked waiting on a run.

---

## 0. Where the runs actually stand

| model | test run (80 articles) | state |
|---|---|---|
| **llama33-70b** | 6c ✅ 6d ✅ 6e ✅ | **complete.** 6d's metrics cell had never executed, so the run looked unfinished; reconstructed offline 2026-08-23 (no LLM calls). 312 → 138 types, 493 → 470 properties, 68.2% type reuse, 42.2% → 44.6% triple schema.org share. |
| **qwen3-32b-think** | none | needs the full chain on the cluster. ~8–11 h GPU (val took 171 min for 20 articles). **Free.** |
| **gemini-2.5-flash** | 6c on 6/80 articles | started 2026-08-23, stopped at article 6 pending an API top-up. The 6 graphs are valid and 6c resumes past them. **The only item on this plan that costs money.** |

Validation is complete for all three models on both tracks, plus deepseek-v4-flash and
gpt-oss-120b as extra e2e points.

### The one budget decision

Gemini's val chain spent ~4.0M tokens on 20 articles (1.58M extraction, 0.19M types, 2.22M
relations). Test is 4× the articles, and the relation stage grows worse than linearly because the
consolidation prompt scales with the vocabulary. **Estimate 14–18M tokens, mostly input**, with the
relation stage the bulk of it. Price it against current Flash rates before topping up.

**Everything else is free.** The grounding judge (nb9b) and the MINE benchmark (nb10) both run on
open-weight backends over vLLM — the judge benchmark below says qwen3-32b-think is the better judge
anyway. So the API spend buys exactly one thing: gemini's own *pipeline* run. Its *evaluation*
costs nothing.

If the top-up doesn't happen, the fallback is that gemini is reported on validation only and the
test-set comparison is llama vs qwen — stated as a scope limit, not hidden. It is a weaker thesis,
because gemini is the default backend of Chapter 4, but it is a coherent one.

---

## 1. Day by day

**Sun 23 (today)** — cluster up. Launch qwen test overnight; it is the long pole and it is free.

```bash
cluster/labia-tunnel.sh                     # then confirm :8001 answers
MODEL=qwen3-32b-think SPLIT=test REPEAT=1 EXPERIMENT=main ./run_e2e.sh
```

Decide the gemini top-up. If yes, resume it in parallel — different backend, no contention:

```bash
MODEL=gemini-2.5-flash SPLIT=test REPEAT=1 EXPERIMENT=main SKIP_SERVER_CHECK=1 ./run_e2e.sh
```

Writing: Chapter 5 §§1–4 (objectives, corpus, instruments, setup). Every number these need already
exists.

**Mon 24** — qwen test lands. Run MINE (nb10) on the llama and qwen test graphs — cluster, free.
Writing: Chapter 5 results for extraction/review, type canonicalisation, relation canonicalisation.

**Tue 25** — grounding (nb9b) on llama + qwen test graphs. Gemini test finishes if it was launched.
Writing: grounding and MINE result sections.

**Wed 26** — evaluate the gemini test graphs (MINE + grounding, both judged on the cluster, free).
Writing: finish Chapter 5 — cross-model comparison, cost, threats to validity.

**Thu 27** — Chapter 6 (conclusion, limitations, perspectives) and the appendix. Reconcile the
promises Chapter 4 makes that this week cannot keep (§3 below).

**Fri 28** — figures (§2), full compile, cross-references, and the per-chapter intro/outro
paragraphs the supervisor's general remarks require.

**Sat 29** — supervisor read, buffer.

The order matters in one specific way: the free cluster work is front-loaded, so a gemini
top-up that arrives late still lands inside the schedule.

---

## 2. Figures — 9 placeholders, none of them run-dependent

`\phbox` grey boxes still stand at Chapter1:133, Chapter1:180, Chapter2:61, Chapter3:218,
Chapter3:289, Chapter4:96, Chapter4:236, Chapter4:328, Chapter4:379.

`Chapter4:96` is `fig:pipeline`, which Chapter 4's own comment calls the single most important
picture in the report; the comment says a polished version exists on the "Open extraction, then
canonicalize, validate, ground" slide of `~/Documents/Thesis/Lab_Seminar_M2.pptx` and needs
exporting as `pipeline.pdf`. Chapter 5 also needs its own pipeline-to-components figure per the
supervisor's stub.

These are independent of every run and can be done in any gap. They are the most likely thing to be
left undone, so do not leave them to Friday alone.

---

## 3. Two promises Chapter 4 makes that this week cannot keep

Both need an honest edit rather than a silent omission.

**Repeats.** `sec:m_protocol` states each claimed arm is run three times and reported as a mean with
its range. No `_r2`/`_r3` run exists for any model or arm. The evidence that repeats matter is real
and already documented (69 vs 91 raw minted properties under identical settings, 45% of individual
property decisions identical), so the honest form is: state the protocol as designed, report that
single runs were made under deadline, and carry that documented spread as the uncertainty on every
vocabulary number. **The one cheap exception worth buying:** 3 repeats of `6d 6e` on *validation*
for one model is a few hours on the cluster and turns "we know runs vary" into a measured range.
Do it for qwen if Tuesday has slack.

**The ablation grid.** `tab:ablations` promises ten arms, each run on validation and repeated three
times, "for each model" — thirty-plus chains. All ten config files exist
(`no-review.json`, `no-rerank.json`, `inline-vocab.json`, `no-consolidation.json`,
`one-call-consolidation.json`, `no-generalize.json`, `no-guard-schema-parent.json`,
`consolidate-at-end.json`, `concise-prompt.json`, `no-rerank-cons.json`); none has been run.
Rewrite the table's caption as the designed grid, and report what the main runs already measure
without any ablation — the review pass, the reranker and consolidation all write their own effect
numbers into `metrics_*.json` (REPORTING_PLAN §5). That covers three of the ten questions from the
reported runs themselves. The other seven move to Chapter 6 as further work.

---

## 4. What the numbers already say

Enough of Chapter 5 is already determined that the writing is not a gamble on the remaining runs.

**The judge benchmark is a genuine result, and it is finished.** 500 items (250 clean, 225
corrupted across six operators), human-reviewed key, three repeats per judge:

| judge | recall | FPR | precision | accuracy |
|---|---|---|---|---|
| qwen3-32b-think (current prompt) | 84.6 / 86.6 / 85.8 | 2.2–3.5% | 96.4–97.7% | 90.7–92.0% |
| llama33-70b (current prompt) | 81.4 ×3 (identical) | 3.5% | 96.2% | 88.6% |
| qwen3-32b-think (previous prompt) | 76.5 / 79.4 / 77.3 | 1.3–3.1% | 96.4–98.5% | 86.3–88.4% |

Three things fall out of this table: qwen is the judge to use; the prompt revision is worth +8.0
recall points, which is a measured prompt-engineering result; and llama is bit-identical across
three runs while qwen is not, which is a concrete data point on greedy-decoding determinism.
Per-operator, `direction_flip` is the hard case (~50% caught by both judges) and `property_swap`
separates them (36–39/45 qwen vs 26/45 llama); `object_swap`, `subject_swap` and `number_edit` are
caught at ≥96%. That per-operator profile is what lets Chapter 5 bound its own grounding numbers.

**The judge dominates the MINE signal, and that has to be said out loud.** The *same* gemini
validation graphs score 86.0% e2e under the llama judge and 62.7% under the qwen judge. Any MINE
number is a statement about a (graph, judge) pair. Fix one judge for every test-set MINE run —
qwen3-32b-think, on the benchmark evidence above — and report the val cross-judge gap as the
measurement's own error bar.

**Watch for the self-judging bias.** qwen's validation grounding summaries are flagged
`self_judged: true`. Either re-judge qwen's graphs with llama, or state the bias where the number
appears.

**Consolidation behaves very differently per model**, which is the most interesting cross-model
finding available: type vocabulary reduction is 3.6% for gemini (448→432) against 27.5% for qwen
(80→58) and 55.8% for llama on the test split (312→138). Gemini mints far more (36.5% mint rate,
776 distinct type labels on 20 articles) and consolidates far less. This is a real result about
model behaviour under the same pipeline, not noise.

**The review pass separates the models sharply**: +98.0% relations for llama, +8.2% for gemini,
+1.1% for qwen. The second call is doing very different work depending on how complete the first
one was.

---

## 5. Gotcha to not get caught by

`metrics_extraction_dedup.json` is tagged by **variant alone**, not by run tag, because 6c writes it
before a run tag exists. A test-set 6c therefore overwrites the validation extraction numbers for
that model. llama's file already holds the 80-article test numbers; its validation extraction
numbers are gone. Gemini's and qwen's validation extraction metrics were snapshotted to the
scratchpad on 2026-08-23 before their test runs — copy them somewhere permanent before writing them
up. `run_metrics.save` guards against a *subset* replacing a fuller run, which is the opposite
direction and does not help here.
