# Reporting plan — what to run, and which number answers which question

The runs whose numbers go into the write-up, and the measurement each component's claim rests on.
Everything here is executable today: the configs exist, the metrics are computed by
`notebooks/run_metrics.py`, and every run records itself in `results/<model>/_runs/<run_id>.json`.

---

## 1. The corpus split

`dataset/splits.json`, produced by `python dataset/select_splits.py` (deterministic, keyed by URL):

| split | n | what it is |
|---|---|---|
| **validation** | 20 | 13 pinned + 7 filled to 2 per domain (Culture has only 2 articles, so 1). Every component claim is measured here. |
| **test** | 59 | everything else. Disjoint. The reported end-to-end numbers. |

The 13 pins are the articles whose side was never in question:

- the **10 monitoring articles** — they carry the 200 manually-checked grounding facts, so the
  human-verified anchor sits where the components are judged;
- the **4 development articles** (art1, art2, art4, art5) — every prompt, threshold and retrieval
  setting was chosen while looking at their output. A test score computed on them would be scoring a
  corpus the method was fitted to. art1 is in both sets, hence 13 rather than 14.

Pins **consume** the 20 budget rather than adding to it; the remaining 7 slots go to the article
closest to the **corpus median length** in each domain that still needs one.

**Length.** The sampled slots deliberately span the corpus's range rather than skimming the longest
article off each domain — length drives graph size, fragmentation and how much there is to
canonicalize at all, so a uniformly long validation set would measure the components on a corpus the
test set does not contain. The result tracks the corpus closely:

| | n | min | median | mean | max | <2,400 chars |
|---|---|---|---|---|---|---|
| corpus | 79 | 1,088 | 3,842 | 4,364 | 11,252 | 14 |
| validation | 20 | 1,261 | 3,918 | 4,592 | 11,252 | 2 |
| test | 59 | 1,088 | 3,363 | 4,287 | 10,549 | 12 |

**Two known skews, both stated rather than engineered away.** Validation is 45% BBC against a corpus
that is 33% — the four development articles are all BBC, which is a fact about how the pipeline was
built, not a sampling choice. Tightening the per-outlet cap to fix it drops Environment to a single
validation article, and domain coverage is worth more than outlet share for measuring per-component
effects. Every domain keeps at least one test article.

```bash
python dataset/select_splits.py --check     # disjoint · 13/13 pins in validation · no leakage
```

```bash
SPLIT=val  ./run_e2e.sh          # tag -> main_val
SPLIT=test ./run_e2e.sh          # tag -> main_test
```

The split is part of the tag, so a validation number can never land on top of a test number.

---

## 2. The run matrix

**e2e on both splits; staged on validation only.** The staged track's job in this thesis is to
establish that the single call does not cost anything the decomposition was buying — once validation
shows that, the test runs are e2e alone. That is a validation set being used for exactly what it is
for, but it only holds if the rule is fixed **before** the numbers exist (§3), and it costs the
thesis the right to say anything about the staged track on the test set (§3, *what this gives up*).

```bash
# ── e2e (6c -> 6d -> 6e): both splits, every model ────────────────────────────
for m in gemini-2.5-flash qwen3-32b-think llama33-70b; do
  MODEL=$m SPLIT=val  REPEAT=1 EXPERIMENT=main ./run_e2e.sh          # full chain -> main_val_r1
  MODEL=$m SPLIT=val  REPEAT=2 EXPERIMENT=main ./run_e2e.sh 6d 6e    #            -> main_val_r2
  MODEL=$m SPLIT=val  REPEAT=3 EXPERIMENT=main ./run_e2e.sh 6d 6e    #            -> main_val_r3
  MODEL=$m SPLIT=test REPEAT=1 EXPERIMENT=main ./run_e2e.sh          # reported   -> main_test_r1
done

# ── staged, n-ary chain (the like-for-like one): VALIDATION ONLY ──────────────
for m in gemini-2.5-flash qwen3-32b-think llama33-70b; do
  MODEL=$m SPLIT=val REPEAT=1 EXPERIMENT=main ./run_staged.sh 2 4 6b 7b
  MODEL=$m SPLIT=val REPEAT=2 EXPERIMENT=main ./run_staged.sh 4 7b
  MODEL=$m SPLIT=val REPEAT=3 EXPERIMENT=main ./run_staged.sh 4 7b
done
```

**Give even the first run its `REPEAT=1`.** An unnumbered `main_val` and a numbered `main_val_r2`
are the same arm under two naming schemes, and `run_metrics tracks` (§3) looks for `_r1/_r2/_r3`.
The test run is `r1` for the same reason — it leaves room for repeats without renaming the run the
thesis already cites.

The repeats re-run only the canonicalization stages — they tag their own outputs and reuse the
extraction, which is what makes them cheap and what the `run_manifest` guard enforces.

**The staged track has two relation halves and they answer different questions.** `2 4 6b 7b` is the
reified n-ary chain — the like-for-like comparison against e2e, and the only one this decision needs.
`2 4 6 7` is the *binary* chain, which is the "reification vs. plain triples" question. If the thesis
makes that claim too it is a separate validation-only arm, not part of this comparison.

**6c only runs once per model.** Its output is named by `variant` alone
(`nary_graph_dedup.json`), so the val and test runs extract disjoint articles into their own
folders and together cover the corpus exactly once. Every ablation below except `no-review` reuses
those graphs and needs only `./run_e2e.sh 6d 6e` — which is most of the budget saved.
A second *extraction* sample is a new `variant`, not a `REPEAT` (the guard in
`run_manifest.py` enforces this and says so).

---

## 3. The staged-vs-e2e decision, fixed in advance

Write this down before the runs finish; a threshold chosen after seeing the gap is not a threshold.

**The rule.** On the validation split, across the three models, e2e continues alone to test iff:

1. **Correctness is not worse.** Grounding precision (nb9/9b) and MINE recall (nb10) for e2e are
   within the repeat spread of, or above, the staged track's. This is the binding criterion — the
   others are economy, and no amount of economy buys back a less faithful graph.
2. **Shape is not worse.** `mean_largest_component_share` and `isolated_nodes` do not favour staged
   outside the spread. (Fragmentation is the thing the decomposition was supposed to cost.)
3. **Economy is not worse.** `vocab_consolidated` and `triple_schema_org_pct_consolidated` likewise.

"Within the spread" is mechanical: two tracks are indistinguishable on a key when the ranges of
their three repeats overlap. This corpus has produced 69 vs 91 raw mints under identical settings,
so a gap narrower than that range is not a finding.

```bash
cd notebooks
python -m run_manifest list                              # confirm every arm actually finished
python -m run_metrics tracks gemini-2.5-flash main_val   # the table, repeat-aware
python -m run_metrics tracks qwen3-32b-think  main_val
python -m run_metrics tracks llama33-70b      main_val
```

It prints each key as `e2e BETTER` / `e2e worse` / `~within spread`, and leaves raw graph size
unscored on purpose: a track emitting more relations may be finding more facts or splitting the same
facts more finely, and only grounding and MINE can tell those apart.

**If the staged track wins.** Then the plan changes rather than the conclusion — you cannot drop the
better track because it is more expensive. Either run staged on test as well, or scope the thesis to
the staged track and use e2e as the baseline it was built to be. Deciding this now is what stops the
validation comparison from being a formality.

**What this gives up.** Every staged-track number in the thesis is then a *20-article validation*
number and has to say so. Nothing about the staged track generalises to the 59 test articles, and
"e2e ≈ staged" is a claim about validation only. That is an acceptable, explicit limitation — it
stops being acceptable if it is left unstated.

## 4. Repeats — this is not optional

Temperature 0 is greedy, not deterministic. Two runs of the same configuration on the same corpus
gave **69 vs 91** raw minted properties (65 vs 67 after consolidation), and only 45% of the
individual property decisions were identical (`notebooks/e2e/CURRENT_CONFIG.txt` §7). A single run
cannot distinguish two prompt arms, and several of the ablations below are expected to move things
by less than that.

```bash
REPEAT=1 SPLIT=val EXPERIMENT=main ./run_e2e.sh 6d 6e     # tag main_val_r1
REPEAT=2 SPLIT=val EXPERIMENT=main ./run_e2e.sh 6d 6e     # main_val_r2
REPEAT=3 SPLIT=val EXPERIMENT=main ./run_e2e.sh 6d 6e     # main_val_r3
```

Protocol: **3 repeats of the canonicalization chain on validation for every arm whose effect is
being claimed** — including both tracks, since §3's "within spread" test is meaningless without the
spread. Report mean + range, and report **post-consolidation** numbers, which absorb most of the
sampling noise. On test, 3 repeats for the headline arm per model if budget allows; otherwise 1 run,
with the spread carried over from validation and stated as such.

---

## 5. Component effects — which number answers which question

Everything in the "from" column is written by the run itself and lands in
`metrics_<stage><tag>.json` plus the run manifest.

### Already measured, from the reported run itself (no ablation needed)

| Component | The question it was added for | Number | From |
|---|---|---|---|
| **Review pass** (6c/6b, 2nd call) | does a second look recover facts the first call missed? | `review_added_relations`, `review_added_entities`, `review_added_nary_nodes`, `review_relations_delta_pct`, `review_removed_relations`, `review_fixed_evidence`, `review_guard_dropped`, `review_failed_articles` | `review_stats`, now saved into each graph (it used to be printed and lost) → `extraction()` |
| **Inline reranker** (6d/6e) | does the cross-encoder surface schema.org terms e5 buried too deep to show? | `rr_promoted_pct` (share of the shortlist the cross-encoder rewrote), `rr_promoted_schema_org`, **`rr_chosen_promoted`** and **`rr_chosen_promoted_schema`** (promotions the LLM then took), `rr_mean_embed_rank_of_chosen`, `rr_mean_positions_gained` | each candidate already carries `embed_rank` = where e5 had it → `_rerank_effect()` in `types()`/`relations()` |
| **Consolidation** (6d/6e) | does global re-consolidation collapse the duplicates that schema-only inline retrieval creates by design? | `cons_merge_groups`, `cons_removed`, `cons_merged_into_schema`, `cons_reparented`, `cons_reparented_under_schema`, `cons_generalized`, `cons_blocked_degrounding`, `vocab_raw` → `vocab_consolidated`, `cons_in_shortlist_pct` | the consolidation report |
| **Consolidation, on the output** | …and what did that do to the actual triples? | `cons_triples_property_changed(_pct)`, `cons_triples_to_schema_org`, `triple_schema_org_pct` → `triple_schema_org_pct_consolidated` | diff of `*_extended.json` vs `*_extended_consolidated.json` |
| **Minting economy over time** | does minting saturate as the corpus grows, or keep growing? | `cum_minted` per article, in corpus order, in `per_article` | running sum of the mint decisions |
| **Fragmentation** (staged vs e2e) | does the single call produce a more connected graph? | `largest_component_share`, `isolated_nodes`, `relations_per_node` | `extraction()` |
| **Cost** | what does each model/stage cost? | per-stage `token_usage` and `llm_seconds` | the run manifest |

On the archived 4-article exploratory run these already produce, for example: the cross-encoder
rewrote **48.7%** of the shortlist, and of the decisions where the chosen property was in the
shortlist, **24.8% (28 decisions) were promotions the bi-encoder would not have shown — all 28 of
them schema.org**; consolidation changed the property on **50 of 593 triples (8.4%)**, **47** of them
onto schema.org, lifting reuse **65.4% → 73.4%**.

### Needs an ablation run (each isolates one mechanism against `main`)

| Config | Isolates | Read |
|---|---|---|
| `no-review` | the second extraction call | needs its own 6c (variant `noreview`); compare graph size, evidence quality, and the downstream reuse/vocabulary |
| `no-rerank` | both cross-encoder stages | reuse %, vocabulary size |
| `no-rerank-inline` *(new)* | **only** the inline shortlist | splits `no-rerank`, which conflated two stages acting on different pools (~1.5k schema.org terms vs a ~100-entry vocabulary) |
| `no-rerank-cons` *(new)* | **only** the consolidation shortlist | watch `cons_in_shortlist_pct` — a drop means the judge is reaching past what the embedder proposed |
| `inline-vocab` | inline reuse vs deferred global consolidation | raw mints vs post-consolidation vocabulary |
| `no-consolidation` | the global pass itself | the upper bound on vocabulary size |
| `one-call-consolidation` | merge-then-reparent as two passes | ratio of merges to re-parents: is "subproperty" an escape hatch from merging? |
| `no-generalize` | minting one general property to collapse specifics | `cons_generalize_groups`, final vocabulary size |
| `no-guard-schema-parent` *(new)* | the re-parenting guard | `cons_blocked_degrounding` counts how often it fires in `main`; this shows what those moves would have done |
| `consolidate-at-end` *(new)* | the consolidation SCHEDULE | `main` now re-consolidates every 5 articles and feeds the result back; this runs the single end-of-run pass instead. Compare `vocab_consolidated`, `triple_schema_org_pct_consolidated` and the consolidation prompt size — the periodic schedule exists to keep that prompt inside the context window on the 59-article test split. |
| `concise-prompt` | whether the long prompt's repetition is load-bearing | **needs 3 repeats** — the effect is below single-run noise |

All ten run on all three models via `MODEL=`, no extra config files:

```bash
for m in gemini-2.5-flash qwen3-32b-think llama33-70b; do
  for e in no-rerank no-rerank-inline no-rerank-cons inline-vocab \
           no-consolidation one-call-consolidation no-generalize \
           no-guard-schema-parent concise-prompt; do
    MODEL=$m SPLIT=val EXPERIMENT=$e ./run_e2e.sh 6d 6e
  done
  MODEL=$m SPLIT=val EXPERIMENT=no-review ./run_e2e.sh          # this one re-extracts
done
```

---

## 6. What the plan is still missing

**(a) Nothing above says the graph is *correct*.** Every number is economy — size, reuse, merges. A
run that maps every relation onto `about` scores beautifully. The quality side is nb9 (grounding),
nb9b (full-text grounding), nb9c (NLI) and nb10 (MINE recall), and it has to be in the matrix per
model × split, or the reuse numbers have nothing holding them down.

**(b) Canonicalization correctness is the biggest hole.** `schema_org_reuse_pct` counts *whether* a
schema.org term was chosen, never whether it was the *right* one. A wrong-sense reuse is worse than
an honest mint, and nothing currently distinguishes them. This needs a judged sample — a few hundred
(phrase → chosen property) and (type → chosen class) pairs from the validation run, labelled
correct/wrong-sense/too-general/should-have-minted. Without it, "73% reuse" is a proxy a reviewer can
attack. **This is the measurement I would add before any further ablation.**

**(c) Article order is a confound.** Canonicalization is incremental: what article *k* mints depends
on what *k−1* added, so the vocabulary numbers are partly an artefact of corpus order. One
shuffled-order repeat on validation (same config, `selected_articles` permuted) quantifies it — and
if the effect is small, that is itself a result worth reporting.

**(d) Cross-model vocabulary agreement.** Three models mint three vocabularies over the same 20
articles. How much do they overlap? That answers "is the schema the LLM invents a property of the
domain or of the model?" — cheap to compute from the three `type_vocabulary_*_consolidated.json`
files, and a more interesting finding than three separate reuse percentages.

**(e) Robustness/failure rates differ by model far more than quality does.** JSON parse failures,
`review_failed_articles`, guard drops, empty graphs, and (for the open-weight arms) truncation
against the context limit. Structured mode vs thinking mode will differ sharply here. Some of this is
now captured; make it a reported column, not a footnote.

**(f) Cost comparison needs a stated basis.** Gemini bills tokens, the open-weight arms bill GPU
hours on Lab-IA. The manifest records tokens and seconds for both; say which one the comparison uses
and why, rather than putting them in one column.

**(g) Statistics.** With 3 repeats, arm-level numbers are mean + range (n=3 — do not compute a
t-test on that). The per-decision metrics are different: one validation run yields thousands of
retrieval decisions, so `rr_chosen_promoted_pct` and `chosen_at_rank1_pct` support a real confidence
interval. Use the right one for each.

**(h) No leakage to declare, but say so.** e5 and the cross-encoder are used off-the-shelf with no
training or tuning on this corpus, so the val/test split protects against *human* overfitting only.
One sentence preempts the question.

---

## 7. Budget

Measured on the archived 4-article Gemini run: ≈**225k tokens per article** for a full e2e chain
(6c ≈ 80k, 6d ≈ 36k, 6e ≈ 109k). Per chain: validation (20 articles) ≈ **4.5M tokens**, test (59) ≈
**13M**. Thinking tokens are roughly a third of that and are already included.

With staged on validation only, the `main` matrix is, per model: one full e2e val chain (4.5M) + two
e2e canonicalization repeats (2 × 2.9M) + one e2e test chain (13M) ≈ **23M**, plus the staged val
chain and its two repeats. The staged track has no measured cost yet — it adds nb2's entity call
ahead of reification, so expect it above e2e per article; the manifest will record the real number
on the first run. Running staged on test as well would have added ≈ 3 × 13M+ for a comparison the
validation split already settles.

Ablations are cheaper because they reuse 6c: 6d+6e ≈ 145k tokens/article ≈ **2.9M per validation
chain**, so ten ablations × three models ≈ **87M tokens** — against ≈ 135M if each re-extracted.
`no-review` is the one that must, at ≈ 4.5M more per model.

Check the real number as you go — it is recorded per stage:

```bash
cd notebooks && python -m run_manifest list     # tokens per run
```
