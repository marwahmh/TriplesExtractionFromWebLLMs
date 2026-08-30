# Vocabulary audit: are the chosen terms the right terms?

A hand-verifiable audit of the **naming** decisions in the consolidated validation
graphs. The grounding and MINE instruments ask whether a fact is supported by the
article and whether the expected facts were found. Neither asks whether the
`schema.org` term the pipeline attached to that fact is the right one. This audit
does.

## What was sampled

For each of the four backends, **250 consolidated relations** were drawn from the
**single-call track** on the validation split (the same 20 articles for every model).
The draw is **proportional**: the share of sampled relations carrying a `schema.org`
property reproduces that backend's own share, so a rate measured on the sample is a
rate over that model's graphs and needs no re-weighting. Those shares differ a lot,
which is why the sheets differ in size:

| backend | e2e relations | schema.org | minted | sampled as |
|---|---|---|---|---|
| gemini-2.5-flash | 2,834 | 64% | 36% | 161 / 89 |
| deepseek-v4-flash | 1,506 | 80% | 20% | 201 / 49 |
| llama33-70b | 371 | 50% | 50% | 125 / 125 |
| qwen3-32b-think | 533 | 84% | 16% | 211 / 39 |

For `llama33-70b` the 250 relations are 67% of everything it produced on these
articles, so its figures are closer to a census than a sample. Within a stratum the
draw is round-robin over articles so no article dominates.
Seed `1234`; the sampler is `scripts/sample.py` and is deterministic. `scripts/`
holds the whole chain (`sample.py` -> `sheets.py` -> `domainrange.py` ->
`merge.py` -> `combine.py` -> `stats2.py`) plus `scripts/judged/`, the raw
judgment TSVs merged into the sheets. Re-running `sample.py` regenerates the
sample; re-running the rest rebuilds the sheets and the summary.

Source graphs:

```
results/<model>/art*/e2e/nary_graph_dedup_main_val_r1_extended_consolidated.json
```

## Files

Per model (`gemini-2.5-flash/`, `deepseek-v4-flash/`, `llama33-70b/`, `qwen3-32b-think/`):

| file | rows | what it is |
|---|---|---|
| `sample_relations.csv` | 250 | the drawn relations, with both endpoint nodes: text, node kind, type, type source, definition, parent, the model's own justification, aliases and evidence. This is the audit trail; the two sheets below are derived from it. |
| `judgments_schemaorg.csv` | 83–127 | one row per distinct element (property or node type) that the run tagged `schema.org`, with the real `schema.org` definition, domain and range, the usages it covers in this sample, objective domain/range compliance counts, and the judgment. |
| `judgments_minted.csv` | 82–239 | one row per distinct element that is **not** `schema.org`: either `minted` (deliberately coined, carries a definition and a parent) or `never_canonicalised` (`Claim` hub types, which the single-call track never puts through matching). Includes lexically retrieved `schema.org` candidates as a check against missed reuse. |

Pooled across models: `ALL_sample_relations.csv` (1,000), `ALL_judgments_schemaorg.csv`
(450), `ALL_judgments_minted.csv` (562, of which 469 minted and 93 never-canonicalised). `audit_summary.json` holds every count
behind the reported percentages. `ISSUE_CODES.md` is the codebook.

## How to hand-verify

Every judged row has two blank columns, `hand_verified` and `hand_notes`. Fill
`hand_verified` with `agree` / `disagree` / `unsure`. Nothing else needs editing:
`audit_summary.json` can be regenerated from the sheets.

The columns that are **not** judgment and can be trusted without review, because
they are computed from the `schema.org` release in `schema_org_files/`:

- `in_schema_org` — whether a term tagged `schema.org` actually exists in it.
- `domain_ok` / `domain_bad` / `range_ok` / `range_bad` / `domain_range_violations` —
  domain and range compliance, counted over the usages in this sample where both
  endpoint types are themselves `schema.org` classes (so the counts are lower than
  `n_occurrences`; the rest are not checkable).
- `exists_verbatim_in_schema_org` — whether a coined label collides with an existing
  `schema.org` term. A collision is not automatically a missed reuse: `benefits`,
  `originatesFrom`, `relatedTo`, `follows` and `affectedBy` all exist in
  `schema.org` with narrow, unrelated senses, which is why `NAME_CLASH_WITH_SCHEMA_ORG`
  is a separate code from `SCHEMA_EXACT_EXISTS`.
- `label_has_whitespace` — whether the label is a phrase rather than an identifier.
- `justification_is_fallback` — whether the run recorded no reasoned verdict for this
  match and took the top retrieval hit. These carry a 66% defect rate against 23% for
  reasoned matches, and fail to denote the assertion at all 40% of the time against 8%.
  This is the single most useful filter for a reviewer with limited time.
- `schema_org_candidates` — top lexical matches from `schema.org`, offered as a
  prompt for the reviewer, not as a verdict.

## Judging rubric

Elements are judged once per model, not once per occurrence, because consolidation
already made these vocabulary-level decisions: one term is chosen for a set of
usages, and the justification the model records is written for that set.
`n_occurrences`, `n_e2e` and `n_staged` allow any figure to be re-weighted by how
often the term is actually used.

**Report properties and types separately.** They are not comparable tasks. Type
decisions run at 88-98% because most of them are `Person`, `Country` and `City`;
property decisions run at 72-75%. Pooling them hides the decision the pipeline
exists to make.

**The judged property column does not separate the backends, and should not be
reported as if it did.** At n=47-80 property terms per backend the 95% intervals are
nested. `scripts/resolution.py` prints every
column with its interval. What does separate the backends is the computed side:
domain compliance runs 72% / 53% / 49% / 39% and the fallback share 13% / 0% / 4% /
45%, with non-overlapping intervals at the extremes. The relation-level defect rate
(n=250, the best power here) also separates one backend: llama33-70b at 54% [48,60]
clears gemini 40%, deepseek 39% and qwen 33%. Those carry no annotator error
at all, which is a second reason to lead with them.

The reported verdicts use the **lenient** calibration set out in `ISSUE_CODES.md`:
the term is judged, not its worst individual assignment; a signature violation is
counted once, in the computed columns, not twice; and a defensible term counts as
the best fit. The **strict** verdicts are preserved in `strict_fits_meaning`,
`strict_is_best_fit` and `strict_issue_code`, so either reading can be reported
without re-judging. `scripts/recalibrate.py` derives one from the other and lists
every hand-classified exception; re-running it is idempotent only against the strict
columns, so run it once on a freshly merged sheet.

Neither calibration moves a verdict of *does not denote the assertion*, an inverted
direction, or a term that does not exist in schema.org.

## Relation-level audit (2026-08-30, the one reported in the thesis)

The judgment sheets above score each *distinct term* once. The thesis reports a simpler
reading of the same 1,000 sampled relations, made the way a consumer of the graph would
make it, relation by relation:

- `ALL_judgments_relation_level.csv` — one row per sampled relation with three yes/no
  verdicts (`subject_type_fits`, `property_makes_sense` in the stated direction,
  `object_type_fits`), `relation_ok` (all three), `better_schema_term` (an existing
  schema.org term that would have fitted better, asked of standard and minted terms alike),
  `defects` (from: WRONG_SENSE, UNREADABLE, INVERTED, WRONG_TYPE, STANDARD_EXISTS,
  OVERSPECIFIC, GENERIC, LABEL_FORM, METACLASS, NEAR_DUPLICATE, NOT_IN_VOCAB) and a note.
- `audit_summary_relation_level.json` — per-backend rates with 95% Wilson intervals.

Headline: relations sound (all three verdicts) gemini 81.2%, llama 76.8%, qwen 64.0%,
deepseek 59.2%; standard properties make sense in 75.9% of uses pooled, minted ones in 80.5%.
One annotator; verdicts are lenient on readability (a wrong-sense term that still reads
correctly is `y` with a WRONG_SENSE defect recorded).
