# Phase 2 — LLM triple extraction & schema.org canonicalization

Extract entities and relations from news articles with **Gemini** open information extraction
(OIE), then canonicalize each free-form entity **type** to a **schema.org class** and each free-form
**relation** to a **schema.org property** via embedding retrieval — minting a new `ex:` term (and
growing a persistent vocabulary) only when nothing in schema.org fits.

The pipeline has two symmetric halves — an **entity/type** side (nb1–4) and a **relation/property**
side (nb5–7) — that share the same *define → embed → retrieve → reuse-best-match-else-mint* recipe:

## Configuring a run — one name, one file

Every notebook's config cell is now a single line:

```python
EXPERIMENT = "main"          # -> configs/main.json
```

Everything else — which `results/<model>/` tree, which articles and in what order, retrieval / rerank
/ consolidation settings, the output tag — comes from `configs/<EXPERIMENT>.json`, read through
`notebooks/experiment_config.py`. Both tracks read the **same** file, so the staged and e2e chains can
be run under one experiment and compared without hand-matching two sets of settings, and the stages
of a chain can no longer disagree about the results tree or the run tags.

```bash
cd notebooks
python -m experiment_config list           # every experiment, one line each
python -m experiment_config show main      # the fully-resolved config (paste into an appendix)
python -m experiment_config check main     # consistency, before spending API/GPU time
```

Headless, without touching a notebook:

```bash
EXPERIMENT=no-rerank ./run_e2e.sh          # 6c -> 6d -> 6e
EXPERIMENT=no-rerank ./run_staged.sh       # 2 -> 4 -> 6 -> 7   (or: ./run_staged.sh 2 4 6b 7b)
SPLIT=val ./run_e2e.sh                     # the validation articles (SPLIT=test for the reported set)
MODEL=qwen3-32b-think ./run_e2e.sh         # another model's tree + backend, same experiment
REPEAT=2 ./run_e2e.sh 6d 6e                # the same arm again, under its own tag (main -> main_r2)
```

Each override becomes part of the tag, so a run stays identifiable from its filenames alone:
`MODEL=llama33-70b SPLIT=val REPEAT=2 EXPERIMENT=no-rerank` writes
`results/llama33-70b/…_dedup_no_rerank_val_r2…`.

`run_tag` is what keeps runs from overwriting each other: everything a stage writes is suffixed with
it, and the next stage reads it back as its `input_tag`. It **defaults to the experiment name**, so
an experiment cannot inherit another one's tag by forgetting to set it. See
**[`configs/README.md`](configs/README.md)** for the full key reference and the list of ablations.

## Provenance — which run produced a file

Every stage writes a **run manifest** at `results/<model>/_runs/<run_id>.json`
([`notebooks/run_manifest.py`](notebooks/run_manifest.py)), opened by the config cell and closed by
the metrics cell at the bottom of the same notebook. Per stage it holds the fully resolved config,
the git commit and dirty flag, the article order, every output file, the token spend, and the stage
metrics — so a number in the write-up resolves to a run, and a run resolves to a commit and a config
file. Nothing to remember to do: it is a side effect of the cells the notebook already has.

```bash
cd notebooks
python -m run_manifest list                      # every run on disk, one line each
python -m run_manifest show e2e__dedup__main     # one run in full
python -m run_manifest diff main_r1 main_r2         # which config keys differ between two runs
python -m run_manifest index                     # regenerate results/RUNS.md
```

A stage **refuses to start** when its manifest says it already finished under the same tag — the
guard against a re-run quietly replacing outputs that are already reported. `REPEAT=<n>` for another
sample of the same arm, a new experiment file for a different configuration, `PHASE2_OVERWRITE=1` to
redo one in place.

Repeats matter here: temperature 0 is greedy, not deterministic, and two identical runs of this
corpus produced 69 vs 91 raw minted properties (65 vs 67 after consolidation). Report an arm as
mean + range over three repeats, and report POST-consolidation numbers — see
[`notebooks/e2e/CURRENT_CONFIG.txt`](notebooks/e2e/CURRENT_CONFIG.txt) §7.

Results written before 2026-08-15 were exploratory (four development articles, hand-invented tags)
and now live in [`results_archive/`](results_archive/); `results/` holds only runs made under this
scheme.

## Which runs, and which number answers which question

The corpus is split once, reproducibly, into **validation (20 articles, including all 10 manually
grounded ones)** and **test (59)** by `python dataset/select_splits.py` → `dataset/splits.json`.
Component effects are measured on validation, reported end-to-end numbers come from test, and the
split is part of the run tag so the two can never overwrite each other.

**[`REPORTING_PLAN.md`](REPORTING_PLAN.md)** is the matrix: the three models × two tracks × two
splits under `main`, the ten ablations and what each isolates, the repeat protocol (temperature 0 is
not deterministic — 69 vs 91 raw mints on identical settings), the token budget, and the measurement
gaps that the current metrics do *not* close.

## Reporting — every number a run produced

Each stage ends with a metrics cell (`notebooks/run_metrics.py`) that re-reads the files the stage
just wrote and saves `results/<model>/_global[/e2e]/metrics_<stage><tag>.json`: graph size and
**fragmentation** for extraction; the reuse-vs-mint split, the retrieval quality it depends on (was
the chosen label in the shortlist, and at which rank) and the vocabulary size before/after
consolidation for the canonicalization stages; plus consolidation's own counts and the token cost.
The cells are pure reading — no LLM call — so they can be re-run at any time, including on a run that
finished days ago. `run_metrics.compare(...)` puts several experiments side by side, which is the
table the ablations exist to fill.

Notebooks now live under `notebooks/<track>/` (`0_shared`, `two_staged`, `e2e`). Per-article outputs
under `results/<id>/` are split into `two_staged/` (staged pipeline) and `e2e/` (single-call baseline);
cross-article vocabularies/reports live in `results/_global/`.

| # | Notebook | Scope | Reads | Writes |
|---|----------|-------|-------|--------|
| 0 | `dataset/build_dataset.ipynb` | build corpus | `dataset/articles.csv` | `dataset/articles/*.txt` |
| **Entity / type side** (`two_staged`) ||||
| 1 | `0_shared/1_schema_enrichment.ipynb` | **once** | `schemaorg-current-https.jsonld` | `schema_classes{,_enriched}.json`, `enriched_defs_cache.json`, `reference/*` |
| 2 | `two_staged/2_oie_extraction.ipynb` | per article | `dataset/articles/<id>.txt` | `<id>/two_staged/1_entities/oie_entities.json` |
| 3 | `two_staged/3_schema_matching.ipynb` | per article | enriched class pool + `oie_entities.json` | `<id>/two_staged/1_entities/schema_matching_*.json` *(optional schema.org-only baseline)* |
| 4 | `two_staged/4_type_extension.ipynb` | per article, incremental | enriched class pool + `oie_entities.json` | `_global/type_vocabulary.json`, `<id>/two_staged/1_entities/entities_typed.json` (+ `_consolidated`), `entity_type_matching_review.json` |
| **Relation / property side** (`two_staged`) ||||
| 5 | `0_shared/5_property_enrichment.ipynb` | **once** | `schemaorg-current-https.jsonld` | `schema_properties{,_enriched}.json`, `enriched_prop_defs_cache.json` |
| 6 | `two_staged/6_relation_extraction.ipynb` | per article | `<id>.txt` + `1_entities/entities_typed*.json` (nb4) | `<id>/two_staged/2_binary_relations/oie_relations.json` |
| 7 | `two_staged/7_property_extension.ipynb` | per article, incremental | enriched property pool + `oie_relations.json` | `_global/relation_vocabulary.json`, `<id>/two_staged/2_binary_relations/relations.json` (+ `_consolidated`) |
| **N-ary event reification** ||||
| 6b | `two_staged/6b_event_reification.ipynb` | per article | `1_entities/entities_typed*.json` (nb4) | `<id>/two_staged/3_nary_graph/nary_graph.json`, `_global/hub_type_vocabulary.json` |
| 7b | `two_staged/7b_nary_property_extension.ipynb` | per article, incremental | `3_nary_graph/nary_graph.json` + property pool | `<id>/two_staged/3_nary_graph/nary_graph_extended.json`, `_global/nary_relation_vocabulary.json` |
| 6c | `e2e/6c_e2e_reification.ipynb` | per article (**baseline**) | `dataset/articles/<id>.txt` (raw) | `<id>/e2e/nary_graph.json` |
| 6d | `e2e/6d_e2e_type_canonicalization.ipynb` | per article, incremental | `<id>/e2e/nary_graph.json` | `_global/e2e/type_vocabulary.json`, `<id>/e2e/nary_graph_typed.json` |
| 6e | `e2e/6e_e2e_relation_canonicalization.ipynb` | per article, incremental | `<id>/e2e/nary_graph_typed.json` (6d) | `_global/e2e/nary_relation_vocabulary.json`, `<id>/e2e/nary_graph_extended.json` |
| **E2E retrieval experiments** (archived) ||||
| 6f–6k | `e2e/experiments/*.ipynb` | per article (**offline study**) | `<id>/e2e/nary_graph_typed.json` (6d) | `_global/e2e/experiments/*_dedup_<variant>.json` |

| **Visualization** ||||
| 8 | `0_shared/8_kg_visualization.ipynb` | per article | any graph file above | `<id>/two_staged/viz/kg_{binary,reified}.{png,html}`, `<id>/e2e/viz/kg.{png,html}` |

Run order: **1 & 5 once**, then per article **2 → 4** (entities), then **6 → 7** (relations, since
relations are extracted between nb4's typed entities); reified track adds **6b → 7b**. The **e2e track**
is independent: **6c → 6d → 6e** (single-call graph, then its own type then relation canonicalization
against SEPARATE `_global/e2e/` vocabularies). 6d and 6e each end with the same **optional global
re-consolidation + hierarchy** tail as nb4/nb7 (writes `*_consolidated.*` + `*_hierarchy*.txt` under
`_global/e2e/`, leaving the raw incremental outputs untouched). nb3 is an optional schema.org-only baseline;
nb4 and nb7 are self-contained. Each per-article notebook has an `ARTICLE_ID` / `SELECTED_ARTICLES`
config cell plus a final **batch** cell. nb4 and nb7 each end with the same **global
re-consolidation** block, cleaning the grown vocabulary into `*_consolidated.*` files and leaving the
raw incremental outputs untouched.

The staged and e2e notebooks are deliberately kept in **step with each other**: nb4 is 6d's twin,
nb7b is 6e's, nb7 is the binary twin of both, and nb6b runs the same review/repair pass as 6c. They
take the same knobs from the same config section names and write the same metrics files, so a
difference between the two tracks' results is a difference between the *pipelines* — not between two
implementations that drifted apart. The one deliberate asymmetry is nb4's within-group deduplication:
nb2 emits every mention, whereas the single 6c call has already merged coreferent ones.

### Parallel batches

Where a step's articles are **independent**, its batch cell runs several at once through
`notebooks/nb_parallel.py` — set `WORKERS` in the config cell (`WORKERS = 1` restores the plain
sequential loop). That covers **6c** (extraction), **9 / 9b** (grounding) and **10** (MINE-1); each
article's prints are buffered and shown as one block when it finishes, and a failed article is
reported and skipped instead of ending the run. Since the time goes almost entirely on waiting for
the API, WORKERS≈4 turns a 100-article grounding run into a ~4× shorter one. Ceiling: Gemini paid
tier is comfortable at 4–8, the free tier (10 RPM) wants 1–2, and a vLLM server batches internally
so 4–8 there too.

The canonicalization notebooks (**4, 7, 7b, 6d, 6e**) stay sequential *by design*: they grow one
shared vocabulary article by article, and what article *k* mints depends on what *k−1* added — running
them in parallel would change the output, not just the speed.

### Token & time accounting

Every notebook that calls an LLM ends its batch cell with the same report, from the shared ledger in
`nb_parallel.py` — each call's prompt / output / **thinking** tokens and its latency, broken down per
article:

```
LLM USAGE — nb4 · type extension
               calls     prompt     output  thinking      total    llm_s
art1               1      7,577      3,722     3,331     14,630     12.4
art2               1      6,550      4,331     3,516     14,397     12.3
------------------------------------------------------------------------
TOTAL              2     14,127      8,053     6,847     29,027     24.7
wall clock: 25.1s   (summed LLM time 24.7s)
```

**Thinking tokens are reported separately because they are billed as output but excluded from the
API's output count** — on a reasoning model they are routinely the larger half, so a total that
folds them in silently, or drops them, misstates the cost. A backend that does no reasoning reports
zero and says so. Each report is also written to `results/<model>/_global/token_usage_<nb>.json`
(the enrichment notebooks write theirs next to the schema cache), so a run's cost survives the
kernel. `wall clock` is the batch's real elapsed time; for a parallel batch the summed LLM time is
larger, and the ratio is the overlap the thread pool actually bought.

The **e2e/experiments/** notebooks (6f–6k) are archived variants of the 6e relation-canonicalization
step, each changing one thing about how candidate properties are retrieved (full-schema, hybrid
type-signature, dual-embedding, quota, schema-only + reconsolidation) to cut avoidable minting. They
are an offline study, not part of the run order; the winner was **6j (quota retrieval)** — separate
schema.org / vocab retrieval lanes. See `notebooks/e2e/experiments/EXPERIMENTS.txt` for the full aims
and findings.

## Layout

```
Phase2_LLM_Extraction/
├── configs/                               # ONE FILE PER EXPERIMENT — see configs/README.md
│   ├── _base.json                         # every default; experiments extend it
│   ├── main.json                          # the reported configuration
│   └── no-rerank.json, inline-vocab.json, ...   # ablations + open-weight model arms
├── run_e2e.sh / run_staged.sh             # run a whole track headlessly under one experiment
├── notebooks/
│   ├── experiment_config.py               # loads configs/<name>.json into a notebook's globals
│   ├── run_manifest.py                    # one record per run: config + commit + outputs + tokens
│   ├── run_metrics.py                     # collects every stage's numbers from the files it wrote
│   ├── llm_backend.py                     # swaps gemini_json for a vLLM server (open-weight models)
│   ├── nb_parallel.py                     # thread pool + per-article output capture for the batch cells
│   ├── 0_shared/
│   │   ├── 1_schema_enrichment.ipynb      # once:  parse + Gemini-enrich schema.org CLASSES
│   │   ├── 5_property_enrichment.ipynb    # once:  parse + Gemini-enrich schema.org PROPERTIES
│   │   └── 8_kg_visualization.ipynb       # draw the KG (binary / reified / e2e) -> png + interactive html
│   ├── two_staged/
│   │   ├── 2_oie_extraction.ipynb         # Gemini entities + free-form types
│   │   ├── 3_schema_matching.ipynb        # embedding retrieval to schema.org (baseline)
│   │   ├── 4_type_extension.ipynb         # canonicalize types -> schema.org ∪ type-vocab, else mint
│   │   ├── 6_relation_extraction.ipynb    # Gemini open relations between nb4's typed entities
│   │   ├── 6b_event_reification.ipynb     # reify facts into Event/Observation/Claim n-ary hubs
│   │   ├── 7_property_extension.ipynb     # canonicalize relations -> schema.org ∪ prop-vocab, else mint
│   │   └── 7b_nary_property_extension.ipynb  # canonicalize n-ary relation phrases -> properties
│   └── e2e/
│       ├── 6c_e2e_reification.ipynb            # BASELINE: raw text -> entities+n-ary graph in ONE call
│       ├── 6d_e2e_type_canonicalization.ipynb  # canonicalize e2e entity types  (twin of nb4)
│       ├── 6e_e2e_relation_canonicalization.ipynb  # canonicalize e2e relations (twin of nb7b)
│       └── experiments/                        # archived 6e retrieval variants (6f-6k) — see EXPERIMENTS.txt
├── dataset/
│   ├── articles.csv               # catalog: title, url, domain, source
│   ├── build_dataset.ipynb        # scrapes article bodies from the catalog
│   ├── articles/   art<N>.txt      # raw article text  (pipeline INPUT)
│   └── metadata/   art<N>.json     # per-article title / date / url / facts
├── schema_org_files/
│   ├── schemaorg-current-https.jsonld   # source vocabulary dump
│   ├── schema_classes.json              # parsed classes (baseline pool)
│   ├── schema_classes_enriched.json     # Gemini-enriched class pool     ← used by nb3/nb4
│   ├── schema_properties.json           # parsed properties (baseline pool)
│   ├── schema_properties_enriched.json  # Gemini-enriched property pool  ← used by nb7
│   ├── enriched_defs_cache.json         # resumable class-enrichment cache
│   ├── enriched_prop_defs_cache.json    # resumable property-enrichment cache
│   └── reference/                       # human-readable type/property trees
├── results_archive/                  # frozen pre-2026-08-15 exploration; read by nothing
└── results/<model>/                  # ONLY runs made under the manifest scheme (see above);
    │                                 # one tree per `results_model`, indexed by ../RUNS.md
    ├── _runs/<run_id>.json           # THE RUN MANIFEST: config + commit + outputs + tokens, per stage
    ├── _global/                      # cross-article vocabularies & consolidation reports
    │   ├── type_vocabulary.json           # grown minted-type vocab (nb4)      (+ _consolidated)
    │   ├── relation_vocabulary.json       # grown minted-property vocab (nb7)  (+ _consolidated)
    │   ├── hub_type_vocabulary.json       # n-ary node ('hub') type vocab (nb6b/7b), parented into the type tree
    │   ├── nary_relation_vocabulary.json  # n-ary relation-phrase vocab (nb7b)
    │   ├── e2e/                           # SEPARATE e2e vocabularies (6d, 6e) — kept comparable, never mixed
    │   │   ├── type_vocabulary.json           # e2e entity-type vocab (6d)
    │   │   ├── nary_relation_vocabulary.json  # e2e relation-property vocab (6e)
    │   │   └── experiments/                   # archived per-variant outputs of 6f-6k (*_dedup_<variant>.json)
    │   └── *_hierarchy*.txt, *_consolidation_report.json, *_changes.json
    └── art<N>/                        # per-article outputs (one folder per article)
        ├── two_staged/                    # staged pipeline (nb2–7b)
        │   ├── 1_entities/
        │   │   ├── oie_entities.json                   # nb2
        │   │   ├── entities_typed.json                 # nb4  (+ _consolidated)
        │   │   └── entity_type_matching_review.json    # nb4
        │   ├── 2_binary_relations/
        │   │   ├── oie_relations.json                  # nb6
        │   │   ├── relations.json                      # nb7  (+ _consolidated)
        │   │   └── relation_matching_review.json       # nb7
        │   ├── 3_nary_graph/
        │   │   ├── nary_graph.json                     # nb6b
        │   │   ├── nary_graph_extended.json            # nb7b
        │   │   └── nary_relation_matching_review.json  # nb7b
        │   └── viz/  kg_binary.{png,html}, kg_reified.{png,html}   # nb8
        └── e2e/                           # single-call baseline (nb6c) + its own canonicalization
            ├── nary_graph.json                         # nb6c  (free-form types & phrases)
            ├── nary_graph_typed.json                   # 6d    (entity types canonicalized)
            ├── nary_graph_extended.json                # 6e    (relations canonicalized — final e2e KG)
            ├── entity_type_matching_review.json        # 6d
            ├── nary_relation_matching_review.json      # 6e
            └── viz/  kg.{png,html}                     # nb8
```

Article ids line up end to end: `dataset/articles/art7.txt` → `results/art7/`.

## Setup

```bash
pip install google-genai sentence-transformers numpy pandas beautifulsoup4 requests
export GEMINI_API_KEY=...   # https://aistudio.google.com/apikey  (or GOOGLE_API_KEY)
```

Notebooks 1, 2, 5 and 6 call Gemini (`gemini-2.5-flash`) and need the key; so do the LLM-selection
and consolidation stages of nb4 and nb7. The retrieval in nb3/nb4/nb7 is a local sentence-transformer
(`intfloat/e5-base-v2`, downloaded on first run) doing cosine retrieval — no API key required for the
embedding step itself.

### Running on open-weight models instead of Gemini

Any notebook can be pointed at an open-weight model (Qwen3, Gemma 3, Llama 3.x, …) served by vLLM on
the Lab-IA GPU cluster, by adding **one cell** that swaps `gemini_json` for `notebooks/llm_backend.py`.
Full setup, per-session runbook, how to switch models, and troubleshooting: **[`cluster/README.md`](cluster/README.md)**.

## Type matching details (nb3 / nb4)

- Matching is at the **OIE type** level: one query per distinct free-form type
  (`type: definition`, mention excluded); every entity inherits its type's match.
- Two document pools are compared — **baseline** (terse schema.org comments) vs **gemini**
  (enriched definitions). The persisted `schema_matching_*` files use the **gemini** pool.
- Each embedding family gets its own query/document prefixes (bge instruction vs e5
  `query:`/`passage:`); see `embed_prefixes()` in notebook 3.

## Relation matching details (nb6 / nb7)

- **nb6** hands Gemini the article's already-typed, deduplicated entities (from nb4, each stamped with
  a stable id) and extracts every stated relation as a directed `[subject, relation_phrase, object]`
  triple with a verbatim evidence span — inventing the free-form `relation_phrase`, exactly as nb2
  invents type labels. Only entities from the given list may be arguments.
- **nb7** retrieves at the **relation-phrase** level (one query per distinct phrase, against
  **schema.org properties ∪ property-vocabulary**) but resolves at the **instance** level: for each
  phrase the judge returns one or more **assignments**, each a subset of the phrase's instances plus the
  property chosen for that subset. So a single phrase can **split across properties** when its instances
  have different signatures (e.g. *"located in"* → `containedInPlace` for City→Country but `location`
  for Event→Place). Per assignment it checks direction and `domain → range` fit (loosely — see below),
  prefers a schema.org property on overlap (**schema.org-first**), and mints a new `ex:` property (with
  a `subPropertyOf` parent + domain/range) only when nothing fits. Instances are then deduplicated
  deterministically on `(subject, chosen_property, object)`, so instances that land on the same property
  between the same two entities collapse into one triple (evidence unioned). The review file records
  each phrase's assignments, so a split is fully auditable.
- **Domain/range is a soft signal at the judge, not a hard filter.** schema.org's domain/range are
  permissive suggestions and entity types are often sub-/super-/sibling types of the declared ones, so
  the judge reuses a meaning-preserving candidate even when the argument types aren't an exact match —
  it won't mint merely because types differ, only when reuse would change or lose the relation's meaning.
- **Inverse reuse.** A phrase can be matched to a candidate that expresses the *opposite* direction:
  the judge sets `direction: "inverse"` and nb7 **swaps subject and object** so the stored triple runs
  in the property's direction — reusing an existing property instead of minting a redundant new one
  (e.g. an `employs` phrase reuses `worksFor` as *object → subject*). Each schema.org candidate also
  shows its `inverseOf` name as a hint. Because inverse reuse orients the triple to the property, the
  `relations_ext.json` records carry `phrase_directions` (which phrase was flipped) and the
  preservation check compares **unordered** entity pairs.
- **Embedding style is symmetric.** Both the relation-phrase **query** and every property **document**
  (schema.org and vocab) share one shape: `name: definition [Also called synonyms]`, authored to the
  same template in nb5 and nb6. The `domain → range` **signature is deliberately not embedded** (the
  query has no counterpart for it); it lives in the `signature` field and is given to the **LLM judge**.
- **Definitions are direction-NEUTRAL** — they describe the relation as an association between two kinds
  of entity without fixing which is subject/object (e.g. *"An employment relation between a person and
  an organization."*). This is deliberate: a direction-explicit gloss would make an inverse property
  look like a *different* relation and push it out of the retrieved top-k, defeating **inverse reuse**.
  Neutral glosses let a property **and its inverse** both surface as candidates; the judge then decides
  `as_is` vs `inverse` from the `signature` + instance types (the reliable direction signal), not from
  the comment wording. So: retrieval finds the relation *concept* (direction-agnostic); the judge
  resolves *direction*.
