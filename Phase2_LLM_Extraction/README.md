# Phase 2 — LLM triple extraction & schema.org canonicalization

Extract entities and relations from news articles with **Gemini** open information extraction
(OIE), then canonicalize each free-form entity **type** to a **schema.org class** and each free-form
**relation** to a **schema.org property** via embedding retrieval — minting a new `ex:` term (and
growing a persistent vocabulary) only when nothing in schema.org fits.

The pipeline has two symmetric halves — an **entity/type** side (nb1–4) and a **relation/property**
side (nb5–7) — that share the same *define → embed → retrieve → reuse-best-match-else-mint* recipe:

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
config cell plus a final **batch** cell. nb4 and nb7 each end with an **optional global
re-consolidation** block that cleans the grown vocabulary into `*_consolidated.*` files, leaving the
raw incremental outputs untouched.

The **e2e/experiments/** notebooks (6f–6k) are archived variants of the 6e relation-canonicalization
step, each changing one thing about how candidate properties are retrieved (full-schema, hybrid
type-signature, dual-embedding, quota, schema-only + reconsolidation) to cut avoidable minting. They
are an offline study, not part of the run order; the winner was **6j (quota retrieval)** — separate
schema.org / vocab retrieval lanes. See `notebooks/e2e/experiments/EXPERIMENTS.txt` for the full aims
and findings.

## Layout

```
Phase2_LLM_Extraction/
├── notebooks/
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
└── results/
    ├── _global/                       # cross-article vocabularies & consolidation reports
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
