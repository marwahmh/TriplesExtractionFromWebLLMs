# Phase 2 — LLM triple extraction & schema.org canonicalization

Extract entities and relations from news articles with **Gemini** open information extraction
(OIE), then canonicalize each free-form entity **type** to a **schema.org class** and each free-form
**relation** to a **schema.org property** via embedding retrieval — minting a new `ex:` term (and
growing a persistent vocabulary) only when nothing in schema.org fits.

The pipeline has two symmetric halves — an **entity/type** side (nb1–4) and a **relation/property**
side (nb5–7) — that share the same *define → embed → retrieve → reuse-best-match-else-mint* recipe:

| # | Notebook | Scope | Reads | Writes |
|---|----------|-------|-------|--------|
| 0 | `dataset/build_dataset.ipynb` | build corpus | `dataset/articles.csv` | `dataset/articles/*.txt` |
| **Entity / type side** ||||
| 1 | `1_schema_enrichment.ipynb` | **once** | `schemaorg-current-https.jsonld` | `schema_classes{,_enriched}.json`, `enriched_defs_cache.json`, `reference/*` |
| 2 | `2_oie_extraction.ipynb` | per article | `dataset/articles/<id>.txt` | `results/<id>/oie_entities.json` |
| 3 | `3_schema_matching.ipynb` | per article | enriched class pool + `oie_entities.json` | `results/<id>/schema_matching_*.json` *(optional schema.org-only baseline)* |
| 4 | `4_type_extension.ipynb` | per article, incremental | enriched class pool + `oie_entities.json` | `type_vocabulary.json`, `results/<id>/typed_deduped_entities_ext.json` (+ consolidated) |
| **Relation / property side** ||||
| 5 | `5_property_enrichment.ipynb` | **once** | `schemaorg-current-https.jsonld` | `schema_properties{,_enriched}.json`, `enriched_prop_defs_cache.json` |
| 6 | `6_relation_extraction.ipynb` | per article | `dataset/articles/<id>.txt` + `typed_deduped_entities_ext*.json` (nb4) | `results/<id>/oie_relations.json` |
| 7 | `7_property_extension.ipynb` | per article, incremental | enriched property pool + `oie_relations.json` | `relation_vocabulary.json`, `results/<id>/relations_ext.json` (+ consolidated) |

Run order: **1 & 5 once**, then per article **2 → 4** (entities), then **6 → 7** (relations, since
relations are extracted between nb4's typed entities). nb3 is an optional schema.org-only baseline;
nb4 and nb7 are self-contained. Each per-article notebook has an `ARTICLE_ID` / `SELECTED_ARTICLES`
config cell plus a final **batch** cell. nb4 and nb7 each end with an **optional global
re-consolidation** block that cleans the grown vocabulary into `*_consolidated.*` files, leaving the
raw incremental outputs untouched.

## Layout

```
Phase2_LLM_Extraction/
├── 1_schema_enrichment.ipynb      # once:        parse + Gemini-enrich schema.org CLASSES
├── 2_oie_extraction.ipynb         # per-article: Gemini entities + free-form types
├── 3_schema_matching.ipynb        # per-article: embedding retrieval to schema.org (baseline)
├── 4_type_extension.ipynb         # per-article: canonicalize types -> schema.org u type-vocab, else mint
├── 5_property_enrichment.ipynb    # once:        parse + Gemini-enrich schema.org PROPERTIES
├── 6_relation_extraction.ipynb    # per-article: Gemini open relations between nb4's typed entities
├── 7_property_extension.ipynb     # per-article: canonicalize relations -> schema.org u prop-vocab, else mint
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
    ├── type_vocabulary.json           # grown minted-type vocab (nb4)      (+ _consolidated)
    ├── relation_vocabulary.json       # grown minted-property vocab (nb7)  (+ _consolidated)
    └── art<N>/                        # per-article outputs (one folder per article)
        ├── oie_entities.json                       # nb2
        ├── typed_deduped_entities_ext.json         # nb4  (+ _consolidated)
        ├── oie_relations.json                      # nb6
        └── relations_ext.json                      # nb7  (+ _consolidated)
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
- **nb7** canonicalizes at the **relation-phrase** level: one query per distinct phrase, retrieved
  against **schema.org properties ∪ property-vocabulary**. The selection LLM checks direction and
  `domain → range` fit (the property analogue of a class's ancestor `path`), prefers a schema.org
  property on overlap (**schema.org-first**), and mints a new `ex:` property (with a `subPropertyOf`
  parent + domain/range) only when nothing fits. Instances are then deduplicated deterministically on
  `(subject, chosen_property, object)`, so two phrases that map to the same property between the same
  two entities collapse into one triple (evidence spans unioned).
- The property pool encodes the `domain → range` signature in each property's `embedding_text`, the
  same way the class pool encodes the ancestor `path`, so nb5's pool is a drop-in twin of nb1's.
