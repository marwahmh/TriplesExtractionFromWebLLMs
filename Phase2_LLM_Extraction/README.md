# Phase 2 — LLM triple extraction & schema.org canonicalization

Extract entities and relations from news articles with **Gemini** open information extraction
(OIE), then canonicalize each free-form entity type to a **schema.org** class via embedding
retrieval.

## Pipeline

| # | Notebook | Scope | Reads | Writes |
|---|----------|-------|-------|--------|
| 0 | `dataset/build_dataset.ipynb` | build corpus | `dataset/articles.csv` | `dataset/articles/*.txt` |
| 1 | `1_schema_enrichment.ipynb` | **once** | `schema_org_files/schemaorg-current-https.jsonld` | `schema_classes.json`, `schema_classes_enriched.json`, `enriched_defs_cache.json`, `reference/*` |
| 2 | `2_oie_extraction.ipynb` | per article | `dataset/articles/<id>.txt` | `results/<id>/oie_entities.json`, `oie_triples.json` |
| 3 | `3_schema_matching.ipynb` | per article | enriched pool + `results/<id>/oie_entities.json` | `results/<id>/schema_matching_<model>{,_by_type}.json` |

Run order: **1 once**, then **2 → 3** for each article. Notebooks 2 and 3 have an `ARTICLE_ID`
config cell (single article) and a final **batch** cell that loops over the whole dataset.

## Layout

```
Phase2_LLM_Extraction/
├── 1_schema_enrichment.ipynb      # one-time: parse + Gemini-enrich schema.org
├── 2_oie_extraction.ipynb         # per-article: Gemini entities + triples
├── 3_schema_matching.ipynb        # per-article: embedding retrieval to schema.org
├── dataset/
│   ├── articles.csv               # catalog: title, url, domain, source
│   ├── build_dataset.ipynb        # scrapes article bodies from the catalog
│   ├── articles/   art<N>.txt      # raw article text  (pipeline INPUT)
│   └── metadata/   art<N>.json     # per-article title / date / url / facts
├── schema_org_files/
│   ├── schemaorg-current-https.jsonld   # source vocabulary dump
│   ├── schema_classes.json              # parsed classes (baseline pool)
│   ├── schema_classes_enriched.json     # Gemini-enriched pool  ← used for matching
│   ├── enriched_defs_cache.json         # resumable Gemini enrichment cache
│   └── reference/                       # human-readable type/property trees
└── results/
    └── art<N>/                    # per-article outputs (one folder per article)
```

Article ids line up end to end: `dataset/articles/art7.txt` → `results/art7/`.

## Setup

```bash
pip install google-genai sentence-transformers numpy pandas beautifulsoup4 requests
export GEMINI_API_KEY=...   # https://aistudio.google.com/apikey  (or GOOGLE_API_KEY)
```

Notebooks 1 and 2 call Gemini (`gemini-2.5-flash`) and need the key. Notebook 3 is local only:
it downloads two sentence-transformer models (`BAAI/bge-small-en-v1.5`, `intfloat/e5-base-v2`)
on first run and does cosine retrieval — no API key required.

## Matching details

- Matching is at the **OIE type** level: one query per distinct free-form type
  (`type: definition`, mention excluded); every entity inherits its type's match.
- Two document pools are compared — **baseline** (terse schema.org comments) vs **gemini**
  (enriched definitions). The persisted `schema_matching_*` files use the **gemini** pool.
- Each embedding family gets its own query/document prefixes (bge instruction vs e5
  `query:`/`passage:`); see `embed_prefixes()` in notebook 3.
