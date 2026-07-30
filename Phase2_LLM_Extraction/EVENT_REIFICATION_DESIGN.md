# Event-centric reification — design spec

**Status:** implemented (nb6b + nb7b, staged track) · **Decisions locked:** hybrid reification, schema.org-first grounding.

> **As-built note (read first).** This document is the original design. The implementation kept the
> concepts (three n-ary node kinds, schema.org-first grounding, open/extensible predicate vocabulary,
> the binary-vs-reified comparison) but simplified the on-disk shape:
> - Output is **`results/<id>/two_staged/3_nary_graph/nary_graph.json`** (nb6b), **not** `reified_graph.json`.
> - Per-node `roles[]` + a separate `binary_relations[]` were flattened into **one top-level `relations[]`**
>   list of `{subject_id, relation_phrase, object_id, evidence, confidence, …}` (subjects/objects may be
>   entity *or* n-ary node ids) plus a `relation_definitions[]` glossary — i.e. the "roles" became **open
>   `relation_phrase` values**, exactly like the binary path's OIE relations.
> - Canonicalizing those phrases to schema.org properties / a minted vocab is a **separate stage, nb7b**
>   (`7b_nary_property_extension`), which writes **`nary_graph_extended.json`** — the n-ary analogue of nb7.
> - The cross-article vocabularies live in **`results/_global/`**: `hub_type_vocabulary.json` (n-ary
>   node-type vocab, called `nary_type_vocabulary.json` below) and `nary_relation_vocabulary.json`
>   (the consolidated relation-phrase → property vocab). Code/files sometimes call an n-ary node a **"hub"**.
> - A single-call **end-to-end baseline** (`notebooks/e2e/6c_e2e_reification.ipynb` → `results/<id>/e2e/nary_graph.json`)
>   builds the same-shaped graph from raw text in one Gemini call, for comparison against this staged track.
> §5 and §6 below have been updated to the as-built paths/schema; §1–§4 remain the conceptual design.

A second relational layer for the pipeline. Instead of only stating facts as binary
`[subject, property, object]` triples between entities, we introduce a small set of **n-ary nodes**
(events, observations, attributed claims) that gather the participants of *one* news fact under
*one* node via an **open, extensible set of semantic roles**. The binary path (`6_relation_extraction` → `7`)
stays untouched; this is a parallel `6b` that consumes the **same** typed entities so the two graphs
are directly comparable.

---

## 1. Motivation (grounded in art1)

One news fact:

> "In France, the number of deaths recorded between 22 and 28 June increased by 2,025 — nearly 30%.
> Deaths rose by 62% in the Paris region alone."

Today this is shattered across disconnected binary triples, each on a different subject node:

```
France                    --recordedValue-->        "more than 2,000 deaths"
"more than 2,000 deaths"   --occurredDuring-->        "last week of ... heatwave in June"
"22 and 28 June"           --experiencedIncreaseOf--> "2,025"
"the Paris region"         --experiencedIncreaseOf--> "62%"
```

Three structural symptoms, all consequences of the binary shape:

1. **Context-less triples** — `France --recordedValue--> "2,025"` has lost the *when*, the *what*
   (mortality), and the *cause* (heatwave); they live on other nodes.
2. **Junk hops / promoted pseudo-nodes** — a measurement (`"more than 2,000 deaths"`) or a date
   (`"22 and 28 June"`) is forced to become a subject, because a binary triple *needs* one. This is
   accidental, inconsistent reification already happening out of necessity.
3. **Predicate explosion** — with nowhere to hang context, it is crammed into the predicate name:
   `notedIncreaseInDeathsForAgeGroup`, `expectsConditionsFrom`, `experiencedIncreaseOf`. Each is an
   n-ary frame collapsed into a string, which is why `relation_vocabulary` keeps growing.

**Framing (thesis):** neo-Davidsonian event reification / frame semantics, realized as an
event-centric knowledge graph. Grounded in schema.org's `Event`, `Observation`, `Claim`, and the
W3C n-ary-relations pattern.

---

## 2. The model — three n-ary node kinds

| N-ary node kind        | Mint one when…                                                         | schema.org base            |
|-----------------|------------------------------------------------------------------------|----------------------------|
| **Event**       | something happens: heatwave, wildfire, evacuation, forecast, alert     | `Event` subtree + minted   |
| **Observation** | a statistic / measurement / quantified change is stated                | `Observation`              |
| **Claim**       | a statement is attributed to a source ("X said Y", "ministry called it…")| `Claim` / `Statement`    |

`Observation` is a close schema.org fit for the excess-death numbers: it already defines
`observationAbout`, `measuredProperty`, `observationDate`, `marginOfError`, and inherits `value` /
`unitText` from `QuantitativeValue`.

**Two typing axes + grounding (stage 4).** Every n-ary node has a coarse `node_kind` (one of the three above)
*and* a fine `type`. The fine type is **grounded** — reconciled against the same hierarchy the entities
use — so it is truthful and connected, not a free-floating label:

1. **Reuse** the nb4 entity type vocabulary (or the n-ary node vocabulary) when the type already exists there
   (`WeatherEvent → Event`, `NaturalDisaster → EnvironmentalPhenomenon`), inheriting that parent.
2. **schema.org** if it's a *real* class, **verified against the JSON-LD** (the model's own `schema.org`
   claim is not trusted — `Event` has 35 real subclasses, all secular; `Observation`/`Claim` are leaves)
   → `type_source: "schema.org"`, authoritative parent.
3. **Mint** otherwise, with an explicit `type_parent` chaining up to the n-ary node root.

**One extension namespace.** `type_source` (like `role_source` and `property_source`) is just
`schema.org` **or** `ex` — there is no separate `"vocab"`. A reused extension type (case 1) is still
`ex`; "reused vs freshly minted" is tracked by `support`/`first_seen` in the vocabulary files, not by a
third source value.

Result fields on each n-ary node: `type`, `type_source ∈ {schema.org, ex}`, `type_parent`. New minted
n-ary node types accrue (`support`/`provenance`) into `results/_global/hub_type_vocabulary.json` (the
`{types:[…], next_id}` file; `hub_kind` = the node kind) across articles — the n-ary node analogue of nb4's
`type_vocabulary_consolidated.json`, and parented *into* it so the two form one hierarchy. Before grounding, the model mislabeled `WeatherEvent`/`DeathEvent`/`WeatherForecast` as
`schema.org` (none exist); after, they are correctly `ex` with real parents.

**Hybrid rule (locked).** Reify only these three kinds. Stable, lasting relations (part-of, membership,
office held, ownership, authorship, identity) stay as plain edges in `binary_relations` rather than
being wrapped in an n-ary node.

**Binary-relations pass is MANDATORY** (added after the first art1 run, where the model produced
`binary_relations: []` and silently dropped 25 stable facts — all the geography, affiliations, and
titles the binary graph had). The prompt requires an explicit sweep for stable relations, kept general
(no named entity examples), populating `binary_relations` **even when the nodes also appear in an n-ary node**.
Endpoints may be **entities OR n-ary nodes** (entity↔entity, entity↔n-ary node, n-ary node↔n-ary node), so binary relations tie
into the n-ary node structure and the whole graph stays one connected component. Each entry carries a
`property_source` (`schema.org` | `ex`) so extension properties render dashed.

---

## 3. Extensible role vocabulary (seed + invention)

Roles are the edges from an n-ary node to its participants. The vocabulary is **open and extensible** — same
philosophy as the binary path (`6_relation_extraction` → `7`): the model invents the label that fits,
gives it a definition and an optional parent, and a later consolidation pass folds it into a growing
`role_vocabulary` (mirroring `relation_vocabulary_consolidated.json`). It is **not** a closed set.

The table below is the **seed** — a preferred starter vocabulary injected into the prompt. The model
should reuse a seed role when one fits and only mint a new one otherwise.

**Invention mechanics** (per role instance): `role`, `role_source` (`"schema.org"` **or** `"ex"` — the
single extension namespace, exactly like `type_source`/`property_source`), `definition` (required when
the role is invented), `parent` (optional existing role it specializes), plus the value (see below).
Whether an `ex` role was reused or freshly invented is tracked in `new_roles`, not by the source value.

**The one discipline that keeps reification's benefit:** invented roles must stay **atomic and
reusable** — a single participant slot (`evacuees`, `warnsOf`, `burnedArea`), *never* a
sentence-fragment predicate that re-crams context (`notedIncreaseInDeathsForAgeGroup`). Context like
time / place / value goes on its *own* role edge, which is the whole point of the n-ary node.

The seed is a small, **domain-neutral** backbone of thematic roles that apply to *any* article —
who/what/where/when/why/how. Domain specifics (`changeRate`, `reportedOn`, `evacuees`, `burnedArea`)
are deliberately **left out** so they surface as *invented* roles — that is the extensibility working.

| Seed role      | schema.org / ex:   | Generic sense — connects n-ary node → …                     |
|----------------|--------------------|------------------------------------------------------|
| `agent`        | schema:agent       | who/what brings the n-ary node about (the doer)             |
| `affected`     | ex:affected        | who/what undergoes or is affected (patient/theme/experiencer) |
| `about`        | schema:about       | what it concerns / what is measured (topic)          |
| `location`     | schema:location    | where                                                |
| `time`         | schema:startDate   | when — a point or interval (value may be a Date/Duration node) |
| `value`        | schema:value       | how much — a quantity/measurement/magnitude          |
| `cause`        | ex:cause           | why — the event/condition that brought it about      |
| `result`       | ex:result          | the outcome/effect it produces                       |
| `instrument`   | ex:instrument      | by what means / method                               |
| `source`       | schema:source      | origin — where it comes from, or who states/reports it |
| `participant`  | schema:participant | any other involved entity not captured above         |
| `partOf`       | schema:superEvent  | the larger event/whole this belongs to               |

Twelve generic roles. Everything domain-specific is invented on top (with `parent` usually pointing
at one of these — e.g. `changeRate` → `value`, `reportedOn` → `time`, `reportedBy` → `source`).

A role whose value is a scalar (`value`, or invented `changeRate` etc.) may point at an existing
entity node **or** carry a literal, so we don't have to promote every number to a node.

---

## 4. Worked example — art1 France mortality

**Before:** 4+ binary triples across 3 subject nodes (§1).

**After:** one n-ary node, fully contextualized, predicate vocabulary collapsed to roles.

```json
{
  "id": "OBS1",
  "node_kind": "Observation",
  "type": "ExcessMortalityStatistic",
  "type_source": "ex",
  "label": "France excess deaths, 22–28 June 2026",
  "roles": [
    {"role": "about",    "role_source": "schema.org", "value_text": "excess deaths"},
    {"role": "affected", "role_source": "ex", "value_ref": "E1"},                         // France (ex:affected)
    {"role": "location", "role_source": "schema.org", "value_ref": "E46"},                // Paris region
    {"role": "time",     "role_source": "schema.org", "value_ref": "E72"},                // 22 and 28 June
    {"role": "value",    "role_source": "schema.org", "value_ref": "E73"},                // 2,025
    {"role": "cause",    "role_source": "ex", "value_ref_nary": "EV1"},                    // June heatwave (ex:cause)
    {"role": "source",   "role_source": "schema.org", "value_text": "Public Health France"},
    // INVENTED atomic roles (also ex), each parented to a seed role:
    {"role": "changeRate", "role_source": "ex", "parent": "value",
     "definition": "Rate/percentage of change in the measured value.", "value_ref": "E26", "value_text": "+29%"},
    {"role": "reportedOn", "role_source": "ex", "parent": "time",
     "definition": "When the fact was reported/announced.", "value_text": "2026-07-03"}
  ],
  "evidence": ["the number of deaths recorded between 22 and 28 June increased by 2,025 — nearly 30%"],
  "confidence": 1.0
}
```

And the **backbone**: a main `Event` (`EV1 = June 2026 European heatwave`) that per-country
observations, wildfires, and alerts hang off via `cause` / `partOf`. Salient nodes (heatwave ↔
death toll ↔ country) become 1 hop apart instead of 4.

---

## 5. Output schema (as-built)

**nb6b → `results/<id>/two_staged/3_nary_graph/nary_graph.json`** (open, pre-canonicalization):

```jsonc
{
  "article_id": "art1",
  "reference_date": "2026-07-03",
  "entity_file": "…/1_entities/entities_typed_consolidated.json",   // which entity set it reified over
  "entities": [ /* same id-stamped cards passed in (E1, E2, …) */ ],
  "nary_nodes": [
    {
      "id": "CL1", "node_kind": "Event|Observation|Claim",
      "type": "WeatherEvent",            // fine free-form type (grounded/canonicalized downstream)
      "label": "June 2026 European heatwave", "is_main": true,
      "evidence": ["…"], "confidence": 0.9
    }
  ],
  // ONE flat list — the "roles" of §3, realized as OPEN relation_phrase values.
  // subject_id / object_id may be an ENTITY id (E..) or an N-ARY NODE id (CL../OBS../EV..).
  "relations": [
    {"subject_id": "CL1", "relation_phrase": "has_quantity", "object_id": "E9",
     "evidence": ["more than 2,000 deaths"], "confidence": 0.9,
     "subject_text": "…", "subject_type": "…", "object_text": "…", "object_type": "QuantitativeValue"}
  ],
  "relation_definitions": [ {"relation_phrase": "has_quantity", "definition": "…"} ]   // glossary of phrases used
}
```

**nb7b → `results/<id>/two_staged/3_nary_graph/nary_graph_extended.json`** — canonicalizes each
`relation_phrase` to a schema.org property or minted `ex`/vocab term (the n-ary analogue of nb7). Same
`entities` / `nary_nodes`, but every relation gains: `property`, `property_source ∈ {schema.org, ex}`,
`vocab_id`, `property_is_new_to_vocab`, `property_definition`, `property_parent`, `property_justification`.
The review trace goes to `nary_relation_matching_review.json` in the same folder.

Cross-article side artifacts in **`results/_global/`**:
- `hub_type_vocabulary.json` — growing n-ary node-type vocabulary (`{types:[{id, canonical_label, parent,
  hub_kind, definition, support, provenance}], next_id}`), parented into the nb4 entity vocab / schema.org.
- `nary_relation_vocabulary.json` — consolidated relation-phrase → property vocabulary
  (`{properties:[{id, canonical_label, definition, parent, domain, …}], next_id}`), the n-ary analogue of
  `relation_vocabulary_consolidated.json`.

Notes:
- Object endpoints may be an entity, another n-ary node, or a literal-ish entity (numbers/dates kept as
  `QuantitativeValue`/`Date` entity cards) — so scalars need not each be promoted to a node.
- `property_source`: `schema.org` (standard property) or `ex` (extension — invented or reused; the single
  non-standard namespace). Whether an `ex` term was reused vs freshly minted lives in the `support` /
  `property_is_new_to_vocab` fields, not a third source value.
- `is_main` flags the article's central event(s) for the backbone / layout.

---

## 6. Pipeline placement (as-built — three graphs from the same entities/text)

| Notebook                                        | Track        | Role                                                                 |
|-------------------------------------------------|--------------|---------------------------------------------------------------------|
| `two_staged/4_type_extension` (nb4)             | shared in    | produces `1_entities/entities_typed*.json`                          |
| `two_staged/6_relation_extraction` → `7`        | binary       | binary-triple baseline → `2_binary_relations/relations*.json`        |
| **`two_staged/6b_event_reification`**           | reified      | same typed entities → `3_nary_graph/nary_graph.json`                |
| **`two_staged/7b_nary_property_extension`**     | reified      | canonicalize phrases → `3_nary_graph/nary_graph_extended.json`      |
| **`e2e/6c_e2e_reification`**                    | e2e baseline | raw text → `e2e/nary_graph.json` in ONE call (no staged entities)   |
| **`e2e/6d_e2e_type_canonicalization`**          | e2e baseline | canonicalize e2e entity types (twin of nb4) → `e2e/nary_graph_typed.json`   |
| **`e2e/6e_e2e_relation_canonicalization`**      | e2e baseline | canonicalize e2e relation phrases (twin of nb7b) → `e2e/nary_graph_extended.json` |
| `0_shared/8_kg_visualization`                   | viz          | render binary / reified / e2e graphs (`…/viz/kg_*.{png,html}`)      |

The e2e baseline gets the **same** canonicalization the staged track has (types like nb4, relation phrases
like nb7b) but grows its **own** vocabularies under `results/_global/e2e/`, so the two methods' type- and
predicate-economy stay independently measurable (§7). 6d relabels entities **by id** (no merge — the single
call already resolved coreference), so relation endpoints stay valid. 6d and 6e each carry the same
**optional global re-consolidation + hierarchy** tail as nb4/nb7 (synonym/subtype merges → `*_consolidated.*`
and `*_hierarchy*.txt` under `_global/e2e/`; raw incremental outputs left intact).

Each per-article notebook has a single-article demo cell + a `SELECTED_ARTICLES` batch cell, mirroring nb6.
See the E2E-vs-staged comparison for why the single-call baseline matters (orphan entities, predicate
economy, entity recall).

---

## 7. Evaluation — "is it better?"

Same articles, two graphs, measure:

- **Compactness / navigability** — node & edge counts; avg. shortest-path between *salient* entities
  (heatwave ↔ death toll ↔ country). Reification should shrink hops.
- **Predicate economy** — both vocabularies grow, so compare their *shape*: role-vocab size &
  growth curve vs. `relation_vocabulary`, and mean predicate specificity (roles should stay atomic /
  high-reuse; binary predicates trend toward one-off sentence fragments). Reuse rate = uses per
  distinct role.
- **Context completeness** — % of facts carrying both *time* and *location* on one node.
- **Fragmentation ratio** — binary triples collapsed per n-ary node (the 4→1 above).
- **Downstream QA** (most persuasive) — a few questions per article ("how many excess deaths in
  France and when?") answered by traversal / an LLM over each graph; score faithfulness & completeness.

---

## 8. Open questions to revisit after v1

- Cross-article event linking (same heatwave across art1/art2)? Out of scope for v1.
- Do `Claim` n-ary nodes and `Observation` n-ary nodes overlap when a stat is attributed? → allow a Claim whose
  `about` points at an Observation n-ary node.
- Whether recurring invented roles (`changeRate`, comparison `baseline`) should promote into the seed
  once consolidation shows them common across articles, or stay flat invented roles.
