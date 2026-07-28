# Event-centric reification — design spec

**Status:** design, pre-implementation · **Decisions locked:** hybrid reification, schema.org-first grounding.

A second relational layer for the pipeline. Instead of only stating facts as binary
`[subject, property, object]` triples between entities, we introduce a small set of **hub nodes**
(events, observations, attributed claims) that gather the participants of *one* news fact under
*one* node via a **closed set of semantic roles**. The binary path (`6_relation_extraction` → `7`)
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

## 2. The model — three hub kinds

| Hub kind        | Mint one when…                                                         | schema.org base            |
|-----------------|------------------------------------------------------------------------|----------------------------|
| **Event**       | something happens: heatwave, wildfire, evacuation, forecast, alert     | `Event` subtree + minted   |
| **Observation** | a statistic / measurement / quantified change is stated                | `Observation`              |
| **Claim**       | a statement is attributed to a source ("X said Y", "ministry called it…")| `Claim` / `Statement`    |

`Observation` is a close schema.org fit for the excess-death numbers: it already defines
`observationAbout`, `measuredProperty`, `observationDate`, `marginOfError`, and inherits `value` /
`unitText` from `QuantitativeValue`.

**Hybrid rule (locked).** Reify only these three. Genuinely-binary facts stay as plain edges — either
entity→entity (`Paris partOf France`, `Rist ministerOf France`) or, when they qualify a hub,
hub→entity. Do **not** route simple binary relations through a hub.

**Binary-relations pass is MANDATORY** (added after the first art1 run, where the model produced
`binary_relations: []` and silently dropped 25 stable facts — all the geography, affiliations, and
titles the binary graph had). The prompt now requires an explicit second sweep with named categories
— geographic containment/part-of, organizational membership/affiliation, role/title, ownership/identity
— populating `binary_relations` **even when the entities also appear in a hub**. Each entry carries a
`property_source` (`schema.org` | `ex`) so minted binary properties render dashed like the binary graph.

---

## 3. Extensible role vocabulary (seed + invention)

Roles are the edges from a hub to its participants. The vocabulary is **open and extensible** — same
philosophy as the binary path (`6_relation_extraction` → `7`): the model invents the label that fits,
tags it new, gives it a definition and an optional parent, and a later consolidation pass folds it
into a growing `role_vocabulary` (mirroring `relation_vocabulary_consolidated.json`). It is **not** a
closed set.

The table below is the **seed** — a preferred starter vocabulary injected into the prompt. The model
should reuse a seed role when one fits and only mint a new one otherwise.

**Invention mechanics** (per role instance, mirroring nb6's property fields):
`role`, `role_source` (`"seed"` | `"schema.org"` | `"new"`), `definition` (required if new),
`parent` (optional seed/existing role, for hierarchy), plus the value (see below).

**The one discipline that keeps reification's benefit:** invented roles must stay **atomic and
reusable** — a single participant slot (`evacuees`, `warnsOf`, `burnedArea`), *never* a
sentence-fragment predicate that re-crams context (`notedIncreaseInDeathsForAgeGroup`). Context like
time / place / value goes on its *own* role edge, which is the whole point of the hub.

The seed is a small, **domain-neutral** backbone of thematic roles that apply to *any* article —
who/what/where/when/why/how. Domain specifics (`changeRate`, `reportedOn`, `evacuees`, `burnedArea`)
are deliberately **left out** so they surface as *invented* roles — that is the extensibility working.

| Seed role      | schema.org / ex:   | Generic sense — connects hub → …                     |
|----------------|--------------------|------------------------------------------------------|
| `agent`        | schema:agent       | who/what brings the hub about (the doer)             |
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

**After:** one hub, fully contextualized, predicate vocabulary collapsed to roles.

```json
{
  "id": "OBS1",
  "hub_kind": "Observation",
  "type": "ExcessMortalityStatistic",
  "type_source": "ex",
  "label": "France excess deaths, 22–28 June 2026",
  "roles": [
    {"role": "about",    "role_source": "schema.org", "value_text": "excess deaths"},
    {"role": "affected", "role_source": "seed", "value_ref": "E1"},                       // France
    {"role": "location", "role_source": "schema.org", "value_ref": "E46"},                // Paris region
    {"role": "time",     "role_source": "seed", "value_ref": "E72"},                      // 22 and 28 June
    {"role": "value",    "role_source": "schema.org", "value_ref": "E73"},                // 2,025
    {"role": "cause",    "role_source": "seed", "value_ref_hub": "EV1"},                  // June heatwave (hub)
    {"role": "source",   "role_source": "schema.org", "value_text": "Public Health France"},
    // INVENTED atomic roles (not in the generic seed), each parented to a seed role:
    {"role": "changeRate", "role_source": "new", "parent": "value",
     "definition": "Rate/percentage of change in the measured value.", "value_ref": "E26", "value_text": "+29%"},
    {"role": "reportedOn", "role_source": "new", "parent": "time",
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

## 5. Output schema — `results/<id>/reified_graph.json`

```jsonc
{
  "article_id": "art1",
  "reference_date": "2026-07-03",
  "entities": [ /* same id-stamped cards passed in (E1, E2, …) */ ],
  "hubs": [
    {
      "id": "EV1", "hub_kind": "Event", "type": "WeatherEvent", "type_source": "schema.org",
      "label": "June 2026 European heatwave", "is_main": true,
      "roles": [ {"role": "..", "role_source": "seed|schema.org|new", "parent": "..?",
                  "definition": "..?", "value_ref": "E..", "value_ref_hub": "..", "value_text": ".."} ],
      "evidence": ["…"], "confidence": 1.0
    }
    // Observation, Claim hubs …
  ],
  "binary_relations": [ {"subject_id","property","object_id","property_source","evidence"} ],
  "seed_roles": [ /* the starter vocabulary passed in */ ],
  "new_roles": [ /* roles the model invented this article: {role, parent, definition, count} —
                    feeds the cross-article consolidation, like relation_vocabulary */ ]
}
```

- `value_ref` → an entity id (`E7`); `value_ref_hub` → another hub id (`EV1`); `value_text` → a
  literal when neither node fits. Exactly one is primary per role instance (a `value_text` gloss may
  accompany a `value_ref`).
- `role_source`: `seed` (reused a seed role), `schema.org` (a seed role that maps to a schema.org
  property), or `new` (invented — then `definition` is required and `parent` optional).
- `new_roles` is collected post-hoc from the hubs so a later pass can consolidate the growing role
  vocabulary across articles.
- `is_main` flags the article's central event(s) for the backbone / layout.

---

## 6. Pipeline placement (both methods preserved)

| Notebook                         | Status   | Role                                              |
|----------------------------------|----------|---------------------------------------------------|
| `5_property_enrichment` / nb4    | shared   | produces `typed_deduped_entities_ext*.json`       |
| `6_relation_extraction` → `7`    | **unchanged** | binary-triple baseline                       |
| **`6b_event_reification`** (new) | new      | same entities in → `reified_graph.json`           |
| `8_kg_visualization`             | +mode    | render both graphs from one article for eyeballing|

Single-article demo cell + `SELECTED_ARTICLES` batch cell, mirroring nb6.

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
- **Fragmentation ratio** — binary triples collapsed per hub (the 4→1 above).
- **Downstream QA** (most persuasive) — a few questions per article ("how many excess deaths in
  France and when?") answered by traversal / an LLM over each graph; score faithfulness & completeness.

---

## 8. Open questions to revisit after v1

- Cross-article event linking (same heatwave across art1/art2)? Out of scope for v1.
- Do `Claim` hubs and `Observation` hubs overlap when a stat is attributed? → allow a Claim whose
  `about` points at an Observation hub.
- Whether recurring invented roles (`changeRate`, comparison `baseline`) should promote into the seed
  once consolidation shows them common across articles, or stay flat invented roles.
