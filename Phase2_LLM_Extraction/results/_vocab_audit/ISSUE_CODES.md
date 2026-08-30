# Vocabulary audit codebook

One `issue_code` per judged element. Codes are mutually exclusive; pick the most
specific one that applies. `NONE` means the choice is sound.

| code | meaning |
|---|---|
| `NONE` | No issue: the term fits and no better choice was available. |
| `SCHEMA_EXACT_EXISTS` | A schema.org term with the same meaning exists; minting was unnecessary. |
| `SCHEMA_BROADER_ADEQUATE` | No exact schema.org equivalent, but an existing broader schema.org term would have been adequate. |
| `USAGE_MISASSIGNED` | The term is the right term; the defect is in one individual assignment, not in the choice. Typically a class used correctly for most of its nodes and wrongly for one, or a property used in the right semantic role but outside its declared domain or range. |
| `WRONG_SENSE` | The term exists but the sense it carries in schema.org is not the sense used here. |
| `NOT_IN_SCHEMA_ORG` | Labelled `schema.org` but the term does not exist in the schema.org vocabulary. |
| `DIRECTION_INVERTED` | The property is asserted in the inverse of its defined direction. |
| `OVERSPECIFIC` | Term is bound to this article's instance or topic and will not generalise. |
| `TOO_GENERIC` | Term is so broad it carries almost no meaning. |
| `NEAR_DUPLICATE` | Near-synonym of another term in the same run's vocabulary (a consolidation miss). |
| `TYPE_PROPERTY_CONFUSION` | A type is used where a property is needed, or a property where a type is needed. |
| `VALUE_AS_TYPE` | A measurement, quantity or literal is modelled as a node type. |
| `EVENT_ENTITY_CONFUSION` | An event is modelled as an entity type, or an entity as an event type. |
| `ATTRIBUTE_AS_ENTITY` | An attribute, state or role is modelled as a node type in its own right. |
| `VERBOSE_LABEL` | Sentence-like or multi-concept label rather than a reusable term. |
| `BAD_PARENT` | The declared parent type/property is wrong or uninformative. |
| `METACLASS_LABEL` | A class named `...Type` / `...Category` is used to type instances directly, so the class stands for a category rather than for the things themselves. |
| `NAME_CLASH_WITH_SCHEMA_ORG` | The coined label is spelled exactly like an existing schema.org term whose sense is unrelated, so the graph now carries two meanings under one name. |
| `NOT_CANONICALISED` | Staged-arm hub type that never went through schema.org matching and has no other defect. Reserved for terms whose *only* problem is that they were never matched; where a more specific defect applies, that code is used instead and the `source` column still records the term as `never_canonicalised`. |

## Column semantics

**`judgments_schemaorg.csv`** (elements the run tagged `property_source`/`type_source = schema.org`)
- `fits_meaning` — `yes` / `partial` / `no`: does the schema.org term denote what the text asserts?
- `is_best_fit` — `yes` / `no`: is it the best available schema.org term?
- `better_fit` — a closer schema.org term, where one was noted. Populated independently of
  `is_best_fit`, so it can hold a suggestion even where the chosen term is judged acceptable.
- `closer_term_exists` — `yes` where `better_fit` is populated. This is the strict reading of
  "best fit" and is kept separate so either calibration can be reported.

**`judgments_minted.csv`** (everything else)
- `source` — `minted` (deliberately coined, carries a vocab id and definition) vs
  `never_canonicalised` (staged-arm hub types that never entered schema.org matching).
- `fits_meaning` — `yes` / `partial` / `no`: does the coined term denote what the text asserts?
- `schemaorg_alternative` — `exact` / `broader` / `none`: could schema.org have been used instead?
- `suggested_schemaorg` — the term that could have been used, when not `none`.

Domain and range compliance is deliberately **not** an issue code: it is counted
exactly, per usage, in the `domain_ok` / `domain_bad` / `range_ok` / `range_bad`
columns of the schema.org sheet, so it never competes with a semantic code for the
one code slot.

## Calibration

The reported verdicts use a **lenient** calibration, on three rules:

1. **`fits_meaning` judges the term, not its worst assignment.** A class used correctly for
   nine of its ten nodes and wrongly for the tenth is a right term with one misassigned node,
   which is `USAGE_MISASSIGNED`, not a wrong term. This audit is about vocabulary decisions;
   whether an individual node was typed correctly is what grounding measures.
2. **A signature violation is not also counted as a semantic defect.** A property used in the
   right semantic role but outside its declared domain or range is recorded once, in the
   computed `domain_ok` / `domain_bad` / `range_ok` / `range_bad` columns.
3. **`is_best_fit` is lenient.** A defensible term counts as the best fit; a closer term that
   is nicer but not materially different (`QuantitativeValue` for `Number`, `attendee` for the
   superseded `attendees`) leaves the verdict at `yes` and is recorded in `closer_term_exists`.

Untouched by all three: verdicts of `no`, `DIRECTION_INVERTED`, `NOT_IN_SCHEMA_ORG`, and every
computed column. A term that does not denote the assertion at all is still `no` however
leniently it is read.

The **strict** verdicts are preserved verbatim alongside, in `strict_fits_meaning`,
`strict_is_best_fit` and `strict_issue_code`, so the audit can be re-reported under either
calibration without re-judging. Under the strict reading, `fits_meaning = yes` falls from
77-87% to 49-66% on the schema.org sheet; the difference is entirely the three rules
above, not a change of opinion about any term. `scripts/recalibrate.py` derives one from the
other and lists every hand-classified exception.

`hand_verified` / `hand_notes` are left blank for manual review.
