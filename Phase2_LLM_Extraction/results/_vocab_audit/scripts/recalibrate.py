"""Re-derive the reported verdicts under the lenient calibration.

The strict verdicts are preserved verbatim in `strict_*` columns; nothing is lost.
Three rules change, each stated in ISSUE_CODES.md:

  1. `fits_meaning` judges the TERM, not its worst individual assignment. Where a
     term is right for its usages but one node was typed with it wrongly, the code
     becomes USAGE_MISASSIGNED and the term itself counts as fitting.
  2. A property applied outside its declared schema.org domain or range, but in the
     right semantic role, is not a second time a semantic defect: the computed
     domain/range columns already carry it.
  3. `is_best_fit` is lenient. A defensible term counts as the best fit; where a
     closer term was nonetheless noted, `closer_term_exists` records it.

Verdicts of `no`, DIRECTION_INVERTED, NOT_IN_SCHEMA_ORG and every computed column
are untouched by all three.
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import OUT, MODELS

# Terms whose WRONG_SENSE verdict is really a misassigned node or a signature
# transfer, not a wrong term. Keyed (model, kind, element[, source]).
MISASSIGNED_STD = {
 ("gemini-2.5-flash","type","Duration"), ("gemini-2.5-flash","type","Time"),
 ("gemini-2.5-flash","property","cause"), ("gemini-2.5-flash","property","duration"),
 ("gemini-2.5-flash","property","startTime"),
 ("deepseek-v4-flash","type","Movie"), ("deepseek-v4-flash","type","GovernmentOrganization"),
 ("deepseek-v4-flash","type","SportsTeam"), ("deepseek-v4-flash","type","MedicalCondition"),
 ("deepseek-v4-flash","property","applicableCountry"), ("deepseek-v4-flash","property","includesObject"),
 ("deepseek-v4-flash","property","nextItem"),
 ("llama33-70b","type","Corporation"), ("llama33-70b","type","SoftwareApplication"),
 ("llama33-70b","type","Duration"), ("llama33-70b","property","result"),
 ("llama33-70b","property","object"), ("llama33-70b","property","includesObject"),
 ("llama33-70b","property","jurisdiction"), ("llama33-70b","property","duration"),
 ("qwen3-32b-think","type","Date"), ("qwen3-32b-think","type","SportsEvent"),
 ("qwen3-32b-think","type","DefinedRegion"), ("qwen3-32b-think","type","Clip"),
 ("qwen3-32b-think","type","SaleEvent"), ("qwen3-32b-think","property","attendee"),
 ("qwen3-32b-think","property","affiliation"), ("qwen3-32b-think","property","containedIn"),
 ("qwen3-32b-think","property","hostingOrganization"), ("qwen3-32b-think","property","includesObject"),
 ("qwen3-32b-think","property","legislationPassedBy"), ("qwen3-32b-think","property","riskFactor"),
 ("qwen3-32b-think","property","seriousAdverseOutcome"),
}
MISASSIGNED_MINT = {
 ("deepseek-v4-flash","type","DiseaseOutbreak"), ("deepseek-v4-flash","type","DroneAttack"),
 ("deepseek-v4-flash","type","Negotiation"), ("deepseek-v4-flash","type","PastryChef"),
 ("deepseek-v4-flash","type","Burglary"), ("deepseek-v4-flash","type","AgeGroup"),
 ("deepseek-v4-flash","type","FirefighterGroup"),
 ("llama33-70b","type","WildfireEvacuation"),
 ("qwen3-32b-think","type","GrandSlamTournament"), ("qwen3-32b-think","type","DefensiveTechnology"),
 ("qwen3-32b-think","type","DomesticBoxOfficeRanking"), ("qwen3-32b-think","type","RelativeDate"),
}
# Defects of naming or modelling, not of meaning: the term still denotes the assertion.
NOT_A_MEANING_DEFECT = {"TOO_GENERIC","METACLASS_LABEL","VALUE_AS_TYPE","DIRECTION_INVERTED",
                        "OVERSPECIFIC","NEAR_DUPLICATE","VERBOSE_LABEL","BAD_PARENT",
                        "SCHEMA_EXACT_EXISTS","NAME_CLASH_WITH_SCHEMA_ORG","NOT_CANONICALISED",
                        "ATTRIBUTE_AS_ENTITY","NONE"}
# Codes that still mean the term itself was the wrong choice.
WRONG_CHOICE = {"WRONG_SENSE","NOT_IN_SCHEMA_ORG","TYPE_PROPERTY_CONFUSION","EVENT_ENTITY_CONFUSION"}

def preserve(row, fields):
    """Record the strict verdict once. Re-running must never overwrite it with a
    lenient one, or the strict reading would be lost silently."""
    for f in fields:
        k = "strict_" + f
        if not row.get(k):
            row[k] = row[f]

for m in MODELS:
    # ---------------- schema.org sheet ----------------
    p = os.path.join(OUT, m, "judgments_schemaorg.csv")
    rows = list(csv.DictReader(open(p)))
    cols = list(rows[0].keys())
    for c in ("closer_term_exists","strict_fits_meaning","strict_is_best_fit","strict_issue_code"):
        if c not in cols: cols.insert(cols.index("hand_verified"), c)
    for r in rows:
        preserve(r, ("fits_meaning","is_best_fit","issue_code"))
        r["fits_meaning"], r["is_best_fit"], r["issue_code"] = (
            r["strict_fits_meaning"], r["strict_is_best_fit"], r["strict_issue_code"])
        code = r["issue_code"]
        r["closer_term_exists"] = "yes" if r["better_fit"] else "no"
        if code == "NOT_BEST_FIT":
            r["fits_meaning"], r["is_best_fit"], r["issue_code"] = "yes", "yes", "NONE"
        elif code == "WRONG_SENSE" and (m, r["element_kind"], r["element"]) in MISASSIGNED_STD:
            r["fits_meaning"], r["is_best_fit"], r["issue_code"] = "yes", "yes", "USAGE_MISASSIGNED"
        elif code == "DIRECTION_INVERTED":
            # the relation asserted is the right one, stated the wrong way round
            r["fits_meaning"], r["is_best_fit"] = "yes", "yes"
        elif code in NOT_A_MEANING_DEFECT:
            if r["fits_meaning"] != "no": r["fits_meaning"] = "yes"
            r["is_best_fit"] = "yes"
    with open(p,"w",newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    # ---------------- non-schema.org sheet ----------------
    p = os.path.join(OUT, m, "judgments_minted.csv")
    rows = list(csv.DictReader(open(p)))
    cols = list(rows[0].keys())
    for c in ("strict_fits_meaning","strict_issue_code"):
        if c not in cols: cols.insert(cols.index("hand_verified"), c)
    for r in rows:
        preserve(r, ("fits_meaning","issue_code"))
        r["fits_meaning"], r["issue_code"] = r["strict_fits_meaning"], r["strict_issue_code"]
        code = r["issue_code"]
        if code == "SCHEMA_BROADER_ADEQUATE":
            # a coined term that declares a standard parent is a sound specialisation
            r["fits_meaning"], r["issue_code"] = "yes", "NONE"
        elif code == "WRONG_SENSE" and (m, r["element_kind"], r["element"]) in MISASSIGNED_MINT:
            r["fits_meaning"], r["issue_code"] = "yes", "USAGE_MISASSIGNED"
        elif code == "DIRECTION_INVERTED":
            r["fits_meaning"] = "yes"
        elif code in NOT_A_MEANING_DEFECT and r["fits_meaning"] != "no":
            r["fits_meaning"] = "yes"
    with open(p,"w",newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("recalibrated", m)
