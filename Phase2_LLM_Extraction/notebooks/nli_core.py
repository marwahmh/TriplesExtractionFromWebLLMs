"""Entailment verification — the scoring core shared by 9c (local, interactive) and the cluster
batch job (cluster/nli_verify.py).

There is exactly one implementation of the premise construction and the scoring here, deliberately:
if the notebook and the cluster job split a sentence differently or chunk a document differently,
they produce different probabilities for the same assertion, and the agreement numbers 9c reports
become meaningless. Import this from both; never re-implement either half.

Nothing in here touches IPython, notebook globals, or the results/ layout — it takes text and
assertions and returns scores, so it runs identically under Jupyter, a SLURM job, or pytest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# The verified default. See 9c section 2 for why the probe is mandatory and why FactCG's HF export
# is not usable (it loads cleanly with an untrained head and scores everything ~0.9).
DEFAULT_MODEL = "lytang/MiniCheck-DeBERTa-v3-Large"

# Verbaliser: turns a graph edge into a sentence an entailment model can actually judge. Open
# weights, run locally — the point of this whole notebook is a verification path that does not
# depend on a closed model, and that has to include the verbalisation step.
DEFAULT_VERBALISER = "Qwen/Qwen2.5-3B-Instruct"

VERBALISE_SYSTEM = (
    "You rewrite one edge of a knowledge graph as a single plain English sentence.\n"
    "Rules:\n"
    "- State exactly the given fact, using only the given subject, relation and object.\n"
    "- Turn the relation into a VERB PHRASE. Never copy the relation name verbatim: write "
    "\"was caused by\", not \"source\"; \"took place in\", not \"location\".\n"
    "- Do not add any information that is not in the input.\n"
    "- Do not hedge, explain, or comment. Output the sentence and nothing else.\n"
    "- Keep the subject's and object's own wording, but you may add function words and inflect "
    "verbs so it reads as English."
)

# A 3B model follows these far better than it follows the rules above: without exemplars it echoes
# the relation name roughly half the time ("risk of an outbreak source weather conditions"), which
# produces exactly the entity-bag hypotheses the entailment model cannot judge. The examples cover
# the awkward property shapes in this graph — bare nominals (`source`, `location`, `about`) and
# measurement edges (`hasQuantity`, `hasIncrease`).
VERBALISE_SHOTS = [
    ("Subject: Excess deaths in France\nRelation: source\nObject: European heatwave in June\nSentence:",
     "Excess deaths in France were caused by the European heatwave in June."),
    ("Subject: red heat alert\nRelation: applicable location\nObject: half the nation\nSentence:",
     "A red heat alert applied to half the nation."),
    ("Subject: Drowning deaths in France\nRelation: temporal\nObject: 18 June\nSentence:",
     "Drowning deaths in France occurred on 18 June."),
    ("Subject: Wildfires in France this summer\nRelation: has quantity\nObject: nearly 7,000 fires\nSentence:",
     "There were nearly 7,000 wildfires in France this summer."),
    ("Subject: European heatwave in June\nRelation: about\nObject: Europe\nSentence:",
     "The European heatwave in June affected Europe."),
]


# nb9 and nb9b each write a flat judgment log; both are keyed by the same per-assertion `id`,
# because both derive it from the same graph.
JUDGMENT_LOGS = {"evidence": "grounding_judgments.json",
                 "fulltext": "grounding_fulltext_judgments.json"}
VERDICTS_NAME = "nli_verdicts.json"


# --------------------------------------------------------------------------------------------
# Reading the claims to verify. Shared so the notebook and the cluster job score the SAME
# assertion strings for the same ids — the verdict files are keyed by id, so as long as both read
# the logs the same way, either side can produce them and the other can consume them.
# --------------------------------------------------------------------------------------------
def load_claims(results_dir, article_id, track="e2e", sources=("evidence", "fulltext")):
    """One row per (kind, id) for an article, unioned across the requested judgment logs.

    Two classes of row carry `<source>_judged = False`: relations orphaned by a dropped endpoint,
    and (nb9 only) assertions whose evidence failed the deterministic quote check and so never
    reached the judge. They are still scored — the entailment model has no such gate, and its
    verdict on them is informative — but 9c holds them out of the agreement statistics, because
    comparing against a default is not comparing against a decision.
    """
    results_dir = Path(results_dir)
    rows = {}
    for src in sources:
        p = results_dir / article_id / track / JUDGMENT_LOGS[src]
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8")):
            key = (r["kind"], r["id"])
            row = rows.setdefault(key, {"article_id": article_id, "kind": r["kind"],
                                        "id": r["id"], "assertion": r["assertion"],
                                        # the parts the verbaliser needs to build a sentence;
                                        # None on nodes, which carry their label in `assertion`
                                        "subject": r.get("subject"), "property": r.get("property"),
                                        "object": r.get("object")})
            row[f"{src}_grounded"] = r.get("grounded")
            row[f"{src}_supported"] = r.get("supported")
            row[f"{src}_judged"] = (r.get("decision") != "unjudged") and not r.get("orphaned")
            row[f"{src}_failure"] = r.get("failure_type")
            row[f"{src}_note"] = r.get("justification")
            if src == "fulltext":
                row["judge_quote"] = r.get("judge_quote")
    return list(rows.values())


def has_claims(results_dir, article_id, track="e2e", sources=("evidence", "fulltext")):
    return any((Path(results_dir) / article_id / track / JUDGMENT_LOGS[s]).exists()
               for s in sources)


def write_verdicts(results_dir, article_id, track, scores_by_id, config):
    """Verdict file the notebook reads back. `config` records how the scores were produced, so a
    run scored on the cluster and analysed locally can prove the two used the same settings."""
    path = Path(results_dir) / article_id / track / VERDICTS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    # encoding is explicit everywhere in this module on purpose: the cluster's compute nodes run
    # under an ASCII locale, so a bare write_text() dies on the first accented name in an article
    # ("Stephanie" is fine, "Stéphanie" is not). macOS defaults to UTF-8 and hides this entirely.
    path.write_text(json.dumps({**config, "scores": scores_by_id}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------
# Premise construction. Dependency-free on purpose: no nltk/spacy download on a compute node that
# reaches the internet only through a proxy.
# --------------------------------------------------------------------------------------------
# The lookbehinds must include the period: they are evaluated at the split point, which sits AFTER
# the ".", so `(?<!\bMr)` would test "r." and never fire, while `(?<!\bMr\.)` tests "Mr." and does.
_ABBREV = (r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bProf\.)"
           r"(?<!\bSt\.)(?<!\bNo\.)(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)")
_SPLIT = re.compile(_ABBREV + r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z\"'(\[])")


def sentences(text):
    """Split into sentences. A split that is slightly wrong costs a window boundary, never a
    verdict, because every window overlaps its neighbours."""
    out = [s.strip() for s in _SPLIT.split(text) if s.strip()]
    return out or [text.strip()]


def sentence_windows(sents, k):
    """Every window of k consecutive sentences, stride 1, so support that spans a sentence boundary
    (a figure in one sentence, its subject in the next) is seen whole by at least one window."""
    if len(sents) <= k:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + k]) for i in range(len(sents) - k + 1)]


def doc_chunks(text, tokenizer, max_len=512, reserve=128):
    """Overlapping chunks that fit the model's budget, cut on sentence boundaries. The overlap is
    one sentence, so a claim supported across a chunk seam is still seen whole by one chunk."""
    budget = max_len - reserve                     # leave room for the hypothesis
    sents = sentences(text)
    lens = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sents]
    chunks, start = [], 0
    while start < len(sents):
        end, total = start, 0
        while end < len(sents) and (end == start or total + lens[end] <= budget):
            total += lens[end]
            end += 1                               # always take >=1 sentence, even if oversized
        chunks.append(" ".join(sents[start:end]))
        if end >= len(sents):
            break
        start = max(start + 1, end - 1)            # one-sentence overlap, but never stall
    return chunks


def pick_device(prefer=None):
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------------------------
# The verifier
# --------------------------------------------------------------------------------------------
PROBE_DOC = (
    "France has said it recorded an increase of more than 2,000 deaths during the last week of a "
    "record-breaking European heatwave in June. Deaths rose 29% in the last week of June compared "
    "with the week before, with French health minister Stephanie Rist adding there had been a clear "
    "increase in deaths among those over 45. France saw its hottest day ever on average country-wide "
    "on 24 June, with temperatures hitting almost 41C in Paris and half the nation placed under a "
    "red heat alert. Copernicus said Europe is the fastest-warming continent."
)
PROBE_TRUE = [
    "France recorded more than 2,000 excess deaths during the last week of June.",
    "Deaths in France rose 29% in the last week of June compared with the week before.",
    "France saw its hottest country-wide day on record on 24 June.",
    "Temperatures in Paris reached almost 41C.",
    "Stephanie Rist is the French health minister.",
    "Europe is warming faster than any other continent.",
]
PROBE_FALSE = [
    "France recorded no excess deaths during the June heatwave.",
    "The mayor of Paris resigned on Thursday.",
    "Temperatures in Paris stayed below 30C throughout June.",
    "Japan recorded its coldest June on record.",
    "The French health minister is Emmanuel Macron.",
    "Deaths in France fell 29% in the last week of June.",
]


def auc(pos, neg):
    """P(a random supported claim outranks a random unsupported one). Rank-based, so a single badly
    scored outlier nudges it instead of deciding the comparison — which a min/max margin criterion
    does, and it will happily pick a label orientation out of pure noise."""
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


class Verifier:
    """A loaded entailment model plus the orientation the probe established.

    `supported_idx` is DERIVED, never assumed: these checkpoints ship placeholder label names
    (LABEL_0/LABEL_1) and a checkpoint can load with no missing-key warning and still carry an
    untrained head, in which case it scores every claim alike. `probe()` refuses such a model.
    """

    def __init__(self, model_name=DEFAULT_MODEL, device=None, max_len=512, batch_size=8):
        self.model_name = model_name
        self.device = pick_device(device)
        self.max_len = max_len
        self.batch_size = batch_size
        self._bs = batch_size                      # current size; lowered in place after an OOM
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.supported_idx = 1                     # provisional until probe() runs
        self.probe_auc = None

    @property
    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def free_gpu(self):
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def _raw_probs(self, premises, hypotheses):
        """P over the 2-way head per pair. The document goes FIRST: `only_first` truncation drops
        the tail of the first sequence when a pair overflows, which must be the document — passing
        the claim first makes transformers raise "Sequence to truncate too short".

        An out-of-memory error is recoverable, not fatal: how many 512-token DeBERTa pairs fit
        depends on the card and on what else holds memory right now, so the batch is halved and
        retried down to a single pair. The reduced size sticks, so the run pays that cost once.
        """
        out, i = [], 0
        while i < len(premises):
            bs = self._bs
            try:
                enc = self.tok(premises[i:i + bs], hypotheses[i:i + bs], truncation="only_first",
                               max_length=self.max_len, padding=True, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.float()
                out += torch.softmax(logits, dim=-1).cpu().tolist()
                i += bs
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if "out of memory" not in str(exc).lower() or bs <= 1:
                    raise
                self._bs = max(1, bs // 2)
                self.free_gpu()
                print(f"  [out of memory at batch {bs} -> retrying at {self._bs}]", flush=True)
        return out

    def entail_probs(self, premises, hypotheses):
        """P(supported) per (premise, hypothesis) pair, using the probe-established label index."""
        return [p[self.supported_idx] for p in self._raw_probs(premises, hypotheses)]

    def probe(self, threshold=0.5, verbose=True, min_auc=0.9):
        """Derive the label index from the model's own ranking on claims whose answers are not in
        dispute, and refuse to continue unless the separation is real. -> dict of probe results."""
        claims = PROBE_TRUE + PROBE_FALSE
        gold = [True] * len(PROBE_TRUE) + [False] * len(PROBE_FALSE)
        probs = self._raw_probs([PROBE_DOC] * len(claims), claims)
        aucs = {i: auc([p[i] for p, g in zip(probs, gold) if g],
                       [p[i] for p, g in zip(probs, gold) if not g]) for i in (0, 1)}
        self.supported_idx = max(aucs, key=aucs.get)
        self.probe_auc = aucs[self.supported_idx]
        scores = [p[self.supported_idx] for p in probs]

        if verbose:
            print(f"label index for 'supported' = {self.supported_idx}   "
                  f"(AUC {aucs[1]:.3f} for idx 1, {aucs[0]:.3f} for idx 0)")
            for c, g, p in zip(claims, gold, scores):
                bad = (g and p < threshold) or (not g and p >= threshold)
                print(f"  expect {str(g):5}  P(supported)={p:.3f}  {c[:60]}"
                      f"{'   <-- MISRANKED' if bad else ''}")

        if self.probe_auc < min_auc:
            raise AssertionError(
                f"{self.model_name} ranks supported vs unsupported claims at AUC "
                f"{self.probe_auc:.3f} on unambiguous probe data — it is not a usable verifier. The "
                "likeliest cause is a checkpoint whose classification head did not really load (it "
                "will NOT warn you: an untrained head loads 'cleanly' and scores everything ~0.9).")
        if verbose:
            print(f"\nprobe AUC {self.probe_auc:.3f} — verifier usable")
        return {"supported_idx": self.supported_idx, "auc": self.probe_auc,
                "scores": scores, "gold": gold}

    def score_claims(self, article_text, assertions, window=3, sentence_pass=True, threshold=0.5):
        """Score every assertion against one article.

        Two granularities, on purpose:
          * document pass  — max over token-budget chunks. Mirrors "is this supported ANYWHERE in
            the article?", so it is like-for-like with nb9b's full-article judge. This is the
            headline verdict.
          * localisation pass — max over sliding sentence windows, keeping the argmax window. That
            is the model's own answer to "which sentence supports this?", directly comparable to
            nb9b's judge_quote, and it costs roughly 10x the forward passes of the document pass.

        -> list of dicts aligned with `assertions`.
        """
        sents = sentences(article_text)
        chunks = doc_chunks(article_text, self.tok, self.max_len)

        doc_best = [0.0] * len(assertions)
        for ch in chunks:
            for i, p in enumerate(self.entail_probs([ch] * len(assertions), assertions)):
                doc_best[i] = max(doc_best[i], p)

        loc_best = [None] * len(assertions)
        loc_win = [None] * len(assertions)
        if sentence_pass:
            best = [-1.0] * len(assertions)
            for w in sentence_windows(sents, window):
                for i, p in enumerate(self.entail_probs([w] * len(assertions), assertions)):
                    if p > best[i]:
                        best[i], loc_win[i] = p, w
            loc_best = best

        return [{"nli_doc_score": round(doc_best[i], 4),
                 "nli_local_score": None if loc_best[i] is None else round(loc_best[i], 4),
                 "nli_best_window": loc_win[i],
                 "nli_supported": bool(doc_best[i] >= threshold)}
                for i in range(len(assertions))]


# --------------------------------------------------------------------------------------------
# Verbalisation
#
# Why this exists: the arrow notation nb9/nb9b hand their LLM judge
#     "Drowning deaths in France" (Event) --[temporal]--> "18 June" (Date)
# is unreadable to an entailment encoder, and the obvious mechanical fix ("subject property
# object") is worse than useless — it produces "Drowning deaths in France temporal 18 June", which
# is a bag of entities rather than a proposition. Measured on art1: the encoder scored a
# deliberately CORRUPTED triple (object swapped for an unrelated one) at 0.898 against 0.888 for
# the true one. It was matching topic, not judging support.
#
# So an edge has to become a real sentence first. That is done here with a small open-weight
# instruct model at greedy decoding: open so the verification path never touches a closed model,
# greedy so the same graph always yields the same hypothesis. The model supplies SURFACE FORM
# ONLY — it is never asked whether the claim is true; that stays with the entailment model.
#
# Never trust this step blindly either: `discrimination_check` below is the gate that proves the
# resulting hypotheses can actually be told apart from corrupted ones.
# --------------------------------------------------------------------------------------------
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def spaced_property(prop):
    """'forecastsTemperature' -> 'forecasts temperature'"""
    return _CAMEL.sub(" ", (prop or "").replace("_", " ")).lower().strip()


def strip_type(s):
    """'"high pressure" (WeatherSystem)' -> 'high pressure'"""
    if s is None:
        return ""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(s).strip()).strip('"').strip()


def template_hypothesis(row):
    """The LLM-free fallback: `subject property object`. Kept because every hypothesis needs SOME
    value even if the verbaliser is unavailable — but see the note above, it does not discriminate
    well on its own. Used only when `verbalise=False`."""
    if row.get("kind") == "relation" and row.get("subject") is not None:
        return (f"{strip_type(row.get('subject'))} {spaced_property(row.get('property'))} "
                f"{strip_type(row.get('object'))}.").strip()
    m = re.search(r'of type "([^"]+)" exists, described as "([^"]+)"', row.get("assertion", ""))
    if m:
        return f"{m.group(2)} is a {m.group(1).lower()}."
    return row.get("assertion", "")


class Verbaliser:
    """Small open-weight instruct model that rewrites an edge as one sentence. Greedy, batched."""

    def __init__(self, model_name=DEFAULT_VERBALISER, device=None, batch_size=16, max_new_tokens=48):
        from transformers import AutoModelForCausalLM        # local: only needed when verbalising
        self.model_name = model_name
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        # Qwen ships temperature/top_p/top_k in its generation_config; with do_sample=False they are
        # unused but transformers warns on every call. Clear them so greedy really means greedy.
        for k in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, k, None)
        self.model.generation_config.do_sample = False

    @staticmethod
    def _prompt(row):
        if row.get("kind") == "relation" and row.get("subject") is not None:
            return (f"Subject: {strip_type(row.get('subject'))}\n"
                    f"Relation: {spaced_property(row.get('property'))}\n"
                    f"Object: {strip_type(row.get('object'))}\n"
                    "Sentence:")
        m = re.search(r'of type "([^"]+)" exists, described as "([^"]+)"', row.get("assertion", ""))
        if m:
            return (f"Subject: {m.group(2)}\nRelation: is a\nObject: {m.group(1)}\nSentence:")
        return f"Rewrite as one sentence: {row.get('assertion', '')}\nSentence:"

    @staticmethod
    def _clean(text, row):
        """One sentence, no preamble, no quotes. Falls back to the template if the model rambles
        or returns something empty — a bad rewrite must never silently become an empty hypothesis."""
        s = (text or "").strip().split("\n")[0].strip()
        s = re.sub(r'^(sentence|output|answer)\s*:\s*', "", s, flags=re.I).strip()
        s = s.strip('"').strip()
        if len(s) < 3 or len(s) > 400:
            return template_hypothesis(row)
        return s if s.endswith((".", "!", "?")) else s + "."

    @torch.inference_mode()
    def verbalise(self, rows, verbose=False):
        """-> one sentence per row, in order."""
        out = []
        for i in range(0, len(rows), self.batch_size):
            batch = rows[i:i + self.batch_size]
            shots = [m for u, a in VERBALISE_SHOTS
                     for m in ({"role": "user", "content": u}, {"role": "assistant", "content": a})]
            texts = [self.tok.apply_chat_template(
                        [{"role": "system", "content": VERBALISE_SYSTEM}, *shots,
                         {"role": "user", "content": self._prompt(r)}],
                        tokenize=False, add_generation_prompt=True) for r in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
            for r, g in zip(batch, gen):
                new = g[enc["input_ids"].shape[1]:]
                out.append(self._clean(self.tok.decode(new, skip_special_tokens=True), r))
            if verbose:
                print(f"  verbalised {min(i + self.batch_size, len(rows))}/{len(rows)}", flush=True)
        return out


def corrupt_rows(rows, seed=0):
    """Object-swapped copies of `rows`: a real subject+property with ANOTHER row's object. The
    article almost never supports these, and they keep the exact vocabulary of a real claim, so
    they test judgment rather than topical overlap."""
    import random
    rng = random.Random(seed)
    objs = [r.get("object") for r in rows if r.get("object")]
    out = []
    for r in rows:
        alts = [o for o in objs if strip_type(o) != strip_type(r.get("object"))]
        if not alts:
            continue
        out.append({**r, "object": rng.choice(alts)})
    return out


def discrimination_check(verifier, article_text, rows, hypotheses, corrupt_hypotheses,
                         threshold=0.5, min_auc=0.75, verbose=True):
    """THE GATE. Score real hypotheses against object-swapped ones on the same article.

    A verbalisation that scores both alike has produced hypotheses the encoder cannot judge, and
    every agreement number computed from them would be measuring topical overlap. This is exactly
    what the mechanical `subject property object` template does (AUC 0.70 on art1), so it is
    checked rather than assumed. Raises if the separation is too weak to build on.
    """
    chunks = doc_chunks(article_text, verifier.tok, verifier.max_len)

    def best(hyps):
        b = [0.0] * len(hyps)
        for ch in chunks:
            for i, p in enumerate(verifier.entail_probs([ch] * len(hyps), hyps)):
                b[i] = max(b[i], p)
        return b

    real, fake = best(hypotheses), best(corrupt_hypotheses)
    score = auc(real, fake)
    if verbose:
        print(f"  real     n={len(real):4} >={threshold}: {sum(x >= threshold for x in real)/len(real):6.1%}"
              f"  mean={sum(real)/len(real):.3f}")
        print(f"  corrupted n={len(fake):4} >={threshold}: {sum(x >= threshold for x in fake)/len(fake):6.1%}"
              f"  mean={sum(fake)/len(fake):.3f}")
        print(f"  discrimination AUC = {score:.3f}  (need >= {min_auc})")
    if score < min_auc:
        raise AssertionError(
            f"the verbalised hypotheses do not separate true claims from object-swapped ones "
            f"(AUC {score:.3f} < {min_auc}). The encoder is matching topic, not judging support, "
            "so any agreement number built on these would be meaningless. Fix the verbalisation "
            "before trusting anything downstream.")
    return {"auc": score, "real": real, "corrupt": fake}
