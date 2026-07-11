"""Alias-aware categorical answer matching for higher-order reasoning tasks.

Some reasoning benchmarks ask for a short *categorical* answer (e.g.
``"Valid"``/``"Invalid"``, ``"Breach"``/``"No Breach"``, ``"Yes"``/``"No"``)
where a single gold label may surface in several accepted forms.  Scoring a
free-form model generation against such a label raises two problems:

1. **Extraction** -- the model usually emits a chain of thought before the
   answer, so the generation must be reduced to the categorical label it
   finishes with.
2. **Alias acceptance** -- the gold answer ships with an *accepted-alias
   set*; a prediction is correct if it matches *any* accepted alias after
   normalization.

``alias_match`` combines both steps.  It adapts the higher-order-reasoning
evaluation contract of *HOLMES: Evaluating Higher-Order Logical Reasoning
in LLMs* (arXiv:2606.23238) to lm-evaluation-harness.  That contract is the
union of two existing harness patterns -- legalbench's ``exact_match`` over
an accepted-answer set and mmlu_pro's regex answer extraction -- folded into
a single scorer for categorical (non multiple-choice) answers.

Results are returned as ``{"alias_match": fraction}`` in ``[0, 1]``, matching
the harness convention used by ``exact_match``.  Register the metric on a
``generate_until`` task with ``doc_to_target`` returning the accepted-alias
list (so ``multiple_target`` is set); no separate regex filter is required.
"""

import re


# Articles to drop during normalization.
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
# Punctuation to drop during normalization (keep word chars, spaces, hyphens).
_PUNCT_RE = re.compile(r"[^\w\s-]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Phrasings that commonly introduce the final answer in a chain of thought.
# Ordered most-specific first; within a pattern, the *last* match wins.  Each
# captures up to the end of the sentence/line so a trailing rationale is not
# pulled into the answer.
_ANSWER_PATTERNS = (
    re.compile(r"\bthe answer is\s*:?\s*([^.\n]+)", flags=re.IGNORECASE),
    re.compile(r"\banswer is\s*:?\s*([^.\n]+)", flags=re.IGNORECASE),
    re.compile(r"\bfinal answer\s*[:=]?\s*([^.\n]+)", flags=re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]\s*([^.\n]+)", flags=re.IGNORECASE),
)


def normalize_label(text: str) -> str:
    """Lowercase, drop articles and punctuation, and collapse whitespace.

    Lets surface variants such as ``"A Valid Contract"`` and
    ``"valid contract"`` compare equal.
    """
    if text is None:
        return ""
    text = _ARTICLE_RE.sub(" ", str(text))
    text = _PUNCT_RE.sub(" ", text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


def extract_answer(text: str) -> str:
    """Pull the categorical answer out of a free-form generation.

    Looks for explicit ``answer is X`` / ``answer: X`` / ``final answer: X``
    cues (the mmlu_pro-style extraction contract) and falls back to the last
    non-empty line.  Returns the empty string for empty input.
    """
    if not text:
        return ""
    text = str(text)

    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def alias_match(predictions, references, **kwargs):
    """Score predictions against an accepted-alias set.

    Each prediction is extracted and normalized; each reference (a gold
    alias) is normalized.  A prediction is correct when its normalized form
    equals the normalized reference.

    ``ConfigurableTask.process_results`` calls this metric once per accepted
    alias in the multiple-target path (``references=[alias]``,
    ``predictions=[generation]``) and credits the document if *any* alias
    matches; the per-pair mean reported here is therefore ``0.0`` or ``1.0``
    in that path.  When called with equal-length aligned lists, the fraction
    of matching pairs is returned.

    Args:
        predictions: iterable of model generations.
        references: iterable of gold aliases, aligned 1:1 with
            ``predictions`` when the lengths match.

    Returns:
        ``{"alias_match": fraction_correct}``
    """
    predictions = list(predictions)
    references = list(references)

    if len(predictions) == 0:
        return {"alias_match": 0.0}

    pred_norm = [normalize_label(extract_answer(p)) for p in predictions]
    ref_norm = [normalize_label(r) for r in references]

    if len(pred_norm) == len(ref_norm):
        scores = [
            1.0 if p == r else 0.0 for p, r in zip(pred_norm, ref_norm, strict=True)
        ]
    else:
        # Defensive: a single prediction broadcast against several aliases.
        ref_set = set(ref_norm)
        scores = [1.0 if p in ref_set else 0.0 for p in pred_norm]

    return {"alias_match": sum(scores) / len(scores)}
