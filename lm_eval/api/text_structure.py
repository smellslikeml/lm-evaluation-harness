"""Deterministic checkers for structured *text* outputs.

Implements the Summarization-domain rule-based evaluators of *StructTest:
Benchmarking LLMs' Reasoning through Compositional Structured Outputs*
(arXiv:2412.18011). StructTest scores a model by asking it to produce an output
that satisfies compositionally-specified *structural* instructions -- e.g. "give
exactly five bullet points, each at most ten words" -- and checking the result
with a deterministic, rule-based evaluator instead of matching a gold string.

The sibling :mod:`lm_eval.api.structured_output` covers the paper's *JSON*
constraints (key/type/regex over a parsed object). This module covers the
Summarization domain, whose constraints live in the free-form *text* itself:
bullet counts, per-bullet word limits, sentence counts, total length, and
required-keyword inclusion. Both share StructTest's two scoring views
(constraint-level fraction and prompt-level all-or-nothing), so
:func:`lm_eval.api.structured_output.score_response` can score a single response
against a mix of JSON and text constraints.

Everything here is stdlib-only (``re``) -- no NLTK, no learned estimators, no
external artifacts -- matching the harness-native, dependency-free posture of
the JSON validator. Sentence and bullet detection use simple deterministic
regexes rather than a parser, which is faithful to the paper's rule-based (not
model-based) evaluation and keeps scoring cheap and reproducible.
"""

import re
from typing import Any, Protocol


__all__ = [
    "TEXT_CONSTRAINT_TYPES",
    "check_text_constraint",
    "count_bullets",
    "count_words",
    "split_sentences",
]


class _ConstraintLike(Protocol):
    """Structural type for a constraint: a ``type`` string and a ``params`` map.

    :class:`lm_eval.api.structured_output.Constraint` satisfies this, so callers
    can pass the same dataclass used for JSON constraints without this module
    importing it (which would create a circular import).
    """

    type: str
    params: dict[str, Any]


# A bullet is a line that, after leading whitespace, opens with an unordered
# marker (-, *, +, bullet) or an ordered marker (``1.`` / ``1)``) followed by
# text. This mirrors how StructTest's summarization evaluator locates list items.
_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+\S")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")

# Split on sentence-final punctuation followed by whitespace or end-of-text.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")

# Words are runs of alphanumerics allowing internal apostrophes/hyphens, so
# "state-of-the-art" and "don't" each count as one word.
_WORD_RE = re.compile(r"[^\W_](?:[\w'-]*[^\W_])?")


# The constraint ``type`` values handled here (as opposed to the JSON checkers
# in :mod:`lm_eval.api.structured_output`). ``score_response`` uses this set to
# route each constraint to the raw text rather than the parsed JSON object.
TEXT_CONSTRAINT_TYPES = frozenset(
    {
        "num_bullets",
        "max_words_per_bullet",
        "num_sentences",
        "num_words",
        "keywords_present",
    }
)


def _bullet_lines(text: str) -> list[str]:
    """Return the lines of ``text`` that are formatted as bullets."""
    return [line for line in text.splitlines() if _BULLET_RE.match(line)]


def _bullet_body(line: str) -> str:
    """Strip the leading bullet marker from ``line`` and trim surrounding space."""
    return _BULLET_PREFIX_RE.sub("", line).strip()


def count_bullets(text: str) -> int:
    """Count the bullet-formatted lines in ``text``."""
    return len(_bullet_lines(text))


def count_words(text: str) -> int:
    """Count word tokens in ``text``."""
    return len(_WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into non-empty sentences on ``.``/``!``/``?`` boundaries."""
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _within(value: int, params: dict[str, Any]) -> bool:
    """Check ``value`` against optional ``n`` (exact), ``min``, and ``max`` bounds.

    An empty bound set is treated as unsatisfiable so that a mistyped spec can
    never inflate a score (matching the JSON validator's fail-closed policy).
    """
    if not any(k in params for k in ("n", "min", "max")):
        return False
    return all(
        (
            "n" not in params or value == int(params["n"]),
            "min" not in params or value >= int(params["min"]),
            "max" not in params or value <= int(params["max"]),
        )
    )


def check_text_constraint(text: str, constraint: _ConstraintLike) -> bool:
    """Return whether ``text`` satisfies a single text-structure ``constraint``.

    Supported ``constraint.type`` values (all read ``constraint.params``):

    - ``num_bullets``          -- bullet count satisfies ``n``/``min``/``max``.
    - ``max_words_per_bullet`` -- every bullet body has ``<= params["max"]``
                                  words; false when there are no bullets.
    - ``num_sentences``        -- sentence count satisfies ``n``/``min``/``max``.
    - ``num_words``            -- total word count satisfies ``n``/``min``/``max``.
    - ``keywords_present``     -- every string in ``params["keywords"]`` appears
                                  (case-insensitive unless ``case_sensitive``).

    Any unknown type is treated as unsatisfied rather than ignored.
    """
    if not isinstance(text, str):
        return False

    kind = constraint.type
    params = constraint.params or {}

    if kind == "num_bullets":
        return _within(count_bullets(text), params)

    if kind == "max_words_per_bullet":
        bullets = _bullet_lines(text)
        if not bullets:
            return False
        limit = int(params.get("max", 0))
        return all(count_words(_bullet_body(b)) <= limit for b in bullets)

    if kind == "num_sentences":
        return _within(len(split_sentences(text)), params)

    if kind == "num_words":
        return _within(count_words(text), params)

    if kind == "keywords_present":
        keywords = params.get("keywords") or []
        if not keywords:
            return False
        case_sensitive = bool(params.get("case_sensitive"))
        haystack = text if case_sensitive else text.lower()
        return all(
            (kw if case_sensitive else kw.lower()) in haystack for kw in keywords
        )

    return False
