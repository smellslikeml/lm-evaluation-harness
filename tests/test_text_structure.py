"""Tests for the StructTest Summarization-domain text-structure checkers.

The integration tests drive the paper's text constraints through the *existing*
``lm_eval.api.structured_output.score_response`` scorer and the registered
``structured_output_acc`` metric -- the same code path the harness uses -- to
prove the new checkers are wired end to end rather than living in isolation.
The remaining tests exercise the checker module directly.
"""

import json

from lm_eval.api.metrics import structured_output_acc_fn
from lm_eval.api.registry import get_metric
from lm_eval.api.structured_output import Constraint, score_response
from lm_eval.api.text_structure import (
    check_text_constraint,
    count_bullets,
    count_words,
    split_sentences,
)


# A model summary that follows a compositional StructTest instruction:
# "Summarize in exactly three bullet points, each at most eight words."
SUMMARY = (
    "- Revenue grew twelve percent this fiscal year\n"
    "- New markets opened across three continents\n"
    "- Costs stayed flat despite rapid hiring"
)


def _spec(*constraints):
    return json.dumps({"constraints": list(constraints)})


# ---------------------------------------------------------------------------
# Integration: text constraints scored through the shared score_response path
# ---------------------------------------------------------------------------


def test_score_response_scores_summarization_constraints():
    constraints = [
        {"type": "num_bullets", "params": {"n": 3}},
        {"type": "max_words_per_bullet", "params": {"max": 8}},
    ]
    scores = score_response(SUMMARY, constraints)
    assert scores["prompt_level_acc"] == 1
    assert scores["constraint_level_acc"] == 1.0
    # A bulleted summary is not JSON; text scoring must not depend on that.
    assert scores["json_valid"] == 0
    assert scores["num_constraints"] == 2


def test_score_response_reports_partial_text_satisfaction():
    # Four bullets fails the count; the per-bullet word limit still holds.
    four_bullets = SUMMARY + "\n- Guidance raised for the coming quarter"
    scores = score_response(
        four_bullets,
        [
            {"type": "num_bullets", "params": {"n": 3}},
            {"type": "max_words_per_bullet", "params": {"max": 8}},
        ],
    )
    assert scores["constraint_level_acc"] == 0.5
    assert scores["prompt_level_acc"] == 0


def test_score_response_mixes_json_and_text_constraints():
    # One JSON constraint and one text constraint in the same spec.
    response = '{"title": "Q3 report"}\n- only one bullet here'
    scores = score_response(
        response,
        [
            Constraint(type="key_present", path="title"),
            Constraint(type="num_bullets", params={"n": 1}),
        ],
    )
    assert scores["json_valid"] == 1
    assert scores["prompt_level_acc"] == 1


def test_metric_scores_text_constraints_end_to_end():
    """The registered metric routes text constraints to the new checkers."""
    metric_fn = get_metric("structured_output_acc")
    assert metric_fn is structured_output_acc_fn

    reference = _spec(
        {"type": "num_bullets", "params": {"n": 3}},
        {"type": "num_words", "params": {"max": 40}},
        {"type": "keywords_present", "params": {"keywords": ["revenue", "costs"]}},
    )
    result = metric_fn([reference], [SUMMARY])
    assert result["structured_output_acc"] == 1.0
    assert result["structured_output_prompt_acc"] == 1


# ---------------------------------------------------------------------------
# Checker unit tests
# ---------------------------------------------------------------------------


def test_bullet_and_word_and_sentence_counting():
    assert count_bullets(SUMMARY) == 3
    # Ordered lists and the bullet glyph are recognized as bullets too.
    assert count_bullets("1. first\n2) second\n• third") == 3
    assert count_words("state-of-the-art, don't count punctuation!") == 4
    assert len(split_sentences("One idea. Two ideas! Three? ")) == 3


def test_bounds_and_keyword_and_unknown_checks():
    within = Constraint(type="num_sentences", params={"min": 2, "max": 4})
    assert check_text_constraint("A. B. C.", within) is True
    assert check_text_constraint("Only one.", within) is False

    # An empty bound set is unsatisfiable, so a typo can never inflate a score.
    assert check_text_constraint(SUMMARY, Constraint(type="num_bullets")) is False

    case = Constraint(type="keywords_present", params={"keywords": ["REVENUE"]})
    assert check_text_constraint(SUMMARY, case) is True  # case-insensitive default
    strict = Constraint(
        type="keywords_present",
        params={"keywords": ["REVENUE"], "case_sensitive": True},
    )
    assert check_text_constraint(SUMMARY, strict) is False  # only "Revenue" appears

    # max_words_per_bullet requires at least one bullet to exist.
    limit = Constraint(type="max_words_per_bullet", params={"max": 8})
    assert check_text_constraint("no bullets in this text", limit) is False

    # Unknown types fail closed.
    assert check_text_constraint(SUMMARY, Constraint(type="does_not_exist")) is False
