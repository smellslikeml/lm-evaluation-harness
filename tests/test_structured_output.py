"""Tests for the StructTest-style structured-output constraint metric.

The integration test drives the registered ``structured_output_acc`` metric
through a real ``ConfigurableTask.process_results`` path -- the same code the
evaluator uses -- to prove the wiring end to end. The remaining tests exercise
the capability module directly.
"""

import json

from lm_eval.api.metrics import structured_output_acc_fn
from lm_eval.api.registry import get_metric, get_metric_aggregation
from lm_eval.api.structured_output import Constraint, score_response
from lm_eval.api.task import ConfigurableTask
from lm_eval.config.task import TaskConfig


class MockGenerateTask(ConfigurableTask):
    """A generate_until task whose only metric is structured_output_acc."""

    def __init__(self, target):
        config = {
            "task": "test_structured_output",
            "output_type": "generate_until",
            "metric_list": [{"metric": "structured_output_acc"}],
            "doc_to_target": target,
            "target_delimiter": " ",
        }
        self._config = TaskConfig(**config)
        self.OUTPUT_TYPE = "generate_until"
        self.multiple_input = 0
        self.multiple_target = 0
        self._metric_fn_list = {"structured_output_acc": structured_output_acc_fn}
        self._metric_fn_kwargs = {"structured_output_acc": {}}
        self._aggregation_list = {
            "structured_output_acc": get_metric_aggregation("structured_output_acc")
        }
        self._higher_is_better = {"structured_output_acc": True}

    def doc_to_target(self, doc):
        # The spec is a constant JSON string; return it verbatim so no Jinja
        # rendering touches the braces in the constraint spec.
        return self._config.doc_to_target

    # --- minimal abstract-method stubs ---------------------------------
    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def download(self, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Integration: metric invoked through the harness process_results path
# ---------------------------------------------------------------------------


def _spec(*constraints):
    return json.dumps({"constraints": list(constraints)})


def test_metric_registered_and_callable():
    """The metric is registered and is the function we wired up."""
    assert get_metric("structured_output_acc") is structured_output_acc_fn
    # aggregation falls back to mean for the expanded dict keys.
    assert get_metric_aggregation("structured_output_acc") is not None


def test_integration_scores_valid_response():
    target = _spec(
        {"type": "key_present", "path": "name"},
        {"type": "type_is", "path": "items", "params": {"type": "array"}},
        {"type": "min_length", "path": "items", "params": {"min": 2}},
    )
    task = MockGenerateTask(target)

    response = json.dumps({"name": "Ada", "items": ["a", "b", "c"]})
    result = task.process_results({"target": target}, [response])

    assert result["structured_output_acc"] == 1.0
    assert result["structured_output_prompt_acc"] == 1
    assert result["structured_output_json_valid"] == 1


def test_integration_partial_and_invalid_responses():
    target = _spec(
        {"type": "key_present", "path": "name"},
        {"type": "min_length", "path": "items", "params": {"min": 2}},
    )
    task = MockGenerateTask(target)

    # Satisfies only the first constraint -> fraction 0.5, prompt-level 0.
    half = task.process_results({}, [json.dumps({"name": "Ada", "items": ["a"]})])
    assert half["structured_output_acc"] == 0.5
    assert half["structured_output_prompt_acc"] == 0
    assert half["structured_output_json_valid"] == 1

    # Unparseable response -> everything is 0.
    bad = task.process_results({}, ["no structure here"])
    assert bad["structured_output_acc"] == 0.0
    assert bad["structured_output_json_valid"] == 0


def test_integration_extracts_json_from_markdown_fence():
    target = _spec({"type": "key_present", "path": "ok"})
    task = MockGenerateTask(target)

    fenced = 'Here you go:\n```json\n{"ok": true}\n```\nDone.'
    result = task.process_results({}, [fenced])
    assert result["structured_output_acc"] == 1.0
    assert result["structured_output_json_valid"] == 1


# ---------------------------------------------------------------------------
# Capability unit tests
# ---------------------------------------------------------------------------


def test_extract_structured_tolerates_surrounding_prose():
    # The brace-span scan recovers the JSON even with leading/trailing prose.
    scores = score_response('Sure! {"a": 1, "b": [1, 2]} hope that helps', [])
    assert scores["json_valid"] == 1


def test_constraint_type_and_equality_checkers():
    obj = {"verified": True, "score": 7, "label": "cat"}
    assert (
        score_response("{}", [Constraint(type="key_present", path="missing")])[
            "constraint_level_acc"
        ]
        == 0.0
    )

    def prompt_level(constraint):
        return score_response(json.dumps(obj), [constraint])["prompt_level_acc"]

    assert (
        prompt_level(
            Constraint(type="type_is", path="verified", params={"type": "boolean"})
        )
        == 1
    )
    assert (
        prompt_level(
            Constraint(type="field_equals", path="label", params={"value": "cat"})
        )
        == 1
    )
    assert prompt_level(Constraint(type="count_keys", params={"min": 2})) == 1
    # "number" must not match a boolean (bool is an int subclass).
    assert (
        prompt_level(
            Constraint(type="type_is", path="verified", params={"type": "number"})
        )
        == 0
    )


def test_score_response_reports_both_views():
    constraints = [
        Constraint(type="key_present", path="a"),
        Constraint(type="key_present", path="b"),
        Constraint(type="key_present", path="c"),
    ]
    scores = score_response(json.dumps({"a": 1, "b": 2}), constraints)
    assert scores["constraint_level_acc"] == 2 / 3
    assert scores["prompt_level_acc"] == 0
    assert scores["num_constraints"] == 3


# ---------------------------------------------------------------------------
# Edge-case coverage exercised through the registered metric function
# (imported from the non-new lm_eval.api.metrics module, so these prove the
# wiring, not just the capability module in isolation).
# ---------------------------------------------------------------------------


def _metric(target_spec, prediction):
    """Invoke the registered metric the way process_results does."""
    return structured_output_acc_fn([target_spec], [prediction])


def test_metric_empty_predictions_list_scores_zero():
    # An empty (or missing) prediction must never raise and must score 0 on
    # every view -- otherwise a model that emitted nothing could be scored as
    # if it were correct.
    spec = _spec({"type": "key_present", "path": "name"})
    result = structured_output_acc_fn([spec], [])
    assert result == {
        "structured_output_acc": 0.0,
        "structured_output_prompt_acc": 0,
        "structured_output_json_valid": 0,
    }
    # Empty references as well -> no constraints, no output -> still all zero.
    assert structured_output_acc_fn([], [])["structured_output_json_valid"] == 0


def test_metric_malformed_json_is_not_valid():
    # Looks structured (has braces) but is not parseable JSON (trailing comma);
    # extraction must fail rather than the brace-scan silently "recovering" it.
    spec = _spec({"type": "json_valid"})
    result = _metric(spec, '{"a": 1, "b": 2,}')
    assert result["structured_output_json_valid"] == 0
    assert result["structured_output_acc"] == 0.0
    assert result["structured_output_prompt_acc"] == 0


def test_metric_structurally_valid_but_semantically_wrong():
    # Correct JSON shape and correct types, but a value that violates a
    # field_equals constraint: JSON validity is 1 while the constraint views
    # drop below full -- structural correctness must not imply a passing score.
    spec = _spec(
        {"type": "type_is", "path": "status", "params": {"type": "string"}},
        {"type": "field_equals", "path": "status", "params": {"value": "ok"}},
    )
    result = _metric(spec, json.dumps({"status": "error"}))
    assert result["structured_output_json_valid"] == 1
    assert result["structured_output_acc"] == 0.5
    assert result["structured_output_prompt_acc"] == 0


def test_metric_scores_first_prediction_of_a_batch():
    # generate_until can hand the metric several candidate responses per doc
    # (e.g. repeats / self-consistency). The per-document contract is to score
    # the first candidate; the extra candidates must not change the result.
    spec = _spec({"type": "key_present", "path": "name"})
    good = json.dumps({"name": "Ada"})
    bad = "no structure here"
    first_good = structured_output_acc_fn([spec], [good, bad])
    first_bad = structured_output_acc_fn([spec], [bad, good])
    assert first_good["structured_output_prompt_acc"] == 1
    assert first_bad["structured_output_prompt_acc"] == 0


def test_metric_partial_score_never_rounds_up_to_prompt_level_pass():
    # A "tie" at a fractional constraint score (here exactly 0.5) is reported
    # deterministically and must never be promoted to a prompt-level pass:
    # prompt_level is strictly all-or-nothing, so partial credit stays partial.
    spec = _spec(
        {"type": "key_present", "path": "a"},
        {"type": "key_present", "path": "b"},
    )
    result = _metric(spec, json.dumps({"a": 1}))
    assert result["structured_output_acc"] == 0.5
    assert result["structured_output_prompt_acc"] == 0
    # Deterministic: scoring the same response again yields the same numbers.
    assert _metric(spec, json.dumps({"a": 1})) == result
