"""Compositional structured-output constraint validation.

Adapted from the core evaluation mechanism of *StructTest: Benchmarking LLMs'
Reasoning through Compositional Structured Outputs* (arXiv:2412.18011).

StructTest scores an LLM by checking whether its *structured* outputs (e.g.
JSON) satisfy a set of compositional constraints, rather than by matching a
single target answer. Because there is no fixed gold string to memorize, the
signal is resistant to the data-contamination and cheating that target-answer
benchmarks are vulnerable to.

This module implements that constraint-satisfaction scoring as a reusable,
dependency-free validator (stdlib ``json``/``re`` only -- no learned
estimators, no external artifacts). It is wired into the harness as the
``structured_output_acc`` metric registered in :mod:`lm_eval.api.metrics`; any
``generate_until`` task can opt in by listing its constraints in
``doc_to_target`` as a JSON spec (see :func:`score_response`).

Mode-2 adaptation: the paper's bespoke benchmark dataset and task suite are
replaced by this harness-native metric; the core compositional-constraint
mechanism (per-constraint checking + prompt-level / constraint-level scoring)
is kept at full fidelity.
"""

import dataclasses
import json
import re
from collections.abc import Sequence
from typing import Any


__all__ = [
    "Constraint",
    "check_constraint",
    "extract_structured",
    "score_response",
]


@dataclasses.dataclass
class Constraint:
    """One compositional constraint over a structured output.

    ``path`` addresses a (possibly nested) value with dot-separated keys, e.g.
    ``"address.city"`` or ``"items.0"``; an empty path targets the root object.
    ``type`` selects the checker; ``params`` carries its arguments (e.g.
    ``{"min": 2}`` for ``min_length``).

    Supported ``type`` values:

    - ``json_valid``    -- the parsed output is an object or array (no path).
    - ``key_present``   -- the value at ``path`` exists.
    - ``type_is``       -- the value at ``path`` is of JSON ``params.type``
                           (``object``/``array``/``string``/``number``/``boolean``).
    - ``min_length`` / ``max_length`` -- ``len(value)`` within ``params.min`` /
                           ``params.max`` (for arrays, strings, or objects).
    - ``count_keys``    -- the object at ``path`` has ``>= params.min`` keys.
    - ``regex_match``   -- the string value at ``path`` matches ``params.pattern``.
    - ``field_equals``  -- the value at ``path`` equals ``params.value``.
    """

    type: str
    path: str | None = None
    params: dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structured-output extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_BRACKETS = (("{", "}"), ("[", "]"))


def extract_structured(response: str) -> tuple[Any, bool]:
    """Best-effort parse of a model response into a Python object.

    Tries, in order: a direct ``json.loads``; extraction from a ```json fence;
    and a brace/bracket span scan that tolerates leading or trailing prose.
    Returns ``(obj, ok)`` where ``ok`` is ``False`` when no structured payload
    could be recovered.
    """
    if not isinstance(response, str):
        return response, False
    text = response.strip()

    def _try(value: str) -> Any | None:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None

    parsed = _try(text)
    if parsed is not None:
        return parsed, True

    for match in _FENCE_RE.finditer(text):
        parsed = _try(match.group(1))
        if parsed is not None:
            return parsed, True

    for opener, closer in _BRACKETS:
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            parsed = _try(text[start : end + 1])
            if parsed is not None:
                return parsed, True

    return None, False


# ---------------------------------------------------------------------------
# Path resolution + constraint checkers
# ---------------------------------------------------------------------------


def _resolve(obj: Any, path: str | None) -> tuple[Any, bool]:
    """Resolve a dot-separated ``path`` against ``obj`` -> ``(value, found)``."""
    if not path:
        return obj, True
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None, False
        else:
            return None, False
    return cur, True


_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "boolean": bool,
}


def check_constraint(obj: Any, constraint: Constraint) -> bool:
    """Return whether ``obj`` satisfies a single ``Constraint``."""
    kind = constraint.type
    params = constraint.params or {}

    if kind == "json_valid":
        return isinstance(obj, (dict, list))

    value, found = _resolve(obj, constraint.path)

    if kind == "key_present":
        return constraint.path is not None and found
    if kind == "type_is":
        if not found:
            return False
        expected = _JSON_TYPES.get(str(params.get("type")))
        # bool is an int subclass; keep the JSON notion of "number" distinct.
        if expected == (int, float) and isinstance(value, bool):
            return False
        return expected is not None and isinstance(value, expected)
    if kind in ("min_length", "max_length"):
        if not found or not hasattr(value, "__len__"):
            return False
        length = len(value)
        if kind == "min_length":
            return length >= int(params.get("min", 0))
        return length <= int(params.get("max", 0))
    if kind == "count_keys":
        if not found or not isinstance(value, dict):
            return False
        return len(value) >= int(params.get("min", 0))
    if kind == "regex_match":
        if not found or not isinstance(value, str):
            return False
        return re.search(str(params.get("pattern", "")), value) is not None
    if kind == "field_equals":
        return found and value == params.get("value")
    # Unknown constraint type is treated as unsatisfied rather than ignored,
    # so a typo in a spec can never inflate a score.
    return False


# ---------------------------------------------------------------------------
# StructTest-style scoring
# ---------------------------------------------------------------------------


def _coerce(raw: Constraint | dict[str, Any]) -> Constraint:
    if isinstance(raw, Constraint):
        return raw
    if isinstance(raw, dict):
        return Constraint(
            type=raw["type"],
            path=raw.get("path"),
            params=raw.get("params") or {},
        )
    raise TypeError(f"Unsupported constraint spec: {raw!r}")


def score_response(
    response: str, constraints: Sequence[Constraint | dict[str, Any]]
) -> dict[str, Any]:
    """Score one response against compositional ``constraints``.

    Mirrors StructTest's two views: ``constraint_level_acc`` is the fraction of
    constraints satisfied (the per-instruction view), while ``prompt_level_acc``
    is 1.0 only when *every* constraint holds (the all-or-nothing view). Both
    collapse to ``json_valid`` when no constraints are supplied, so the metric
    can also be used as a plain structured-output validity check.
    """
    parsed = [_coerce(c) for c in constraints]
    obj, ok = extract_structured(response)

    if not ok or not parsed:
        return {
            "json_valid": int(ok),
            "constraint_level_acc": float(ok),
            "prompt_level_acc": int(ok),
            "num_constraints": len(parsed),
        }

    flags = [check_constraint(obj, c) for c in parsed]
    return {
        "json_valid": 1,
        "constraint_level_acc": sum(flags) / len(flags),
        "prompt_level_acc": int(all(flags)),
        "num_constraints": len(parsed),
    }
