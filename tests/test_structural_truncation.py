"""Regression tests for structural response truncation (#556 item 17, audit F-22).

``SKTIME_MCP_MAX_RESPONSE_TOKENS`` used to slice the serialized JSON text
mid-string and append a notice, so every truncated response was invalid JSON.
Truncation now operates on the result structure before serialization: the
envelope stays valid JSON, ``success`` is preserved and a ``truncated`` object
describes what was cut.
"""

import json
import os

import pytest

from sktime_mcp.server import _apply_response_token_limit

ENV = "SKTIME_MCP_MAX_RESPONSE_TOKENS"


@pytest.fixture(autouse=True)
def clean_env():
    old = os.environ.pop(ENV, None)
    yield
    if old is not None:
        os.environ[ENV] = old
    else:
        os.environ.pop(ENV, None)


def _limited(tool_name, result, tokens):
    """Apply the limit and return (parsed dict, serialized text)."""
    os.environ[ENV] = str(tokens)
    limited = _apply_response_token_limit(tool_name, result)
    text = json.dumps(limited, indent=2, default=str)
    return json.loads(text), text


def test_huge_predictions_dict_stays_valid_json():
    result = {
        "success": True,
        "handle_id": "fc_1",
        "predictions": {f"2000-{i:05d}": float(i) for i in range(5000)},
    }
    parsed, text = _limited("predict", result, tokens=200)

    assert len(text) <= 200 * 4
    assert parsed["success"] is True
    assert parsed["handle_id"] == "fc_1"
    # The first entries are kept, in order.
    assert list(parsed["predictions"])[:2] == ["2000-00000", "2000-00001"]
    assert len(parsed["predictions"]) < 5000

    truncated = parsed["truncated"]
    assert ENV in truncated["notice"]
    assert "(tool: predict)" in truncated["notice"]
    (entry,) = truncated["fields"]
    assert entry["field"] == "predictions"
    assert entry["total"] == 5000
    assert entry["shown"] == len(parsed["predictions"])


def test_huge_single_string_is_shortened_with_marker():
    result = {"success": True, "code": "x = 1\n" * 3000}
    parsed, text = _limited("export_code", result, tokens=400)

    assert len(text) <= 400 * 4
    assert parsed["success"] is True
    assert parsed["code"].startswith("x = 1\n")
    assert parsed["code"].endswith("[truncated]")
    (entry,) = parsed["truncated"]["fields"]
    assert entry["field"] == "code"
    assert entry["total"] == 18000
    assert entry["unit"] == "chars"
    assert entry["shown"] < entry["total"]


def test_nested_list_is_capped_and_records_path():
    result = {
        "success": True,
        "summary": {"mean": 1.0},
        "fold_results": [
            {"fold": i, "score": 0.1 * i, "y_pred": list(range(2000))} for i in range(3)
        ],
    }
    parsed, text = _limited("evaluate", result, tokens=300)

    assert len(text) <= 300 * 4
    assert parsed["success"] is True
    assert parsed["summary"] == {"mean": 1.0}
    fields = {e["field"]: e for e in parsed["truncated"]["fields"]}
    # Something inside fold_results was capped and its path is reported.
    assert any(f.startswith("fold_results") for f in fields)
    for entry in fields.values():
        assert entry["shown"] < entry["total"]


def test_unchanged_when_under_limit():
    result = {"success": True, "predictions": {"a": 1, "b": 2}}
    os.environ[ENV] = "1000"
    limited = _apply_response_token_limit("predict", result)
    assert limited == result
    assert "truncated" not in limited


def test_unlimited_by_default():
    result = {"success": True, "data": list(range(10000))}
    assert _apply_response_token_limit("x", result) is result


def test_notice_mentions_env_var_and_result_not_mutated():
    result = {"success": True, "rows": list(range(10000))}
    snapshot = json.dumps(result)
    parsed, _ = _limited("inspect_data", result, tokens=100)
    assert ENV in parsed["truncated"]["notice"]
    assert "limit of 100 tokens" in parsed["truncated"]["notice"]
    assert json.dumps(result) == snapshot  # caller's dict untouched


def test_error_envelope_survives_truncation():
    result = {"success": False, "error": "boom " * 5000}
    parsed, _ = _limited("fit", result, tokens=400)
    assert parsed["success"] is False
    assert parsed["error"].startswith("boom ")
    assert parsed["truncated"]["fields"][0]["field"] == "error"


def test_budget_too_small_for_notice_still_yields_valid_json():
    result = {"success": True, "predictions": {str(i): i for i in range(500)}}
    parsed, _ = _limited("predict", result, tokens=5)
    assert parsed["success"] is True
    assert "truncated" in parsed


def test_old_behaviour_is_gone_text_slicing_never_happens():
    """Regression for F-22: a truncated response is always parseable."""
    result = {
        "success": True,
        "results": [{"name": f"Est{i}", "tags": {"a": i}} for i in range(400)],
    }
    for tokens in (10, 50, 150, 400, 900):
        parsed, text = _limited("query_registry", result, tokens=tokens)
        assert parsed["success"] is True
        json.loads(text)  # never raises
