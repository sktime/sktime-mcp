"""
Regression tests for query_registry filter semantics (#556 item 13; audit F-17, F-42, F-47).

- F-17: task/tag filters must give the same results with and without ``query``.
- F-42: tag filter values are coerced by the tag's ``value_type`` (bool strings),
  bad values give a structured error, list-valued tags filter by membership.
- F-47: every ``task`` returned is a valid, filterable scitype (skchange
  ``interval_scorer`` objects resolve to ``estimator``).
"""

import pytest

from sktime_mcp.registry.interface import EstimatorNode, get_registry
from sktime_mcp.tools.query_registry import query_registry_tool

# Every EstimatorNode.module is "<package>.<module>.<Class>", so "." matches all.
MATCH_ALL_QUERY = "."

FILTERS = [
    {"task": "metric_forecasting"},
    {"task": "estimator"},
    {"tags": {"capability:pred_int": [True]}},
    # y_inner_mtype is list-valued on many estimators: any-of + membership
    {"tags": {"y_inner_mtype": ["pd.Series", "pd.DataFrame"]}},
]


@pytest.mark.parametrize("filters", FILTERS, ids=lambda f: str(f))
def test_same_total_with_and_without_query(filters):
    """F-17: the query path must apply the same task/tag semantics as the plain path."""
    plain = query_registry_tool(**filters, limit=1)
    with_query = query_registry_tool(**filters, query=MATCH_ALL_QUERY, limit=1)
    assert plain["success"] and with_query["success"]
    assert plain["total"] > 0
    assert with_query["total"] == plain["total"]


@pytest.mark.parametrize("filters", FILTERS, ids=lambda f: str(f))
def test_query_results_are_subset_of_filtered(filters):
    """F-17: a narrowing query returns a non-empty subset of the filtered list."""
    plain = {r["name"] for r in query_registry_tool(**filters, limit=1000)["results"]}
    narrowed = query_registry_tool(**filters, query="e", limit=1000)
    assert narrowed["success"]
    names = {r["name"] for r in narrowed["results"]}
    assert names
    assert names <= plain


def test_task_and_query_combined():
    """task='forecaster' + query='naive' ranks NaiveForecaster first."""
    result = query_registry_tool(task="forecaster", query="naive", limit=5)
    assert result["success"]
    assert result["results"][0]["name"] == "NaiveForecaster"
    assert all(r["task"] == "forecaster" for r in result["results"])


@pytest.mark.parametrize("raw", ["true", "True", "1", 1])
def test_bool_tag_value_is_coerced(raw):
    """F-42: string/int bools are coerced to bool for bool-typed tags."""
    expected = query_registry_tool(tags={"capability:pred_int": True}, limit=1)["total"]
    assert expected > 0
    result = query_registry_tool(tags={"capability:pred_int": raw}, limit=1)
    assert result["success"]
    assert result["total"] == expected
    assert result["tag_filter"] == {"capability:pred_int": True}


def test_false_bool_string_is_coerced():
    expected = query_registry_tool(tags={"capability:pred_int": False}, limit=1)["total"]
    result = query_registry_tool(tags={"capability:pred_int": "false"}, limit=1)
    assert result["success"]
    assert result["total"] == expected > 0


def test_uncoercible_tag_value_is_structured_error():
    """F-42: a value that cannot be coerced names the tag and the expected type."""
    result = query_registry_tool(tags={"capability:pred_int": "maybe"}, limit=1)
    assert result["success"] is False
    assert "capability:pred_int" in result["error"]
    assert "bool" in result["error"]
    assert result["invalid_tag_values"] == [
        {"tag": "capability:pred_int", "value": "maybe", "expected": "bool (true/false)"}
    ]


def test_allowed_values_are_checked():
    """A tag with an enumerated value_type rejects values outside the list."""
    result = query_registry_tool(tags={"task": "bogus"}, limit=1)
    assert result["success"] is False
    assert result["invalid_tag_values"][0]["tag"] == "task"
    assert "change_point_detection" in result["invalid_tag_values"][0]["expected"]


@pytest.mark.parametrize("value", ["statsmodels", ["statsmodels"]])
def test_list_valued_tag_filters_by_membership(value):
    """F-42: python_dependencies (str or list on the estimator) filters by membership."""
    result = query_registry_tool(tags={"python_dependencies": value}, limit=500)
    assert result["success"], result.get("error")
    names = {r["name"] for r in result["results"]}
    # ExponentialSmoothing declares the plain string, AutoETS a list containing it
    assert {"ExponentialSmoothing", "AutoETS"} <= names
    assert "NaiveForecaster" not in names


def test_every_result_task_is_a_valid_task():
    """F-47: no result carries a task that cannot be used as a filter."""
    valid = set(get_registry().get_available_tasks())
    result = query_registry_tool(limit=5000)
    assert result["success"]
    assert result["total"] == result["count"]
    assert {r["task"] for r in result["results"]} <= valid


def test_interval_scorers_resolve_to_estimator_and_are_filterable():
    """F-47: skchange cost/score objects keep object_type in tags but task='estimator'."""
    hits = [
        r for r in query_registry_tool(query="CUSUM", limit=10)["results"] if r["name"] == "CUSUM"
    ]
    if not hits:
        pytest.skip("CUSUM (skchange interval scorer) not available in this sktime")
    node = hits[0]
    assert node["task"] == "estimator"
    assert node["tags"]["object_type"] == "interval_scorer"
    filtered = query_registry_tool(task="estimator", query="CUSUM", limit=10)
    assert "CUSUM" in {r["name"] for r in filtered["results"]}


def test_resolve_task_prefers_most_specific_valid_scitype():
    """Unit check of task resolution on synthetic classes."""
    from sktime.base import BaseEstimator

    class _Scorer(BaseEstimator):
        _tags = {"object_type": "interval_scorer"}

    class _Metric(BaseEstimator):
        _tags = {"object_type": ["metric", "metric_forecasting"]}

    registry = get_registry()
    assert registry._resolve_task(_Scorer) == "estimator"
    assert registry._resolve_task(_Metric) == "metric_forecasting"
    node = registry._create_node("_Scorer", _Scorer)
    assert isinstance(node, EstimatorNode)
    assert node.task == "estimator"
    assert node.tags["object_type"] == "interval_scorer"
