"""Cross-representation contracts for load_data / inspect_data / prepare_data / save_data."""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from sktime.datatypes import check_is_scitype, get_examples

from sktime_mcp.runtime.executor import Executor
from sktime_mcp.tools.prepare_data import prepare_data_tool
from sktime_mcp.tools.save_data import save_data_tool
from tests.test_mcp_wire import _body, _connected_session

MTYPES = [
    "pd.Series",
    "pd.DataFrame",
    "np.ndarray",
    "numpy3D",
    "df-list",
    "nested_univ",
    "pd-multiindex",
    "pd_multiindex_hier",
    "numpy1D",
    "numpy2D",
    "pd_DataFrame_Table",
    "pd_Series_Table",
    "list_of_dict",
    "alignment",
    "pred_interval",
    "pred_quantiles",
]


@pytest.fixture
def executor(monkeypatch):
    ex = Executor()
    monkeypatch.setattr("sktime_mcp.tools.save_data.get_executor", lambda: ex)
    monkeypatch.setattr("sktime_mcp.tools.prepare_data.get_executor", lambda: ex)
    monkeypatch.setattr("sktime_mcp.tools.inspect_data.get_executor", lambda: ex)
    return ex


@pytest.mark.parametrize("mtype", MTYPES)
async def test_native_sktime_examples_load_and_inspect_identically(executor, mtype):
    from sktime_mcp.tools.inspect_data import inspect_data_tool

    examples = get_examples(mtype)
    example = next(value for value in examples.values() if value is not None)
    config = {"type": "memory", "data": example, "mtype": mtype}
    sync = executor.load_data(config)
    async_result = await executor.load_data_async(config)
    assert sync["success"], sync
    assert async_result["success"], async_result
    assert sync["inspection"] == async_result["inspection"]
    assert sync["validation"]["valid"], sync["validation"]
    inspected = inspect_data_tool(sync["data_handle"])
    assert inspected["success"]
    assert inspected["mtype"] == mtype
    json.dumps(inspected, allow_nan=False)


def test_load_preserves_multivariate_missing_duplicate_unsorted_data(executor):
    data = pd.DataFrame({"a": [3.0, np.nan, 1.0], "b": [4.0, 2.0, 5.0]}, index=[2, 1, 1])
    result = executor.load_data({"type": "memory", "data": data})
    assert result["success"]
    record = executor._data_handles[result["data_handle"]]
    pd.testing.assert_frame_equal(record["y"], data)
    assert record["X"] is None
    assert result["validation"]["warnings"]
    data.iloc[0, 0] = 999
    assert record["y"].iloc[0, 0] == 3.0


def test_table_with_mixed_labels_is_inspectable_without_temporal_assumptions(executor):
    data = pd.DataFrame({"label": ["yes", "no", "yes"]}, index=[1, "sample", None])
    result = executor.load_data({"type": "memory", "data": data, "scitype": "Table"})
    assert result["success"], result
    assert result["validation"]["valid"], result["validation"]
    for name in ("start_date", "end_date", "cutoff", "frequency"):
        assert result["inspection"][name] is None
    pd.testing.assert_frame_equal(executor._data_handles[result["data_handle"]]["y"], data)


@pytest.mark.parametrize("start", ["2024-03-09 12:00", "2024-11-02 12:00"])
def test_daily_regularization_preserves_local_time_across_dst(executor, start):
    grid = pd.date_range(start, periods=4, freq="D", tz="America/New_York")
    source = pd.Series([1.0, 4.0], index=grid[[0, 3]])
    loaded = executor.load_data({"type": "memory", "data": source, "frequency": "D"})
    assert loaded["success"], loaded
    prepared = executor.prepare_data(
        loaded["data_handle"], "format", frequency="D", regularize=True
    )
    assert prepared["success"], prepared
    result = executor._data_handles[prepared["data_handle"]]["y"]
    pd.testing.assert_index_equal(result.index, grid)
    pd.testing.assert_series_equal(result.iloc[[0, 3]], source, check_freq=False)
    assert result.isna().sum() == 2


def test_monthly_period_frequency_uses_period_alias(executor):
    source = pd.Series([1.0, 3.0], index=pd.PeriodIndex(["2024-01", "2024-03"], freq="M"))
    loaded = executor.load_data({"type": "memory", "data": source, "frequency": "M"})
    assert loaded["success"], loaded
    prepared = executor.prepare_data(
        loaded["data_handle"], "format", frequency="M", regularize=True
    )
    assert prepared["success"], prepared
    result = executor._data_handles[prepared["data_handle"]]["y"]
    pd.testing.assert_index_equal(result.index, pd.period_range("2024-01", periods=3, freq="M"))
    pd.testing.assert_series_equal(result.iloc[[0, 2]], source)
    assert result.isna().sum() == 1


@pytest.mark.parametrize("missing", ["keep", "ffill"])
def test_group_local_format_preserves_interleaved_instance_order(executor, missing):
    index = pd.MultiIndex.from_tuples(
        [("a", 0), ("b", 0), ("a", 1), ("b", 1)], names=["instance", "time"]
    )
    source = pd.DataFrame({"value": [1.0, 20.0, np.nan, np.nan]}, index=index)
    loaded = executor.load_data({"type": "memory", "data": source, "scitype": "Panel"})
    assert loaded["success"], loaded
    prepared = executor.prepare_data(loaded["data_handle"], "format", missing=missing)
    assert prepared["success"], prepared
    expected = source.copy()
    if missing == "ffill":
        expected["value"] = [1.0, 20.0, 1.0, 20.0]
    pd.testing.assert_frame_equal(executor._data_handles[prepared["data_handle"]]["y"], expected)
    pd.testing.assert_frame_equal(executor._data_handles[loaded["data_handle"]]["y"], source)


def test_month_start_grid_converts_to_monthly_periods_explicitly(executor):
    source = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="MS"))
    loaded = executor.load_data({"type": "memory", "data": source})
    prepared = executor.prepare_data(
        loaded["data_handle"], "format", frequency="MS", index_type="period"
    )
    assert prepared["success"], prepared
    pd.testing.assert_series_equal(
        executor._data_handles[prepared["data_handle"]]["y"], source.to_period()
    )


@pytest.mark.parametrize("exists", [True, False])
def test_export_failure_leaves_destination_and_directory_unchanged(
    executor, tmp_path, monkeypatch, exists
):
    loaded = executor.load_data({"type": "memory", "data": {"y": [1.0, 2.0, 3.0]}})
    target = tmp_path / "values.parquet"
    if exists:
        target.write_bytes(b"previous complete export")

    def failing_writer(self, output, **kwargs):
        output.write(b"partial write")
        raise OSError("Simulated serialization failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_writer)
    result = save_data_tool(loaded["data_handle"], str(target), "parquet", overwrite=exists)
    assert result["success"] is False
    assert "Simulated serialization failure" in result["error"]
    if exists:
        assert target.read_bytes() == b"previous complete export"
    else:
        assert not target.exists()
    assert list(tmp_path.iterdir()) == ([target] if exists else [])


def test_explicit_multivariate_targets_and_features_preserved_after_conversion(executor):
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0], "promo": [0, 1, 0]})
    loaded = executor.load_data(
        {"type": "memory", "data": data, "target_columns": ["a", "b"], "exog_columns": ["promo"]}
    )
    result = prepare_data_tool(loaded["data_handle"], "convert", to_mtype="np.ndarray")
    assert result["success"], result
    record = executor._data_handles[result["data_handle"]]
    np.testing.assert_array_equal(record["y"], data[["a", "b"]].values)
    pd.testing.assert_frame_equal(record["X"], data[["promo"]])
    assert loaded["data_handle"] in executor._data_handles


@pytest.mark.parametrize(
    "config",
    [
        {"target_column": "a", "target_columns": ["b"]},
        {"target_columns": ["a"], "exog_columns": ["a"]},
        {"target_columns": ["a"], "exog_columns": ["b"]},
        {"role": "X", "target_column": "a"},
    ],
)
def test_ambiguous_or_incomplete_mapping_rejected(executor, config):
    result = executor.load_data(
        {"type": "memory", "data": {"a": [1, 2], "b": [3, 4], "c": [5, 6]}, **config}
    )
    assert result["success"] is False
    assert not executor._data_handles


def test_forecast_features_resolve_without_target_leakage(executor):
    target = executor.load_data({"type": "memory", "data": {"y": [1, 2, 3]}})
    features = executor.load_data({"type": "memory", "data": {"promo": [0, 1, 0]}, "role": "X"})
    assert not executor._resolve_source(target["data_handle"], prefer="X")["success"]
    assert executor._resolve_source(features["data_handle"], prefer="X")["success"]
    assert not executor._resolve_xy_inputs(y_handle=features["data_handle"])["success"]


def test_custom_panel_classifier_uses_explicit_x_and_labels(executor):
    X = np.arange(24, dtype=float).reshape(6, 1, 4)
    x = executor.load_data({"type": "memory", "data": X, "mtype": "numpy3D", "role": "X"})
    labels = executor.load_data({"type": "memory", "data": [0, 1, 0, 1, 0, 1], "mtype": "numpy1D"})
    model = executor.instantiate(spec="DummyClassifier(strategy='most_frequent')")
    try:
        resolved = executor._resolve_xy_inputs(
            X_handle=x["data_handle"], y_handle=labels["data_handle"]
        )
        fit = executor.fit(model["handle"], y=resolved["y"], X=resolved["X"])
        assert fit["success"], fit
        prediction = executor.predict(model["handle"], X=resolved["X"])
        assert prediction["success"], prediction
        assert len(prediction["predictions"]) == 6
    finally:
        executor._handle_manager.release_handle(model["handle"])


def test_format_panel_never_fills_between_instances(executor):
    data = pd.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "time": [0, 1, 0, 1],
            "y": [1.0, np.nan, np.nan, 9.0],
            "X": [2.0, np.nan, np.nan, 8.0],
        }
    )
    loaded = executor.load_data(
        {
            "type": "memory",
            "data": data,
            "instance_columns": ["id"],
            "time_column": "time",
            "target_column": "y",
        }
    )
    result = prepare_data_tool(loaded["data_handle"], "format", missing="ffill")
    assert result["success"], result
    record = executor._data_handles[result["data_handle"]]
    assert record["y"].loc[("a", 1), "y"] == 1
    assert pd.isna(record["y"].loc[("b", 0), "y"])
    assert pd.isna(record["X"].loc[("b", 0), "X"])
    assert result["changes_applied"]["y"]["missing_filled"] == 1
    assert result["changes_applied"]["X"]["missing_filled"] == 1


def test_frequency_expansion_bounded_before_allocation(executor, monkeypatch):
    loaded = executor.load_data(
        {
            "type": "memory",
            "data": pd.Series([1.0, 2.0], index=pd.to_datetime(["2000-01-01", "2020-01-01"])),
        }
    )
    monkeypatch.setenv("SKTIME_MCP_MAX_PREPARED_ROWS", "10")
    before = set(executor._data_handles)
    result = prepare_data_tool(loaded["data_handle"], "format", frequency="ns", regularize=True)
    assert not result["success"]
    assert "maximum" in result["error"]
    assert set(executor._data_handles) == before


def test_load_frequency_is_validation_not_resampling(executor):
    data = {"t": ["2020-01-01", "2020-01-03"], "y": [1.0, 3.0]}
    result = executor.load_data(
        {"type": "memory", "data": data, "time_column": "t", "frequency": "D"}
    )
    assert result["success"]
    assert result["metadata"]["rows"] == 2
    expanded = prepare_data_tool(result["data_handle"], "format", frequency="D", regularize=True)
    assert expanded["success"], expanded
    assert expanded["metadata"]["rows"] == 3
    assert expanded["inspection"]["n_missing"] == 1
    invalid = executor.load_data(
        {"type": "memory", "data": data, "time_column": "t", "frequency": "bogus"}
    )
    assert not invalid["success"]


@pytest.mark.parametrize("axis,expected", [("time", (3, 2, 6)), ("instance", (2, 2, 8))])
def test_panel_split_respects_axes(executor, axis, expected):
    loaded = executor.load_data(
        {"type": "memory", "data": np.arange(48).reshape(3, 2, 8), "mtype": "numpy3D"}
    )
    result = prepare_data_tool(loaded["data_handle"], "split", axis=axis, test_size=0.25)
    assert result["success"], result
    assert executor._data_handles[result["train_handle"]]["y"].shape == expected
    assert loaded["data_handle"] in executor._data_handles


def test_split_capacity_failure_does_not_evict_source(executor):
    loaded = executor.load_data({"type": "memory", "data": pd.Series(range(10))})
    executor._max_data_handles = 2
    result = prepare_data_tool(loaded["data_handle"], "split", fh=2)
    assert not result["success"]
    assert list(executor._data_handles) == [loaded["data_handle"]]


@pytest.mark.parametrize("fmt", ["csv", "parquet", "json"])
def test_hierarchy_save_load_preserves_values_and_index_columns(executor, tmp_path, fmt):
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
    data = pd.DataFrame(
        {"time": np.arange(16)},
        index=pd.MultiIndex.from_product(
            [["r"], ["a", "b"], range(8)], names=["region", "store", "step"]
        ),
    )
    loaded = executor.load_data({"type": "memory", "data": data})
    saved = save_data_tool(loaded["data_handle"], str(tmp_path / f"data.{fmt}"), format=fmt)
    assert saved["success"], saved
    again = executor.load_data(saved["load_config"])
    assert again["success"], again
    actual = executor._data_handles[again["data_handle"]]["y"]
    np.testing.assert_array_equal(actual.values, data.values)
    assert actual.index.nlevels == 3
    assert actual.index.tolist() == data.index.tolist()


def test_numpy_save_reload_retains_axes(executor, tmp_path):
    data = np.arange(48).reshape(3, 2, 8)
    loaded = executor.load_data({"type": "memory", "data": data, "mtype": "numpy3D"})
    saved = save_data_tool(loaded["data_handle"], str(tmp_path / "panel.npy"), format="npy")
    assert saved["success"], saved
    again = executor.load_data(saved["load_config"])
    assert again["success"], again
    np.testing.assert_array_equal(executor._data_handles[again["data_handle"]]["y"], data)


async def test_new_tools_and_strict_action_schema_over_mcp():
    async with _connected_session() as session:
        names = {tool.name for tool in (await session.list_tools()).tools}
        assert len(names) == 25
        assert {"load_data", "prepare_data"} <= names
        assert (
            not {"load_data_source", "split_data", "transform_data", "auto_format_on_load"} & names
        )
        for name in ["load_data_source", "split_data", "transform_data", "auto_format_on_load"]:
            assert (await session.call_tool(name, {})).is_error
        loaded = _body(
            await session.call_tool(
                "load_data", {"config": {"type": "memory", "data": {"a": [1, 2, 3, 4]}}}
            )
        )
        try:
            assert loaded["success"]
            bad = await session.call_tool(
                "prepare_data",
                {
                    "data_handle": loaded["data_handle"],
                    "action": "convert",
                    "to_mtype": "np.ndarray",
                    "fh": 1,
                },
            )
            assert bad.is_error
            split = _body(
                await session.call_tool(
                    "prepare_data",
                    {"data_handle": loaded["data_handle"], "action": "split", "fh": 1},
                )
            )
            assert split["success"], split
            for handle in [split["train_handle"], split["test_handle"]]:
                await session.call_tool("release_data_handle", {"data_handle": handle})
        finally:
            await session.call_tool("release_data_handle", {"data_handle": loaded["data_handle"]})
