"""Scitype-specific fit/predict/format paths (#556 item 10, audit F-10/F-11/F-12/F-31/F-54).

``Executor.predict`` assumed the forecaster signature ``predict(fh, X)`` for
every scitype it did not special-case, and the JSON serialiser assumed a flat
index. Detectors, hierarchical forecasts, multivariate ``format``,
transformers given ``y_handle`` and clusterers without X all crashed with raw
sktime/pandas errors after a successful fit.
"""

import contextlib

import numpy as np
import pandas as pd
import pytest
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.detection.bs import BinarySegmentation
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.boxcox import BoxCoxTransformer

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.fit_predict import fit_tool, predict_tool
from sktime_mcp.tools.transform_data import transform_data_tool


@pytest.fixture
def executor():
    ex = get_executor()
    before = set(ex._data_handles)
    yield ex
    for h in list(ex._data_handles):
        if h not in before:
            ex._data_handles.pop(h, None)


def _register(ex, handle, y, X=None):
    ex._data_handles[handle] = {"y": y, "X": X, "metadata": {}, "validation": {}, "config": {}}
    return handle


@pytest.fixture
def estimator(executor):
    """Create one estimator handle; released on teardown."""
    handles = []

    def _make(name, instance):
        h = executor._handle_manager.create_handle(name, instance, {})
        handles.append(h)
        return h

    yield _make
    for h in handles:
        with contextlib.suppress(KeyError):
            executor._handle_manager.release_handle(h)


def _step_series(n=50):
    rng = np.random.RandomState(0)
    return pd.Series(np.r_[np.zeros(n // 2), np.full(n // 2, 10.0)] + rng.normal(0, 0.1, n))


# --------------------------------------------------------------------------
# F-10: detectors
# --------------------------------------------------------------------------


class TestDetector:
    @pytest.fixture
    def fitted(self, executor, estimator):
        _register(executor, "sp_det", _step_series())
        h = estimator("BinarySegmentation", BinarySegmentation(threshold=1))
        res = fit_tool(estimator_handle=h, y_handle="sp_det")
        assert res["success"], res
        return h

    def test_predict_returns_detection_frame(self, fitted):
        res = predict_tool(estimator_handle=fitted, y_handle="sp_det")
        assert res["success"], res
        # BinarySegmentation reports the change point's integer location
        assert [row["ilocs"] for row in res["predictions"].values()] == [24]
        assert "horizon" not in res
        assert "warnings" not in res

    def test_predict_accepts_series_as_x(self, executor, fitted):
        # a handle whose exogenous frame is the series to annotate
        _register(executor, "sp_det_x", _step_series(), X=_step_series().to_frame("v"))
        res = predict_tool(estimator_handle=fitted, X_handle="sp_det_x")
        assert res["success"], res
        assert [row["ilocs"] for row in res["predictions"].values()] == [24]

    def test_predict_without_series_is_structured(self, fitted):
        res = predict_tool(estimator_handle=fitted)
        assert not res["success"]
        assert "fh" not in res["error"]
        assert "y_handle" in res["error"]

    def test_other_modes_point_to_call_method(self, fitted):
        res = predict_tool(estimator_handle=fitted, y_handle="sp_det", mode="predict_interval")
        assert not res["success"]
        assert "call_method" in res["error"]

    def test_fit_with_x_does_not_collide(self, executor, estimator):
        # F-10: fit(y, X=X) raised "got multiple values for argument 'X'"
        h = estimator("BinarySegmentation", BinarySegmentation(threshold=1))
        res = executor.fit(h, y=None, X=_step_series())
        assert res["success"], res
        exog = pd.DataFrame({"e": np.arange(50.0)})
        res = executor.fit(h, y=_step_series(), X=exog)
        assert res["success"], res
        assert any("ignored" in w for w in res["warnings"])

    def test_native_methods_via_call_method(self, executor, fitted):
        res = executor.call_method(fitted, "predict_points", {"X_data_handle": "sp_det"})
        assert res["success"], res
        assert res["result"]["ilocs"] == {0: 24}
        res = executor.call_method(fitted, "transform", {"X_data_handle": "sp_det"})
        assert res["success"], res
        assert len(res["result"]["labels"]) == 50


# --------------------------------------------------------------------------
# F-11: hierarchical / panel forecasts
# --------------------------------------------------------------------------


class TestHierarchicalForecast:
    def test_predict_flattens_multiindex(self, executor, estimator):
        idx = pd.MultiIndex.from_product(
            [["a", "b"], pd.period_range("2000-01", periods=8, freq="M")],
            names=["series", "time"],
        )
        _register(executor, "sp_hier", pd.DataFrame({"v": np.arange(16.0)}, index=idx))
        h = estimator("NaiveForecaster", NaiveForecaster())
        assert fit_tool(estimator_handle=h, y_handle="sp_hier")["success"]

        res = predict_tool(estimator_handle=h, horizon=2)
        assert res["success"], res
        assert list(res["predictions"]) == ["a_2000-09", "a_2000-10", "b_2000-09", "b_2000-10"]
        assert res["predictions"]["b_2000-10"] == {"v": 15.0}
        # the handle keeps the real MultiIndex for plotting/scoring
        stored = executor._data_handles[res["prediction_handle"]]["y"]
        assert isinstance(stored.index, pd.MultiIndex)
        assert stored.index.names == ["series", "time"]

    def test_predict_interval_flattens_multiindex(self, executor, estimator):
        idx = pd.MultiIndex.from_product([["a", "b"], range(8)])
        _register(executor, "sp_hier_int", pd.DataFrame({"v": np.arange(16.0)}, index=idx))
        h = estimator("NaiveForecaster", NaiveForecaster())
        assert fit_tool(estimator_handle=h, y_handle="sp_hier_int")["success"]
        res = predict_tool(estimator_handle=h, horizon=1, mode="predict_interval")
        assert res["success"], res
        assert list(res["intervals"]) == ["a_8", "b_8"]


# --------------------------------------------------------------------------
# F-12: transform_data(action="format") on a multivariate frame
# --------------------------------------------------------------------------


class TestFormatMultivariate:
    def test_format_fills_missing_across_columns(self, executor):
        frame = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        _register(executor, "sp_mv", frame)
        res = transform_data_tool(data_handle="sp_mv", action="format")
        assert res["success"], res
        assert "Filled 2 missing values (forward/backward fill)" in res["changes_applied"]
        out = executor._data_handles[res["data_handle"]]["y"]
        assert not out.isna().values.any()

    def test_format_series_still_counts_missing(self, executor):
        s = pd.Series([1.0, np.nan, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="D"))
        _register(executor, "sp_uv", s)
        res = transform_data_tool(data_handle="sp_uv", action="format")
        assert res["success"], res
        assert "Filled 1 missing values (forward/backward fill)" in res["changes_applied"]


# --------------------------------------------------------------------------
# F-31: transformer predict with y_handle
# --------------------------------------------------------------------------


class TestTransformerPredict:
    @pytest.fixture
    def fitted(self, executor, estimator):
        _register(executor, "sp_air", executor.load_dataset("airline")["y"])
        h = estimator("BoxCoxTransformer", BoxCoxTransformer())
        assert fit_tool(estimator_handle=h, y_handle="sp_air")["success"]
        return h

    def test_predict_with_y_handle_transforms(self, fitted):
        res = predict_tool(estimator_handle=fitted, y_handle="sp_air")
        assert res["success"], res
        assert len(res["predictions"]) == 144
        assert "horizon" not in res
        assert "warnings" not in res
        assert "prediction_handle" in res

    def test_predict_without_series_is_structured(self, fitted):
        res = predict_tool(estimator_handle=fitted)
        assert not res["success"]
        assert "has no attribute 'predict'" not in res["error"]
        assert "y_handle" in res["error"] and "X_handle" in res["error"]


# --------------------------------------------------------------------------
# F-54: clusterer predict without X
# --------------------------------------------------------------------------


class TestClustererPredict:
    @pytest.fixture
    def fitted(self, executor, estimator):
        idx = pd.MultiIndex.from_product([[0, 1, 2, 3], range(5)])
        panel = pd.DataFrame({"v": np.r_[np.zeros(10), np.ones(10)]}, index=idx)
        h = estimator(
            "TimeSeriesKMeans",
            TimeSeriesKMeans(n_clusters=2, max_iter=1, metric="euclidean", random_state=0),
        )
        assert executor.fit(h, y=None, X=panel)["success"]
        return h, panel

    def test_predict_without_x_is_structured(self, fitted):
        h, _ = fitted
        res = predict_tool(estimator_handle=h)
        assert not res["success"]
        assert "fh" not in res["error"]
        assert "X_handle" in res["error"] and "X_dataset" in res["error"]

    def test_predict_with_x_labels(self, executor, fitted):
        h, panel = fitted
        res = executor.predict(h, X=panel)
        assert res["success"], res
        assert sorted(set(res["predictions"])) == [0, 1]
