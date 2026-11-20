"""The demo dataset catalogue must be loadable and correctly bucketed.

Previously ``load_dataset`` decided whether a loader's tuple was
``(X, y)`` or ``(y, X)`` from a hard-coded list of six names, so every
other classification/regression dataset (unit_test, japanese_vowels,
acsf1, covid_3month, tecator, ...) came back with X and y swapped and
``fit`` failed with "X must be in an sktime compatible format";
``list_available_data`` bucketed by another name list, filing
classification, regression and segmentation loaders under "forecasting"
next to loaders that need a network download.
"""

import numpy as np
import pandas as pd

from sktime_mcp.runtime.executor import (
    _classify_dataset,
    _discover_demo_datasets,
    _split_dataset_tuple,
    get_executor,
)
from sktime_mcp.tools.list_available_data import list_available_data_tool


def _nested_panel(n: int, n_cols: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {f"dim_{c}": [pd.Series(np.arange(5.0)) for _ in range(n)] for c in range(n_cols)}
    )


class TestSplitByShape:
    def test_panel_and_labels_maps_to_x_and_y(self):
        panel, labels = _nested_panel(4), np.array(["a", "b", "a", "b"])
        y, X = _split_dataset_tuple((panel, labels))
        assert X is panel and y is labels

    def test_series_and_frame_maps_to_y_and_x(self):
        y_in = pd.Series(np.arange(6.0))
        x_in = pd.DataFrame({"exog": np.arange(6.0)})
        y, X = _split_dataset_tuple((y_in, x_in))
        assert y is y_in and X is x_in

    def test_segmentation_tuple_keeps_only_the_series(self):
        series = pd.Series(np.arange(10.0))
        y, X = _split_dataset_tuple((series, 5, np.array([3])))
        assert y is series and X is None

    def test_classify_buckets(self):
        panel = _nested_panel(30)
        labels = np.array(["a", "b"] * 15)
        assert _classify_dataset(labels, panel, (panel, labels)) == "classification"
        floats = np.linspace(0.0, 1.0, 30, dtype="float32")
        assert _classify_dataset(floats, panel, (panel, floats)) == "regression"
        series = pd.Series(np.arange(10.0))
        assert _classify_dataset(series, None, (series, 5, np.array([3]))) == "detection"
        assert _classify_dataset(series, None, series) == "forecasting"


class TestCatalogue:
    def test_heavy_loaders_are_not_advertised(self):
        names = set(_discover_demo_datasets())
        assert not {"m5", "solar", "unit_test_tsf"} & names
        assert "airline" in names

    def test_unit_test_loads_with_panel_x_and_label_y(self):
        res = get_executor().load_dataset("unit_test")
        assert res["success"], res
        assert isinstance(res["X"], pd.DataFrame)
        assert np.asarray(res["y"]).ndim == 1
        assert len(res["y"]) == len(res["X"])

    def test_regression_dataset_has_float_targets(self):
        res = get_executor().load_dataset("covid_3month")
        assert res["success"], res
        assert isinstance(res["X"], pd.DataFrame)
        assert np.asarray(res["y"]).dtype.kind == "f"

    def test_segmentation_dataset_returns_series_target(self):
        res = get_executor().load_dataset("gun_point_segmentation")
        assert res["success"], res
        assert isinstance(res["y"], pd.Series) and res["X"] is None

    def test_buckets_follow_dataset_shape(self):
        demos = list_available_data_tool(True)["system_demos"]
        assert "unit_test" in demos["classification"]
        assert "japanese_vowels" in demos["classification"]
        assert "covid_3month" in demos["regression"]
        assert "tecator" in demos["regression"]
        assert "gun_point_segmentation" in demos["detection"]
        assert "airline" in demos["forecasting"]
        assert "longley" in demos["forecasting"]
        for bucket in ("classification", "regression", "detection"):
            assert not set(demos[bucket]) & set(demos["forecasting"])
