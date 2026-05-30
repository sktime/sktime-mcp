"""Tests for the data source layer (adapters, registry, executor integration)."""

import pandas as pd

from sktime_mcp.data import DataSourceRegistry, FileAdapter, PandasAdapter, SQLAdapter, UrlAdapter
from sktime_mcp.runtime.executor import get_executor


class TestDataSourceImports:
    """Verify all data module components are importable."""

    def test_adapter_classes_importable(self):
        assert PandasAdapter is not None
        assert SQLAdapter is not None
        assert FileAdapter is not None
        assert UrlAdapter is not None

    def test_registry_importable(self):
        assert DataSourceRegistry is not None


class TestDataSourceRegistry:
    """DataSourceRegistry lists and describes adapters."""

    def test_list_adapters(self):
        adapters = DataSourceRegistry.list_adapters()
        assert isinstance(adapters, list)
        assert len(adapters) > 0
        assert "pandas" in adapters
        assert "file" in adapters

    def test_get_adapter_info(self):
        for adapter_type in DataSourceRegistry.list_adapters():
            info = DataSourceRegistry.get_adapter_info(adapter_type)
            assert "class" in info


class TestPandasAdapter:
    """PandasAdapter loads, validates, and converts dict-based data."""

    def test_load_validate_convert(self):
        config = {
            "type": "pandas",
            "data": {
                "date": pd.date_range(start="2020-01-01", periods=10, freq="D"),
                "value": list(range(10, 20)),
            },
            "time_column": "date",
            "target_column": "value",
        }

        adapter = DataSourceRegistry.create_adapter(config)
        assert isinstance(adapter, PandasAdapter)

        data = adapter.load()
        assert len(data) == 10

        is_valid, validation = adapter.validate(data)
        assert is_valid

        y, X = adapter.to_sktime_format(data)
        assert y.shape[0] == 10

        metadata = adapter.get_metadata()
        assert metadata["rows"] == 10


class TestExecutorDataIntegration:
    """Executor can load data sources and run fit_predict via data handles."""

    def test_load_and_predict_with_data_handle(self):
        executor = get_executor()

        config = {
            "type": "pandas",
            "data": {
                "date": pd.date_range(start="2020-01-01", periods=50, freq="D"),
                "sales": [100 + i for i in range(50)],
            },
            "time_column": "date",
            "target_column": "sales",
        }

        result = executor.load_data_source(config)
        assert result["success"]
        assert "data_handle" in result
        assert result["metadata"]["rows"] == 50

        handles = executor.list_data_handles()
        assert handles["count"] >= 1

        est_result = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        assert est_result["success"]

        pred_result = executor.fit_predict(
            est_result["handle"],
            "",
            5,
            data_handle=result["data_handle"],
        )
        assert pred_result["success"]
        assert len(pred_result["predictions"]) == 5

        cleanup = executor.release_data_handle(result["data_handle"])
        assert cleanup["success"]
