"""Persistence edge cases (#556 items 21/22: audit F-28, F-29, F-30, F-32, N-2).

- F-28: ``export_code`` output must run outside the server for specs that
  reference ``np.``/``pd.`` and for datasets that return ``(y, X)`` tuples.
- F-29: ``save_model`` must treat ``file://`` URIs as local paths.
- F-30: ``transform_data(action="convert", to_mtype="pd.DataFrame")`` must
  carry the handle's exogenous ``X`` across.
- F-32: ``save_data(format="json")`` must handle MultiIndex handles and a
  column literally named ``time``.
- N-2: ``save_data`` must report the time column it wrote so the file can be
  loaded back with ``load_data_source(time_column=...)``.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.tools.codegen import export_code_tool
from sktime_mcp.tools.instantiate import instantiate_tool
from sktime_mcp.tools.save_data import save_data_tool
from sktime_mcp.tools.save_model import resolve_model_path, save_model_tool
from sktime_mcp.tools.transform_data import transform_data_tool


def _run_exported(code: str) -> subprocess.CompletedProcess:
    """Execute generated code in a fresh interpreter (no server namespace)."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def estimator(request):
    """Instantiate a spec and release the handle afterwards."""
    handles = []

    def _make(spec: str) -> str:
        res = instantiate_tool(spec=spec)
        assert res["success"], res
        handles.append(res["handle"])
        return res["handle"]

    yield _make
    hm = get_handle_manager()
    for h in handles:
        hm.release_handle(h)


@pytest.fixture
def register_handle():
    """Register an inline (y, X) pair as a data handle; clean up afterwards."""
    ex = get_executor()
    created = []

    def _register(handle_id, y, X=None):
        ex._register_data_handle(
            handle_id,
            {
                "y": y,
                "X": X,
                "metadata": {"rows": len(y)},
                "validation": {"valid": True, "errors": [], "warnings": []},
                "config": {},
            },
        )
        created.append(handle_id)
        return handle_id

    yield _register
    for h in created:
        ex._data_handles.pop(h, None)


# ---------------------------------------------------------------------------
# F-28: export_code output runs outside the server
# ---------------------------------------------------------------------------


class TestExportCodeRunsOutsideServer:
    def test_np_spec_constructs_in_subprocess(self, estimator):
        handle = estimator("CurveFitForecaster(function=np.exp)")
        res = export_code_tool(handle, var_name="model")
        assert res["success"], res
        code = res["code"]
        assert "import numpy as np" in code
        assert "from sktime.forecasting.trend import CurveFitForecaster" in code
        assert "craft(" not in code

        proc = _run_exported(code + "\nprint(type(model).__name__)")
        assert proc.returncode == 0, proc.stderr
        assert "CurveFitForecaster" in proc.stdout

    def test_np_spec_with_fit_example_runs(self, estimator):
        handle = estimator("NaiveForecaster(sp=int(np.sqrt(144)))")
        res = export_code_tool(handle, include_fit_example=True, dataset="airline")
        assert res["success"], res
        proc = _run_exported(res["code"])
        assert proc.returncode == 0, proc.stderr

    def test_pd_spec_imports_pandas(self, estimator):
        handle = estimator("NaiveForecaster(sp=int(pd.to_numeric('12')))")
        res = export_code_tool(handle)
        assert res["success"], res
        assert "import pandas as pd" in res["code"]
        assert "import numpy" not in res["code"]
        proc = _run_exported(res["code"])
        assert proc.returncode == 0, proc.stderr

    def test_plain_spec_still_uses_craft(self, estimator):
        handle = estimator("NaiveForecaster(sp=12)")
        res = export_code_tool(handle)
        assert res["success"], res
        assert "from sktime.registry import craft" in res["code"]
        assert "import numpy" not in res["code"]

    def test_block_spec_with_np_keeps_craft_and_warns(self):
        import numpy as np
        from sktime.forecasting.trend import CurveFitForecaster

        # Multi-statement specs cannot be built via instantiate on sktime 1.0.1
        # (craft's exec path mis-indents them), so register the handle directly.
        hm = get_handle_manager()
        handle = hm.create_handle(
            "CurveFitForecaster",
            CurveFitForecaster(function=np.exp),
            params={"spec": "f = CurveFitForecaster(function=np.exp)\nreturn f"},
        )
        try:
            res = export_code_tool(handle)
        finally:
            hm.release_handle(handle)
        assert res["success"], res
        assert "craft(" in res["code"]
        assert any("np" in w for w in res.get("warnings", [])), res

    def test_fit_example_unpacks_exogenous_dataset(self, estimator):
        handle = estimator("NaiveForecaster(strategy='last')")
        res = export_code_tool(handle, include_fit_example=True, dataset="longley")
        assert res["success"], res
        code = res["code"]
        assert "y, X = load_longley()" in code
        assert "X=" in code, "exogenous X must be passed to fit/predict"

        proc = _run_exported(code)
        assert proc.returncode == 0, proc.stderr

    def test_transformer_example_unpacks_exogenous_dataset(self, estimator):
        handle = estimator("Detrender()")
        res = export_code_tool(handle, include_fit_example=True, dataset="longley")
        assert res["success"], res
        assert "y, X = load_longley()" in res["code"]
        proc = _run_exported(res["code"])
        assert proc.returncode == 0, proc.stderr

    def test_fit_example_single_return_dataset_unchanged(self, estimator):
        handle = estimator("NaiveForecaster(strategy='last')")
        res = export_code_tool(handle, include_fit_example=True, dataset="airline")
        assert res["success"], res
        assert "y = load_airline()" in res["code"]
        assert "y, X" not in res["code"]


# ---------------------------------------------------------------------------
# F-29: save_model with a file:// URI
# ---------------------------------------------------------------------------


class TestSaveModelFileUri:
    def test_resolve_file_uri_to_local_path(self, tmp_path):
        target = tmp_path / "model_dir"
        assert resolve_model_path(f"file://{target}") == str(target)
        assert resolve_model_path(f"file://localhost{target}") == str(target)
        assert resolve_model_path("file://~/model_dir") == str(Path("~/model_dir").expanduser())

    def test_other_schemes_still_pass_through(self):
        for uri in ("s3://bucket/model", "runs:/abc/model", "models:/m/1"):
            assert resolve_model_path(uri) == uri

    def test_save_model_file_uri_targets_local_dir(self, monkeypatch, tmp_path):
        import sktime_mcp.tools.save_model as save_model_module

        calls = {}

        def fake_save_model(**kwargs):
            calls.update(kwargs)

        monkeypatch.setattr(save_model_module, "_get_mlflow_save_model", lambda: fake_save_model)
        monkeypatch.chdir(tmp_path)

        hm = get_handle_manager()
        handle = hm.create_handle("NaiveForecaster", object())
        hm.mark_fitted(handle)
        try:
            target = tmp_path / "saved" / "model"
            result = save_model_tool(handle, f"file://{target}")
        finally:
            hm.release_handle(handle)

        assert result["success"], result
        assert calls["path"] == str(target)
        assert result["saved_path"] == str(target)
        assert not (tmp_path / "file:").exists()


# ---------------------------------------------------------------------------
# F-30: convert to pd.DataFrame keeps X
# ---------------------------------------------------------------------------


class TestConvertKeepsExogenous:
    def test_convert_to_dataframe_carries_exog(self, register_handle):
        idx = pd.date_range("2024-01-01", periods=6, freq="D")
        y = pd.Series(range(6), index=idx, dtype=float, name="value")
        X = pd.DataFrame({"temp": range(6), "promo": [0, 1, 0, 1, 0, 1]}, index=idx)
        dh = register_handle("test_convert_keeps_x", y, X)

        res = transform_data_tool(data_handle=dh, action="convert", to_mtype="pd.DataFrame")
        assert res["success"], res
        ex = get_executor()
        new = ex._data_handles.pop(res["data_handle"])
        assert isinstance(new["y"], pd.DataFrame)
        assert new["X"] is not None, "convert dropped the exogenous X"
        pd.testing.assert_frame_equal(new["X"], X)
        assert any("exogenous" in c.lower() for c in res["changes_applied"]), res


# ---------------------------------------------------------------------------
# F-32 / N-2: save_data json edge cases and the reported time column
# ---------------------------------------------------------------------------


def _load_back(ex, path, fmt, time_column, target_column):
    return ex.load_data_source(
        {
            "type": "file",
            "path": str(path),
            "format": fmt,
            "time_column": time_column,
            "target_column": target_column,
        }
    )


class TestSaveDataEdges:
    def test_json_multiindex_handle(self, register_handle, tmp_path):
        idx = pd.MultiIndex.from_product(
            [["a", "b"], pd.period_range("2024-01", periods=3, freq="M")],
            names=["instance", "period"],
        )
        y = pd.DataFrame({"value": range(6)}, index=idx, dtype=float)
        dh = register_handle("test_json_multiindex", y)

        out = tmp_path / "panel.json"
        res = save_data_tool(dh, path=str(out), format="json")
        assert res["success"], res
        assert res["time_column"] == "period"
        assert res["index_columns"] == ["instance", "period"]
        records = json.loads(out.read_text())
        assert records[0] == {"instance": "a", "period": "2024-01", "value": 0.0}

    def test_json_column_named_time(self, register_handle, tmp_path):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        y = pd.DataFrame({"time": [1.0, 2.0, 3.0], "value": [4.0, 5.0, 6.0]}, index=idx)
        dh = register_handle("test_json_time_col", y)

        out = tmp_path / "t.json"
        res = save_data_tool(dh, path=str(out), format="json")
        assert res["success"], res
        assert res["time_column"] == "time_index"
        records = json.loads(out.read_text())
        assert set(records[0]) == {"time_index", "time", "value"}

    @pytest.mark.parametrize("fmt", ["csv", "json"])
    def test_named_index_round_trips_with_reported_time_column(
        self, register_handle, tmp_path, fmt
    ):
        idx = pd.date_range("2024-01-01", periods=4, freq="D", name="date")
        y = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="value")
        dh = register_handle(f"test_roundtrip_{fmt}", y)

        out = tmp_path / f"series.{fmt}"
        res = save_data_tool(dh, path=str(out), format=fmt)
        assert res["success"], res
        assert res["time_column"] == "date"

        ex = get_executor()
        loaded = _load_back(ex, out, fmt, res["time_column"], "value")
        try:
            assert loaded["success"], loaded
            y_back = ex._data_handles[loaded["data_handle"]]["y"]
            assert isinstance(y_back.index, (pd.DatetimeIndex, pd.PeriodIndex))
            assert list(y_back.values) == [1.0, 2.0, 3.0, 4.0]
            assert loaded["metadata"].get("exog_columns", []) == []
        finally:
            ex._data_handles.pop(loaded.get("data_handle"), None)

    def test_unnamed_index_written_as_time(self, register_handle, tmp_path):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        y = pd.Series([1.0, 2.0, 3.0], index=idx, name="value")
        dh = register_handle("test_unnamed_idx", y)

        out = tmp_path / "s.csv"
        res = save_data_tool(dh, path=str(out), format="csv")
        assert res["success"], res
        assert res["time_column"] == "time"
        assert out.read_text().splitlines()[0] == "time,value"

    def test_parquet_reports_index_kept_in_file(self, register_handle, tmp_path):
        pytest.importorskip("pyarrow")
        idx = pd.date_range("2024-01-01", periods=3, freq="D", name="date")
        y = pd.Series([1.0, 2.0, 3.0], index=idx, name="value")
        dh = register_handle("test_parquet_idx", y)

        res = save_data_tool(dh, path=str(tmp_path / "s.parquet"), format="parquet")
        assert res["success"], res
        assert res["time_column"] is None
        assert "index" in res["note"].lower()
