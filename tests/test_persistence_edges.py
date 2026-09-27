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
