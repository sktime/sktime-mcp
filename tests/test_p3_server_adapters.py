"""P3 roundup for #556: server/registry and adapter/infra nits.

Covers F-46 (integral floats), F-48 (dead JSON-string tags branch),
F-49 (sets in sanitize_for_json), F-50 (run_command timeout with a
backgrounded child), F-61 (file adapter csv_options / parse_dates),
F-62 (SQL adapter URL escaping and filter validation) and F-64 (Dockerfile
extras, xlrd for .xls).
"""

import contextlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from sktime_mcp.data.adapters.file_adapter import FileAdapter
from sktime_mcp.data.adapters.sql_adapter import SQLAdapter
from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager
from sktime_mcp.server import sanitize_for_json
from sktime_mcp.tools import run_command as run_command_module
from sktime_mcp.tools.fit_predict import _validate_horizon, fit_tool, predict_tool
from sktime_mcp.tools.instantiate import instantiate_tool
from sktime_mcp.tools.job_tools import list_jobs_tool
from sktime_mcp.tools.query_registry import query_registry_tool
from sktime_mcp.tools.run_command import run_command_tool

REPO_ROOT = Path(__file__).resolve().parents[1]


def _release(handle):
    with contextlib.suppress(KeyError):
        get_handle_manager().release_handle(handle)


# ---------------------------------------------------------------------------
# F-46: integral floats accepted by the schema must not crash the slice
# ---------------------------------------------------------------------------


class TestIntegralFloats:
    def test_query_registry_accepts_integral_float_limit_and_offset(self):
        res = query_registry_tool(task="forecaster", limit=5.0, offset=0.0)
        assert res["success"], res
        assert res["count"] == 5
        assert res["limit"] == 5 and isinstance(res["limit"], int)
        assert res["offset"] == 0 and isinstance(res["offset"], int)

    def test_query_registry_tag_page_accepts_integral_float_limit(self):
        res = query_registry_tool(task="tag", limit=3.0)
        assert res["success"], res
        assert res["count"] == 3

    @pytest.mark.parametrize("kwargs", [{"limit": 5.5}, {"offset": 1.5}])
    def test_query_registry_rejects_non_integral_float(self, kwargs):
        res = query_registry_tool(task="forecaster", **kwargs)
        assert res["success"] is False
        assert "integer" in res["error"].lower()
        assert "slice" not in res["error"].lower()

    def test_list_jobs_accepts_integral_float_limit_and_offset(self):
        res = list_jobs_tool(limit=5.0, offset=0.0)
        assert res["success"], res
        assert res["limit"] == 5 and isinstance(res["limit"], int)
        assert res["offset"] == 0 and isinstance(res["offset"], int)

    @pytest.mark.parametrize("kwargs", [{"limit": 2.5}, {"offset": 0.5}])
    def test_list_jobs_rejects_non_integral_float(self, kwargs):
        res = list_jobs_tool(**kwargs)
        assert res["success"] is False
        assert "integer" in res["error"].lower()

    def test_validate_horizon_coerces_integral_float(self):
        res = _validate_horizon(12.0)
        assert res["valid"], res
        assert res["horizon"] == 12 and isinstance(res["horizon"], int)

    def test_validate_horizon_rejects_non_integral_float_and_bool(self):
        assert _validate_horizon(2.5)["valid"] is False
        assert "integer" in _validate_horizon(2.5)["error"]
        assert _validate_horizon(True)["valid"] is False

    def test_predict_accepts_integral_float_horizon(self):
        h = instantiate_tool(spec="NaiveForecaster()")["handle"]
        try:
            fit_res = fit_tool(estimator_handle=h, y_dataset="airline")
            assert fit_res["success"], fit_res
            res = predict_tool(estimator_handle=h, horizon=3.0)
            assert res["success"], res
            assert len(res["predictions"]) == 3
        finally:
            _release(h)

    def test_predict_rejects_non_integral_float_horizon(self):
        res = predict_tool(estimator_handle="est_does_not_matter", horizon=2.5)
        assert res["success"] is False
        assert "integer" in res["error"]
