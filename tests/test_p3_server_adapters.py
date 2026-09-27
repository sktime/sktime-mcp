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


# ---------------------------------------------------------------------------
# F-48: the JSON-string `tags` branch is unreachable (schema says object)
# ---------------------------------------------------------------------------


class TestTagsStringBranchRemoved:
    def test_string_tags_is_structured_error_not_parsed(self):
        res = query_registry_tool(task="forecaster", tags='{"capability:pred_int": true}')
        assert res["success"] is False
        assert "dict" in res["error"].lower()

    def test_docstring_no_longer_claims_json_string_support(self):
        assert "JSON string" not in (query_registry_tool.__doc__ or "")


# ---------------------------------------------------------------------------
# F-49: sets serialise as lists, not as their repr string
# ---------------------------------------------------------------------------


class TestSanitizeSets:
    def test_set_becomes_sorted_list(self):
        assert sanitize_for_json({3, 1, 2}) == [1, 2, 3]

    def test_frozenset_becomes_sorted_list(self):
        assert sanitize_for_json(frozenset({"b", "a"})) == ["a", "b"]

    def test_nested_set_in_dict_is_json_serialisable(self):
        out = sanitize_for_json({"tags": {"x", "y"}})
        assert out == {"tags": ["x", "y"]}
        json.dumps(out)

    def test_unsortable_set_falls_back_to_list(self):
        out = sanitize_for_json({1, "a"})
        assert isinstance(out, list)
        assert set(out) == {1, "a"}
        json.dumps(out)


# ---------------------------------------------------------------------------
# F-50: run_command with a backgrounded child holding stdout
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
class TestRunCommandBackgroundChild:
    def test_timeout_returns_partial_output_and_kills_process_group(self, monkeypatch):
        monkeypatch.setattr(run_command_module, "_TIMEOUT_SECONDS", 1)
        started = time.monotonic()
        res = run_command_tool("sleep 3 & pid=$!; echo started $pid")
        elapsed = time.monotonic() - started

        assert elapsed < 2.5, f"took {elapsed:.1f}s: waited for the background child"
        assert res["success"] is False
        assert res["timed_out"] is True
        assert "started" in res["output"]
        assert "timed out" in res["error"].lower()

        # The background sleep must not outlive the tool call.
        pid = int(res["output"].split()[1])
        status = Path(f"/proc/{pid}/status")
        if status.exists():
            state = re.search(r"^State:\s+(\S)", status.read_text(), re.M)
            assert state and state.group(1) == "Z", f"pid {pid} still alive"

    def test_fast_command_is_not_marked_timed_out(self):
        res = run_command_tool("echo hello")
        assert res["success"] is True
        assert res["timed_out"] is False
        assert res["output"] == "hello"


# ---------------------------------------------------------------------------
# F-61: file adapter csv_options / parse_dates
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "tiny.csv"
    pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02", "2020-01-03"], "value": [1.0, 2.0, 3.0]}
    ).to_csv(p, index=False)
    return p


class TestFileAdapterOptions:
    def test_parse_dates_false_is_honoured(self, csv_path):
        adapter = FileAdapter(
            {
                "type": "file",
                "path": str(csv_path),
                "time_column": "date",
                "target_column": "value",
                "parse_dates": False,
            }
        )
        df = adapter.load()
        assert not isinstance(df.index, pd.DatetimeIndex)
        assert list(df.index) == ["2020-01-01", "2020-01-02", "2020-01-03"]

    def test_parse_dates_default_still_parses(self, csv_path):
        adapter = FileAdapter({"type": "file", "path": str(csv_path), "time_column": "date"})
        assert isinstance(adapter.load().index, pd.DatetimeIndex)

    def test_csv_options_string_is_structured_error(self, csv_path):
        adapter = FileAdapter({"type": "file", "path": str(csv_path), "csv_options": "sep=,"})
        with pytest.raises(ValueError, match="csv_options"):
            adapter.load()

        res = get_executor().load_data_source(
            {
                "type": "file",
                "path": str(csv_path),
                "csv_options": "sep=,",
                "target_column": "value",
            }
        )
        assert res["success"] is False
        assert res.get("error_type") != "AttributeError"
        assert "csv_options" in res["error"]

    def test_csv_options_not_mutated(self, csv_path):
        opts = {"encoding": "utf-8"}
        FileAdapter({"type": "file", "path": str(csv_path), "csv_options": opts}).load()
        assert opts == {"encoding": "utf-8"}

    def test_excel_options_string_is_structured_error(self, tmp_path):
        adapter = FileAdapter(
            {"type": "file", "path": str(tmp_path / "x.xlsx"), "excel_options": "sheet=0"}
        )
        with pytest.raises(ValueError, match="excel_options"):
            adapter._load_excel(tmp_path / "x.xlsx")


# ---------------------------------------------------------------------------
# F-62: SQL adapter URL escaping and filter validation
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db(tmp_path):
    db = tmp_path / "tiny.db"
    con = sqlite3.connect(db)
    try:
        con.execute("CREATE TABLE sales (date TEXT, value REAL)")
        con.executemany(
            "INSERT INTO sales VALUES (?, ?)",
            [
                ("2020-01-01", 1.0),
                ("2020-01-02", 2.0),
                ("2020-01-03", 3.0),
                ("2020-01-04", 4.0),
            ],
        )
        con.commit()
    finally:
        con.close()
    return db


class TestSQLAdapter:
    def test_component_url_escapes_credentials(self):
        adapter = SQLAdapter(
            {
                "type": "sql",
                "dialect": "postgresql",
                "username": "us@er",
                "password": "p@ss/w:rd",
                "host": "db.example",
                "port": 5432,
                "database": "sales",
            }
        )
        url = adapter._get_connection_string()
        assert isinstance(url, str)
        assert "p@ss/w:rd" not in url
        assert "us@er" not in url

        sqlalchemy = pytest.importorskip("sqlalchemy")
        parsed = sqlalchemy.engine.make_url(url)
        assert parsed.username == "us@er"
        assert parsed.password == "p@ss/w:rd"
        assert parsed.host == "db.example"
        assert parsed.port == 5432
        assert parsed.database == "sales"

    def test_component_url_escapes_without_sqlalchemy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sqlalchemy", None)
        monkeypatch.setitem(sys.modules, "sqlalchemy.engine", None)
        adapter = SQLAdapter(
            {"dialect": "mysql", "username": "u", "password": "p@ss", "database": "db"}
        )
        url = adapter._get_connection_string()
        assert url == "mysql://u:p%40ss@localhost/db"

    def test_scalar_filters_still_work(self, sqlite_db):
        pytest.importorskip("sqlalchemy")
        adapter = SQLAdapter(
            {
                "type": "sql",
                "dialect": "sqlite",
                "database": str(sqlite_db),
                "table": "sales",
                "filters": {"value": ">=2"},
                "time_column": "date",
            }
        )
        df = adapter.load()
        assert len(df) == 3
        assert list(df["value"]) == [2.0, 3.0, 4.0]

    @pytest.mark.parametrize("bad", [[1, 2], {"in": [1, 2]}, (1, 2)])
    def test_non_scalar_filter_is_structured_error_without_sql(self, sqlite_db, bad):
        pytest.importorskip("sqlalchemy")
        adapter = SQLAdapter(
            {
                "type": "sql",
                "dialect": "sqlite",
                "database": str(sqlite_db),
                "table": "sales",
                "filters": {"value": bad},
                "time_column": "date",
            }
        )
        with pytest.raises(ValueError) as excinfo:
            adapter.load()
        msg = str(excinfo.value)
        assert "[SQL:" not in msg
        assert "parameters:" not in msg
        assert "value" in msg and "scalar" in msg.lower()


# ---------------------------------------------------------------------------
# F-64: Dockerfile extras and xlrd for .xls
# ---------------------------------------------------------------------------


class TestPackagingExtras:
    def test_dockerfile_installs_sql_and_files_extras(self):
        dockerfile = (REPO_ROOT / "Dockerfile").read_text()
        install_lines = [ln for ln in dockerfile.splitlines() if "pip install" in ln]
        assert install_lines, "no pip install line in Dockerfile"
        assert any(re.search(r"\[sql,\s*files\]", ln) for ln in install_lines), install_lines

    def test_files_extra_includes_xlrd(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        files_block = re.search(r"^files\s*=\s*\[(.*?)^\]", pyproject, re.S | re.M)
        assert files_block, "no [files] extra"
        assert "xlrd" in files_block.group(1)
        all_block = re.search(r"^all\s*=\s*\[(.*?)^\]", pyproject, re.S | re.M)
        assert all_block and "xlrd" in all_block.group(1)

    def test_xls_missing_engine_error_names_xlrd(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "xlrd", None)
        adapter = FileAdapter({"type": "file", "path": str(tmp_path / "old.xls")})
        with pytest.raises(ImportError) as excinfo:
            adapter._load_excel(tmp_path / "old.xls")
        assert "xlrd" in str(excinfo.value)
        assert "openpyxl" not in str(excinfo.value)

    def test_xlsx_missing_engine_error_names_openpyxl(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "openpyxl", None)
        adapter = FileAdapter({"type": "file", "path": str(tmp_path / "new.xlsx")})
        with pytest.raises(ImportError) as excinfo:
            adapter._load_excel(tmp_path / "new.xlsx")
        assert "openpyxl" in str(excinfo.value)
