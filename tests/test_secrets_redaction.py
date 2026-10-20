"""Regression tests for #556 item 23 (audit F-33): secrets must not leak into
adapter metadata, error messages or the server log."""

import http.server
import json
import logging
import threading
from unittest.mock import patch

import pandas as pd
import pytest

from sktime_mcp.data.adapters.sql_adapter import SQLAdapter
from sktime_mcp.data.adapters.url_adapter import UrlAdapter
from sktime_mcp.redaction import redact_connection_string, redact_for_logging, redact_url
from sktime_mcp.server import call_tool

SECRET = "s3cr3tPW"
TOKEN = "tok3nXYZ"


# ---------------------------------------------------------------------------
# Connection strings (SQL adapter)
# ---------------------------------------------------------------------------


class TestConnectionStringRedaction:
    @pytest.mark.parametrize(
        "conn, must_keep",
        [
            (f"postgresql://alice:{SECRET}@db.example.com:5432/sales", "db.example.com:5432/sales"),
            (
                f"postgresql://db.example.com/sales?password={SECRET}&sslmode=require",
                "sslmode=require",
            ),
            (f"Driver={{ODBC Driver 17}};Server=srv;Database=d;UID=u;PWD={SECRET}", "Server=srv"),
            (f"mysql://db.example.com/sales?token={TOKEN}", "db.example.com/sales"),
            (f"bigquery://proj?credentials_path=/home/u/{SECRET}.json", "bigquery://proj"),
            (f"mssql+pyodbc:///?odbc_connect=Driver%3Dx%3BPWD%3D{SECRET}", "mssql+pyodbc:///"),
            (f"https://{TOKEN}@github.com/org/repo.csv", "github.com/org/repo.csv"),
        ],
    )
    def test_sanitize_connection_string_masks_credentials(self, conn, must_keep):
        adapter = SQLAdapter({})
        out = adapter._sanitize_connection_string(conn)
        assert SECRET not in out
        assert TOKEN not in out
        assert "***" in out
        assert must_keep in out

    def test_non_credential_strings_untouched(self):
        assert redact_connection_string("sqlite:///tmp/data.db") == "sqlite:///tmp/data.db"
        assert redact_connection_string("Driver={x};Server=h;Database=d") == (
            "Driver={x};Server=h;Database=d"
        )
        assert (
            redact_url("https://example.com/a.csv?sep=%3B") == "https://example.com/a.csv?sep=%3B"
        )

    def test_userinfo_password_masked_but_username_kept(self):
        out = redact_url(f"postgresql://alice:{SECRET}@host/db")
        assert out == "postgresql://alice:***@host/db"

    def test_sql_metadata_connection_is_redacted(self):
        pytest.importorskip("sqlalchemy")
        conn = f"postgresql://alice:{SECRET}@host/db?password={SECRET}"
        adapter = SQLAdapter({"connection_string": conn, "table": "t", "time_column": "date"})
        frame = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=3, freq="D"), "value": [1.0, 2.0, 3.0]}
        )

        class _Engine:
            def dispose(self):
                pass

        with (
            patch("sqlalchemy.create_engine", return_value=_Engine()),
            patch("pandas.read_sql", return_value=frame),
        ):
            adapter.load()

        meta = adapter.get_metadata()
        assert SECRET not in json.dumps(meta)
        assert meta["connection"] == "postgresql://alice:***@host/db?password=***"


# ---------------------------------------------------------------------------
# URLs (URL adapter)
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_http_server(tmp_path):
    """Serve a tiny CSV over a local http.server; always shut down."""
    (tmp_path / "data.csv").write_text("date,value\n2020-01-01,1\n2020-01-02,2\n2020-01-03,3\n")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def log_message(self, *args):  # keep test output quiet
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


class TestUrlRedaction:
    def test_metadata_url_is_redacted(self, csv_http_server):
        url = f"{csv_http_server}/data.csv?token={TOKEN}&sep=%2C"
        adapter = UrlAdapter({"type": "url", "url": url, "time_column": "date"})
        df = adapter.load()
        assert len(df) == 3
        meta = adapter.get_metadata()
        assert TOKEN not in json.dumps(meta)
        assert meta["url"] == f"{csv_http_server}/data.csv?token=***&sep=%2C"

    async def test_async_metadata_url_is_redacted(self, csv_http_server):
        pytest.importorskip("aiohttp")
        url = f"{csv_http_server}/data.csv?token={TOKEN}"
        adapter = UrlAdapter({"type": "url", "url": url, "time_column": "date"})
        await adapter.load_async()
        assert TOKEN not in json.dumps(adapter.get_metadata())

    def test_error_message_is_redacted(self):
        url = f"https://alice:{SECRET}@files.example.com/data.csv?token={TOKEN}"
        adapter = UrlAdapter({"type": "url", "url": url})
        with (
            patch("urllib.request.urlretrieve", side_effect=OSError("boom")),
            pytest.raises(ValueError) as excinfo,
        ):
            adapter.load()
        msg = str(excinfo.value)
        assert SECRET not in msg
        assert TOKEN not in msg
        assert "alice:***@files.example.com/data.csv?token=***" in msg
        assert "boom" in msg

    async def test_async_http_error_is_redacted(self, csv_http_server):
        pytest.importorskip("aiohttp")
        url = f"{csv_http_server}/missing.csv?token={TOKEN}"
        adapter = UrlAdapter({"type": "url", "url": url})
        with pytest.raises(ValueError) as excinfo:
            await adapter.load_async()
        assert TOKEN not in str(excinfo.value)
        assert "HTTP 404" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Structured redaction used by the server log
# ---------------------------------------------------------------------------


class TestRedactForLogging:
    def test_masks_credential_keys_anywhere(self):
        obj = {
            "config": {
                "connection_string": f"postgresql://u:{SECRET}@h/db?password={SECRET}",
                "api_key": "sk-live-123",
                "nested": {"Authorization": "Bearer abc", "token": "t"},
            },
            "url": f"https://h/x?token={TOKEN}",
            "password": SECRET,
        }
        out = redact_for_logging(obj)
        text = json.dumps(out)
        for needle in (SECRET, TOKEN, "sk-live-123", "Bearer abc"):
            assert needle not in text
        assert out["config"]["api_key"] == "***"
        assert out["config"]["nested"] == {"Authorization": "***", "token": "***"}
        assert out["config"]["connection_string"] == "postgresql://u:***@h/db?password=***"
        assert out["url"] == "https://h/x?token=***"

    def test_truncates_long_strings_and_large_payloads(self):
        out = redact_for_logging(
            {"image": "A" * 5000, "data": {"value": list(range(500))}, "rows": [1, 2, 3]}
        )
        assert len(out["image"]) < 300
        assert "truncated, 5000 chars" in out["image"]
        assert out["data"]["value"] == "<list: 500 items>"
        assert out["rows"] == [1, 2, 3]

    def test_leaves_ordinary_values_alone(self):
        obj = {"handle": "est_123", "fh": [1, 2, 3], "run_async": False, "n": None, "x": 1.5}
        assert redact_for_logging(obj) == obj


# ---------------------------------------------------------------------------
# Server log via call_tool
# ---------------------------------------------------------------------------

SQL_ARGS = {
    "config": {
        "type": "sql",
        "connection_string": f"postgresql://alice:{SECRET}@127.0.0.1:1/db?password={SECRET}",
        "table": "t",
        "api_key": "sk-live-123",
    }
}


def _log_text(caplog):
    return "\n".join(r.getMessage() for r in caplog.records if r.name == "sktime_mcp.server")


class TestServerLogRedaction:
    async def test_info_log_has_summary_but_no_secrets(self, caplog):
        caplog.set_level(logging.INFO, logger="sktime_mcp.server")
        await call_tool("load_data_source", SQL_ARGS)
        text = _log_text(caplog)
        assert SECRET not in text
        assert "sk-live-123" not in text
        assert "=== Tool Call: load_data_source ===" in text
        assert "alice:***@127.0.0.1:1/db?password=***" in text
        assert "=== Result for load_data_source: success=False keys=[" in text
        assert "bytes=" in text
        # The full result body is not logged at INFO.
        assert "Result body for load_data_source" not in text

    async def test_debug_log_has_redacted_body(self, caplog):
        caplog.set_level(logging.DEBUG, logger="sktime_mcp.server")
        await call_tool("load_data_source", SQL_ARGS)
        text = _log_text(caplog)
        assert SECRET not in text
        assert "Result body for load_data_source" in text
        assert '"success": false' in text

    async def test_inline_data_is_not_logged_verbatim(self, caplog):
        caplog.set_level(logging.DEBUG, logger="sktime_mcp.server")
        values = [1000.5 + i for i in range(300)]
        dates = [str(d.date()) for d in pd.date_range("2020-01-01", periods=300, freq="D")]
        args = {
            "config": {
                "type": "pandas",
                "data": {"date": dates, "value": values},
                "time_column": "date",
                "target_column": "value",
            }
        }
        response = await call_tool("load_data_source", args)
        text = _log_text(caplog)
        assert "<list: 300 items>" in text
        assert "1150.5" not in text  # a mid-payload row is not in the log

        # Release the handle the call created so other tests see a clean registry.
        from sktime_mcp.tools.data_tools import release_data_handle_tool

        handle = json.loads(response[0].text).get("data_handle")
        if handle:
            release_data_handle_tool(handle)
