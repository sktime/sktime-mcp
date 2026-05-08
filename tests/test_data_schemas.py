"""Tests for load_data_source config schema validation."""

from sktime_mcp.tools.data_schemas import validate_config


def test_missing_type():
    result = validate_config({})
    assert not result["valid"]
    assert "type" in result["missing_fields"]


def test_unknown_type():
    result = validate_config({"type": "unknown"})
    assert not result["valid"]
    assert "unknown" in result["error"]


def test_pandas_missing_data():
    result = validate_config({"type": "pandas"})
    assert not result["valid"]
    assert "data" in result["missing_fields"]


def test_pandas_valid():
    result = validate_config({"type": "pandas", "data": {"value": [1, 2]}})
    assert result["valid"]


def test_file_missing_path():
    result = validate_config({"type": "file"})
    assert not result["valid"]
    assert "path" in result["missing_fields"]


def test_file_valid():
    result = validate_config({"type": "file", "path": "/data.csv"})
    assert result["valid"]


def test_url_missing_url():
    result = validate_config({"type": "url"})
    assert not result["valid"]
    assert "url" in result["missing_fields"]


def test_sql_missing_connection():
    result = validate_config({"type": "sql", "query": "SELECT 1"})
    assert not result["valid"]


def test_sql_missing_query():
    result = validate_config({"type": "sql", "connection_string": "sqlite:///db"})
    assert not result["valid"]


def test_sql_valid_with_connection_string():
    result = validate_config({
        "type": "sql",
        "connection_string": "sqlite:///db",
        "query": "SELECT * FROM data",
    })
    assert result["valid"]


def test_sql_valid_with_dialect():
    result = validate_config({
        "type": "sql",
        "dialect": "postgresql",
        "database": "mydb",
        "table": "sales",
    })
    assert result["valid"]