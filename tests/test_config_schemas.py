"""Tests for structured per-type config schemas (#350)."""

import pandas as pd
import pytest

from sktime_mcp.data import (
    CONFIG_SCHEMAS,
    DataSourceRegistry,
    get_config_schema,
    validate_config,
)
from sktime_mcp.tools.data_tools import list_data_sources_tool


class TestConfigSchemaCoverage:
    """Every registered adapter type publishes a schema."""

    def test_schema_for_every_adapter(self):
        for source_type in DataSourceRegistry.list_adapters():
            schema = get_config_schema(source_type)
            assert schema["title"] == source_type
            assert "required" in schema
            assert "properties" in schema

    def test_registry_delegates_to_schemas(self):
        assert DataSourceRegistry.get_config_schema("file") is get_config_schema("file")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown data source type"):
            get_config_schema("s3")

    def test_schemas_are_json_serializable(self):
        import json

        json.dumps(CONFIG_SCHEMAS)


class TestValidateConfig:
    """validate_config reports problems; empty list means acceptable."""

    def test_pandas_missing_data(self):
        problems = validate_config("pandas", {"type": "pandas"})
        assert any("'data'" in p for p in problems)

    def test_pandas_valid_dict_and_dataframe(self):
        assert validate_config("pandas", {"type": "pandas", "data": {"y": [1, 2]}}) == []
        df = pd.DataFrame({"y": [1, 2]})
        assert validate_config("pandas", {"type": "pandas", "data": df}) == []

    def test_pandas_wrong_data_type(self):
        problems = validate_config("pandas", {"type": "pandas", "data": "nope"})
        assert any("'data' must be" in p for p in problems)

    def test_file_missing_path(self):
        problems = validate_config("file", {"type": "file"})
        assert any("'path'" in p for p in problems)

    def test_file_bad_format(self):
        problems = validate_config("file", {"type": "file", "path": "a.csv", "format": "docx"})
        assert any("Unsupported format" in p for p in problems)

    def test_file_valid(self):
        assert validate_config("file", {"type": "file", "path": "a.csv"}) == []
        assert validate_config("file", {"type": "file", "path": "a.xlsx", "format": "excel"}) == []

    def test_sql_needs_connection_and_selector(self):
        problems = validate_config("sql", {"type": "sql"})
        assert any("connection_string" in p for p in problems)
        assert any("'query' or 'table'" in p for p in problems)

    def test_sql_dialect_plus_table_ok(self):
        config = {"type": "sql", "dialect": "sqlite", "table": "sales"}
        assert validate_config("sql", config) == []

    def test_sql_bad_dialect(self):
        config = {"type": "sql", "dialect": "oracle", "query": "SELECT 1"}
        problems = validate_config("sql", config)
        assert any("Unsupported dialect" in p for p in problems)

    def test_url_missing_url(self):
        problems = validate_config("url", {"type": "url"})
        assert any("'url'" in p for p in problems)

    def test_url_valid(self):
        config = {"type": "url", "url": "https://example.com/data.csv"}
        assert validate_config("url", config) == []

    def test_unknown_source_type(self):
        problems = validate_config("s3", {"type": "s3"})
        assert any("Unknown source type" in p for p in problems)

    def test_unknown_keys_allowed(self):
        # Adapters accept pass-through option dicts, so unknown keys must not fail.
        config = {"type": "file", "path": "a.csv", "csv_options": {"sep": ";"}}
        assert validate_config("file", config) == []


class TestCreateAdapterValidation:
    """create_adapter rejects bad configs with an agent-friendly error."""

    def test_missing_type_still_rejected(self):
        with pytest.raises(ValueError, match="must specify 'type'"):
            DataSourceRegistry.create_adapter({})

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown source type"):
            DataSourceRegistry.create_adapter({"type": "s3"})

    def test_missing_required_key_rejected(self):
        with pytest.raises(ValueError, match="Missing required key: 'path'"):
            DataSourceRegistry.create_adapter({"type": "file"})

    def test_sql_missing_selector_rejected(self):
        with pytest.raises(ValueError, match="Missing data selector"):
            DataSourceRegistry.create_adapter(
                {"type": "sql", "connection_string": "sqlite:///x.db"}
            )

    def test_valid_config_still_creates_adapter(self):
        adapter = DataSourceRegistry.create_adapter({"type": "pandas", "data": {"y": [1, 2, 3]}})
        assert adapter is not None


class TestListDataSourcesExposesSchemas:
    """list_data_sources publishes the per-type contracts."""

    def test_config_schemas_present(self):
        result = list_data_sources_tool()
        assert result["success"] is True
        schemas = result["config_schemas"]
        for source_type in result["sources"]:
            assert source_type in schemas
            assert "required" in schemas[source_type]
