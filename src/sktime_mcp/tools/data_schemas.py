"""Structured config schemas for load_data_source by source type."""

from typing import Any

SCHEMAS = {
    "pandas": {
        "required": ["type", "data"],
        "optional": ["time_column", "target_column", "exog_columns", "frequency"],
        "description": "Load from a pandas-compatible dict or DataFrame.",
        "example": {
            "type": "pandas",
            "data": {"date": ["2020-01", "2020-02"], "value": [100, 200]},
            "time_column": "date",
            "target_column": "value",
        },
    },
    "sql": {
        "required": ["type"],
        "one_of": [
            ["connection_string"],
            ["dialect", "database"],
        ],
        "query_one_of": [["query"], ["table"]],
        "optional": [
            "time_column", "target_column", "exog_columns",
            "host", "port", "username", "password",
            "filters", "parse_dates", "frequency",
        ],
        "description": "Load from a SQL database. Provide connection_string or dialect+database, and query or table.",
        "example": {
            "type": "sql",
            "connection_string": "postgresql://user:pass@host:5432/db",
            "query": "SELECT date, value FROM sales",
            "time_column": "date",
            "target_column": "value",
        },
    },
    "file": {
        "required": ["type", "path"],
        "optional": [
            "format", "time_column", "target_column", "exog_columns",
            "csv_options", "excel_options", "parse_dates", "frequency",
        ],
        "description": "Load from a local CSV, Excel, or Parquet file.",
        "example": {
            "type": "file",
            "path": "/path/to/data.csv",
            "time_column": "date",
            "target_column": "value",
            "csv_options": {"sep": ",", "encoding": "utf-8"},
        },
    },
    "url": {
        "required": ["type", "url"],
        "optional": [
            "format", "time_column", "target_column", "exog_columns",
            "csv_options", "parse_dates", "frequency",
        ],
        "description": "Load from a remote URL pointing to a CSV, Excel, or Parquet file.",
        "example": {
            "type": "url",
            "url": "https://example.com/data.csv",
            "time_column": "date",
            "target_column": "value",
            "csv_options": {"sep": ","},
        },
    },
}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a load_data_source config against its source type schema.

    Returns a dict with:
    - valid: bool
    - missing_fields: list of missing required fields
    - suggestion: example config for the given type
    - error: human-readable error message if invalid
    """
    source_type = config.get("type")

    if not source_type:
        return {
            "valid": False,
            "missing_fields": ["type"],
            "error": "Missing required field 'type'. Must be one of: "
            + ", ".join(SCHEMAS.keys()),
            "suggestion": None,
        }

    if source_type not in SCHEMAS:
        return {
            "valid": False,
            "missing_fields": [],
            "error": f"Unknown source type '{source_type}'. "
            f"Valid types are: {', '.join(SCHEMAS.keys())}",
            "suggestion": None,
        }

    schema = SCHEMAS[source_type]
    missing = [f for f in schema["required"] if f not in config]

    if missing:
        return {
            "valid": False,
            "missing_fields": missing,
            "error": f"Config for type '{source_type}' is missing: {missing}",
            "suggestion": schema["example"],
        }

    # SQL conditional validation
    if source_type == "sql":
        has_connection = "connection_string" in config or (
            "dialect" in config and "database" in config
        )
        if not has_connection:
            return {
                "valid": False,
                "missing_fields": ["connection_string or dialect+database"],
                "error": "SQL config must have 'connection_string' or both 'dialect' and 'database'",
                "suggestion": schema["example"],
            }
        has_query = "query" in config or "table" in config
        if not has_query:
            return {
                "valid": False,
                "missing_fields": ["query or table"],
                "error": "SQL config must have 'query' or 'table'",
                "suggestion": schema["example"],
            }

    return {"valid": True, "missing_fields": [], "error": None, "suggestion": None}