"""
Structured config schemas for data source adapters.

Each adapter accepts a ``config`` dict whose exact keys depend on the source
``type``. This module publishes one JSON-Schema-style contract per source type
so agents calling ``load_data_source`` get a clear, predictable contract
instead of guessing keys (#350).

The schemas are informational: ``additionalProperties`` stays ``True`` because
adapters accept pass-through option dicts (``csv_options`` and friends).
``validate_config`` enforces only what the adapters genuinely require.
"""

from typing import Any

#: Supported file formats for the ``file`` and ``url`` adapters.
FILE_FORMATS = ("csv", "excel", "parquet", "json")

#: Supported SQL dialects for the ``sql`` adapter.
SQL_DIALECTS = ("postgresql", "mysql", "sqlite", "mssql")

CONFIG_SCHEMAS: dict[str, dict[str, Any]] = {
    "pandas": {
        "title": "pandas",
        "description": "In-memory pandas DataFrame or column mapping.",
        "required": ["data"],
        "properties": {
            "data": {
                "type": "object",
                "description": (
                    "Column-name to values mapping, or a pandas DataFrame object. Required."
                ),
            },
            "time_column": {
                "type": ["string", "array"],
                "description": (
                    "Column (or list of columns) to use as the time index. "
                    "Optional; auto-detected when omitted."
                ),
            },
            "target_column": {
                "type": "string",
                "description": (
                    "Target column for forecasting. Optional; defaults to the first column."
                ),
            },
            "exog_columns": {
                "type": "array",
                "description": "Optional list of exogenous (feature) columns.",
            },
            "frequency": {
                "type": "string",
                "description": (
                    "Optional pandas offset alias (e.g. 'D', 'h'). Inferred when omitted."
                ),
            },
        },
        "additionalProperties": True,
    },
    "file": {
        "title": "file",
        "description": "Local CSV, Excel, Parquet, or JSON file.",
        "required": ["path"],
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the data file. Required.",
            },
            "format": {
                "type": "string",
                "enum": list(FILE_FORMATS),
                "description": (
                    "File format. Optional; auto-detected from the extension when omitted."
                ),
            },
            "time_column": {
                "type": "string",
                "description": "Column to use as the time index.",
            },
            "target_column": {
                "type": "string",
                "description": "Target column for forecasting.",
            },
            "exog_columns": {
                "type": "array",
                "description": "Optional list of exogenous (feature) columns.",
            },
            "csv_options": {
                "type": "object",
                "description": "Extra keyword arguments for pd.read_csv.",
            },
            "excel_options": {
                "type": "object",
                "description": "Extra keyword arguments for pd.read_excel.",
            },
            "json_options": {
                "type": "object",
                "description": "Extra keyword arguments for pd.read_json.",
            },
            "parse_dates": {
                "type": "boolean",
                "description": "Parse the time column to datetime. Default true.",
            },
            "frequency": {
                "type": "string",
                "description": "Optional pandas offset alias (e.g. 'D', 'h').",
            },
        },
        "additionalProperties": True,
    },
    "sql": {
        "title": "sql",
        "description": (
            "SQL database via SQLAlchemy. Provide either 'connection_string' "
            "or the 'dialect' connection components, and either 'query' or "
            "'table' (+ optional 'filters')."
        ),
        "required": [],
        "requirements": ("One of 'connection_string' or 'dialect'; one of 'query' or 'table'."),
        "properties": {
            "connection_string": {
                "type": "string",
                "description": (
                    "Full SQLAlchemy connection string, e.g. 'postgresql://user:pass@host:5432/db'."
                ),
            },
            "dialect": {
                "type": "string",
                "enum": list(SQL_DIALECTS),
                "description": (
                    "Database dialect, used with the host/port/database/ "
                    "username/password components instead of a connection "
                    "string."
                ),
            },
            "host": {"type": "string", "description": "Database host."},
            "port": {"type": "integer", "description": "Database port."},
            "database": {"type": "string", "description": "Database name."},
            "username": {"type": "string", "description": "Database username."},
            "password": {"type": "string", "description": "Database password."},
            "query": {
                "type": "string",
                "description": "SQL query returning the time series.",
            },
            "query_params": {
                "type": "object",
                "description": "Named parameters for 'query'.",
            },
            "table": {
                "type": "string",
                "description": "Table name, as an alternative to 'query'.",
            },
            "filters": {
                "type": "object",
                "description": (
                    "Column to condition mapping, e.g. {'date': '>=2020-01-01'}. Used with 'table'."
                ),
            },
            "time_column": {
                "type": "string",
                "description": "Column to use as the time index.",
            },
            "target_column": {
                "type": "string",
                "description": "Target column for forecasting.",
            },
            "exog_columns": {
                "type": "array",
                "description": "Optional list of exogenous (feature) columns.",
            },
            "parse_dates": {
                "type": "array",
                "description": "Columns to parse as dates.",
            },
            "frequency": {
                "type": "string",
                "description": "Optional pandas offset alias (e.g. 'D', 'h').",
            },
        },
        "additionalProperties": True,
    },
    "url": {
        "title": "url",
        "description": "Remote CSV, Excel, or Parquet file downloaded from a URL.",
        "required": ["url"],
        "properties": {
            "url": {
                "type": "string",
                "description": "Web URL of the data file. Required.",
            },
            "format": {
                "type": "string",
                "enum": ["csv", "excel", "parquet"],
                "description": ("File format. Optional; auto-detected from the URL when omitted."),
            },
            "time_column": {
                "type": "string",
                "description": "Column to use as the time index.",
            },
            "target_column": {
                "type": "string",
                "description": "Target column for forecasting.",
            },
            "exog_columns": {
                "type": "array",
                "description": "Optional list of exogenous (feature) columns.",
            },
            "csv_options": {
                "type": "object",
                "description": "Extra keyword arguments for pd.read_csv.",
            },
            "parse_dates": {
                "type": "boolean",
                "description": "Parse the time column to datetime. Default true.",
            },
            "frequency": {
                "type": "string",
                "description": "Optional pandas offset alias (e.g. 'D', 'h').",
            },
        },
        "additionalProperties": True,
    },
}


def get_config_schema(source_type: str) -> dict[str, Any]:
    """Return the config schema for a source type.

    Parameters
    ----------
    source_type : str
        Source type, e.g. ``"pandas"``, ``"sql"``, ``"file"``, ``"url"``.

    Returns
    -------
    dict
        The JSON-Schema-style contract for that source type's ``config``.

    Raises
    ------
    ValueError
        If the source type is not registered.
    """
    if source_type not in CONFIG_SCHEMAS:
        available = ", ".join(sorted(CONFIG_SCHEMAS))
        raise ValueError(f"Unknown data source type: '{source_type}'. Available types: {available}")
    return CONFIG_SCHEMAS[source_type]


def validate_config(source_type: str, config: dict[str, Any]) -> list[str]:
    """Check a ``load_data_source`` config against its type schema.

    Only enforces what the adapters genuinely require; unknown keys are
    allowed because adapters accept pass-through option dicts.

    Parameters
    ----------
    source_type : str
        Source type the config is for.
    config : dict
        The config to check.

    Returns
    -------
    list of str
        Human-readable problems; empty when the config is acceptable.
    """
    problems: list[str] = []

    if source_type not in CONFIG_SCHEMAS:
        available = ", ".join(sorted(CONFIG_SCHEMAS))
        return [f"Unknown source type '{source_type}'. Available types: {available}."]

    schema = CONFIG_SCHEMAS[source_type]
    properties = schema["properties"]

    for key in schema.get("required", []):
        if key not in config:
            problems.append(f"Missing required key: '{key}'.")

    if source_type == "sql":
        if "connection_string" not in config and "dialect" not in config:
            problems.append("Missing connection info: provide 'connection_string' or 'dialect'.")
        if "query" not in config and "table" not in config:
            problems.append("Missing data selector: provide 'query' or 'table'.")
        dialect = config.get("dialect")
        if dialect is not None and dialect not in SQL_DIALECTS:
            problems.append(
                f"Unsupported dialect '{dialect}'. Supported dialects: {', '.join(SQL_DIALECTS)}."
            )

    if source_type in ("file", "url"):
        file_format = config.get("format")
        allowed = FILE_FORMATS if source_type == "file" else ("csv", "excel", "parquet")
        if file_format is not None and file_format not in allowed:
            problems.append(
                f"Unsupported format '{file_format}'. Supported formats: {', '.join(allowed)}."
            )

    if source_type == "pandas":
        data = config.get("data")
        if data is not None and not isinstance(data, dict):
            try:
                import pandas as pd

                is_df = isinstance(data, pd.DataFrame)
            except ImportError:  # pragma: no cover
                is_df = False
            if not is_df:
                problems.append(
                    "'data' must be a dict of columns or a pandas DataFrame, "
                    f"got {type(data).__name__}."
                )

    for key in ("path", "url", "connection_string", "query", "table"):
        if key in properties and key in config and not isinstance(config[key], str):
            problems.append(f"'{key}' must be a string.")

    return problems
