"""
SQL adapter for database connections.

Supports loading data from SQL databases using SQLAlchemy.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..base import DataSourceAdapter, check_frequency_alignment, coerce_time_index


class SQLAdapter(DataSourceAdapter):
    """
    Adapter for SQL databases.

    Config example::

        {
            "type": "sql",
            "connection_string": "postgresql://user:pass@host:5432/db",
            # OR individual components:
            "dialect": "postgresql",  # postgresql, mysql, sqlite, mssql
            "host": "localhost",
            "port": 5432,
            "database": "mydb",
            "username": "user",
            "password": "pass",

            # Query
            "query": "SELECT date, value FROM sales WHERE date >= '2020-01-01'",
            # OR
            "table": "sales",
            "filters": {"date": ">=2020-01-01"},

            # Column mapping
            "time_column": "date",
            "target_column": "value",
            "exog_columns": ["feature1", "feature2"],

            # Optional
            "parse_dates": ["date"],
            "frequency": "D"
        }
    """

    def load(self) -> pd.DataFrame:
        """Load from SQL database."""
        try:
            from sqlalchemy import create_engine
        except ImportError as e:
            raise ImportError(
                "SQLAlchemy is required for SQL adapter. Install with: pip install sqlalchemy"
            ) from e

        # Get connection string
        conn_string = self._get_connection_string()
        self._check_sqlite_file_exists(conn_string)

        # Get query
        query, query_params = self._get_query()

        # Create engine and load data
        engine = create_engine(conn_string)

        try:
            # Parse dates only if explicitly requested; the time column itself is
            # coerced below so integer columns are not read as epoch datetimes.
            parse_dates = self.config.get("parse_dates", [])

            df = pd.read_sql(
                query,
                engine,
                params=query_params if query_params else None,
                parse_dates=parse_dates if parse_dates else None,
            )
        finally:
            engine.dispose()

        # Set time index. Integer columns become an integer index; only
        # date-like values are parsed to datetime; with no time_column the
        # RangeIndex is kept (shared with the pandas and file adapters, F-13/F-14).
        time_col = self.config.get("time_column")
        df, index_warnings = coerce_time_index(df, time_col)

        # Sort by time
        df = df.sort_index()

        # Set frequency if specified
        freq = self.config.get("frequency")
        if freq:
            # Refuse a freq whose anchor misses the timestamps: asfreq would
            # silently replace every row with NaN (F-15)
            check_frequency_alignment(df.index, freq)
            df = df.asfreq(freq)

        self._data = df

        # Determine frequency for metadata (infer_freq needs >= 3 points)
        if isinstance(df.index, pd.DatetimeIndex):
            from .pandas_adapter import _safe_infer_freq

            freq_str = str(df.index.freq) if df.index.freq else _safe_infer_freq(df.index)
        else:
            freq_str = "Integer"

        self._metadata = {
            "source": "sql",
            "connection": self._sanitize_connection_string(conn_string),
            "rows": len(df),
            "columns": list(df.columns),
            "frequency": freq_str,
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
        }
        if index_warnings:
            self._metadata["validation"] = {"valid": True, "errors": [], "warnings": index_warnings}

        return df

    @staticmethod
    def _check_sqlite_file_exists(conn_string: str) -> None:
        """Refuse a file-backed sqlite URL whose file is missing.

        ``create_engine`` + connect would silently create an empty database at
        that path and then fail with "no such table" (F-14).
        """
        from sqlalchemy.engine import make_url

        try:
            url = make_url(conn_string)
        except Exception:
            return  # let create_engine report a malformed URL
        if url.get_backend_name() != "sqlite":
            return
        database = url.database
        if not database or database == ":memory:" or database.startswith("file:"):
            return
        if not Path(database).exists():
            raise FileNotFoundError(
                f"SQLite database file not found: {database!r} (resolved from the "
                f"connection string). Check the path; an absolute path needs four "
                f"slashes, e.g. 'sqlite:////abs/path/db.sqlite'."
            )

    def _get_connection_string(self) -> str:
        """Build connection string from config."""
        # Check if connection string is provided directly
        if "connection_string" in self.config:
            return self.config["connection_string"]

        # Build from components
        dialect = self.config.get("dialect")
        if not dialect:
            raise ValueError("Must provide 'connection_string' or 'dialect'")

        # SQLite special case
        if dialect == "sqlite":
            database = self.config.get("database", "database.db")
            return f"sqlite:///{database}"

        # Other databases
        username = self.config.get("username", "")
        password = self.config.get("password", "")
        host = self.config.get("host", "localhost")
        port = self.config.get("port", "")
        database = self.config.get("database", "")

        # Build connection string
        auth = f"{username}:{password}@" if username else ""
        port_str = f":{port}" if port else ""

        return f"{dialect}://{auth}{host}{port_str}/{database}"

    def _get_query(self) -> tuple[Any, dict[str, Any]]:
        """Get SQL query and parameters from config."""
        from sqlalchemy import text

        # Check if query is provided directly
        if "query" in self.config:
            return self.config["query"], self.config.get("query_params", {})

        # Build query from table and filters
        table = self.config.get("table")
        if not table:
            raise ValueError("Must provide 'query' or 'table'")
        table = self._validate_identifier(table, "table")

        # Simple query builder
        query = f"SELECT * FROM {table}"
        query_params: dict[str, Any] = {}

        # Add filters if provided
        filters = self.config.get("filters", {})
        if filters:
            conditions = []
            for param_idx, (col, value) in enumerate(filters.items()):
                column_name = self._validate_identifier(col, "column")
                param_name = f"filter_{param_idx}"

                # Simple filter handling
                if isinstance(value, str) and value.startswith((">=", "<=", ">", "<", "!=")):
                    operator = value[:2] if value[:2] in [">=", "<=", "!="] else value[0]
                    val = value[2:] if len(operator) == 2 else value[1:]
                    conditions.append(f"{column_name} {operator} :{param_name}")
                    query_params[param_name] = val
                else:
                    conditions.append(f"{column_name} = :{param_name}")
                    query_params[param_name] = value

            query += " WHERE " + " AND ".join(conditions)

        return text(query), query_params

    def _validate_identifier(self, identifier: str, kind: str) -> str:
        """Allow only safe SQL identifiers."""
        if not isinstance(identifier, str):
            raise ValueError(f"Invalid {kind} identifier: {identifier}")

        if not re.fullmatch(r"[a-zA-Z0-9_.]+", identifier):
            raise ValueError(
                f"Invalid {kind} identifier '{identifier}'. Only [a-zA-Z0-9_.] are allowed."
            )
        return identifier

    def _sanitize_connection_string(self, conn_string: str) -> str:
        """Remove credentials from connection string for metadata."""
        # Hide password in connection string but preserve the dialect/protocol
        if "@" in conn_string:
            try:
                protocol_auth, rest = conn_string.split("@", 1)
                if "://" in protocol_auth:
                    protocol, _ = protocol_auth.split("://", 1)
                    return f"{protocol}://***@{rest}"
                return f"***@{rest}"
            except Exception:
                return f"***@{conn_string.split('@')[-1]}"
        return conn_string

    def validate(self, data: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """Validate SQL data using pandas adapter validation."""
        # Reuse pandas validation logic
        from .pandas_adapter import PandasAdapter

        pandas_adapter = PandasAdapter(
            {
                "data": data,
                "target_column": self.config.get("target_column"),
                "exog_columns": self.config.get("exog_columns", []),
            }
        )
        return pandas_adapter.validate(data)
