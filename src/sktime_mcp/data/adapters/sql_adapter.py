"""
SQL adapter for database connections.

Supports loading data from SQL databases using SQLAlchemy.
"""

from typing import Any

import pandas as pd

from ..base import DataSourceAdapter


class SQLAdapter(DataSourceAdapter):
    """
    Adapter for SQL databases.

    Config example:
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

        # Get query
        query = self._get_query()

        # Create engine and load data
        engine = create_engine(conn_string)

        try:
            # Parse dates if specified
            parse_dates = self.config.get("parse_dates", [])
            if not parse_dates and self.config.get("time_column"):
                parse_dates = [self.config["time_column"]]

            df = pd.read_sql(query, engine, parse_dates=parse_dates if parse_dates else None)
        finally:
            engine.dispose()

        # Set time index
        time_col = self.config.get("time_column")
        if time_col and time_col in df.columns:
            df = df.set_index(time_col)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to datetime: {e}") from e

        # Sort by time
        df = df.sort_index()

        # Set frequency if specified
        freq = self.config.get("frequency")
        if freq:
            df = df.asfreq(freq)

        self._data = df
        self._metadata = {
            "source": "sql",
            "connection": self._sanitize_connection_string(conn_string),
            "rows": len(df),
            "columns": list(df.columns),
            "frequency": str(df.index.freq) if df.index.freq else pd.infer_freq(df.index),
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
        }

        return df

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

    def _get_query(self) -> str:
        """Get SQL query from config."""
        # Check if query is provided directly
        if "query" in self.config:
            return self.config["query"]

        # Build query from table and filters
        table = self.config.get("table")
        if not table:
            raise ValueError("Must provide 'query' or 'table'")

        # Simple query builder
        query = f"SELECT * FROM {table}"

        # Add filters if provided
        filters = self.config.get("filters", {})
        if filters:
            conditions = []
            for col, value in filters.items():
                # Simple filter handling
                if isinstance(value, str) and value.startswith((">=", "<=", ">", "<", "!=")):
                    operator = value[:2] if value[:2] in [">=", "<=", "!="] else value[0]
                    val = value[2:] if len(operator) == 2 else value[1:]
                    conditions.append(f"{col} {operator} '{val}'")
                else:
                    conditions.append(f"{col} = '{value}'")

            query += " WHERE " + " AND ".join(conditions)

        return query

    def _sanitize_connection_string(self, conn_string: str) -> str:
        """Remove credentials from connection string for metadata."""
        # Hide password in connection string
        if "@" in conn_string:
            parts = conn_string.split("@")
            if len(parts) == 2:
                return f"***@{parts[1]}"
        return conn_string

    def validate(self, data: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """Validate SQL data using pandas adapter validation."""
        # Reuse pandas validation logic
        from .pandas_adapter import PandasAdapter

        pandas_adapter = PandasAdapter({"data": data})
        return pandas_adapter.validate(data)
