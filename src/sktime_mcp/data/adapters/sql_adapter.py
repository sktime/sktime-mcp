"""
SQL adapter for database connections.

Supports loading data from SQL databases using SQLAlchemy.
"""

import re
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
        query, query_params = self._get_query()

        # Create engine and load data
        engine = create_engine(conn_string)

        try:
            # Parse dates if specified
            parse_dates = self.config.get("parse_dates", [])
            if not parse_dates and self.config.get("time_column"):
                parse_dates = [self.config["time_column"]]

            df = pd.read_sql(
                query,
                engine,
                params=query_params if query_params else None,
                parse_dates=parse_dates if parse_dates else None,
            )
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
        
    async def load_async(
        self,
        progress_callback=None,
    ) -> pd.DataFrame:
        """Asynchronously load from SQL database."""
        dialect = self.config.get("dialect")
        conn_string = self._get_connection_string()
        
        # Async callback helper
        import asyncio
        async def cb(pct, msg):
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(pct, msg)
                else:
                    progress_callback(pct, msg)

        # For sqlite, we can use aiosqlite natively for non-blocking IO
        if dialect == "sqlite" or conn_string.startswith("sqlite://"):
            try:
                import aiosqlite
            except ImportError:
                raise ImportError(
                    "aiosqlite is required for async SQLite fetching. "
                    "Install with: pip install aiosqlite"
                )
            
            await cb(0.0, "Connecting to SQLite (async)...")
            
            # Extract database path from connection string
            # sqlite:///path/to/db -> path/to/db
            db_path = conn_string.replace("sqlite:///", "")
            if not db_path:
                db_path = ":memory:"
                
            query = self._get_query()
            
            async with aiosqlite.connect(db_path) as db:
                await cb(5.0, "Executing query...")
                
                async with db.execute(query) as cursor:
                    # Get column names
                    columns = [description[0] for description in cursor.description]
                    
                    rows = []
                    total_fetched = 0
                    chunk_size = 1000
                    
                    while True:
                        chunk = await cursor.fetchmany(chunk_size)
                        if not chunk:
                            break
                        rows.extend(chunk)
                        total_fetched += len(chunk)
                        await cb(0.0, f"Fetched {total_fetched} rows...")
                        
            await cb(95.0, "Parsing results into DataFrame...")
            df = pd.DataFrame(rows, columns=columns)
            
        else:
            # Fallback to run_in_executor for other dialects 
            # (they could be converted to SQLAlchemy 2.0 AsyncEngine later)
            await cb(0.0, f"Connecting to {dialect or 'SQL'} via standard engine...")
            
            loop = asyncio.get_event_loop()
            def fetch_data():
                return self.load()
                
            # If we fallback, the base `load()` does all formatting and metadata assignment implicitly.
            return await loop.run_in_executor(None, fetch_data)

        # For the native async path (e.g. SQLite), apply standard sktime formatting
        time_col = self.config.get("time_column")
        parse_dates = self.config.get("parse_dates", [])
        if not parse_dates and time_col:
            parse_dates = [time_col]
            
        if parse_dates:
            for date_col in parse_dates:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])

        if time_col and time_col in df.columns:
            df = df.set_index(time_col)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to datetime: {e}")
        
        # Sort by time
        df = df.sort_index()
        
        # Set frequency if specified
        freq = self.config.get("frequency")
        if freq:
            try:
                df = df.asfreq(freq)
            except Exception:
                pass
        
        self._data = df
        
        # Calculate freq for empty / 1-row dataframes cleanly
        freq_str = str(df.index.freq) if getattr(df.index, 'freq', None) else None
        if not freq_str and isinstance(df.index, pd.DatetimeIndex) and len(df) > 2:
            freq_str = str(pd.infer_freq(df.index))
        freq_str = freq_str or "None"

        # Safe min/max for empty dfs
        min_date = str(df.index.min()) if len(df) > 0 else "None"
        max_date = str(df.index.max()) if len(df) > 0 else "None"
            
        self._metadata = {
            "source": "sql",
            "connection": self._sanitize_connection_string(conn_string),
            "rows": len(df),
            "columns": list(df.columns),
            "frequency": freq_str,
            "start_date": min_date,
            "end_date": max_date,
        }
        
        await cb(100.0, "Data loaded successfully!")
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

        pandas_adapter = PandasAdapter({"data": data})
        return pandas_adapter.validate(data)
