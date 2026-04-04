"""
Data source layer for sktime-mcp.

Provides adapters for loading data from various sources:
- Pandas DataFrames (in-memory)
- SQL databases (PostgreSQL, MySQL, SQLite, etc.)
- Files (CSV, Excel, Parquet)

Usage:
    from sktime_mcp.data import DataSourceRegistry

    # Create adapter from config
    config = {
        "type": "pandas",
        "data": df,
        "time_column": "date",
        "target_column": "value"
    }

    adapter = DataSourceRegistry.create_adapter(config)
    data = adapter.load()
    is_valid, report = adapter.validate(data)
    y, X = adapter.to_sktime_format(data)
"""

from .adapters import FileAdapter, PandasAdapter, SQLAdapter
from .base import DataSourceAdapter
from .registry import DataSourceRegistry

__all__ = [
    "DataSourceAdapter",
    "DataSourceRegistry",
    "PandasAdapter",
    "SQLAdapter",
    "FileAdapter",
]
