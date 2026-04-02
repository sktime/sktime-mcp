"""
Data source adapters.

Available adapters:
- PandasAdapter: In-memory DataFrames
- SQLAdapter: SQL databases
- FileAdapter: CSV, Excel, Parquet files
- UrlAdapter: Datasets from Web URLs
"""

from .file_adapter import FileAdapter
from .pandas_adapter import PandasAdapter
from .sql_adapter import SQLAdapter
from .url_adapter import UrlAdapter

__all__ = [
    "PandasAdapter",
    "SQLAdapter",
    "FileAdapter",
    "UrlAdapter",
]
