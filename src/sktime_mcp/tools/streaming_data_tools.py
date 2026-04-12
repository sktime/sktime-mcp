"""
Streaming data loading tools for MCP.

Provides tools for loading large datasets efficiently without loading
entire files into memory at once.
"""

import logging
import uuid
from typing import Any

from sktime_mcp.data.lazy_loader import LazyDataLoader, PaginatedSQLLoader
from sktime_mcp.runtime.handles import get_handle_manager

logger = logging.getLogger(__name__)


def load_data_source_streaming_tool(
    source_type: str,
    config: dict[str, Any],
    chunk_size: int | None = None,
    preview_size: int = 1000,
) -> dict[str, Any]:
    """
    Load a data source with streaming/lazy loading.

    Uses chunked loading for large files to minimize memory usage.
    Perfect for datasets larger than available RAM.

    Args:
        source_type: Type of source: "streaming" (recommended for files)
        config: Data source configuration dict:
            - path: File path (for streaming type)
            - format: File format - csv, parquet
            - chunk_size: Rows per chunk (default: 10000)
            - time_column: Name of time/date column
            - target_column: Target variable column name
        chunk_size: Override chunk size
        preview_size: Rows to preview for metadata

    Returns:
        {
            "success": bool,
            "handle": str (data handle for use with fit_predict_streaming),
            "metadata": {
                "total_rows_estimate": int,
                "columns": [str],
                "file_size_bytes": int,
                "chunk_size": int,
                "memory_estimate_mb": float
            },
            "preview": DataFrame sample as dict
        }

    Example:
        >>> result = load_data_source_streaming_tool(
        ...     "streaming",
        ...     {
        ...         "path": "/data/large_sales.csv",
        ...         "time_column": "date",
        ...         "target_column": "sales"
        ...     },
        ...     chunk_size=50000
        ... )
    """
    try:
        # Create lazy loader
        if source_type != "streaming":
            config = {**config, "type": source_type}

        loader = LazyDataLoader(config, chunk_size=chunk_size)

        # Get metadata without loading all data
        metadata = loader.get_metadata()

        # Get preview data
        try:
            preview_data = None
            for chunk in loader.iterate_chunks(chunk_size=preview_size):
                preview_data = chunk.head(preview_size)
                break

            preview_dict = preview_data.to_dict("records") if preview_data is not None else []
        except Exception as e:
            logger.warning(f"Could not generate preview: {e}")
            preview_dict = []

        # Store handle for later use
        handle_manager = get_handle_manager()
        data_handle = f"data_{uuid.uuid4().hex[:12]}"

        # Store the loader for later retrieval
        if not hasattr(handle_manager, "_data_handles"):
            handle_manager._data_handles = {}
        handle_manager._data_handles[data_handle] = loader

        return {
            "success": True,
            "handle": data_handle,
            "metadata": metadata,
            "preview_rows": len(preview_dict),
            "preview": preview_dict[:10] if preview_dict else [],
        }

    except Exception as e:
        logger.exception("Error loading streaming data source")
        return {
            "success": False,
            "error": str(e),
            "hint": "Ensure path exists, format is specified, and columns are correct",
        }


def get_data_source_metadata_tool(
    config: dict[str, Any],
    sample_size: int = 1000,
) -> dict[str, Any]:
    """
    Get metadata about a data source without loading all data.

    Useful for understanding large files before deciding how to process them.

    Args:
        config: Data source configuration
        sample_size: Rows to sample for analysis

    Returns:
        Metadata dict with estimated row count, columns, file size, etc.

    Example:
        >>> result = get_data_source_metadata_tool({
        ...     "type": "streaming",
        ...     "path": "/data/sales.parquet",
        ...     "format": "parquet"
        ... })
    """
    try:
        if config.get("type") != "streaming":
            config = {**config, "type": "streaming"}

        loader = LazyDataLoader(config)
        metadata = loader.get_metadata()

        return {
            "success": True,
            "metadata": metadata,
        }

    except Exception as e:
        logger.exception("Error getting metadata")
        return {
            "success": False,
            "error": str(e),
        }


def load_data_paginated_sql_tool(
    connection_config: dict[str, Any],
    page_number: int = 0,
    page_size: int = 10000,
) -> dict[str, Any]:
    """
    Load a page of data from SQL database.

    For large SQL tables, load data page-by-page to minimize memory usage.

    Args:
        connection_config: SQL connection configuration:
            - dialect: postgresql, mysql, sqlite, mssql
            - host, port, database, username, password
            - (or) connection_string
            - table: Table name
            - filters: Optional filter conditions
        page_number: Page to load (0-indexed)
        page_size: Rows per page

    Returns:
        {
            "success": bool,
            "page_number": int,
            "rows_in_page": int,
            "data": [dict],  # Array of row dicts
            "offset": int,
            "handle": str  # For paginated loader
        }

    Example:
        >>> result = load_data_paginated_sql_tool(
        ...     {
        ...         "dialect": "sqlite",
        ...         "database": "sales.db",
        ...         "table": "transactions"
        ...     },
        ...     page_number=0,
        ...     page_size=50000
        ... )
    """
    try:
        loader = PaginatedSQLLoader(connection_config, page_size=page_size)
        df = loader.load_page(page_number)

        if df.empty:
            return {
                "success": True,
                "page_number": page_number,
                "rows_in_page": 0,
                "data": [],
                "message": "No more pages available",
            }

        # Store loader for subsequent pages
        handle_manager = get_handle_manager()
        if not hasattr(handle_manager, "_sql_loaders"):
            handle_manager._sql_loaders = {}

        loader_id = f"sql_loader_{uuid.uuid4().hex[:12]}"
        handle_manager._sql_loaders[loader_id] = loader

        return {
            "success": True,
            "page_number": page_number,
            "rows_in_page": len(df),
            "offset": page_number * page_size,
            "data": df.to_dict("records"),
            "loader_handle": loader_id,
        }

    except Exception as e:
        logger.exception("Error loading paginated SQL data")
        return {
            "success": False,
            "error": str(e),
            "hint": "Check connection config and ensure table exists",
        }


def get_streaming_data_sample_tool(
    data_handle: str,
    sample_size: int = 100,
) -> dict[str, Any]:
    """
    Get a sample of data from a streaming data handle.

    Args:
        data_handle: Handle from load_data_source_streaming_tool
        sample_size: Number of rows to return

    Returns:
        Sample data as list of dicts
    """
    try:
        handle_manager = get_handle_manager()

        if not hasattr(handle_manager, "_data_handles"):
            return {"success": False, "error": "No streaming data handles available"}

        if data_handle not in handle_manager._data_handles:
            return {"success": False, "error": f"Handle not found: {data_handle}"}

        loader = handle_manager._data_handles[data_handle]

        # Get first chunk of sample data
        sample_data = []
        for chunk in loader.iterate_chunks(chunk_size=sample_size):
            sample_data = chunk.head(sample_size).to_dict("records")
            break

        return {
            "success": True,
            "handle": data_handle,
            "sample_size": len(sample_data),
            "data": sample_data,
        }

    except Exception as e:
        logger.exception("Error getting streaming data sample")
        return {
            "success": False,
            "error": str(e),
        }
