"""
Lazy data loader for efficient processing of large datasets.

Provides utilities for progressively loading and processing data without
loading everything into memory at once.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pandas as pd

from sktime_mcp.data.registry import DataSourceRegistry

logger = logging.getLogger(__name__)


class LazyDataLoader:
    """
    Lazy loader for streaming large datasets.

    Supports:
    - Chunk-based iteration
    - Async loading
    - Progressive fitting (partial fit)
    - Memory tracking
    """

    def __init__(
        self,
        config: dict[str, Any],
        chunk_size: int | None = None,
    ):
        """
        Initialize lazy loader.

        Args:
            config: Data source configuration (must use streaming adapter)
            chunk_size: Override chunk size from config
        """
        if chunk_size:
            config = {**config, "chunk_size": chunk_size}

        self.config = config
        self.adapter = DataSourceRegistry.create_adapter(config)
        self._loaded_rows = 0

    def iterate_chunks(
        self,
        chunk_size: int | None = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Iterate over data chunks.

        Args:
            chunk_size: Override chunk size

        Yields:
            DataFrame chunks
        """
        if hasattr(self.adapter, "load_chunks"):
            chunk_iterable = self.adapter.load_chunks(chunk_size=chunk_size)
        else:
            chunk_iterable = [self.adapter.load()]

        for chunk in chunk_iterable:
            self._loaded_rows += len(chunk)
            logger.debug(f"Loaded {self._loaded_rows} rows total")
            yield chunk

    async def iterate_chunks_async(
        self,
        chunk_size: int | None = None,
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Asynchronously iterate over data chunks.

        Args:
            chunk_size: Override chunk size

        Yields:
            DataFrame chunks
        """
        if hasattr(self.adapter, "load_chunks"):
            chunk_iterable = self.adapter.load_chunks(chunk_size=chunk_size)
        else:
            chunk_iterable = [self.adapter.load()]

        for chunk in chunk_iterable:
            self._loaded_rows += len(chunk)
            logger.debug(f"Loaded {self._loaded_rows} rows total (async)")
            yield chunk
            # Yield control to event loop
            await asyncio.sleep(0)

    def get_metadata(self) -> dict[str, Any]:
        """Get file metadata without loading all data."""
        if hasattr(self.adapter, "get_metadata_from_sample"):
            return self.adapter.get_metadata_from_sample()
        return {
            "note": "Metadata not available for this adapter",
            "type": self.config.get("type"),
        }

    def get_rows_loaded(self) -> int:
        """Get number of rows loaded so far."""
        return self._loaded_rows

    def get_all_data(self) -> pd.DataFrame:
        """Load all data into memory (only for small datasets)."""
        logger.warning("Loading all data into memory - use iterate_chunks for large files")
        return self.adapter.load()


class ChunkedFitter:
    """
    Helper for fitting estimators on chunked data.

    Supports progressive fitting using sktime's partial_fit if available.
    """

    def __init__(self, estimator: Any):
        """
        Initialize chunked fitter.

        Args:
            estimator: sktime estimator instance
        """
        self.estimator = estimator
        self._fitted = False
        self._rows_processed = 0

    def fit_on_chunks(
        self,
        data_loader: LazyDataLoader,
        target_column: str | None = None,
    ) -> None:
        """
        Fit estimator on chunked data progressively.

        Uses partial_fit if available, otherwise loads all chunks
        and fits on complete data.

        Args:
            data_loader: LazyDataLoader instance
            target_column: Column to use as target (if needed)
        """
        has_partial_fit = hasattr(self.estimator, "partial_fit")

        if has_partial_fit:
            logger.info("Using partial_fit for progressive training")
            for chunk in data_loader.iterate_chunks():
                y = chunk.iloc[:, 0] if target_column is None else chunk[target_column]
                self.estimator.partial_fit(y)
                self._rows_processed += len(chunk)
                logger.debug(f"Partial fit on {len(chunk)} rows (total: {self._rows_processed})")
        else:
            logger.info("Estimator does not support partial_fit, loading all data")
            # Fall back to loading all data
            df = data_loader.get_all_data()
            y = df.iloc[:, 0] if target_column is None else df[target_column]
            self.estimator.fit(y)
            self._rows_processed = len(df)

        self._fitted = True

    def is_fitted(self) -> bool:
        """Check if estimator has been fitted."""
        return self._fitted

    def get_rows_processed(self) -> int:
        """Get number of rows processed during fitting."""
        return self._rows_processed


class PaginatedSQLLoader:
    """
    Helper for loading data from SQL databases with pagination.

    Reduces memory usage by fetching data page-by-page instead
    of all at once.
    """

    def __init__(
        self,
        config: dict[str, Any],
        page_size: int = 10000,
    ):
        """
        Initialize paginated SQL loader.

        Args:
            config: SQL adapter configuration
            page_size: Rows per page
        """
        if config.get("type") != "sql":
            config = {**config, "type": "sql"}

        self.config = config
        self.page_size = page_size
        self._current_page = 0
        self._total_rows = None

    def load_page(self, page_number: int) -> pd.DataFrame:
        """
        Load a specific page of data.

        Args:
            page_number: Page number (0-indexed)

        Returns:
            DataFrame for that page
        """
        try:
            from sqlalchemy import create_engine
        except ImportError as e:
            raise ImportError(
                "SQLAlchemy is required for SQL adapter. Install with: pip install sqlalchemy"
            ) from e

        config = self.config.copy()

        # Build connection string
        dialect = config.get("dialect")
        if not dialect:
            raise ValueError("Must provide dialect in config")

        if dialect == "sqlite":
            database = config.get("database", "database.db")
            conn_string = f"sqlite:///{database}"
        else:
            username = config.get("username", "")
            password = config.get("password", "")
            host = config.get("host", "localhost")
            port = config.get("port", "")
            database = config.get("database", "")

            auth = f"{username}:{password}@" if username else ""
            port_str = f":{port}" if port else ""
            conn_string = f"{dialect}://{auth}{host}{port_str}/{database}"

        # Get base query
        if "query" in config:
            base_query = config["query"]
        else:
            table = config.get("table")
            filters = config.get("filters", {})
            where_clauses = [f"{k} {v}" for k, v in filters.items()]
            where_clause = " AND ".join(where_clauses) if where_clauses else ""
            base_query = f"SELECT * FROM {table}"
            if where_clause:
                base_query += f" WHERE {where_clause}"

        # Add pagination
        offset = page_number * self.page_size
        paginated_query = f"{base_query} LIMIT {self.page_size} OFFSET {offset}"

        engine = create_engine(conn_string)
        try:
            parse_dates = config.get("parse_dates", [])
            df = pd.read_sql(
                paginated_query, engine, parse_dates=parse_dates if parse_dates else None
            )
            self._current_page = page_number
            return df
        finally:
            engine.dispose()

    def iterate_pages(self) -> Generator[tuple[int, pd.DataFrame], None, None]:
        """
        Iterate over all pages.

        Yields:
            Tuple of (page_number, DataFrame)
        """
        page_number = 0
        while True:
            df = self.load_page(page_number)
            if df.empty:
                break
            yield page_number, df
            page_number += 1

    async def iterate_pages_async(self) -> AsyncGenerator[tuple[int, pd.DataFrame], None]:
        """
        Asynchronously iterate over all pages.

        Yields:
            Tuple of (page_number, DataFrame)
        """
        for page_number, df in self.iterate_pages():
            await asyncio.sleep(0)  # Yield control
            yield page_number, df

    def get_page_size(self) -> int:
        """Get configured page size."""
        return self.page_size

    def get_current_page(self) -> int:
        """Get current page number."""
        return self._current_page
