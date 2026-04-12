"""
Tests for streaming data loading and lazy evaluation.

Includes tests for handling large datasets that would normally
cause out-of-memory errors.
"""

import os
import sqlite3

# Ensure src is in path
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")

from sktime_mcp.data.adapters.streaming_adapter import StreamingDataAdapter
from sktime_mcp.data.lazy_loader import ChunkedFitter, LazyDataLoader, PaginatedSQLLoader


class TestStreamingDataAdapter:
    """Tests for StreamingDataAdapter."""

    @pytest.fixture
    def large_csv_file(self):
        """Create a temporary large CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write header
            f.write("date,value,feature1\n")

            # Write 100k rows
            for i in range(100000):
                date = f"2020-01-{(i % 30) + 1:02d}"
                value = 100 + np.random.randn() * 10
                feature = np.random.rand()
                f.write(f"{date},{value:.2f},{feature:.4f}\n")

            temp_path = f.name

        yield temp_path

        # Cleanup
        path = Path(temp_path)
        if path.exists():
            path.unlink()

    @pytest.fixture
    def large_parquet_file(self):
        """Create a temporary large Parquet file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "large.parquet"

            # Create and save large DataFrame
            dates = pd.date_range("2020-01-01", periods=100000, freq="h")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "value": 100 + np.random.randn(100000) * 10,
                    "feature1": np.random.rand(100000),
                }
            )

            try:
                df.to_parquet(parquet_path, index=False)
                yield str(parquet_path)
            except Exception as e:
                pytest.skip(f"Parquet support not available: {e}")

    def test_csv_chunked_loading(self, large_csv_file):
        """Test loading CSV in chunks."""
        config = {
            "type": "streaming",
            "path": large_csv_file,
            "format": "csv",
            "chunk_size": 10000,
            "time_column": "date",
            "target_column": "value",
        }

        adapter = StreamingDataAdapter(config)
        chunks = list(adapter.load_chunks())

        assert len(chunks) > 0, "Should produce at least one chunk"
        assert len(chunks) == 10, f"Expected 10 chunks of 10k rows, got {len(chunks)}"

        # Verify each chunk is a proper DataFrame
        for chunk in chunks:
            assert isinstance(chunk, pd.DataFrame)
            assert len(chunk) <= 10000
            assert "date" in chunk.index.name or hasattr(chunk.index, "name")

    def test_load_entire_file_small(self, large_csv_file):
        """Test loading entire file into memory (small test file)."""
        config = {
            "type": "streaming",
            "path": large_csv_file,
            "format": "csv",
            "time_column": "date",
            "target_column": "value",
        }

        adapter = StreamingDataAdapter(config)
        df = adapter.load()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100000
        assert "value" in df.columns

    def test_metadata_without_full_load(self, large_csv_file):
        """Test getting metadata without loading entire file."""
        config = {
            "type": "streaming",
            "path": large_csv_file,
            "format": "csv",
            "chunk_size": 5000,
        }

        adapter = StreamingDataAdapter(config)
        metadata = adapter.get_metadata_from_sample(sample_size=1000)

        assert metadata["format"] == "csv"
        assert metadata["estimated_total_rows"] > 0
        assert metadata["file_size_bytes"] > 0
        assert "columns" in metadata
        assert "memory_estimate_mb" in metadata

    def test_parquet_chunked_loading(self, large_parquet_file):
        """Test loading Parquet in chunks (if pyarrow available)."""
        config = {
            "type": "streaming",
            "path": large_parquet_file,
            "format": "parquet",
            "chunk_size": 10000,
            "time_column": "date",
        }

        adapter = StreamingDataAdapter(config)
        chunks = list(adapter.load_chunks())

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, pd.DataFrame)

    def test_validate_chunked_data(self, large_csv_file):
        """Test validation of chunked data."""
        config = {
            "type": "streaming",
            "path": large_csv_file,
            "format": "csv",
            "chunk_size": 10000,
            "time_column": "date",
        }

        adapter = StreamingDataAdapter(config)

        # Validate first chunk
        for chunk in adapter.load_chunks():
            is_valid, report = adapter.validate(chunk)
            assert is_valid, f"Chunk validation failed: {report}"
            break


class TestLazyDataLoader:
    """Tests for LazyDataLoader utility."""

    @pytest.fixture
    def csv_config(self):
        """Fixture for CSV config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create sample time series data
            f.write("datetime,sales,cost\n")
            base_date = "2020-01-01"
            for i in range(50000):
                date = pd.Timestamp(base_date) + pd.Timedelta(days=i)
                sales = 1000 + np.random.randn() * 100
                cost = 500 + np.random.randn() * 50
                f.write(f"{date.date()},{sales:.2f},{cost:.2f}\n")

            temp_path = f.name

        yield {
            "type": "streaming",
            "path": temp_path,
            "chunk_size": 5000,
            "time_column": "datetime",
            "target_column": "sales",
        }

        path = Path(temp_path)
        if path.exists():
            path.unlink()

    def test_lazy_loader_iteration(self, csv_config):
        """Test LazyDataLoader iteration."""
        loader = LazyDataLoader(csv_config, chunk_size=10000)

        total_rows = 0
        chunk_count = 0

        for chunk in loader.iterate_chunks():
            assert isinstance(chunk, pd.DataFrame)
            chunk_count += 1
            total_rows += len(chunk)

        assert chunk_count == 5, f"Expected 5 chunks, got {chunk_count}"
        assert total_rows == 50000

    def test_metadata_retrieval(self, csv_config):
        """Test getting metadata from lazy loader."""
        loader = LazyDataLoader(csv_config)
        metadata = loader.get_metadata()

        assert metadata is not None
        assert metadata.get("format") == "csv"
        assert metadata.get("estimated_total_rows") > 0

    def test_rows_loaded_tracking(self, csv_config):
        """Test tracking of loaded rows."""
        loader = LazyDataLoader(csv_config, chunk_size=5000)

        initial_count = loader.get_rows_loaded()
        assert initial_count == 0

        # Load one chunk
        for _chunk in loader.iterate_chunks():
            assert loader.get_rows_loaded() > 0
            break

        rows_after_one = loader.get_rows_loaded()
        assert rows_after_one == 5000


class TestChunkedFitter:
    """Tests for ChunkedFitter utility."""

    @pytest.fixture
    def simple_estimator(self):
        """Create a simple test estimator with partial_fit."""
        from sktime.forecasting.naive import NaiveForecaster

        return NaiveForecaster()

    def test_chunked_fit_without_partial_fit(self, simple_estimator):
        """Test fitting estimator without partial_fit support."""
        fitter = ChunkedFitter(simple_estimator)

        # Create small data
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100),
                "value": np.random.randn(100),
            }
        )

        config = {
            "type": "pandas",
            "data": df,
            "time_column": "date",
            "target_column": "value",
        }

        loader = LazyDataLoader(config)

        # This should fall back to loading all data
        fitter.fit_on_chunks(loader)

        assert fitter.is_fitted()
        assert fitter.get_rows_processed() > 0

    def test_rows_processed_tracking(self, simple_estimator):
        """Test tracking of rows processed during fitting."""
        fitter = ChunkedFitter(simple_estimator)

        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=1000),
                "value": np.random.randn(1000),
            }
        )

        config = {
            "type": "pandas",
            "data": df,
            "time_column": "date",
            "target_column": "value",
        }

        loader = LazyDataLoader(config)
        fitter.fit_on_chunks(loader)

        assert fitter.get_rows_processed() == 1000


class TestMemoryEfficiency:
    """Tests demonstrating memory efficiency of streaming."""

    def test_large_csv_memory_efficiency(self):
        """Demonstrate that chunked loading handles large files efficiently."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("date,value,feature1,feature2,feature3\n")

            # Create 500k rows (simulating a ~50MB file)
            for i in range(500000):
                date = f"2020-01-{(i % 30) + 1:02d}"
                value = 100 + np.random.randn() * 10
                f1 = np.random.rand()
                f2 = np.random.rand()
                f3 = np.random.rand()
                f.write(f"{date},{value:.2f},{f1:.4f},{f2:.4f},{f3:.4f}\n")

            temp_path = f.name

        try:
            config = {
                "type": "streaming",
                "path": temp_path,
                "format": "csv",
                "chunk_size": 50000,
            }

            adapter = StreamingDataAdapter(config)

            # Verify we can iterate without loading all data
            chunk_count = 0
            for chunk in adapter.load_chunks():
                assert len(chunk) <= 50000
                chunk_count += 1
                # Verify we're processing chunks, not loading all at once
                assert chunk_count <= 11  # 500k / 50k = 10 chunks

            assert chunk_count == 10, f"Expected 10 chunks, got {chunk_count}"

        finally:
            path = Path(temp_path)
            if path.exists():
                path.unlink()


class TestSQLPagination:
    """Tests for SQL pagination (requires SQLite for easy testing)."""

    @pytest.fixture
    def sqlite_db(self):
        """Create a temporary SQLite database with test data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Create and populate database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                date TEXT,
                value REAL,
                cost REAL
            )
        """)

        # Insert 100k rows
        rows = []
        for i in range(100000):
            date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i % 365)
            value = 1000 + np.random.randn() * 100
            cost = 500 + np.random.randn() * 50
            rows.append((date.isoformat(), value, cost))

        cursor.executemany(
            "INSERT INTO sales (date, value, cost) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

        yield db_path

        path = Path(db_path)
        if path.exists():
            path.unlink()

    def test_sql_pagination(self, sqlite_db):
        """Test loading SQL data with pagination."""
        config = {
            "type": "sql",
            "dialect": "sqlite",
            "database": sqlite_db,
            "table": "sales",
            "time_column": "date",
            "target_column": "value",
        }

        try:
            from sktime_mcp.data.adapters.sql_adapter import SQLAdapter

            adapter = SQLAdapter(config)

            page_count = 0
            total_rows = 0

            for page in adapter.load_paginated(page_size=10000):
                assert isinstance(page, pd.DataFrame)
                assert len(page) <= 10000
                page_count += 1
                total_rows += len(page)

            assert page_count == 10, f"Expected 10 pages, got {page_count}"
            assert total_rows == 100000

        except ImportError:
            pytest.skip("SQLAlchemy not available")

    def test_paginated_sql_loader(self, sqlite_db):
        """Test PaginatedSQLLoader utility."""
        config = {
            "type": "sql",
            "dialect": "sqlite",
            "database": sqlite_db,
            "table": "sales",
        }

        try:
            loader = PaginatedSQLLoader(config, page_size=5000)

            # Load first page
            page0 = loader.load_page(0)
            assert len(page0) == 5000

            # Load second page
            page1 = loader.load_page(1)
            assert len(page1) == 5000

            # Verify pages are different
            assert not page0.equals(page1)

        except ImportError:
            pytest.skip("SQLAlchemy not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
