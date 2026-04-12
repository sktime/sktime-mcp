"""
Streaming adapter for large datasets using lazy loading.

Supports memory-efficient loading of large files by reading in chunks
instead of loading the entire dataset into memory at once.
"""

import contextlib
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd

from ..base import DataSourceAdapter


class StreamingDataAdapter(DataSourceAdapter):
    """
    Adapter for streaming/chunked data loading.

    Enables lazy loading of large files (CSV, Parquet) to avoid
    out-of-memory errors. Data is read in chunks and can be consumed
    iteratively or combined progressively.

    Config example:
    {
        "type": "streaming",
        "path": "/path/to/large_file.csv",
        "format": "csv",  # csv, parquet (auto-detected if not specified)
        "chunk_size": 10000,  # Number of rows per chunk (default: 10000)

        # Column mapping
        "time_column": "date",
        "target_column": "value",
        "exog_columns": ["feature1", "feature2"],

        # CSV-specific options
        "csv_options": {
            "sep": ",",
            "header": 0,
            "encoding": "utf-8"
        },

        # Common options
        "parse_dates": True,
        "frequency": "D"
    }
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize streaming adapter."""
        super().__init__(config)
        self._chunk_size = config.get("chunk_size", 10000)
        self._path = None
        self._file_format = None

    def load(self) -> pd.DataFrame:
        """
        Load entire file into memory.

        For large files, use `load_chunks()` or `load_iterator()` instead
        to process data lazily.

        Returns:
            DataFrame with all data
        """
        # Load all chunks and concatenate
        chunks = list(self.load_chunks())
        if not chunks:
            raise ValueError("No data loaded from source")

        df = pd.concat(chunks, ignore_index=True)
        return self._process_dataframe(df)

    def load_chunks(self, chunk_size: int | None = None) -> Generator[pd.DataFrame, None, None]:
        """
        Load data in chunks.

        Args:
            chunk_size: Number of rows per chunk (uses config default if None)

        Yields:
            DataFrame chunks of specified size

        Example:
            >>> adapter = StreamingDataAdapter(config)
            >>> for chunk in adapter.load_chunks(chunk_size=50000):
            ...     process_chunk(chunk)
        """
        if chunk_size is None:
            chunk_size = self._chunk_size

        path_str = self.config.get("path")
        if not path_str:
            raise ValueError("Config must contain 'path' key")

        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Detect or get format
        file_format = self.config.get("format")
        if not file_format:
            file_format = self._detect_format(path)

        self._path = path
        self._file_format = file_format

        if file_format == "csv":
            yield from self._load_csv_chunks(path, chunk_size)
        elif file_format == "parquet":
            yield from self._load_parquet_chunks(path, chunk_size)
        else:
            raise ValueError(
                f"Unsupported format for streaming: {file_format}. Supported formats: csv, parquet"
            )

    def load_iterator(self, chunk_size: int | None = None):
        """
        Get an iterator over data chunks.

        This is an alias for load_chunks() for convenience.

        Args:
            chunk_size: Number of rows per chunk

        Returns:
            Generator yielding DataFrames
        """
        return self.load_chunks(chunk_size=chunk_size)

    def get_metadata_from_sample(self, sample_size: int = 1000) -> dict[str, Any]:
        """
        Get metadata by reading only a sample of the file.

        Useful for understanding large file structure without loading everything.

        Args:
            sample_size: Number of rows to sample

        Returns:
            Metadata dictionary
        """
        path_str = self.config.get("path")
        if not path_str:
            raise ValueError("Config must contain 'path' key")

        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_format = self.config.get("format")
        if not file_format:
            file_format = self._detect_format(path)

        # Read only first chunk
        if file_format == "csv":
            df = self._read_csv_sample(path, sample_size)
        elif file_format == "parquet":
            df = self._read_parquet_sample(path, sample_size)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        df = self._process_dataframe(df)

        # Estimate total rows
        file_size = path.stat().st_size
        chunk_size = self._chunk_size
        # Rough estimate based on first chunk size
        if len(df) > 0:
            avg_row_size = file_size / len(df)
            estimated_rows = int(file_size / avg_row_size)
        else:
            estimated_rows = 0

        return {
            "format": file_format,
            "path": str(path.absolute()),
            "file_size_bytes": file_size,
            "estimated_total_rows": estimated_rows,
            "chunk_size": chunk_size,
            "sample_size": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "index_type": str(type(df.index).__name__),
            "memory_estimate_mb": (estimated_rows * 1024) / (1024 * 1024),  # rough estimate
        }

    def validate(self, data: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """Validate data quality."""
        errors = []
        warnings = []

        # Check for empty data
        if len(data) == 0:
            errors.append("Data is empty")

        # Check for all-NaN columns
        all_nan_cols = data.columns[data.isna().all()].tolist()
        if all_nan_cols:
            warnings.append(f"Columns with all NaN values: {all_nan_cols}")

        # Check for time index
        if (
            not isinstance(data.index, (pd.DatetimeIndex, pd.RangeIndex))
            and "time_column" in self.config
        ):
            warnings.append("Data does not have DatetimeIndex as expected")

        return len(errors) == 0, {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _load_csv_chunks(self, path: Path, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Load CSV file in chunks."""
        csv_options = self.config.get("csv_options", {})
        csv_options = {**csv_options}  # Copy to avoid modifying config

        # Read CSV in chunks
        reader = pd.read_csv(path, chunksize=chunk_size, **csv_options)

        for chunk in reader:
            yield self._process_dataframe(chunk)

    def _load_parquet_chunks(
        self, path: Path, chunk_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """Load Parquet file in chunks."""
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "PyArrow is required for Parquet support. Install with: pip install pyarrow"
            ) from e

        # Parquet doesn't natively support row-based chunking, so we read in batches
        parquet_file = pq.ParquetFile(path)

        # Process batches
        for i in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(i)
            df = table.to_pandas()
            yield self._process_dataframe(df)

    def _read_csv_sample(self, path: Path, nrows: int) -> pd.DataFrame:
        """Read first N rows of CSV."""
        csv_options = self.config.get("csv_options", {})
        csv_options = {**csv_options}
        csv_options["nrows"] = nrows
        return pd.read_csv(path, **csv_options)

    def _read_parquet_sample(self, path: Path, nrows: int) -> pd.DataFrame:
        """Read first N rows of Parquet."""
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "PyArrow is required for Parquet support. Install with: pip install pyarrow"
            ) from e

        parquet_file = pq.ParquetFile(path)
        table = parquet_file.read_row_group(0)
        df = table.to_pandas()
        return df.head(nrows)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard processing to dataframe chunk."""
        # Parse dates if specified
        time_col = self.config.get("time_column")
        if time_col and time_col in df.columns:
            if self.config.get("parse_dates", True):
                with contextlib.suppress(Exception):
                    df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)

        # Ensure datetime index if specified
        if not isinstance(df.index, pd.DatetimeIndex) and time_col:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(
                    f"Could not convert time column '{time_col}' to datetime: {e}"
                ) from e

        # Sort by index
        df = df.sort_index()

        return df

    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        suffix = path.suffix.lower()

        format_map = {
            ".csv": "csv",
            ".txt": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
        }

        file_format = format_map.get(suffix)
        if not file_format:
            raise ValueError(
                f"Could not detect format from extension '{suffix}'. "
                "Please specify 'format' in config. Supported: csv, parquet"
            )

        return file_format
