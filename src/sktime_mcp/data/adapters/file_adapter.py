"""
File adapter for CSV, Excel, and Parquet files.

Supports loading data from local files with automatic format detection.
"""

import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from ..base import DataSourceAdapter


class FileAdapter(DataSourceAdapter):
    """
    Adapter for file-based data sources.

    Config example:
    {
        "type": "file",
        "path": "/path/to/data.csv",
        "format": "csv",  # csv, excel, parquet (auto-detected if not specified)

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

        # Excel-specific options
        "excel_options": {
            "sheet_name": 0,
            "header": 0
        },

        # Common options
        "parse_dates": True,
        "frequency": "D"
    }
    """

    def load(self) -> pd.DataFrame:
        """Load from file."""
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

        # Load based on format
        if file_format == "csv":
            df = self._load_csv(path)
        elif file_format == "excel":
            df = self._load_excel(path)
        elif file_format == "parquet":
            df = self._load_parquet(path)
        else:
            raise ValueError(
                f"Unsupported format: {file_format}. Supported formats: csv, excel, parquet"
            )

        # Set time index
        time_col = self.config.get("time_column")
        if time_col and time_col in df.columns:
            if self.config.get("parse_dates", True):
                with contextlib.suppress(Exception):
                    df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)

        # Only ensure datetime index if it looks like it should be one
        # or if we have a time_column. For RangeIndex, keep it as is.
        if not isinstance(df.index, pd.DatetimeIndex) and time_col:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                # If we explicitly asked for a time_column but can't convert, that's an error
                raise ValueError(
                    f"Could not convert time column '{time_col}' to datetime: {e}"
                ) from e

        # Sort by time
        df = df.sort_index()

        # Set frequency if specified
        freq = self.config.get("frequency")
        if freq:
            with contextlib.suppress(Exception):
                df = df.asfreq(freq)

        self._data = df

        # Determine frequency for metadata
        if isinstance(df.index, pd.DatetimeIndex):
            freq_str = str(df.index.freq) if df.index.freq else pd.infer_freq(df.index)
        else:
            freq_str = "Integer"

        self._metadata = {
            "source": "file",
            "path": str(path.absolute()),
            "format": file_format,
            "file_size_bytes": path.stat().st_size,
            "rows": len(df),
            "columns": list(df.columns),
            "frequency": freq_str,
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
        }

        return df

    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        suffix = path.suffix.lower()

        format_map = {
            ".csv": "csv",
            ".txt": "csv",
            ".tsv": "csv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".parquet": "parquet",
            ".pq": "parquet",
        }

        file_format = format_map.get(suffix)
        if not file_format:
            raise ValueError(
                f"Could not detect format from extension '{suffix}'. "
                "Please specify 'format' in config."
            )

        return file_format

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load CSV file."""
        csv_options = self.config.get("csv_options", {})

        # Set defaults
        csv_options.setdefault("sep", ",")
        csv_options.setdefault("header", 0)

        # Handle TSV files
        if path.suffix.lower() == ".tsv":
            csv_options["sep"] = "\t"

        # Parse dates if specified
        parse_dates = self.config.get("parse_dates", True)
        if parse_dates and self.config.get("time_column"):
            csv_options["parse_dates"] = [self.config["time_column"]]

        try:
            df = pd.read_csv(path, **csv_options)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}") from e

        return df

    def _load_excel(self, path: Path) -> pd.DataFrame:
        """Load Excel file."""
        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openpyxl is required for Excel files. Install with: pip install openpyxl"
            ) from e

        excel_options = self.config.get("excel_options", {})

        # Set defaults
        excel_options.setdefault("sheet_name", 0)
        excel_options.setdefault("header", 0)

        try:
            df = pd.read_excel(path, **excel_options)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}") from e

        return df

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for Parquet files. Install with: pip install pyarrow"
            ) from e

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise ValueError(f"Error reading Parquet file: {e}") from e

        return df

    def validate(self, data: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """Validate file data using pandas adapter validation."""
        from .pandas_adapter import PandasAdapter

        # Reuse pandas validation logic
        pandas_adapter = PandasAdapter({"data": data})
        return pandas_adapter.validate(data)
