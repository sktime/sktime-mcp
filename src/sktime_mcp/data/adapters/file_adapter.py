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
        
    async def load_async(
        self,
        progress_callback=None,
    ) -> pd.DataFrame:
        """Asynchronously load from file."""
        import asyncio
        loop = asyncio.get_event_loop()
        
        async def cb(pct, msg):
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(pct, msg)
                else:
                    progress_callback(pct, msg)

        path_str = self.config.get("path")
        if not path_str:
            raise ValueError("Config must contain 'path' key")
        
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        file_format = self.config.get("format")
        if not file_format:
            file_format = self._detect_format(path)

        await cb(0.0, f"Starting load for {file_format} file...")

        if file_format == "csv":
            # For CSV, we can use chunking in an executor to avoid blocking the loop for long periods
            # and to provide progress updates
            
            csv_options = self.config.get("csv_options", {}).copy()
            csv_options.setdefault("sep", ",")
            csv_options.setdefault("header", 0)
            
            if path.suffix.lower() == ".tsv":
                csv_options["sep"] = "\t"
                
            parse_dates = self.config.get("parse_dates", True)
            if parse_dates and self.config.get("time_column"):
                csv_options["parse_dates"] = [self.config["time_column"]]
            
            # Read in chunks
            chunk_size = 10000
            csv_options["chunksize"] = chunk_size
            
            total_size_bytes = path.stat().st_size
            
            # Using traditional file handle inside the executor for pandas
            def init_reader():
                return pd.read_csv(path, **csv_options)
            
            reader = await loop.run_in_executor(None, init_reader)
            
            chunks = []
            estimated_bytes_read = 0
            # Rough heuristic: chunk_size rows * ~50 bytes per row... better heuristic is just based on total rows but we don't know total rows yet
            # Let's just track rows processed and percentage as an estimate
            rows_processed = 0
            
            import gc
            
            # Consume chunks
            while True:
                def get_next_chunk():
                    try:
                        return next(reader) # reader is a TextFileReader iterator
                    except StopIteration:
                        return None
                        
                chunk = await loop.run_in_executor(None, get_next_chunk)
                if chunk is None:
                    break
                    
                chunks.append(chunk)
                rows_processed += len(chunk)
                
                # We can't perfectly know bytes read without custom wrappers, but we can emit rows and a rough percentage
                # Let's say we assume avg row is 100 bytes for a rough percentage
                estimated_bytes = rows_processed * 100 
                percent = min((estimated_bytes / total_size_bytes) * 100, 99.0) if total_size_bytes > 0 else 0
                await cb(percent, f"Read {rows_processed} rows from CSV...")
                
                # yield to loop
                await asyncio.sleep(0)
            
            await cb(95.0, "Concatenating chunks...")
            def concat_chunks():
                return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                
            df = await loop.run_in_executor(None, concat_chunks)
            
        else:
            # For excel/parquet, full non-blocking wrapper Since chunking isn't natively exposed via these engines as easily as CSV
            await cb(10.0, f"Reading {file_format} file (this may take a moment)...")
            
            def load_full():
                if file_format == "excel":
                    return self._load_excel(path)
                elif file_format == "parquet":
                    return self._load_parquet(path)
                else:
                    raise ValueError(f"Unsupported format: {file_format}")
                    
            df = await loop.run_in_executor(None, load_full)

        await cb(95.0, "Applying index and formatting...")

        # Same formatting block as synchronous load
        time_col = self.config.get("time_column")
        if time_col and time_col in df.columns:
            if self.config.get("parse_dates", True) and file_format != "csv": # csv parsed already via read_csv
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                except Exception:
                    pass
            df = df.set_index(time_col)
        
        if not isinstance(df.index, pd.DatetimeIndex) and time_col:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Could not convert time column '{time_col}' to datetime: {e}")
        
        df = df.sort_index()
        
        freq = self.config.get("frequency")
        if freq:
            try:
                df = df.asfreq(freq)
            except Exception:
                pass
        
        self._data = df
        
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
        
        await cb(100.0, "File loaded successfully!")
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
