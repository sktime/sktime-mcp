"""
Pandas adapter for in-memory DataFrames.

Supports loading data from pandas DataFrames with automatic
time index detection and validation.
"""

from typing import Any

import pandas as pd

from ..base import DataSourceAdapter


def _safe_infer_freq(index: pd.Index) -> str | None:
    """pd.infer_freq requires >= 3 points and raises otherwise; return None instead.

    A 1–2 row series is a valid (if tiny) time series — it should load with no
    frequency rather than crash with pandas' "Need at least 3 dates" ValueError.
    """
    if len(index) < 3:
        return None
    with contextlib.suppress(ValueError, TypeError):
        return pd.infer_freq(index)
    return None


class PandasAdapter(DataSourceAdapter):
    """
    Adapter for in-memory pandas DataFrames.

    Config example::

        {
            "type": "pandas",
            "data": <DataFrame object or dict>,
            "time_column": "date",  # optional, will try to detect
            "target_column": "value",  # optional, defaults to first column
            "exog_columns": ["feature1", "feature2"],  # optional
            "frequency": "D"  # optional, will try to infer
        }
    """

    def load(self) -> pd.DataFrame:
        """Load from in-memory DataFrame or dict."""
        data = self.config.get("data")

        if data is None:
            raise ValueError("Config must contain 'data' key")

        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Data must be a pandas DataFrame or dict, got {type(data)}")

        # Set time index
        time_col = self.config.get("time_column")

        if time_col:
            # User specified time column
            if isinstance(time_col, list):
                missing = [c for c in time_col if c not in df.columns]
                if missing:
                    raise ValueError(f"Time columns {missing} not found in data")
            elif time_col not in df.columns:
                raise ValueError(f"Time column '{time_col}' not found in data")
            df = df.set_index(time_col)
        elif not isinstance(df.index, (pd.DatetimeIndex, pd.RangeIndex, pd.Index)):
            # Try to detect time column
            time_col = self._detect_time_column(df)
            if time_col:
                df = df.set_index(time_col)

        # Only ensure datetime index if it's already specified or looks like one
        if not isinstance(df.index, (pd.DatetimeIndex, pd.MultiIndex)) and time_col:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(
                    f"Could not convert time column '{time_col}' to datetime: {e}"
                ) from e

        # Sort by index
        df = df.sort_index()

        # Infer or set frequency
        freq = self.config.get("frequency")
        if freq:
            try:
                df = df.asfreq(freq)
            except Exception as e:
                raise ValueError(f"Invalid frequency '{freq}': {e}") from e
        elif isinstance(df.index, pd.DatetimeIndex) and df.index.freq is None:
            # Try to infer frequency (needs >= 3 points; short series just skip it)
            inferred_freq = _safe_infer_freq(df.index)
            if inferred_freq:
                df = df.asfreq(inferred_freq)

        self._data = df

        # Determine frequency for metadata
        if isinstance(df.index, pd.DatetimeIndex):
            freq_str = str(df.index.freq) if df.index.freq else _safe_infer_freq(df.index)
        else:
            freq_str = "Integer"

        self._metadata = {
            "source": "pandas",
            "rows": len(df),
            "columns": list(df.columns),
            "frequency": freq_str,
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
            "missing_values": df.isnull().sum().to_dict(),
        }

        return df

    def _detect_time_column(self, df: pd.DataFrame) -> str:
        """
        Try to detect which column is the time column.

        Looks for columns with datetime-like names or types.
        """
        # Common time column names
        time_names = ["date", "time", "datetime", "timestamp", "ds", "period"]

        for col in df.columns:
            # Check by name
            if col.lower() in time_names:
                return col

            # Check by type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        return None

    def validate(self, data: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """Validate pandas DataFrame for time series forecasting."""
        errors = []
        warnings = []

        # Check for sktime-compatible index
        if not isinstance(
            data.index, (pd.DatetimeIndex, pd.RangeIndex, pd.PeriodIndex, pd.Index, pd.MultiIndex)
        ):
            errors.append(
                "Index must be DatetimeIndex, PeriodIndex, RangeIndex, or MultiIndex for sktime forecasting"
            )

        # Additional check: if it's a generic Index, ensure it's at least numeric or string-period-like
        if (
            isinstance(data.index, pd.Index)
            and not isinstance(data.index, (pd.DatetimeIndex, pd.RangeIndex, pd.PeriodIndex))
            and not (
                pd.api.types.is_integer_dtype(data.index) or pd.api.types.is_float_dtype(data.index)
            )
        ):
            warnings.append(
                "Index is not DatetimeIndex or RangeIndex. sktime may have issues if it's not numeric or period-like."
            )

        # Check for empty data
        if len(data) == 0:
            errors.append("DataFrame is empty")

        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.any():
            missing_pct = (missing_counts / len(data) * 100).round(2)
            warnings.append(f"Missing values detected: {missing_pct[missing_pct > 0].to_dict()}")

        # Check for duplicate indices — a warning, not a hard error, so the
        # auto-format step (remove_duplicates) can actually run on the handle.
        # Rejecting here made that documented remedy unreachable (BUG-19).
        if not isinstance(data.index, pd.MultiIndex) and data.index.duplicated().any():
            dup_count = int(data.index.duplicated().sum())
            warnings.append(
                f"Duplicate time indices found: {dup_count}. They will be de-duplicated "
                "by auto-format (keeping the first of each)."
            )

        # Check for monotonic index
        if not data.index.is_monotonic_increasing:
            warnings.append("Time index is not sorted (will be sorted automatically)")

        # Check for sufficient data
        if len(data) < 10:
            warnings.append(
                f"Very small dataset ({len(data)} rows). Consider using more data for reliable forecasting."
            )

        # Check that the target column is numeric — forecasting cannot use an
        # object/string target, and this otherwise passes silently and fails
        # deep inside fit (#533 / NB-07).
        target_col = self.config.get("target_column")
        if target_col is None and len(data.columns) > 0:
            target_col = data.columns[0]
        if (
            target_col is not None
            and target_col in data.columns
            and not pd.api.types.is_numeric_dtype(data[target_col])
        ):
            warnings.append(
                f"Target column '{target_col}' has non-numeric dtype "
                f"'{data[target_col].dtype}'. Forecasting requires numeric values; "
                "convert the column before fitting."
            )

        # Check for constant values
        for col in data.columns:
            if data[col].nunique() == 1:
                warnings.append(f"Column '{col}' has constant values")

        # Check frequency (infer_freq needs >= 3 points; short series skip it)
        if isinstance(data.index, pd.DatetimeIndex) and len(data.index) >= 3:
            freq = _safe_infer_freq(data.index)
            if freq is None:
                warnings.append(
                    "Could not infer frequency. Time series may have irregular intervals."
                )

        is_valid = len(errors) == 0

        return is_valid, {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
        }
