"""
Base adapter for data sources.

Defines the interface that all data source adapters must implement.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


def coerce_time_index(
    df: pd.DataFrame, time_column: str | list[str] | None
) -> tuple[pd.DataFrame, list[str]]:
    """Set ``time_column`` as the index and coerce it to a sktime-usable index.

    Shared by the pandas, file and SQL adapters so they agree on what a time
    column may contain:

    - integer / integer-like values (``year``: 2018..2022) become an integer
      index, reported downstream as ``frequency: "Integer"`` — they are
      **not** fed to ``pd.to_datetime`` (which read them as epoch nanoseconds).
    - datetime dtypes are kept; strings are parsed with ``pd.to_datetime``.
      Timestamps with mixed UTC offsets are normalised to UTC and a warning is
      returned instead of leaving an object index behind.
    - anything else (booleans, ints mixed with strings, unparseable strings)
      raises ``ValueError`` naming the column.

    ``time_column`` may already be the index name (adapters that set the index
    earlier) or a list of columns (MultiIndex, left untouched). With no
    ``time_column`` the frame is returned unchanged, keeping its RangeIndex.

    Returns
    -------
    (df, warnings)
        The re-indexed frame and validation warnings to surface to the caller.
    """
    if time_column is None or time_column == "" or isinstance(time_column, list):
        if isinstance(time_column, list):
            df = df.set_index(time_column)
        return df, []

    if time_column in df.columns:
        df = df.set_index(time_column)
    elif df.index.name != time_column:
        available = ", ".join(repr(c) for c in df.columns)
        raise ValueError(
            f"Time column {time_column!r} not found in data. Available columns: [{available}]"
        )

    index = df.index
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex, pd.MultiIndex)):
        return df, []

    kind = pd.api.types.infer_dtype(index, skipna=True)
    if kind == "integer":
        if index.hasnans:
            raise ValueError(f"Time column {time_column!r} contains missing values.")
        df.index = pd.Index(index.astype("int64"), name=index.name)
        return df, []
    if kind in ("floating", "mixed-integer-float"):
        values = index.to_numpy(dtype="float64")
        if pd.isna(values).any() or (values != values.round()).any():
            raise ValueError(
                f"Time column {time_column!r} has non-integer float values; "
                "use an integer index or date-like values (dates or datetime strings)."
            )
        df.index = pd.Index(values.astype("int64"), name=index.name)
        return df, []
    if kind == "mixed-integer" or (
        kind == "mixed" and any(isinstance(v, (int, float)) for v in index)
    ):
        raise ValueError(
            f"Time column {time_column!r} mixes numbers with non-numeric values; "
            "use all integers (period numbers) or all dates."
        )
    if kind not in ("string", "datetime", "datetime64", "date", "mixed", "empty"):
        raise ValueError(
            f"Could not convert time column {time_column!r} to a time index: "
            f"values are {kind}, expected integers or date-like values."
        )

    notes: list[str] = []
    try:
        with warnings.catch_warnings():
            # pandas < 3 emits a FutureWarning for mixed offsets and returns an
            # object Index; pandas >= 3 raises. Both fall through to utc=True.
            warnings.simplefilter("ignore", FutureWarning)
            converted = pd.to_datetime(index)
        if not isinstance(converted, pd.DatetimeIndex):
            raise ValueError("Mixed timezones detected")
    except (ValueError, TypeError) as e:
        if "mixed" not in str(e).lower() or "timezone" not in str(e).lower():
            raise ValueError(
                f"Could not convert time column {time_column!r} to datetime: {e}"
            ) from e
        try:
            converted = pd.to_datetime(index, utc=True)
        except (ValueError, TypeError) as e2:
            raise ValueError(
                f"Could not convert time column {time_column!r} to datetime: {e2}"
            ) from e2
        notes.append(
            f"Time column {time_column!r} contains timestamps with mixed UTC offsets; "
            "they were normalised to UTC."
        )
    converted.name = index.name
    df.index = converted
    return df, notes


def check_frequency_alignment(index: pd.Index, freq: str) -> None:
    """Raise ``ValueError`` when an explicit ``frequency`` does not fit the timestamps.

    ``df.asfreq(freq)`` reindexes onto ``freq``'s own anchor: monthly ``"M"``
    (month end) on mid-month dates keeps none of the original rows and returns
    all-NaN data. Callers check alignment first and only ``asfreq`` when every
    existing timestamp lies on the ``freq`` grid (gaps are still allowed).

    An unparseable ``freq`` string is not judged here — ``asfreq`` reports it.
    Non-datetime indexes are left to the caller too.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            expected = pd.date_range(start=index.min(), end=index.max(), freq=freq)
    except (ValueError, TypeError):
        return
    misaligned = index[~index.isin(expected)]
    if len(misaligned) == 0:
        return
    first = misaligned[0]
    grid = ", ".join(str(t) for t in expected[:3])
    raise ValueError(
        f"Frequency {freq!r} does not align with the data: {len(misaligned)} of "
        f"{len(index)} timestamps are not on the {freq!r} grid (first: {first}; "
        f"grid starts {grid}{', ...' if len(expected) > 3 else ''}). "
        "Reindexing would replace the data with NaN. Use a frequency anchored on "
        "the actual timestamps (e.g. 'MS' for month starts, 'W-MON' for Mondays) "
        "or omit 'frequency' to have it inferred."
    )


class DataSourceAdapter(ABC):
    """
    Abstract base class for all data source adapters.

    All adapters must implement:
    - load(): Fetch data from source
    - validate(): Check data quality
    - to_sktime_format(): Convert to sktime-compatible format
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the adapter.

        Args:
            config: Configuration dictionary specific to the adapter type
        """
        self.config = config
        self._data = None
        self._metadata = {}

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data from the source (synchronous).

        Returns:
            DataFrame with time index
        """
        pass

    async def load_async(self, job_id: str | None = None) -> pd.DataFrame:
        """
        Load data from the source (asynchronous).

        Default implementation runs the synchronous load() in a separate thread.
        Adapters should override this for true non-blocking async IO.

        Args:
            job_id: Optional job ID for progress reporting

        Returns:
            DataFrame with time index
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load)

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """
        Validate data quality.

        Args:
            data: DataFrame to validate

        Returns:
            Tuple of ``(is_valid, validation_report)``, where
            ``validation_report`` contains the keys ``valid`` (bool),
            ``errors`` (list of str), and ``warnings`` (list of str).
        """
        pass

    def to_sktime_format(self, data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame | None]:
        """
        Convert to sktime format (y, X).

        Args:
            data: DataFrame to convert

        Returns:
            Tuple of (y, X) where:
            - y: Target time series (pd.Series with DatetimeIndex)
            - X: Exogenous variables (pd.DataFrame, optional)
        """
        # Get target column from config
        target_col = self.config.get("target_column")
        exog_cols = self.config.get("exog_columns", [])
        if isinstance(exog_cols, str):
            # A bare string would otherwise be iterated per character (#556 / F-37)
            exog_cols = [exog_cols]

        if target_col is not None and target_col not in data.columns:
            available_columns = ", ".join(repr(col) for col in data.columns)
            raise ValueError(
                f"Target column {target_col!r} not found in data. "
                f"Available columns: [{available_columns}]"
            )

        if target_col is not None:
            y = data[target_col]

            # Get exogenous variables if specified
            if exog_cols:
                missing_exog_cols = [col for col in exog_cols if col not in data.columns]
                if missing_exog_cols:
                    available_columns = ", ".join(repr(col) for col in data.columns)
                    raise ValueError(
                        f"Exogenous column(s) not found in data: {missing_exog_cols!r}. "
                        f"Available columns: [{available_columns}]"
                    )

                X = data[exog_cols]
            else:
                # Use all columns except target as exogenous
                other_cols = [col for col in data.columns if col != target_col]
                X = data[other_cols] if other_cols else None
        else:
            # Default: first column is target, rest are exogenous
            if len(data.columns) == 1:
                y = data.iloc[:, 0]
                X = None
            else:
                y = data.iloc[:, 0]
                X = data.iloc[:, 1:]

                # Add a guideline warning if we're defaulting with multiple columns
                if not hasattr(self, "_metadata") or self._metadata is None:
                    self._metadata = {}

                if "validation" not in self._metadata:
                    self._metadata["validation"] = {"valid": True, "errors": [], "warnings": []}

                # Ensure it's a dict and has warnings list
                val = self._metadata["validation"]
                if isinstance(val, dict) and "warnings" in val:
                    val["warnings"].append(
                        f"Target column not specified. Defaulting to first column '{data.columns[0]}'. "
                        "If this is a time index or feature, please specify 'target_column' in config."
                    )

        return y, X

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata about the data source.

        Returns:
            Dictionary with metadata (rows, columns, frequency, etc.)
        """
        return self._metadata
