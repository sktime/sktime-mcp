"""
Base adapter for data sources.

Defines the interface that all data source adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


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
            Tuple of (is_valid, validation_report)
            validation_report contains:
                - valid: bool
                - errors: List[str]
                - warnings: List[str]
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
                valid_exog_cols = [col for col in exog_cols if col in data.columns]
                X = data[valid_exog_cols] if valid_exog_cols else None
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
