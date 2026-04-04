"""
URL adapter for downloading files directly from the web.

Supports downloading and loading CSV, Excel, and Parquet files from URLs.
"""

import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from ..base import DataSourceAdapter
from .file_adapter import FileAdapter


class UrlAdapter(DataSourceAdapter):
    """
    Adapter for downloading data from Web URLs.

    Config example:
    {
        "type": "url",
        "url": "https://raw.githubusercontent.com/.../data.csv",
        "format": "csv",  # csv, excel, parquet (auto-detected from URL if not specified)

        # Column mapping
        "time_column": "date",
        "target_column": "value",
        "exog_columns": ["feature1", "feature2"],

        # Options are passed identically as FileAdapter
        "csv_options": { ... },
        "parse_dates": True,
        "frequency": "D"
    }
    """

    def load(self) -> pd.DataFrame:
        url = self.config.get("url")
        if not url:
            raise ValueError("Config must contain 'url' key")

        # Determine a filename / extension from the URL if possible
        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = Path(path).name
        if not filename:
            filename = "downloaded_data"

        # Create a temporary directory to store the downloaded file
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = Path(temp_dir.name) / filename

        try:
            # Download the file
            urllib.request.urlretrieve(url, str(temp_file_path))

            # Prepare config for FileAdapter
            # We copy the config and override 'type' and 'path'
            file_config = dict(self.config)
            file_config["type"] = "file"
            file_config["path"] = str(temp_file_path)

            # Use FileAdapter to load the data
            file_adapter = FileAdapter(file_config)
            df = file_adapter.load()

            # Update metadata to reflect the URL source
            self._data = df
            self._metadata = file_adapter.get_metadata()
            self._metadata["source"] = "url"
            self._metadata["url"] = url

            # Remove the temporary local path set by FileAdapter
            if "path" in self._metadata:
                del self._metadata["path"]

            return df

        except Exception as e:
            raise ValueError(f"Error downloading or loading data from URL {url}: {e}") from e

        finally:
            # Clean up the temporary directory
            temp_dir.cleanup()

    def validate(self, data: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """Validate URL data using pandas adapter validation."""
        from .pandas_adapter import PandasAdapter

        # Reuse pandas validation logic
        pandas_adapter = PandasAdapter({"data": data})
        return pandas_adapter.validate(data)
