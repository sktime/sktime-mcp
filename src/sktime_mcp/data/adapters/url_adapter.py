"""
URL adapter for downloading files directly from the web.

Supports downloading and loading CSV, Excel, and Parquet files from URLs.
"""

import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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
        """
        Download and load data from URL (synchronous).
        """
        url = self.config.get("url")
        if not url:
            raise ValueError("Config must contain 'url' key")

        # Determine a filename / extension from the URL if possible
        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)
        if not filename:
            filename = "downloaded_data"

        # Create a temporary directory to store the downloaded file
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = Path(temp_dir.name) / filename

        try:
            # Download the file
            urllib.request.urlretrieve(url, str(temp_file_path))

            # Prepare config for FileAdapter
            file_config = dict(self.config)
            file_config["type"] = "file"
            file_config["path"] = str(temp_file_path)

            # Use FileAdapter to load the data
            file_adapter = FileAdapter(file_config)
            df = file_adapter.load()

            # Update metadata
            self._data = df
            self._metadata = file_adapter.get_metadata()
            self._metadata["source"] = "url"
            self._metadata["url"] = url

            if "path" in self._metadata:
                del self._metadata["path"]

            return df

        except Exception as e:
            raise ValueError(f"Error downloading or loading data from URL {url}: {e}")

        finally:
            temp_dir.cleanup()

    async def load_async(self, job_id: Optional[str] = None) -> pd.DataFrame:
        """
        Download data from URL asynchronously with progress tracking.
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp is required for UrlAdapter.load_async()")

        url = self.config.get("url")
        if not url:
            raise ValueError("Config must contain 'url' key")

        # Determine a filename / extension from the URL if possible
        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)
        if not filename:
            filename = "downloaded_data"

        # Create a temporary directory to store the downloaded file
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = Path(temp_dir.name) / filename

        try:
            from sktime_mcp.runtime.jobs import get_job_manager

            job_manager = get_job_manager()

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download from {url}: HTTP {response.status}")

                    total_size = int(response.headers.get("content-length", 0))
                    downloaded_size = 0

                    with open(temp_file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded_size += len(chunk)

                            # Update progress if job_id is provided
                            if job_id and total_size > 0:
                                percent = (downloaded_size / total_size) * 100
                                job_manager.update_job(
                                    job_id, current_step=f"Downloading: {percent:.1f}%"
                                )

            # Prepare config for FileAdapter
            file_config = dict(self.config)
            file_config["type"] = "file"
            file_config["path"] = str(temp_file_path)

            # Use FileAdapter to load the data
            file_adapter = FileAdapter(file_config)
            df = await file_adapter.load_async(job_id=job_id)

            # Update metadata
            self._data = df
            self._metadata = file_adapter.get_metadata()
            self._metadata["source"] = "url"
            self._metadata["url"] = url

            if "path" in self._metadata:
                del self._metadata["path"]

            return df

        except Exception as e:
            raise ValueError(f"Error downloading or loading data from URL {url}: {e}")

        finally:
            temp_dir.cleanup()

    def validate(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate URL data using pandas adapter validation."""
        from .pandas_adapter import PandasAdapter

        # Reuse pandas validation logic
        pandas_adapter = PandasAdapter({"data": data})
        return pandas_adapter.validate(data)
