"""
URL adapter for downloading files directly from the web.

Supports downloading and loading CSV, Excel, and Parquet files from URLs.
"""

import os
import asyncio
import aiohttp
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
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
            
    async def load_async(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> pd.DataFrame:
        """Asynchronously stream data from a URL using aiohttp."""
        url = self.config.get("url")
        if not url:
            raise ValueError("Config must contain 'url' key")
        
        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)
        if not filename:
            filename = "downloaded_data"
            
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = Path(temp_dir.name) / filename
        
        try:
            if progress_callback:
                async def cb(pct, msg):
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(pct, msg)
                    else:
                        progress_callback(pct, msg)
                await cb(0.0, f"Connecting to {url}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    total_size = response.headers.get("Content-Length")
                    total_size = int(total_size) if total_size else None
                    
                    downloaded = 0
                    chunk_size = 8192
                    
                    with open(temp_file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback:
                                if total_size:
                                    percentage = (downloaded / total_size) * 100
                                    await cb(percentage, f"Downloading: {percentage:.1f}%")
                                else:
                                    await cb(0.0, f"Downloading: {downloaded} bytes")
            
            if progress_callback:
                await cb(100.0, "Download complete. Processing...")
                
            # Parse async-ly just to not block the main thread
            loop = asyncio.get_event_loop()
            
            def parse_file():
                file_config = dict(self.config)
                file_config["type"] = "file"
                file_config["path"] = str(temp_file_path)
                file_adapter = FileAdapter(file_config)
                df = file_adapter.load()
                
                self._data = df
                self._metadata = file_adapter.get_metadata()
                self._metadata["source"] = "url"
                self._metadata["url"] = url
                
                if "path" in self._metadata:
                    del self._metadata["path"]
                return df
                
            return await loop.run_in_executor(None, parse_file)

        except Exception as e:
            raise ValueError(f"Error downloading or loading data from URL {url} async: {e}")
            
        finally:
            temp_dir.cleanup()
            
    def validate(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate URL data using pandas adapter validation."""
        from .pandas_adapter import PandasAdapter

        # Reuse pandas validation logic
        pandas_adapter = PandasAdapter({"data": data})
        return pandas_adapter.validate(data)
