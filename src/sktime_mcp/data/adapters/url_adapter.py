"""Bounded HTTP(S) downloads for CSV, Excel, and Parquet data."""

import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import pandas as pd

from ...config import settings
from ...redaction import redact_text, redact_url
from ..base import DataSourceAdapter
from .file_adapter import FileAdapter


class UrlAdapter(DataSourceAdapter):
    """Download HTTP(S) data, then load it using FileAdapter options.

    Config requires ``type="url"`` and ``url``. Optional ``format``, column
    mappings, ``csv_options``, ``parse_dates`` and ``frequency`` are passed to
    FileAdapter. Server settings bound the download's total time and byte count.
    """

    def _url(self) -> str:
        url = self.config.get("url")
        if not url:
            raise ValueError("Config must contain 'url' key")
        if urlparse(url).scheme.lower() not in {"http", "https"}:
            raise ValueError("URL data sources require an http:// or https:// URL")
        return url

    async def _download(self, url: str, path: Path, job_id: str | None = None) -> None:
        """Enforce one deadline across headers, redirects and streamed body reads."""
        from sktime_mcp.runtime.jobs import get_job_manager

        timeout = settings.url_timeout_seconds
        max_bytes = settings.url_max_bytes
        try:
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout, ceil_threshold=float("inf"))
                ) as session,
                session.get(url) as response,
            ):
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}")
                total_size = response.content_length
                if total_size is not None and total_size > max_bytes:
                    raise ValueError(f"Download exceeds maximum size of {max_bytes} bytes")

                downloaded = 0
                with path.open("wb") as output:
                    # aiohttp decodes compressed bodies. Count decoded bytes too,
                    # rather than trusting Content-Length (which may be absent).
                    async for chunk in response.content.iter_chunked(min(64 * 1024, max_bytes + 1)):
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError(f"Download exceeds maximum size of {max_bytes} bytes")
                        output.write(chunk)
                        if job_id:
                            get_job_manager().update_job(
                                job_id, current_step=f"Downloading: {downloaded} bytes"
                            )
        except asyncio.TimeoutError as exc:
            raise ValueError(f"Download timed out after {timeout} seconds") from exc

    def _parse_download(self, temp_dir, path: Path, url: str) -> pd.DataFrame:
        """Own temporary files until parsing finishes, even if the await is cancelled."""
        with temp_dir:
            adapter = FileAdapter({**self.config, "type": "file", "path": str(path)})
            data = adapter.load()
            self._data = data
            self._metadata = adapter.get_metadata()
            self._metadata.update(source="url", url=redact_url(url))
            self._metadata.pop("path", None)
            return data

    async def load_async(self, job_id: str | None = None) -> pd.DataFrame:
        """Download without blocking the event loop; parse in its shared worker pool."""
        url = self._url()
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / (Path(urlparse(url).path).name or "downloaded_data")
        parsing = None
        try:
            await self._download(url, path, job_id)
            if job_id:
                from sktime_mcp.runtime.jobs import get_job_manager

                get_job_manager().update_job(job_id, current_step="Parsing data file...")
            parsing = asyncio.get_running_loop().run_in_executor(
                None, self._parse_download, temp_dir, path, url
            )
            # A cancelled await must not cancel a queued parser: it owns cleanup.
            return await asyncio.shield(parsing)
        except asyncio.CancelledError:
            if parsing is not None:
                # Retrieve any eventual worker exception after the caller leaves.
                parsing.add_done_callback(
                    lambda future: future.exception() if not future.cancelled() else None
                )
            raise
        except Exception as exc:
            raise ValueError(
                f"Error downloading or loading data from URL {redact_url(url)}: {redact_text(str(exc))}"
            ) from exc
        finally:
            if parsing is None:
                temp_dir.cleanup()

    def load(self) -> pd.DataFrame:
        """Synchronous compatibility API, using the same bounded downloader.

        Async callers should await load_async. If a synchronous caller already
        has an event loop, use a separate thread to avoid nesting asyncio.run.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.load_async())
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(self.load_async())).result()

    def validate(self, data: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """Validate URL data using pandas adapter validation."""
        from .pandas_adapter import PandasAdapter

        return PandasAdapter(self.config).validate(data)
