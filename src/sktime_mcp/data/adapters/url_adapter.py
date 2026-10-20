"""
URL adapter for downloading files directly from the web.

Supports downloading and loading CSV, Excel, and Parquet files from URLs.
Enforces http/https-only scheme to prevent local file exfiltration (audit F-40).
"""

import ipaddress
import logging
import socket
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from ..base import DataSourceAdapter
from .file_adapter import FileAdapter

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}

# Defaults for download safety limits
_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


class UrlAdapter(DataSourceAdapter):
    """
    Adapter for downloading data from Web URLs.

    Config example::

        {
            "type": "url",
            "url": "https://raw.githubusercontent.com/.../data.csv",
            "format": "csv",  # csv, excel, parquet (auto-detected from URL)

            # Column mapping
            "time_column": "date",
            "target_column": "value",
            "exog_columns": ["feature1", "feature2"],

            # Options are passed identically as FileAdapter
            "csv_options": { ... },
            "parse_dates": True,
            "frequency": "D",

            # Security / safety options
            "block_private_ips": False,       # opt-in SSRF protection
            "download_timeout_seconds": 60,   # per-request timeout
            "max_download_bytes": 104857600,   # 100 MB download cap
        }

    Notes
    -----
    * Only ``http`` and ``https`` schemes are allowed.  ``file://``, ``ftp://``
      etc. are rejected to prevent local file exfiltration.
    * The optional ``block_private_ips`` flag resolves the hostname and rejects
      addresses that are private, reserved, loopback, or link-local.  This is
      **opt-in** because self-hosted / LAN data sources are legitimate.
      Note: this check **cannot** prevent DNS-rebinding attacks — a DNS name
      that initially resolves to a public IP may later resolve to a private one.
      The check also does **not** follow HTTP redirects; if the server redirects
      to a private/internal address the request will still proceed.  To
      re-validate each hop, an external HTTP library with redirect hooks is
      required.
    """

    @staticmethod
    def _validate_url(url: str, *, block_private_ips: bool = False) -> None:
        """Validate URL scheme and, optionally, block private/internal IPs.

        Parameters
        ----------
        url : str
            The URL to validate.
        block_private_ips : bool, default False
            If True, resolve the hostname and reject private / reserved /
            loopback / link-local addresses. Note that this check does not
            follow HTTP redirects (or re-validate each hop).

        Raises
        ------
        ValueError
            For disallowed schemes, missing hostnames, unresolvable hosts,
            or (when opted-in) private IP addresses.
        """
        parsed = urlparse(url)

        # 1. Scheme check — always enforced
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"URL scheme '{parsed.scheme}' is not allowed. "
                f"Only {_ALLOWED_SCHEMES} are permitted."
            )

        # 2. Hostname must be present
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("URL must contain a valid hostname.")

        # 3. (Opt-in) Resolve hostname and block private / reserved IP ranges
        if block_private_ips:
            try:
                addr_infos = socket.getaddrinfo(hostname, None)
            except socket.gaierror as e:
                raise ValueError(f"Could not resolve hostname '{hostname}': {e}") from e

            for _family, _type, _proto, _canon, sockaddr in addr_infos:
                ip = ipaddress.ip_address(sockaddr[0])
                if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
                    raise ValueError(
                        f"URL resolves to a private/internal address ({ip}). "
                        "Requests to internal networks are blocked because "
                        "'block_private_ips' is enabled."
                    )

    async def load_async(self, job_id: str | None = None) -> pd.DataFrame:
        """Download and load data asynchronously with progress reporting."""
        url = self.config.get("url")
        if not url:
            raise ValueError("Config must contain 'url' key")

        # Validate URL before making any network request
        block_private = self.config.get("block_private_ips", False)
        self._validate_url(url, block_private_ips=block_private)

        timeout = self.config.get("download_timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
        max_bytes = self.config.get("max_download_bytes", _DEFAULT_MAX_DOWNLOAD_BYTES)

        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name
        if not filename:
            filename = "downloaded_data"

        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = Path(temp_dir.name) / filename

        try:
            import aiohttp

            from sktime_mcp.runtime.jobs import get_job_manager

            job_manager = get_job_manager()

            client_timeout = aiohttp.ClientTimeout(total=timeout)
            async with (
                aiohttp.ClientSession(timeout=client_timeout) as session,
                session.get(url) as response,
            ):
                if response.status != 200:
                    raise ValueError(f"Error downloading from URL {url}: HTTP {response.status}")

                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0

                with temp_file_path.open("wb") as f:
                    async for chunk in response.content.iter_chunked(1024 * 64):
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError(
                                f"Download exceeded maximum size of "
                                f"{max_bytes / 1024 / 1024:.0f} MB. "
                                f"Increase 'max_download_bytes' in config if needed."
                            )
                        f.write(chunk)

                        if job_id and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            job_manager.update_job(
                                job_id,
                                current_step=f"Downloading: {progress:.1f}% ({downloaded / 1024 / 1024:.2f}MB)",
                            )

            # Use FileAdapter to load the downloaded file
            file_config = dict(self.config)
            file_config["type"] = "file"
            file_config["path"] = str(temp_file_path)

            file_adapter = FileAdapter(file_config)

            if job_id:
                job_manager.update_job(job_id, current_step="Parsing data file...")

            import asyncio

            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, file_adapter.load)

            self._data = df
            self._metadata = file_adapter.get_metadata()
            self._metadata["source"] = "url"
            self._metadata["url"] = url

            if "path" in self._metadata:
                del self._metadata["path"]

            return df

        except Exception as e:
            raise ValueError(f"Error downloading or loading data from URL {url}: {e}") from e

        finally:
            temp_dir.cleanup()

    def load(self) -> pd.DataFrame:
        """Download and load data synchronously."""
        url = self.config.get("url")
        if not url:
            raise ValueError("Config must contain 'url' key")

        # Validate URL before making any network request
        block_private = self.config.get("block_private_ips", False)
        self._validate_url(url, block_private_ips=block_private)

        timeout = self.config.get("download_timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
        max_bytes = self.config.get("max_download_bytes", _DEFAULT_MAX_DOWNLOAD_BYTES)

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
            # Download the file with timeout and size cap
            response = urllib.request.urlopen(url, timeout=timeout)
            try:
                downloaded = 0
                with temp_file_path.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 64)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError(
                                f"Download exceeded maximum size of "
                                f"{max_bytes / 1024 / 1024:.0f} MB. "
                                f"Increase 'max_download_bytes' in config if needed."
                            )
                        f.write(chunk)
            finally:
                response.close()

            # Prepare config for FileAdapter
            file_config = dict(self.config)
            file_config["type"] = "file"
            file_config["path"] = str(temp_file_path)

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
