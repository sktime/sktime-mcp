"""URL adapter security hardening tests.

Covers:
- Scheme enforcement (http/https only; blocks file://, ftp://, etc.)
- Opt-in private-IP / SSRF blocking
- Connection-level pinning to prevent DNS-rebinding attacks
- Strict redirect validation (blocks redirects to private IPs and non-http/https schemes)
- Redirect control (allow_redirects, max_redirects)
- Download timeout enforcement
- Max-bytes download cap
- Integration tests against a local ``http.server``
"""

import csv
import http.server
import socket
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import pytest

from sktime_mcp.data.adapters.url_adapter import _ALLOWED_SCHEMES, UrlAdapter


def _write_csv(directory: str, filename: str = "data.csv", rows: int = 20) -> Path:
    """Write a small CSV file and return its path."""
    path = Path(directory) / filename
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "value"])
        for i in range(rows):
            writer.writerow([f"2024-01-{i + 1:02d}", i * 10])
    return path


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Suppress request logging noise in test output."""

    def log_message(self, format, *args):  # noqa: A002
        pass


class _SlowHandler(_QuietHandler):
    """Serves content with a deliberate delay per chunk."""

    delay_seconds = 5

    def do_GET(self):  # noqa: N802
        # Wait longer than the test timeout before sending anything
        time.sleep(self.delay_seconds)
        super().do_GET()


class _RedirectHandler(_QuietHandler):
    """Server providing redirect scenarios for SSRF regression testing."""

    def do_GET(self):  # noqa: N802
        port = self.server.server_port
        if self.path == "/redirect-to-private":
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{port}/data.csv")
            self.end_headers()
        elif self.path == "/redirect-to-10":
            self.send_response(302)
            self.send_header("Location", "http://10.0.0.1/data.csv")
            self.end_headers()
        elif self.path == "/redirect-to-metadata":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
        elif self.path == "/redirect-to-file":
            self.send_response(302)
            self.send_header("Location", "file:///etc/passwd")
            self.end_headers()
        elif self.path == "/redirect-to-ftp":
            self.send_response(302)
            self.send_header("Location", "ftp://example.com/data.csv")
            self.end_headers()
        elif self.path == "/redirect-relative":
            self.send_response(302)
            self.send_header("Location", "/data.csv")
            self.end_headers()
        elif self.path == "/redirect-loop":
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{port}/redirect-loop")
            self.end_headers()
        else:
            super().do_GET()


@pytest.fixture()
def local_csv_server():
    """Spin up a local HTTP server serving a temp directory with a CSV file.

    Yields ``(base_url, csv_filename, temp_dir)``.
    """
    tmp = tempfile.TemporaryDirectory()
    csv_path = _write_csv(tmp.name)
    port = _find_free_port()

    # functools.partial binds the ``directory`` kwarg so each handler
    # instance serves from the temp dir instead of os.getcwd().
    handler_factory = partial(_QuietHandler, directory=tmp.name)

    httpd = http.server.HTTPServer(("127.0.0.1", port), handler_factory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}", csv_path.name, tmp

    httpd.shutdown()
    tmp.cleanup()


@pytest.fixture()
def slow_server():
    """Spin up a server that delays responses (for timeout tests).

    Yields ``(base_url, csv_filename, temp_dir)``.
    """
    tmp = tempfile.TemporaryDirectory()
    _write_csv(tmp.name)
    port = _find_free_port()

    handler_factory = partial(_SlowHandler, directory=tmp.name)

    httpd = http.server.HTTPServer(("127.0.0.1", port), handler_factory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}", "data.csv", tmp

    httpd.shutdown()
    tmp.cleanup()


@pytest.fixture()
def redirect_server():
    """Spin up a server capable of issuing various redirects for security tests.

    Yields ``(base_url, csv_filename, temp_dir)``.
    """
    tmp = tempfile.TemporaryDirectory()
    csv_path = _write_csv(tmp.name)
    port = _find_free_port()

    handler_factory = partial(_RedirectHandler, directory=tmp.name)

    httpd = http.server.HTTPServer(("127.0.0.1", port), handler_factory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}", csv_path.name, tmp

    httpd.shutdown()
    tmp.cleanup()


class TestSchemeEnforcement:
    """URL scheme must be http or https; everything else is rejected."""

    def test_file_scheme_rejected(self):
        with pytest.raises(ValueError, match="not allowed"):
            UrlAdapter._validate_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="not allowed"):
            UrlAdapter._validate_url("ftp://example.com/data.csv")

    def test_gopher_scheme_rejected(self):
        with pytest.raises(ValueError, match="not allowed"):
            UrlAdapter._validate_url("gopher://example.com/data")

    def test_no_scheme_rejected(self):
        with pytest.raises(ValueError, match="not allowed"):
            UrlAdapter._validate_url("/etc/passwd")

    def test_http_scheme_accepted(self):
        UrlAdapter._validate_url("http://example.com/data.csv")

    def test_https_scheme_accepted(self):
        UrlAdapter._validate_url("https://example.com/data.csv")

    def test_missing_hostname_rejected(self):
        with pytest.raises(ValueError, match="hostname"):
            UrlAdapter._validate_url("http:///path/only")

    def test_allowed_schemes_constant(self):
        assert {"http", "https"} == _ALLOWED_SCHEMES


class TestPrivateIpBlocking:
    """Private-IP blocking is off by default and opt-in via config."""

    def test_loopback_allowed_by_default(self):
        """Without block_private_ips, loopback passes scheme check."""
        # Should NOT raise — private-IP check is off by default
        UrlAdapter._validate_url("http://127.0.0.1/data.csv", block_private_ips=False)

    def test_loopback_blocked_when_opted_in(self):
        with pytest.raises(ValueError, match="private/internal"):
            UrlAdapter._validate_url("http://127.0.0.1/data.csv", block_private_ips=True)

    def test_link_local_blocked_when_opted_in(self):
        """Cloud metadata endpoint (169.254.169.254) should be blocked."""
        with pytest.raises(ValueError, match="private/internal"):
            UrlAdapter._validate_url(
                "http://169.254.169.254/latest/meta-data/",
                block_private_ips=True,
            )

    def test_private_10_range_blocked_when_opted_in(self):
        with pytest.raises(ValueError, match="private/internal"):
            UrlAdapter._validate_url("http://10.0.0.1/data.csv", block_private_ips=True)

    def test_unresolvable_host_blocked_when_opted_in(self):
        with pytest.raises(ValueError, match="Could not resolve"):
            UrlAdapter._validate_url(
                "http://this-host-does-not-exist-xyz123.invalid/data",
                block_private_ips=True,
            )

    def test_public_ip_accepted_when_opted_in(self):
        """A public IP should pass even with blocking enabled."""
        # Mock DNS resolution to return a public IP so test doesn't depend on
        # actual network connectivity.
        fake_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            UrlAdapter._validate_url("http://example.com/data.csv", block_private_ips=True)


class TestLoadSchemeEnforcement:
    """load() and load_async() must both reject bad schemes before connecting."""

    def test_load_rejects_file_scheme(self):
        adapter = UrlAdapter({"url": "file:///etc/passwd"})
        with pytest.raises(ValueError, match="not allowed"):
            adapter.load()

    def test_load_rejects_ftp_scheme(self):
        adapter = UrlAdapter({"url": "ftp://example.com/data.csv"})
        with pytest.raises(ValueError, match="not allowed"):
            adapter.load()

    @pytest.mark.asyncio
    async def test_load_async_rejects_file_scheme(self):
        adapter = UrlAdapter({"url": "file:///etc/passwd"})
        with pytest.raises(ValueError, match="not allowed"):
            await adapter.load_async()

    @pytest.mark.asyncio
    async def test_load_async_rejects_ftp_scheme(self):
        adapter = UrlAdapter({"url": "ftp://example.com/data.csv"})
        with pytest.raises(ValueError, match="not allowed"):
            await adapter.load_async()


class TestLocalServerIntegration:
    """End-to-end tests against a real local HTTP server."""

    def test_sync_load_from_local_server(self, local_csv_server):
        base_url, filename, _ = local_csv_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/{filename}",
                "time_column": "date",
                "target_column": "value",
            }
        )
        df = adapter.load()
        assert len(df) == 20
        assert "value" in df.columns
        meta = adapter.get_metadata()
        assert meta["source"] == "url"
        assert "path" not in meta  # temp path scrubbed

    def test_sync_load_metadata_has_url(self, local_csv_server):
        base_url, filename, _ = local_csv_server
        url = f"{base_url}/{filename}"
        adapter = UrlAdapter({"url": url})
        adapter.load()
        assert adapter.get_metadata()["url"] == url

    def test_max_bytes_cap_sync(self, local_csv_server):
        """A tiny max_download_bytes should trigger rejection."""
        base_url, filename, _ = local_csv_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/{filename}",
                "max_download_bytes": 10,  # 10 bytes — way too small
            }
        )
        with pytest.raises(ValueError, match="exceeded maximum size|max"):
            adapter.load()

    def test_private_ip_blocking_with_local_server(self, local_csv_server):
        """With block_private_ips=True, a loopback server should be rejected."""
        base_url, filename, _ = local_csv_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/{filename}",
                "block_private_ips": True,
            }
        )
        with pytest.raises(ValueError, match="private/internal"):
            adapter.load()

    def test_private_ip_default_allows_local_server(self, local_csv_server):
        """Default config allows loopback (block_private_ips defaults False)."""
        base_url, filename, _ = local_csv_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/{filename}",
                # block_private_ips not set → default False
            }
        )
        df = adapter.load()
        assert len(df) == 20


class TestDownloadTimeout:
    """Sync download should respect timeout configuration."""

    def test_timeout_triggers_on_slow_server(self, slow_server):
        """A very short timeout should abort against a slow server."""
        base_url, filename, _ = slow_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/{filename}",
                "download_timeout_seconds": 1,  # 1 second timeout
            }
        )
        # The slow server delays 5s before responding; urlopen(timeout=1)
        # should raise URLError/socket.timeout which we wrap as ValueError.
        with pytest.raises(ValueError):
            adapter.load()


class TestMaxBytesCap:
    """Download size cap must abort when the limit is exceeded."""

    def test_tiny_cap_rejects_download(self, local_csv_server):
        """A cap smaller than the file triggers rejection."""
        base_url, filename, _ = local_csv_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/{filename}",
                "max_download_bytes": 10,  # 10 bytes — way too small
            }
        )
        with pytest.raises(ValueError, match="exceeded maximum size|max"):
            adapter.load()

    def test_generous_cap_allows_download(self, local_csv_server):
        """A generous cap allows a normal download."""
        base_url, filename, _ = local_csv_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/{filename}",
                "max_download_bytes": 10 * 1024 * 1024,  # 10 MB
            }
        )
        df = adapter.load()
        assert len(df) == 20


class TestConfigValidation:
    """Edge cases for adapter config."""

    def test_missing_url_raises(self):
        adapter = UrlAdapter({})
        with pytest.raises(ValueError, match="url"):
            adapter.load()

    def test_config_keys_forwarded_to_file_adapter(self, local_csv_server):
        """csv_options and other keys should pass through to FileAdapter."""
        base_url, filename, _ = local_csv_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/{filename}",
                "csv_options": {"sep": ","},
            }
        )
        df = adapter.load()
        assert len(df) == 20


class TestRedirectSecurity:
    """Redirect validation and SSRF boundary enforcement."""

    def test_redirect_to_loopback_blocked_sync(self, redirect_server):
        """Redirecting to loopback must be rejected when block_private_ips=True."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-private",
                "block_private_ips": True,
            }
        )
        with pytest.raises(ValueError, match="private/internal"):
            adapter.load()

    @pytest.mark.asyncio
    async def test_redirect_to_loopback_blocked_async(self, redirect_server):
        """Async load redirecting to loopback must be rejected when block_private_ips=True."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-private",
                "block_private_ips": True,
            }
        )
        with pytest.raises(ValueError, match="private/internal"):
            await adapter.load_async()

    def test_redirect_to_10_range_blocked_sync(self, redirect_server):
        """Redirecting to private 10.0.0.0/8 must be rejected when block_private_ips=True."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-10",
                "block_private_ips": True,
            }
        )
        with pytest.raises(ValueError, match="private/internal"):
            adapter.load()

    @pytest.mark.asyncio
    async def test_redirect_to_10_range_blocked_async(self, redirect_server):
        """Async redirecting to 10.0.0.0/8 must be rejected when block_private_ips=True."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-10",
                "block_private_ips": True,
            }
        )
        with pytest.raises(ValueError, match="private/internal"):
            await adapter.load_async()

    def test_redirect_to_metadata_endpoint_blocked_sync(self, redirect_server):
        """Redirecting to cloud metadata 169.254.169.254 must be blocked."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-metadata",
                "block_private_ips": True,
            }
        )
        with pytest.raises(ValueError, match="private/internal"):
            adapter.load()

    @pytest.mark.asyncio
    async def test_redirect_to_metadata_endpoint_blocked_async(self, redirect_server):
        """Async redirecting to cloud metadata 169.254.169.254 must be blocked."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-metadata",
                "block_private_ips": True,
            }
        )
        with pytest.raises(ValueError, match="private/internal"):
            await adapter.load_async()

    def test_redirect_to_file_scheme_blocked_sync(self, redirect_server):
        """Redirecting to file:/// must be rejected regardless of block_private_ips."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-file",
                "block_private_ips": False,
            }
        )
        with pytest.raises(ValueError, match="not allowed"):
            adapter.load()

    @pytest.mark.asyncio
    async def test_redirect_to_file_scheme_blocked_async(self, redirect_server):
        """Async redirecting to file:/// must be rejected."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-file",
                "block_private_ips": False,
            }
        )
        with pytest.raises(ValueError, match="not allowed"):
            await adapter.load_async()

    def test_redirect_to_ftp_scheme_blocked_sync(self, redirect_server):
        """Redirecting to ftp:// must be rejected regardless of block_private_ips."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-ftp",
                "block_private_ips": False,
            }
        )
        with pytest.raises(ValueError, match="not allowed"):
            adapter.load()

    @pytest.mark.asyncio
    async def test_redirect_to_ftp_scheme_blocked_async(self, redirect_server):
        """Async redirecting to ftp:// must be rejected."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-to-ftp",
                "block_private_ips": False,
            }
        )
        with pytest.raises(ValueError, match="not allowed"):
            await adapter.load_async()

    def test_allow_redirects_false_blocks_redirect_sync(self, redirect_server):
        """allow_redirects=False must reject any redirect attempt."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-relative",
                "allow_redirects": False,
            }
        )
        with pytest.raises(ValueError, match="redirect"):
            adapter.load()

    @pytest.mark.asyncio
    async def test_allow_redirects_false_blocks_redirect_async(self, redirect_server):
        """Async allow_redirects=False must reject any redirect attempt."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-relative",
                "allow_redirects": False,
            }
        )
        with pytest.raises(ValueError, match="redirect"):
            await adapter.load_async()

    def test_redirect_loop_blocked_sync(self, redirect_server):
        """Redirect loop exceeding max_redirects must abort."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-loop",
                "max_redirects": 3,
            }
        )
        with pytest.raises(ValueError, match="Too many redirects"):
            adapter.load()

    @pytest.mark.asyncio
    async def test_redirect_loop_blocked_async(self, redirect_server):
        """Async redirect loop exceeding max_redirects must abort."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-loop",
                "max_redirects": 3,
            }
        )
        with pytest.raises(ValueError, match="Too many redirects"):
            await adapter.load_async()

    def test_safe_redirect_allowed_by_default_sync(self, redirect_server):
        """Safe redirect on loopback succeeds when block_private_ips is False."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-relative",
                "block_private_ips": False,
            }
        )
        df = adapter.load()
        assert len(df) == 20

    @pytest.mark.asyncio
    async def test_safe_redirect_allowed_by_default_async(self, redirect_server):
        """Async safe redirect on loopback succeeds when block_private_ips is False."""
        base_url, _, _ = redirect_server
        adapter = UrlAdapter(
            {
                "url": f"{base_url}/redirect-relative",
                "block_private_ips": False,
            }
        )
        df = await adapter.load_async()
        assert len(df) == 20


class TestDnsRebindingPrevention:
    """Verify that socket-level peer IP validation catches DNS-rebinding bypasses."""

    def test_dns_rebinding_blocked_sync(self, local_csv_server):
        """Pre-flight DNS passes (public IP), but socket connects to private IP."""
        base_url, filename, _ = local_csv_server
        port = int(urlparse(base_url).port)
        real_getaddrinfo = socket.getaddrinfo

        call_count = 0

        def rebinding_getaddrinfo(host, p, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Pre-flight check simulates resolving to a public IP
                return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", p or 0))]
            # Socket connection time resolves to local loopback (127.0.0.1)
            return real_getaddrinfo("127.0.0.1", port, *args, **kwargs)

        with patch("socket.getaddrinfo", side_effect=rebinding_getaddrinfo):
            adapter = UrlAdapter(
                {
                    "url": f"http://rebind-test.invalid:{port}/{filename}",
                    "block_private_ips": True,
                }
            )
            with pytest.raises(ValueError, match="private/internal"):
                adapter.load()

    @pytest.mark.asyncio
    async def test_dns_rebinding_blocked_async(self, local_csv_server):
        """Async pre-flight DNS passes (public IP), but socket connects to private IP."""
        base_url, filename, _ = local_csv_server
        port = int(urlparse(base_url).port)
        real_getaddrinfo = socket.getaddrinfo

        call_count = 0

        def rebinding_getaddrinfo(host, p, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", p or 0))]
            return real_getaddrinfo("127.0.0.1", port, *args, **kwargs)

        with patch("socket.getaddrinfo", side_effect=rebinding_getaddrinfo):
            adapter = UrlAdapter(
                {
                    "url": f"http://rebind-test.invalid:{port}/{filename}",
                    "block_private_ips": True,
                }
            )
            with pytest.raises(ValueError, match="private/internal"):
                await adapter.load_async()
