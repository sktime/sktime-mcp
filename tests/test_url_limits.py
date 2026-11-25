"""URL download budgets and event-loop responsiveness, using tiny local responses."""

import asyncio
import gzip
import http.server
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from sktime_mcp.data.adapters.url_adapter import UrlAdapter
from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import JobStatus, get_job_manager
from tests.test_mcp_wire import _body, _connected_session

CSV = b"date,value\n2024-01-01,1\n2024-01-02,2\n2024-01-03,3\n"
LIMIT = 128


@pytest.fixture
def download_server(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    monkeypatch.setenv("SKTIME_MCP_URL_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("SKTIME_MCP_URL_MAX_BYTES", str(LIMIT))
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            endpoint = urlparse(self.path).path
            started.set()
            try:
                if endpoint == "/redirect.csv":
                    self.send_response(302)
                    self.send_header("Location", "/large.csv")
                    self.end_headers()
                    return
                if endpoint == "/file-redirect.csv":
                    self.send_response(302)
                    self.send_header("Location", "file:///tmp/private.csv")
                    self.end_headers()
                    return
                if endpoint in {"/stall.csv", "/held.csv"}:
                    release.wait(timeout=3)
                self.send_response(200)
                if endpoint == "/large.csv":
                    self.send_header("Content-Length", str(LIMIT + 1))
                    self.end_headers()
                    return  # Reject the headers, without waiting for any body.
                if endpoint == "/chunked.csv":
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for _ in range(3):
                        chunk = b"x" * 64
                        self.wfile.write(b"40\r\n" + chunk + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                    return
                if endpoint == "/drip.csv":
                    self.end_headers()
                    for _ in range(50):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        if release.wait(timeout=0.05):
                            break
                    return
                body = CSV
                if endpoint in {"/no-length.csv", "/gzip.csv"}:
                    body = b"x" * (LIMIT + 1)
                if endpoint == "/gzip.csv":
                    body = gzip.compress(body)
                    self.send_header("Content-Encoding", "gzip")
                if endpoint != "/no-length.csv":
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when a size limit, deadline or cancellation fires.
            finally:
                finished.set()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            url=f"http://127.0.0.1:{server.server_port}",
            started=started,
            release=release,
            finished=finished,
            temp_path=tmp_path,
        )
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _adapter(server, endpoint):
    return UrlAdapter(
        {
            "type": "url",
            "url": f"{server.url}/{endpoint}.csv?token=private-test-token",
            "time_column": "date",
            "target_column": "value",
        }
    )


async def _load(adapter, mode):
    if mode == "async":
        return await adapter.load_async()
    return await asyncio.to_thread(adapter.load)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_download_at_limit_succeeds_and_cleans_up(download_server, monkeypatch, mode):
    monkeypatch.setenv("SKTIME_MCP_URL_MAX_BYTES", str(len(CSV)))
    adapter = _adapter(download_server, "small")
    data = await _load(adapter, mode)
    assert len(data) == 3
    assert adapter.get_metadata()["source"] == "url"
    assert "private-test-token" not in str(adapter.get_metadata())
    assert not list(download_server.temp_path.iterdir())


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("endpoint", ["large", "no-length", "chunked", "gzip", "redirect"])
async def test_download_rejects_oversized_body(download_server, mode, endpoint):
    with pytest.raises(ValueError, match=f"maximum size of {LIMIT} bytes") as exc:
        await _load(_adapter(download_server, endpoint), mode)
    assert "private-test-token" not in str(exc.value)
    assert not list(download_server.temp_path.iterdir())


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("endpoint", ["stall", "drip"])
async def test_download_total_deadline(download_server, mode, endpoint):
    start = time.monotonic()
    with pytest.raises(ValueError, match="timed out after 1 seconds") as exc:
        await _load(_adapter(download_server, endpoint), mode)
    assert time.monotonic() - start < 2
    assert "private-test-token" not in str(exc.value)
    assert not list(download_server.temp_path.iterdir())


async def test_sync_compatibility_api_inside_event_loop(download_server):
    assert len(_adapter(download_server, "small").load()) == 3
    assert not list(download_server.temp_path.iterdir())


@pytest.mark.parametrize("url", ["file:///tmp/private.csv", "ftp://example.invalid/data.csv"])
async def test_url_adapter_rejects_non_http_schemes(url):
    with pytest.raises(ValueError, match="http:// or https://"):
        await UrlAdapter({"url": url}).load_async()


async def test_redirect_cannot_switch_to_local_file(download_server):
    with pytest.raises(ValueError):
        await _adapter(download_server, "file-redirect").load_async()
    assert not list(download_server.temp_path.iterdir())


@pytest.mark.parametrize("run_async", [False, True])
async def test_mcp_load_keeps_serving_requests(download_server, monkeypatch, run_async):
    monkeypatch.setenv("SKTIME_MCP_URL_TIMEOUT_SECONDS", "10")
    jobs = get_job_manager()
    job_id = None
    data_handle = None
    pending = None
    try:
        async with _connected_session() as session:
            pending = asyncio.create_task(
                session.call_tool(
                    "load_data",
                    {"config": _adapter(download_server, "held").config, "run_async": run_async},
                )
            )
            assert await asyncio.to_thread(download_server.started.wait, 3)
            # A separate MCP request completes while the HTTP server holds the download.
            other = _body(await session.call_tool("list_jobs", {}))
            assert other["success"]
            assert not download_server.finished.is_set(), "URL download blocked the MCP server"
            download_server.release.set()
            result = _body(await pending)
            assert result["success"]
            if run_async:
                job_id = result["job_id"]
                task = jobs._tasks.get(job_id)
                if task is not None:
                    await task
                assert jobs.get_job(job_id).status is JobStatus.COMPLETED
                result = jobs.get_job(job_id).result
            data_handle = result["data_handle"]
    finally:
        download_server.release.set()
        if pending is not None and not pending.done():
            await pending
        if data_handle:
            get_executor().release_data_handle(data_handle)
        if job_id:
            jobs.delete_job(job_id)
    assert not list(download_server.temp_path.iterdir())


async def test_background_download_cancel_closes_download_and_cleans_up(download_server):
    jobs = get_job_manager()
    async with _connected_session() as session:
        response = _body(
            await session.call_tool(
                "load_data",
                {"config": _adapter(download_server, "stall").config, "run_async": True},
            )
        )
        job_id = response["job_id"]
        task = jobs._tasks[job_id]
        try:
            assert await asyncio.to_thread(download_server.started.wait, 3)
            response = _body(await session.call_tool("cancel_job", {"job_id": job_id}))
            assert response["success"]
            with pytest.raises(asyncio.CancelledError):
                await task
            assert jobs.get_job(job_id).status is JobStatus.CANCELLED
            assert not list(download_server.temp_path.iterdir())
        finally:
            download_server.release.set()
            jobs.delete_job(job_id)


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("endpoint, error", [("large", "maximum size"), ("stall", "timed out")])
async def test_mcp_download_limits_return_failure(download_server, run_async, endpoint, error):
    jobs = get_job_manager()
    job_id = None
    try:
        async with _connected_session() as session:
            result = _body(
                await session.call_tool(
                    "load_data",
                    {"config": _adapter(download_server, endpoint).config, "run_async": run_async},
                )
            )
            if run_async:
                job_id = result["job_id"]
                task = jobs._tasks.get(job_id)
                if task is not None:
                    await task
                job = jobs.get_job(job_id)
                assert job.status is JobStatus.FAILED
                assert error in str(job.errors)
                assert "private-test-token" not in str(job.errors)
            else:
                assert result["success"] is False
                assert error in result["error"]
                assert "private-test-token" not in result["error"]
                assert "data_handle" not in result
    finally:
        if job_id:
            jobs.delete_job(job_id)


async def test_cancel_during_parse_defers_cleanup_to_worker(download_server, monkeypatch):
    from sktime_mcp.data.adapters.file_adapter import FileAdapter

    started = threading.Event()
    release = threading.Event()
    original_load = FileAdapter.load

    def held_load(adapter):
        started.set()
        release.wait(timeout=3)
        return original_load(adapter)

    monkeypatch.setattr(FileAdapter, "load", held_load)
    task = asyncio.create_task(_adapter(download_server, "small").load_async())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The parser still owns its file; cancellation must not remove it early.
        assert list(download_server.temp_path.rglob("small.csv"))
    finally:
        release.set()
        for _ in range(300):
            if not list(download_server.temp_path.iterdir()):
                break
            await asyncio.sleep(0.01)
    assert not list(download_server.temp_path.iterdir())
