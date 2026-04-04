# PR Description: Async Data Loading Infrastructure & URL Adapter

## Overview
This PR implements an asynchronous data loading infrastructure for `sktime-mcp`, moving away from blocking synchronous operations that previously caused timeouts in MCP clients (e.g., Claude Desktop). It introduces progress reporting for long-running data loads and integrates seamlessly with the existing `JobManager`.

## Key Changes
- **Async Adapter Interface**: Added `load_async` to the `DataSourceAdapter` base class.
- **Async URL Loading**: Implemented `UrlAdapter.load_async` using `aiohttp` for non-blocking, streaming downloads.
- **Progress Reporting**: Added real-time progress updates (e.g., `"Downloading: 45.2%"`) via the `JobManager`.
- **MCP Tool**: Introduced `load_data_source_async` tool to the server, allowing clients to initiate loads in the background and poll for status.
- **Dependency Update**: Added `aiohttp` to `pyproject.toml`.

## Implementation Details
- `DataSourceAdapter.load_async`: Provides a default thread-based fallback for sync adapters.
- `Executor.load_data_source_async`: Orchestrates the lifecycle of an async load job (Init -> Load -> Validate -> Finalize).
- `UrlAdapter.load_async`: Uses a `ClientSession` to stream data and report bytes downloaded relative to `Content-Length`.

## Verification Status
Verified with a full suite of automated tests (20/20 passed):
- **New Tests**: `tests/test_async_data.py` (Mocks `aiohttp` and verifies progress/executor logic).
- **Regression Tests**: All existing core and background job tests passed.

### Test Command
```bash
python3 -m pytest tests/
```

## How to Test Manually
1. Start the server: `python -m sktime_mcp.server`
2. Call `load_data_source_async` with a URL (e.g., a large CSV from GitHub).
3. Use the returned `job_id` with `check_job_status` to see the progress percentage and current status.
