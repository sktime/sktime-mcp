# Architecture

The MCP layer exposes a small set of workflow operations while delegating estimator
metadata, datatype validation and conversion to sktime.

## Request path

```text
MCP request
  -> advertised tool + JSON schema validation
  -> tool function / executor
  -> sktime, adapter, or persistence operation
  -> redacted JSON result + MCP error flag
```

Unknown and retired tool names are rejected before dispatch. `get_tools()` is the
source for the advertised 25-tool interface. The data schemas live in
`data/schemas.py`, including action-specific preparation constraints.

## Data path

```text
memory / file / SQL / HTTP(S)
  -> adapter reads a native container
  -> explicit index and column mapping
  -> sktime validation and bounded inspection
  -> one shared registration path (sync and async)
  -> data handle
```

| Module | Responsibility |
| --- | --- |
| `data/adapters/` | Source I/O and source metadata |
| `data/contracts.py` | Scitype/mtype declarations, JSON materialization, index/column mapping, validation and inspection |
| `data/preparation.py` | Explicit formatting, sktime conversion, temporal/instance splitting |
| `data/schemas.py` | MCP input contracts for loading and preparation |
| `runtime/executor.py` | Handle registration, shared finalization and estimator execution |
| `tools/prepare_data.py` | Public preparation entry point |

Data records retain primary data, optional X, scientific type, role, metadata,
validation and source configuration. Input containers are copied where applicable;
preparation creates new records and preserves its source. Multi-output preparation
reserves capacity before any outputs are registered. Temporal preparation works
within each instance group; it does not fill across panel boundaries.

Formatting is separate from loading. The forecasting runtime may adapt a private
regular DatetimeIndex view to periods for seasonal-estimator compatibility. The
stored source and non-forecast tasks retain their representation.

## Runtime and concurrency

Estimator handles and data handles are process-wide. Background fitting,
prediction, evaluation and loading are retained asyncio tasks managed by the job
manager. Blocking estimator calls run in the event loop's shared worker pool.
Cancelling the coroutine does not join its worker or force-stop native code.

Foreground and background loading await adapter I/O and use the same finalization.
HTTP(S) downloads have a total deadline and a streamed size limit. Parsing runs in
a worker; cancellation leaves temporary-file cleanup with that worker until it
finishes. Other synchronous tool work can still occupy the event loop; native
method calls are not universally asynchronous.

## Trust boundary

The server account is the execution boundary. Craft specifications are evaluated
by sktime and shell commands run with the account's permissions. HTTP/SSE bearer
authentication and host/origin allowlists protect access, not filesystem isolation.
The `/` health endpoint is public. Stdio uses local process permissions.

Data-source credentials and credential-bearing URLs are redacted from response
errors, metadata and logs. Raw source configuration remains internal to the server.
Resource limits are operation-specific, not a general memory or CPU sandbox.

## Deliberate limits

The public contracts do not claim uniform support across every sktime backend or
task. See {doc}`../support-matrix` for operation eligibility and remaining lifecycle,
evaluation, plotting and persistence gaps. Keep new functionality in the shared
contracts rather than adding another source-specific normalization path.
