# Tool Consolidation Plan

> Reducing the MCP surface from 28 tools to ~17 well-scoped tools, without losing any functionality.

---

## Current Problem

28 tools is a large surface for an LLM to reason about. Many tools overlap, some exist only because of an implementation detail (sync vs async), and a few could be absorbed into tool descriptions or server config. This plan addresses each consolidation.

---

## 1. Merge `fit_predict` + `fit_predict_async` + `fit_predict_with_data` → single `fit_predict`

### Why

These three tools do the same thing (fit an estimator and return predictions) but differ in two orthogonal axes: **data source** (demo name vs data handle) and **execution mode** (sync vs background). The LLM shouldn't need to choose among three tools for one concept.

### Current State

| Tool | Data Source | Execution | Schema |
|------|-----------|-----------|--------|
| `fit_predict` | `dataset` (demo) or `data_handle` | sync | `estimator_handle` req, `dataset` opt, `data_handle` opt, `horizon` opt |
| `fit_predict_async` | `dataset` (demo) only | background job | `estimator_handle` req, `dataset` req, `horizon` opt |
| `fit_predict_with_data` | `data_handle` only | sync | `estimator_handle` req, `data_handle` req, `horizon` opt |

### Proposed Unified Tool

**Name:** `fit_predict`

**Schema:**
```json
{
  "estimator_handle": { "type": "string", "required": true },
  "dataset": { "type": "string", "description": "Demo dataset name (e.g. 'airline'). Ignored if data_handle is provided." },
  "data_handle": { "type": "string", "description": "Handle from load_data_source. Takes priority over dataset." },
  "horizon": { "type": "integer", "default": 12 },
  "background": { "type": "boolean", "default": false, "description": "Run as background job. Returns job_id instead of predictions." }
}
```

**Dispatch logic:**
```python
elif name == "fit_predict":
    bg = arguments.get("background", False)
    data_handle = arguments.get("data_handle")
    dataset = arguments.get("dataset", "")
    handle = arguments["estimator_handle"]
    horizon = arguments.get("horizon", 12)

    if bg:
        result = fit_predict_async_tool(handle, dataset, horizon, data_handle=data_handle)
    elif data_handle:
        result = fit_predict_with_data_tool(handle, data_handle, horizon)
    else:
        result = fit_predict_tool(handle, dataset, horizon)
    result = sanitize_for_json(result)
```

**What changes in the tool functions:**
- `fit_predict_async_tool` gains an optional `data_handle` parameter (currently it only supports demo datasets — this is a gap anyway).
- `fit_predict_tool` already accepts both `dataset` and `data_handle`. No change needed.
- Remove `fit_predict_with_data` and `fit_predict_async` from `list_tools()`.

**Migration:** Both old tool names should keep working in `call_tool()` for one release cycle (dispatch to the unified logic with a deprecation log).

---

## 2. Delete `list_data_sources` — bake into `load_data_source` description

### Why

`list_data_sources` returns a static list: `["pandas", "sql", "file", "url"]` with one-line descriptions. This never changes at runtime. There's no reason for the LLM to burn a tool call on it — the information fits perfectly in the `load_data_source` tool description itself.

### Proposed Change

Remove the `list_data_sources` tool from `list_tools()` and `call_tool()`. Update the `load_data_source` description to:

```python
Tool(
    name="load_data_source",
    description=(
        "Load data from various sources into a data handle for forecasting. "
        "Supported source types: "
        "- 'pandas': from a dict or inline data (keys: data, time_column, target_column). "
        "- 'file': from CSV, Excel (.xlsx), or Parquet files (keys: path, time_column, target_column). "
        "- 'sql': from a SQL database (keys: connection_string, query, time_column, target_column). "
        "- 'url': from a web URL pointing to a CSV/Excel/Parquet file (keys: url, time_column, target_column). "
        "GUIDELINES: "
        "1. NEVER assume a column is a time index unless the user says so. "
        "2. ALWAYS specify 'target_column' if the user mentions a specific variable. "
        "3. The first column is used as target by default — if that's a date column, specify target_column explicitly. "
        "4. For non-standard date formats, omit 'time_column' to use an integer index."
    ),
    ...
)
```

**What to delete:**
- Remove `list_data_sources` from `list_tools()` return list.
- Remove `elif name == "list_data_sources"` from `call_tool()`.
- `list_data_sources_tool()` in `data_tools.py` can stay as a library helper but doesn't need MCP exposure.

---

## 3. `format_time_series` and `auto_format_on_load` — what they do and what to do with them

### What They Actually Do

**`auto_format_on_load`** is a toggle (default: ON). When enabled, every `load_data_source` call automatically runs formatting on the loaded data before returning. The formatting does:
- Remove duplicate timestamps
- Sort by time index
- Infer frequency (daily, hourly, weekly, monthly, etc.) and set it on the index
- Fill gaps in the date range (reindex to a complete range)
- Forward-fill / backward-fill missing values

**`format_time_series`** does exactly the same thing, but manually on an already-loaded data handle. It creates a **new** data handle with the formatted data.

### Why They Exist

sktime estimators are strict about data format: they need a `pd.Series` with a `DatetimeIndex` that has an explicit `freq` attribute, no duplicates, and no NaN gaps. Real-world data almost never satisfies this out of the box. Without auto-formatting, every CSV load would fail at `fit()` time with cryptic frequency errors.

### Recommendation: Keep `format_time_series`, Remove `auto_format_on_load`

**Remove `auto_format_on_load` as a tool.** Make it a server config option instead:

1. Auto-format stays **always on** by default (current behavior).
2. If a power user wants to disable it, they set it in a config file or environment variable (see config suggestion in section 8).
3. The LLM never needs to think about this toggle — it's an implementation detail.

**Keep `format_time_series` as a tool** because it has a legitimate use case: the LLM loads data, looks at the metadata, sees issues, and decides to re-format with specific options (e.g., `fill_missing=False` to preserve NaNs for a model that handles them).

**Changes:**
- Remove `auto_format_on_load` from `list_tools()` and `call_tool()`.
- Move the toggle to `Executor.__init__` defaulting to `True`, configurable via env var `SKTIME_MCP_AUTO_FORMAT=false`.
- Fix the existing bug in `call_tool()` where `arguments["enabled"]` crashes (moot after removal, but fix it in the interim).

---

## 4. Merge `list_estimators` + `search_estimators` → single `list_estimators`

### Why

Both tools browse the estimator registry. `list_estimators` filters by task/tags with pagination. `search_estimators` does substring matching on name/docstring. An LLM has to decide which one to use, but conceptually "find me estimators matching X" is one operation.

### Current State

| Tool | Filter By | Pagination | Returns |
|------|----------|------------|---------|
| `list_estimators` | `task`, `tags`, `limit`, `offset` | Yes | `to_summary()` list |
| `search_estimators` | `query` (substring), `limit` | No offset | `to_summary()` list |

### Proposed Unified Tool

**Name:** `list_estimators`

**Schema:**
```json
{
  "task": { "type": "string", "description": "Filter by task: forecasting, classification, regression, transformation, clustering" },
  "tags": { "type": "object", "description": "Filter by capability tags, e.g. {'capability:pred_int': true}" },
  "query": { "type": "string", "description": "Search by name or description (substring, case-insensitive)" },
  "limit": { "type": "integer", "default": 50 },
  "offset": { "type": "integer", "default": 0 }
}
```

**Implementation:**
```python
def list_estimators_tool(
    task=None, tags=None, query=None, limit=50, offset=0
):
    registry = get_registry()

    if query:
        # Search by name/docstring, then apply task/tag filters
        estimators = registry.search_estimators(query)
        if task:
            estimators = [e for e in estimators if e.task == task]
        if tags:
            estimators = registry._filter_by_tags(estimators, tags)
    else:
        estimators = registry.get_all_estimators(task=task, tags=tags)

    total = len(estimators)
    page = estimators[offset:offset + limit]

    return {
        "success": True,
        "estimators": [e.to_summary() for e in page],
        "count": len(page),
        "total": total,
        "has_more": (offset + limit) < total,
    }
```

**What changes:**
- `search_estimators` removed from `list_tools()` and `call_tool()`.
- `query` added as optional param to unified `list_estimators`.
- The `search_estimators_tool` function stays in `describe_estimator.py` as a library helper.
- When `query` is provided alongside `task`/`tags`, all filters are combined (search first, then filter — giving the LLM the most powerful single call).

---

## 5. `get_available_tags` — what to do with it

### What It Does

Returns a catalog of ~100+ tag names with descriptions, value types, and which estimator types they apply to. It's a prerequisite for using the `tags` filter in `list_estimators`.

### Options

| Option | Pros | Cons |
|--------|------|------|
| **A. Keep as tool** | LLM can query on demand; always up-to-date | Burns a tool call; large response (~100 tags) |
| **B. Move to prompt/description** | Zero tool calls | Too long for a tool description; tags change with sktime versions |
| **C. Merge into `describe_estimator`** | Natural context | Doesn't help when filtering `list_estimators` |
| **D. Keep as tool + add most-common tags to `list_estimators` description** | Best of both worlds | Slight duplication |

### Recommendation: Option D — Keep as tool, but enrich the `list_estimators` description

Add the 10 most commonly useful tags directly into the `list_estimators` description so the LLM rarely needs to call `get_available_tags` separately:

```python
Tool(
    name="list_estimators",
    description=(
        "Discover sktime estimators by task, capability tags, or name search. "
        "Common tags you can filter by: "
        "'capability:pred_int' (bool) - prediction intervals, "
        "'capability:multivariate' (bool) - multivariate support, "
        "'handles-missing-data' (bool) - NaN handling, "
        "'scitype:y' (str) - target type ('univariate'/'multivariate'/'both'), "
        "'requires-fh-in-fit' (bool) - needs forecast horizon at fit time. "
        "Use get_available_tags for the full catalog."
    ),
    ...
)
```

This way `get_available_tags` becomes a power-user fallback, not a required first step.

---

## 6. Merge `instantiate_estimator` + `load_model` → single `instantiate_estimator`

### Why

Both produce the same output: an `est_*` handle pointing to a usable estimator instance. The only difference is the source — one builds from class name + params, the other loads from disk. From the LLM's perspective, "give me a model handle" is one concept.

### Current State

| Tool | Input | Output | Fitted? |
|------|-------|--------|---------|
| `instantiate_estimator` | `estimator` (name), `params` (dict) | `est_*` handle | No |
| `load_model` | `path` (local dir or MLflow URI) | `est_*` handle | Yes (auto-marked) |

### Proposed Unified Tool

**Name:** `instantiate_estimator`

**Schema:**
```json
{
  "estimator": { "type": "string", "description": "Estimator class name (e.g. 'ARIMA') OR omit if loading from path." },
  "params": { "type": "object", "description": "Hyperparameters (only for new instantiation)." },
  "path": { "type": "string", "description": "Load a saved model from this path/URI instead of creating new. The handle is auto-marked as fitted." }
}
```

**Validation rules:**
- Exactly one of `estimator` or `path` must be provided.
- If `path` is given, `estimator` and `params` are ignored.
- If `estimator` is given, `path` is ignored.

**Implementation:**
```python
def instantiate_estimator_tool(
    estimator: str = None,
    params: dict = None,
    path: str = None,
) -> dict:
    if path:
        return load_model_tool(path)
    if not estimator:
        return {"success": False, "error": "Provide 'estimator' name or 'path' to load a saved model."}
    # existing instantiate logic...
```

**What changes:**
- Remove `load_model` from `list_tools()`.
- Update `call_tool()` to pass `path` to the unified function.
- `load_model_tool` stays as an internal function.

---

## 7. Merge `load_data_source` + `load_data_source_async` → single `load_data_source`

### Why

Same logic as the fit_predict merge. The async variant is identical except it returns a `job_id` instead of blocking. A `background` flag handles this cleanly.

### Proposed Unified Tool

**Name:** `load_data_source`

**Schema:** Add one field to existing:
```json
{
  "config": { "type": "object", "required": true },
  "background": { "type": "boolean", "default": false, "description": "Run in background. Returns job_id instead of data_handle." }
}
```

**Implementation:**
```python
elif name == "load_data_source":
    if arguments.get("background", False):
        result = load_data_source_async_tool(arguments["config"])
    else:
        result = load_data_source_tool(arguments["config"])
```

**What changes:**
- Remove `load_data_source_async` from `list_tools()`.
- The underlying async tool function stays for internal use.

---

## 8. `cleanup_old_jobs` → server config, not a tool

### Why

Job cleanup is a housekeeping task, not something an LLM should be making decisions about. It's an operational concern (like log rotation) that belongs in server configuration.

### Proposed Change

Remove `cleanup_old_jobs` from `list_tools()` and `call_tool()`. Instead:

**Option A: Environment variable**
```bash
export SKTIME_MCP_JOB_MAX_AGE_HOURS=24
```

**Option B: Config file** (`sktime_mcp_config.yaml` in project root or `~/.config/sktime-mcp/config.yaml`)
```yaml
server:
  auto_format_on_load: true     # also absorbs the auto_format_on_load toggle
  job_max_age_hours: 24
  max_estimator_handles: 100
  max_data_handles: 100         # currently missing — good to add
  log_level: INFO
```

**Option C (Simplest): Automatic cleanup on a timer**
```python
async def run_server():
    # Start periodic cleanup
    async def _periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # every hour
            get_job_manager().cleanup_old_jobs(max_age_hours=24)

    asyncio.create_task(_periodic_cleanup())

    async with stdio_server() as (read_stream, write_stream):
        await server.run(...)
```

### Recommendation

Use **Option C** (automatic timer) as the default, with **Option A** (env var) to configure the age threshold. This is zero-config for most users and zero-tool-calls for LLMs.

```python
import os

JOB_MAX_AGE_HOURS = int(os.environ.get("SKTIME_MCP_JOB_MAX_AGE_HOURS", "24"))
AUTO_FORMAT_ON_LOAD = os.environ.get("SKTIME_MCP_AUTO_FORMAT", "true").lower() == "true"
```

---

## 9. `cancel_job` vs `delete_job` — differences and what to do

### Current Behavior

| Action | `cancel_job` | `delete_job` |
|--------|-------------|-------------|
| **What it does** | Sets status to `CANCELLED` | Removes the job record entirely |
| **When it works** | Only on `PENDING` or `RUNNING` jobs | On any job regardless of status |
| **Does it stop work?** | **No** — the coroutine keeps running | N/A — just removes the record |
| **Job still visible after?** | Yes — shows as `status: "cancelled"` | No — gone forever |
| **Use case** | "Stop this training" (aspirational) | "Clean up this finished job" |

### The Problem

1. `cancel_job` is misleading — it doesn't actually stop anything. The work continues to completion (or failure), and if it completes, the job may get updated to `COMPLETED` after being marked `CANCELLED` (race condition).
2. LLMs have no clear guidance on when to use cancel vs delete.
3. With automatic cleanup (section 8), `delete_job` becomes less necessary.

### Recommendation: Merge into single `cancel_job` with optional delete

```python
def cancel_job_tool(job_id: str, delete: bool = False) -> dict:
    job_manager = get_job_manager()

    job = job_manager.get_job(job_id)
    if job is None:
        return {"success": False, "error": f"Job '{job_id}' not found"}

    if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        job_manager.cancel_job(job_id)
        msg = f"Job '{job_id}' cancelled"
    elif delete:
        job_manager.delete_job(job_id)
        msg = f"Job '{job_id}' deleted"
    else:
        return {
            "success": False,
            "error": f"Job is already '{job.status.value}'. Use delete=true to remove it.",
        }

    if delete and job_manager.get_job(job_id):
        job_manager.delete_job(job_id)
        msg += " and removed"

    return {"success": True, "message": msg}
```

**Schema:**
```json
{
  "job_id": { "type": "string", "required": true },
  "delete": { "type": "boolean", "default": false, "description": "Also remove the job record after cancelling." }
}
```

Remove `delete_job` from `list_tools()`. The LLM says "cancel this job" and optionally "and delete it".

With automatic cleanup (section 8), manual deletion is rarely needed anyway.

---

## Summary: Before and After

### Before (28 tools)

```
Discovery:      list_estimators, search_estimators, describe_estimator, get_available_tags
Instantiation:  instantiate_estimator, instantiate_pipeline, list_handles, release_handle, load_model
Execution:      fit_predict, fit_predict_async, fit_predict_with_data, evaluate_estimator
Data:           list_available_data, load_data_source, load_data_source_async, list_data_sources,
                release_data_handle, format_time_series, auto_format_on_load
Export:         export_code
Persistence:    save_model
Validation:     validate_pipeline
Jobs:           check_job_status, list_jobs, cancel_job, delete_job, cleanup_old_jobs
```

### After (17 tools)

```
Discovery:      list_estimators (+ query), describe_estimator, get_available_tags
Instantiation:  instantiate_estimator (+ path), instantiate_pipeline, list_handles, release_handle
Execution:      fit_predict (+ background + data_handle), evaluate_estimator
Data:           list_available_data, load_data_source (+ background),
                release_data_handle, format_time_series
Export:         export_code
Persistence:    save_model
Validation:     validate_pipeline
Jobs:           check_job_status, list_jobs, cancel_job (+ delete)
```

### Tools Removed (11)

| Removed Tool | Where It Went |
|-------------|---------------|
| `search_estimators` | Merged into `list_estimators` via `query` param |
| `load_model` | Merged into `instantiate_estimator` via `path` param |
| `fit_predict_async` | Merged into `fit_predict` via `background` param |
| `fit_predict_with_data` | Merged into `fit_predict` via `data_handle` param |
| `load_data_source_async` | Merged into `load_data_source` via `background` param |
| `list_data_sources` | Moved to `load_data_source` tool description text |
| `auto_format_on_load` | Moved to server config (env var / auto-enabled) |
| `cleanup_old_jobs` | Moved to automatic periodic timer + env var config |
| `delete_job` | Merged into `cancel_job` via `delete` param |

### Net Effect

- **28 → 17 tools** (39% reduction)
- **Zero functionality lost** — every capability is still accessible
- **Cleaner LLM experience** — fewer choices, clearer intent per tool
- **`background` pattern** — consistent across `fit_predict` and `load_data_source`
- **Operational concerns** (cleanup, auto-format) moved out of the LLM's decision space

---

## Implementation Priority

| Order | Change | Effort | Risk |
|-------|--------|--------|------|
| 1 | Delete `list_data_sources` (bake into prompt) | Tiny | None |
| 2 | Remove `auto_format_on_load` tool → env var | Small | None |
| 3 | Remove `cleanup_old_jobs` tool → periodic timer | Small | None |
| 4 | Merge `cancel_job` + `delete_job` | Small | Low |
| 5 | Merge `list_estimators` + `search_estimators` | Small | Low |
| 6 | Merge `instantiate_estimator` + `load_model` | Small | Low |
| 7 | Merge `load_data_source` + `load_data_source_async` | Medium | Low |
| 8 | Merge `fit_predict` + `fit_predict_async` + `fit_predict_with_data` | Medium | Medium — async needs `data_handle` support added |
