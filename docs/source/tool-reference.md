# MCP Tool Reference

Complete reference for every tool the `sktime-mcp` server exposes to an MCP client.
The server currently registers **26 tools**.

You normally do not call these by hand — the assistant selects them from your
natural-language request. This page exists so you can see exactly what the
assistant has available, what each argument means, and what comes back.

For the Python API of the package itself (classes, modules), see the
{doc}`api` page instead.

---

## Conventions

### Handles

Most tools return or accept a **handle** — a string ID naming an object the
server holds in memory.

| Prefix | Created by | Refers to |
| :--- | :--- | :--- |
| `est_…` | `instantiate`, `load_model` | An estimator or pipeline |
| `data_…` | `load_data_source`, `split_data`, `transform_data` | A time series (target `y` plus any exogenous `X`) |

Handles live in the server process and are lost when it restarts. Use
`save_data`, `save_model`, or `export_code` to persist anything you need to keep,
and `release_handle` / `release_data_handle` to free memory.

(run-async)=
### Asynchronous execution: the `run_async` flag

Long-running work is made asynchronous by passing **`run_async: true`** to the
tool itself. There are no separate `*_async` tools.

Exactly four tools accept the flag:

- `fit`
- `predict`
- `evaluate`
- `load_data_source`

Behaviour:

| `run_async` | Returns | Use when |
| :--- | :--- | :--- |
| `false` *(default)* | The real result — a handle, predictions, scores | The call is quick |
| `true` | `{"success": true, "job_id": "..."}` immediately | Training or loading is slow |

When you pass `run_async: true`, the work is scheduled on the server's event loop
and you get a `job_id` back straight away. Track it with the job tools:

```text
fit(run_async=true)  ->  job_id
        |
        +-- check_job_status(job_id)   poll status and progress
        +-- list_jobs()                see everything in flight
        +-- cancel_job(job_id)         stop it
```

```{note}
There is **no push notification** when a job finishes. The client must poll
`check_job_status` to find out. An assistant will typically do this for you when
you ask "is it done yet?", but nothing arrives unprompted.
```

Results are retrieved through `check_job_status` once `status` is `completed`.

---

## Discovery

### `query_registry`

Discover sktime estimators, metrics, or capability tags.

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `task` | string | — | — | Filter by scitype: `forecaster`, `classifier`, `regressor`, `transformer`, `clusterer`, `detector`, `splitter`, `metric`, `param_est`, `aligner`, `network`. Set to `tag`/`tags` to list available tags instead. |
| `tags` | object | — | — | Filter by capability tags, e.g. `{"capability:pred_int": true}`. Ignored when `task="tag"`. |
| `query` | string | — | — | Case-insensitive substring search over name and description. Combines with `task` and `tags`. |
| `limit` | integer | — | `50` | Maximum results. |
| `offset` | integer | — | `0` | Skip this many results (pagination). |

Commonly useful tags: `capability:pred_int` (prediction intervals),
`capability:multivariate`, `handles-missing-data`, `scitype:y`.

### `describe_component`

Detailed information about any class in the sktime ecosystem — estimators,
splitters, metrics, transformers.

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `name` | string | ✅ | Class name, e.g. `ARIMA`, `SlidingWindowSplitter`, `MeanAbsolutePercentageError`. |

### `list_available_data`

Lists demo datasets and active user-loaded data handles in one response.

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `is_demo` | boolean | — | `true` = demos only, `false` = live handles only, omit = both. |

---

## Instantiation and handles

### `instantiate`

Create an estimator or pipeline from an sktime **craft** specification string.
Composition is expressed directly in the spec — sktime validates it.

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `spec` | string | ✅ | Craft spec, e.g. `ARIMA(order=(1, 1, 1))` or `Detrender() * ARIMA()`. |

Returns an `est_…` handle.

### `list_handles`

Lists all active estimator handles. Takes no arguments.

### `release_handle`

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `handle` | string | ✅ | Estimator handle to release. |

---

## Execution

### `fit`

Fit an estimator. Supply `X`/`y` as either live handles or demo dataset names,
depending on the estimator's scitype.

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `estimator_handle` | string | ✅ | — | Handle from `instantiate`. |
| `X_handle` | string | — | — | Data handle for `X` (features, panel). |
| `y_handle` | string | — | — | Data handle for `y` (target, labels). |
| `X_dataset` | string | — | — | Demo dataset name for `X`. |
| `y_dataset` | string | — | — | Demo dataset name for `y`. |
| `fh` | int or list | — | — | Forecast horizon passed through to `fit`. |
| `run_async` | boolean | — | `false` | Run in the background, return a `job_id`. See [`run_async`](#run-async). |

### `predict`

Generate predictions from a fitted estimator.

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `estimator_handle` | string | ✅ | — | Handle of a **fitted** estimator. |
| `horizon` | integer | — | `12` | Forecast horizon. |
| `mode` | string | — | `predict` | One of `predict`, `predict_interval`, `predict_quantiles`, `predict_proba`, `predict_var`. |
| `coverage` | float or list | — | `0.9` | Coverage level(s) — used by `predict_interval`. |
| `alpha` | float or list | — | — | Quantile level(s) — used by `predict_quantiles`. |
| `X_handle` | string | — | — | Data handle for `X`. |
| `y_handle` | string | — | — | Data handle for `y` (needed by detectors/annotators). |
| `X_dataset` | string | — | — | Demo dataset name for `X`. |
| `y_dataset` | string | — | — | Demo dataset name for `y`. |
| `run_async` | boolean | — | `false` | Run in the background, return a `job_id`. |

Interval and quantile forecasts are **modes of this tool**, not separate tools.

### `update`

Update a fitted estimator with new data.

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `estimator_handle` | string | ✅ | Handle of a fitted estimator. |
| `X_handle` / `y_handle` | string | — | Data handles for the new data. |
| `X_dataset` / `y_dataset` | string | — | Demo dataset names for the new data. |

### `get_fitted_params`

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `estimator_handle` | string | ✅ | Handle of a fitted estimator. |

### `call_method`

Escape hatch: call any native method on an instantiated component. Use this for
scitypes that do not fit the `fit`/`predict` shape — splitters, metrics, aligners.

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `handle_id` | string | ✅ | Handle of the instantiated component. |
| `method_name` | string | ✅ | Method to call, e.g. `split`, `get_alignment`, `__call__`. |
| `kwargs` | object | — | Keyword arguments. Keys suffixed `_dataset` or `_data_handle` inject server-side data, e.g. `{"y_dataset": "airline"}`. |

### `evaluate`

Cross-validate an estimator.

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `estimator_handle` | string | ✅ | — | Handle from `instantiate`. |
| `y` | string | ✅ | — | Target series: a data handle ID **or** a demo dataset name. |
| `X` | string | — | — | Exogenous series: data handle ID or demo dataset name. |
| `cv_folds` | integer | — | `3` | Number of folds. Ignored when `initial_window` is set. |
| `metric` | string | — | — | Metric name, e.g. `MeanAbsolutePercentageError`. |
| `initial_window` | integer | — | — | Initial training window for expanding-window CV. |
| `run_async` | boolean | — | `false` | Run in the background, return a `job_id`. |

---

## Data

### `load_data_source`

Load data into a `data_…` handle. The `config` object must carry a `type` key.

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `config` | object | ✅ | — | Source configuration; must include `type`. |
| `run_async` | boolean | — | `false` | Load in the background, return a `job_id`. |

Supported `config.type` values:

| `type` | Additional keys |
| :--- | :--- |
| `pandas` | `data`, `time_column`, `target_column` |
| `file` | `path` (CSV / `.xlsx` / Parquet), `time_column`, `target_column` |
| `sql` | connection and query keys, `time_column`, `target_column` |
| `url` | `path`/URL, `time_column`, `target_column` |

```json
{
  "type": "file",
  "path": "/data/sales.csv",
  "time_column": "date",
  "target_column": "revenue"
}
```

### `inspect_data`

Rich metadata for a loaded handle: mtype, scitype, shape, column names, dtypes,
index level names, inferred frequency, cutoff, missing-value count, a 5-row head
preview, and per-column summary statistics.

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `data_handle` | string | ✅ | Handle to inspect. |

### `split_data`

Split a series into temporal train/test handles. Provide **exactly one** of
`test_size` or `fh`.

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `data_handle` | string | ✅ | Handle to split. |
| `test_size` | number | — | Hold-out fraction, exclusive range `(0.0, 1.0)`. Mutually exclusive with `fh`. |
| `fh` | int or list | — | Integer: hold out that many final steps. List: holds out `max(fh)` steps — `fh=[1,5,10]` reserves 10. Mutually exclusive with `test_size`. |

Returns `train_handle` and `test_handle`.

### `transform_data`

Returns a **new** handle; the input handle is unchanged.

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `data_handle` | string | ✅ | — | Handle to transform. |
| `action` | string | — | `format` | `format` or `convert`. |
| `auto_infer_freq` | boolean | — | `true` | *(format)* Infer and set `DatetimeIndex` frequency. |
| `fill_missing` | boolean | — | `true` | *(format)* Forward/backward-fill missing values. |
| `remove_duplicates` | boolean | — | `true` | *(format)* Drop duplicate timestamps, keeping the first. |
| `to_mtype` | string | — | — | *(convert, required)* Target mtype, e.g. `pd.DataFrame`, `pd.Series`, `np.ndarray`. |

`action="format"` also fills index gaps and reports what it did in `changes_applied`.

### `save_data`

Writes the target `y` and any exogenous `X` behind a handle to one file,
creating parent directories as needed.

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `data_handle` | string | ✅ | — | Handle to export. |
| `path` | string | ✅ | — | Destination path. |
| `format` | string | — | `csv` | `csv`, `parquet`, or `json`. |

```{warning}
The format comes from the `format` argument, **not** the file extension.
Writing to `out.parquet` with the default `format` produces a CSV.
```

### `release_data_handle`

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `data_handle` | string | ✅ | Data handle to release. |

### `plot_series`

Plot one or more series. Saves a file when `path` is given, otherwise returns
the image as a base64 string.

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `data_handles` | array of string | ✅ | — | Handles to plot, e.g. train, test, forecast. |
| `labels` | array of string | — | — | Legend label per handle. |
| `title` | string | — | — | Plot title. |
| `path` | string | — | — | Save location, e.g. `/tmp/plot.png`. Omit to get base64 back. |
| `figsize` | array | — | `[12, 6]` | `[width, height]` in inches. |
| `dpi` | integer | — | `150` | Resolution. |
| `markers` | string or array | — | — | Marker style(s), e.g. `"o"` or `[".", "x"]`. |
| `x_label` / `y_label` | string | — | — | Axis labels. |
| `image_format` | string | — | `png` | `png`, `svg`, or `webp`. |

---

## Persistence and code generation

### `export_code`

Emit standalone, runnable Python that reconstructs the estimator.

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `handle` | string | ✅ | — | Estimator/pipeline handle. |
| `var_name` | string | — | `model` | Variable name in the generated code. |
| `include_fit_example` | boolean | — | `false` | Append a fit/predict example. |
| `dataset` | string | — | `airline` | Dataset used in the example. |

### `save_model`

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `estimator_handle` | string | ✅ | Estimator to save. |
| `path` | string | ✅ | Local directory or URI. |
| `mlflow_params` | object | — | Extra parameters for `sktime.utils.mlflow_sktime.save_model`. |

### `load_model`

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `path` | string | ✅ | Path to the saved model directory. |

Registers the loaded model and returns a new `est_…` handle.

---

## Background jobs

See [the `run_async` flag](#run-async) for how jobs are created.

### `check_job_status`

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `job_id` | string | ✅ | Job to check. |

Returns status, progress, and — once `completed` — the result.

### `list_jobs`

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `status` | string | — | — | Filter: `pending`, `running`, `completed`, `failed`, `cancelled`. |
| `limit` | integer | — | `20` | Page size. |
| `offset` | integer | — | `0` | Records to skip. |

Returns the page plus `total` and `has_more` for pagination.

### `cancel_job`

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `job_id` | string | ✅ | — | Job to cancel. |
| `delete` | boolean | — | `false` | Also remove the job record — useful for tidying completed or failed jobs. |

Cancellation is cooperative: the server holds a reference to the running task and
cancels it, but work already inside a blocking sktime call finishes its current
step before stopping.

---

(run-command)=
## System

### `run_command`

```{danger}
**This tool executes arbitrary shell commands** with the full privileges of the
user account running the server. It is not sandboxed, and it is not restricted to
the sktime registry.

A client that is allowed to call `run_command` can read, modify, or delete any
file the server process can reach, and can make network requests.
```

It exists so that an assistant can repair its own environment mid-session —
typically installing an optional dependency that an estimator turned out to need:

```text
pip install mlflow
```

| Argument | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `command` | string | ✅ | The shell command to run. |

**Treat this as opt-in.** Most MCP clients prompt before each tool call and let
you allow or deny individual tools — leave `run_command` on ask-every-time, or
disable it outright, unless you specifically want the assistant installing
packages and inspecting the filesystem on your behalf. If you need a hard
guarantee, run the server inside a container (see the Docker option in the
{doc}`user-guide`) so the blast radius is the container rather than your machine.

See {doc}`developer/architecture` for the full trust boundary.
