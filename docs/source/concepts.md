# Core concepts

## Containers, meanings and roles

A **scitype** describes the data's meaning: Series, Panel, Hierarchical, Table,
Alignment or Proba. An **mtype** describes its physical representation. For example,
`numpy3D` is a Panel with axes `(instances, variables, time)`; `numpy2D` is a Table.
A plain 2D NumPy array can also represent a Series as `np.ndarray`, so declare the
meaning when it is ambiguous.

A data handle contains primary data and optionally an exogenous component. By
default, all variables are primary data. `target_column` or `target_columns` makes
the target selection explicit; other columns are features unless an explicit
`exog_columns` mapping accounts for them. No mapped column is silently discarded.

Use `role="X"` for a standalone feature container, such as a classification panel.
Pass its handle as `X_handle`. Load labels separately as a Table and pass their
handle as `y_handle`. A target-only forecasting handle cannot silently become X.

## Loading is not cleaning

`load_data` reads and describes data. It preserves values, row order, duplicates,
missing values and the existing container where possible. An explicit index or
column mapping is a requested representation change, not an inferred cleaning step.

A successful load means the source was read and registered. Check
`validation.valid` to see whether the current data satisfies its sktime contract.
Repairable pandas inputs can be retained with validation issues so that
`prepare_data` can sort or remove duplicates explicitly. This does not guarantee
that any particular estimator supports the data's properties.

`prepare_data` creates new handles. Its three actions are `format`, `convert` and
`split`. It retains the source and rejects operations whose semantics are undefined
for the selected data. See {doc}`data-guide`.

## Handles and lifecycle

| Handle | Holds | Lifetime |
| --- | --- | --- |
| `data_…` | Loaded, prepared or predicted data | Until release, capacity eviction or server restart |
| `est_…` | An estimator or pipeline | Until release, eviction or server restart |
| Job ID | Background task status and result | Until deletion or completed-job cleanup |

Handles are process-wide, not isolated per client. Releasing a handle frees the
server's reference; it does not delete saved files. Preparation reserves capacity
for its source and outputs before committing a result. A split requires at least
three data-handle slots.

## Background jobs and cancellation

`load_data`, `fit`, `predict` and `evaluate` accept `run_async=true`. They return a
job ID immediately; retrieve progress and results with `check_job_status`.
There are no completion push notifications.

Cancelling a fit stops the awaiting coroutine without joining its worker on the
server loop. An already-running sktime call can still finish and mutate its
estimator. The job remains cancelled. Use a fresh estimator handle afterward;
cancellation is not a rollback of fitted state.

## Execution and reproducibility

Registry descriptions come from the installed sktime version. They report
capabilities and dependencies, not a guarantee of accuracy or workflow success.

`instantiate` delegates a trusted craft expression to sktime. `run_command`
executes a shell command as the server user. Neither interface is a sandbox.
HTTP/SSE requires a configured bearer token unless explicitly started in insecure
local-test mode; stdio uses local process permissions.

`save_data` stores data, `save_model` stores fitted models through MLflow, and
`export_code` reconstructs a specification or model loader. Code export does not
capture custom preprocessing, the conversation, or your complete environment.
Record versions, preparation policies and source data separately.
