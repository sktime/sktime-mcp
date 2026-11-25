# Data guide

The data interface has four tools: **load, inspect, prepare and save**. Loading and
preparation return a compact inspection alongside the result, so a separate
inspection call is needed only when you want to revisit a handle.

## 1. Load without assumptions

| Source `type` | Required source keys | Notes |
| --- | --- | --- |
| `memory` | `data` | JSON records/columns or a native Python container; `pandas` remains a source-type alias |
| `file` | `path` | CSV, Excel, Parquet, JSON or non-pickle NPY |
| `url` | `url` | HTTP(S); the downloaded file uses the file reader |
| `sql` | Connection settings and `query` or `table` | SQLAlchemy; bound `query_params` are supported |

All sources share the same mapping, validation and registration path. Async loads
produce the same data and metadata as foreground loads.

```json
{
  "name": "load_data",
  "arguments": {
    "config": {
      "type": "memory",
      "data": {
        "date": [
          "2024-01-01",
          "2024-01-02",
          "2024-01-03"
        ],
        "sales": [
          10,
          null,
          14
        ],
        "promo": [
          0,
          1,
          0
        ]
      },
      "time_column": "date",
      "target_column": "sales",
      "exog_columns": [
        "promo"
      ]
    }
  }
}
```

The missing sales observation remains missing. The date mapping is explicit.
Integers used as time values remain integers; they are not parsed as epoch dates.

### Column and index policies

- Without a target mapping, **all variables stay in the primary container**.
- `target_column` selects one target; `target_columns` preserves multivariate targets.
- When only targets are specified, remaining columns become exogenous variables.
- Explicit target and feature mappings must account for every data column and must not overlap.
- `instance_columns` identifies panel or hierarchy levels; `time_column` identifies the final time level.
- Existing pandas indexes are retained if no mapping is requested. No date column is guessed from its name.
- `frequency` validates an explicit grid. It does not insert missing timestamps or fill values.
- `parse_dates=false` retains the mapped column's dtype. Unsupported time indexes appear in validation.

### Representations and scitypes

| Data | Declare / use |
| --- | --- |
| One or several time-series variables | Series; `pd.Series`, `pd.DataFrame` or `np.ndarray` |
| Equal-length series collections | Panel; `numpy3D` with axes `(instance, variable, time)` |
| Unequal-length collections | Panel; `df-list`, `nested_univ` or `pd-multiindex` |
| Nested groups of series | Hierarchical; `pd_multiindex_hier` |
| Labels or ordinary feature rows | Table; `numpy1D`, `numpy2D`, pandas Table mtypes or `list_of_dict` |
| Alignments or prediction distributions | Explicit Alignment/Proba mtype; operations are restricted to meaningful ones |

Native containers are accepted through the Python API. JSON cannot carry a pandas
object directly: use records plus explicit index mappings, or nested numeric arrays
with `mtype="numpy3D"`. Use `mtype="df-list"` for a JSON list of per-instance tables;
`nested_univ` can decode lists in cells into nested Series. Optional sktime mtypes
require their dependencies and converter support; the core tests do not certify
all optional or lazy backends.

## 2. Read the inspection

A load or preparation result includes:

- `data_handle`, `metadata`, `validation` and `inspection`;
- scientific/machine type, shape, columns and dtypes;
- index levels, frequency and cutoff where meaningful;
- bounded previews and summary statistics;
- separate exogenous inspection when features are attached.

Previews preserve duplicate index entries as rows. They are bounded to five rows
and twenty columns; nested arrays are bounded on each displayed axis. Missing
counts can be unavailable for containers where a generic count would be misleading.

`inspect_data(data_handle=...)` repeats this read-only inspection. Review the
validation report before fitting. An empty container or categorical data is not
silently converted into a numeric forecasting series.

## 3. Prepare explicitly

### Format

No repair happens unless its option is requested:

| Option | Default | Choices / meaning |
| --- | --- | --- |
| `sort_index` | `false` | Stable index sort |
| `duplicates` | `keep` | `keep`, `first`, `last`, `drop` (all repeated keys), `error` |
| `missing` | `keep` | `keep`, `ffill`, `bfill`, `interpolate` (numeric interior gaps), `constant` |
| `fill_value` | — | Required for constant filling |
| `frequency` | — | Explicit pandas frequency |
| `regularize` | `false` | Insert index gaps on the explicit frequency grid |
| `index_type` | `keep` | Explicit `period` conversion for datetime/period indexes |

```json
{
  "name": "prepare_data",
  "arguments": {
    "data_handle": "<loaded>",
    "action": "format",
    "sort_index": true,
    "duplicates": "first",
    "missing": "ffill",
    "frequency": "D",
    "regularize": true
  }
}
```

Formatting applies independently to each panel/hierarchy series, including attached
features. Values never fill across instance boundaries. The response reports
actual changes separately for primary data and X. The source handle remains intact.

Forward filling leaves leading missing values missing. Backward filling and
interpolation can use future observations: split before preparation when evaluating,
and fit learned imputers inside a sktime pipeline. The formatter does not learn a
training-time transformation or apply it to test data automatically.

Regularization checks alignment and the configured row budget before allocating a
full index. It never guesses a daily, weekly or monthly frequency from irregular
data. Table data has no temporal regularization; Alignment and Proba formatting
are not exposed through this action.

Frequencies follow the index type: a monthly PeriodIndex uses `M`, while datetime
month-end grids use `ME` on current pandas. Daily datetime grids retain local wall
time across daylight-saving changes. Period conversion rejects timezone loss.

### Convert

```json
{
  "name": "prepare_data",
  "arguments": {
    "data_handle": "<loaded>",
    "action": "convert",
    "to_mtype": "np.ndarray",
    "component": "y"
  }
}
```

Conversion delegates to sktime and stays within the current scitype. `component`
selects primary data (`y`), attached features (`X`), or `both`. Unselected components
are preserved. An X-only handle still calls its primary container `y` for this
option; its `role="X"` remains unchanged.

Some conversions cannot represent unequal lengths, multiple variables, or labels.
They fail explicitly or report the representation change; an array can omit a
pandas index and column names. Keep the original handle when those labels matter.
This action does not reinterpret a Series as a Panel or Table.

### Split

```json
{
  "name": "prepare_data",
  "arguments": {
    "data_handle": "<loaded>",
    "action": "split",
    "axis": "time",
    "fh": 6
  }
}
```

- `axis="time"` splits each Series/Panel/Hierarchical series temporally using sktime.
- `fh=6` holds out the final six observations per series; `[1, 6]` holds out the final six too.
- Alternatively, `test_size=0.2` holds out a fraction per series, rounded up.
- `axis="instance"` with `test_size` holds out complete panel/hierarchy series or Table rows, in their existing order. It is not randomized or stratified.
- Every group must retain training data. Attached X follows the same split.

The result contains `train_handle`, `test_handle`, counts and inspections for both.
Split paired classification features and labels with an explicitly coordinated
instance partition; this tool does not infer a relationship between two independent
handles. It also does not equate a temporal split with a classification split.

## 4. Save and reload

```json
{
  "name": "save_data",
  "arguments": {
    "data_handle": "<prepared>",
    "path": "/tmp/prepared.parquet",
    "format": "parquet"
  }
}
```

CSV, JSON and Parquet exports use a canonical table, with explicit index columns
for each hierarchy level. Targets and attached features are aligned and combined.
The response includes a `load_config` for reloading the file. Exact dtype/mtype
round trips can require reader options; these formats are not universal object
serialization. MultiIndex columns require an explicit flat representation.

For an ndarray without attached X, use `format="npy"` to retain its dimensions and
dtype. Object arrays requiring pickle are rejected. The output format comes from
`format`, not the filename extension. Existing files require `overwrite=true`.
Serialization completes before the destination is published. A failed write leaves
an existing file intact and removes its temporary file.

## Resource limits

| Setting | Default | Applies to |
| --- | --- | --- |
| `SKTIME_MCP_URL_TIMEOUT_SECONDS` | `30` | Total HTTP(S) download deadline, including redirects and body reads |
| `SKTIME_MCP_URL_MAX_BYTES` | `52428800` | Downloaded/decoded body limit: 50 MiB |
| `SKTIME_MCP_MAX_PREPARED_ROWS` | `1000000` | Explicit frequency expansion |
| `SKTIME_MCP_MAX_DATA_HANDLES` | `50` | Retained data handles |
| `SKTIME_MCP_MAX_HORIZON` | `10000` | Forecast and holdout horizon |

URL parsing happens after the download deadline. Local files, SQL queries, model
memory and arbitrary native methods do not acquire a universal resource budget from
these settings. Cancellation does not force-stop an already-running parsing thread;
the worker retains its temporary files until it finishes.
