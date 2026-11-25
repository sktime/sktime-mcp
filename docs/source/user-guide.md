# Worked workflows

These examples use the current source interface and lightweight estimators. Replace
placeholder handles with the IDs returned by earlier calls. For exact arguments,
see {doc}`tool-reference`; for representation details, see {doc}`data-guide`.

## Discover before constructing

```json
{
  "name": "query_registry",
  "arguments": {
    "task": "forecaster",
    "tags": {
      "capability:pred_int": true
    },
    "limit": 5
  }
}
```

```json
{
  "name": "describe_component",
  "arguments": {
    "name": "NaiveForecaster"
  }
}
```

The installed registry supplies tags and parameters. Read dependency availability
before choosing an estimator. The demo catalogue is also discovered dynamically:
call `list_available_data(is_demo=true)` rather than relying on an old dataset count.

## Forecast your own monthly series

The following Python example exercises the same runtime used by MCP tools. The
input retains its DatetimeIndex; the forecasting runtime can use a private
PeriodIndex compatibility view without changing the stored source.

```python
import pandas as pd
from sktime_mcp.tools import (
    load_data_tool,
    prepare_data_tool,
    instantiate_tool,
    fit_tool,
    predict_tool,
)

source = pd.DataFrame(
    {"sales": [100.0 + (i % 12) * 5 for i in range(36)]},
    index=pd.date_range("2021-01-01", periods=36, freq="MS"),
)
loaded = load_data_tool({"type": "memory", "data": source, "target_column": "sales"})
assert loaded["success"] and loaded["validation"]["valid"]
partition = prepare_data_tool(loaded["data_handle"], "split", fh=6)
assert partition["success"]
model = instantiate_tool(spec="NaiveForecaster(strategy='last', sp=12)")
assert fit_tool(model["handle"], y_handle=partition["train_handle"])["success"]
forecast = predict_tool(model["handle"], horizon=6)
assert forecast["success"]
```

In an MCP client, pass `train_handle` as `fit.y_handle`, then pass the returned
`prediction_handle` to `plot_series` and `save_data`. Keep the test handle for
scoring. `evaluate` uses expanding-window cross-validation on the selected Series;
it is separate from the single holdout created above.

## Repair a messy series deliberately

First inspect the load response. If duplicate timestamps are genuine repeated
measurements, dropping them may be inappropriate; choose the policy for your data.
This example deliberately keeps the first duplicate and forward-fills after sorting:

```json
{
  "name": "prepare_data",
  "arguments": {
    "data_handle": "<raw-series>",
    "action": "format",
    "sort_index": true,
    "duplicates": "first",
    "missing": "ffill"
  }
}
```

The source remains unchanged. Compare `changes_applied`, the new validation report,
and the missing counts before fitting. No frequency is guessed, and leading
missing values remain missing. For evaluation, split first; avoid preprocessing
that draws on held-out observations.

## Load a panel or hierarchy

```json
{
  "name": "load_data",
  "arguments": {
    "config": {
      "type": "memory",
      "data": {
        "region": [
          "north",
          "north",
          "south",
          "south"
        ],
        "store": [
          "A",
          "A",
          "B",
          "B"
        ],
        "day": [
          0,
          1,
          0,
          1
        ],
        "sales": [
          10,
          12,
          20,
          22
        ]
      },
      "instance_columns": [
        "region",
        "store"
      ],
      "time_column": "day",
      "target_column": "sales"
    }
  }
}
```

Two instance levels plus time produce a Hierarchical container. One instance level
plus time produces a Panel. Formatting never fills values across groups. A temporal
split applies separately within each series; an instance split holds out whole
series in their existing order. Choose an estimator whose tags support the task
and data properties.

## Classify custom series

Load features and labels with different meanings. The features below have
`numpy3D` shape `(6 instances, 1 variable, 4 timepoints)`; labels are a Table vector.

```python
import numpy as np
from sktime_mcp.tools import (
    load_data_tool,
    prepare_data_tool,
    instantiate_tool,
    fit_tool,
    predict_tool,
)

features = load_data_tool(
    {
        "type": "memory",
        "data": np.arange(24, dtype=float).reshape(6, 1, 4),
        "mtype": "numpy3D",
        "role": "X",
    }
)
labels = load_data_tool({"type": "memory", "data": [0, 1, 0, 1, 0, 1], "mtype": "numpy1D"})
model = instantiate_tool(spec="DummyClassifier(strategy='most_frequent')")
assert fit_tool(model["handle"], X_handle=features["data_handle"], y_handle=labels["data_handle"])[
    "success"
]
result = predict_tool(model["handle"], X_handle=features["data_handle"])
assert result["success"] and len(result["predictions"]) == 6
```

For MCP, supply the feature handle as `X_handle` and label handle as `y_handle` to
`fit`. Supply the feature handle to `predict.X_handle`. The example checks plumbing,
not model quality: train/test separation and task-appropriate evaluation are still
required. The `evaluate` tool currently evaluates forecasters only.

## Use a native component method

Instantiate a splitter or metric through `instantiate`. Then call its public method:

```json
{
  "name": "call_method",
  "arguments": {
    "handle_id": "<splitter>",
    "method_name": "get_n_splits",
    "kwargs": {
      "y_dataset": "airline"
    }
  }
}
```

Keys ending in `_dataset` or `_data_handle` inject server-side data. For a
forecaster's native `X_data_handle`, X means its attached/explicit feature component;
it never falls back to the target. For other components, a handle supplies its
primary container, matching native methods such as a detector's `X` input.

Native calls do not provide uniform job tracking, output handles, resource bounds,
or estimator lifecycle management. Prefer the dedicated tool when it fits the task.

## Background execution and persistence

Add `run_async=true` to loading, fitting, prediction or evaluation. Poll
`check_job_status(job_id=...)` until the job completes or fails. A cancelled worker
can still finish; create a fresh estimator for subsequent work.

Use `save_data` for data and predictions, `save_model` for fitted model state, and
`export_code` for reconstruction code. Keep the chosen source configuration,
preparation policies and environment versions with the artifacts. Exported code
is not a complete replay of the conversation or a guarantee of portability.
