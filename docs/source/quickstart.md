# Quickstart

This page uses the unreleased source interface. Install the current checkout as
shown in {doc}`installation`; PyPI 0.1.3 still uses the earlier data-tool names.

## Connect your MCP client

Configure a stdio server using the absolute path to your environment's Python.
Clients that use an `mcpServers` configuration can use this shape:

```json
{
  "mcpServers": {
    "sktime": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": [
        "-m",
        "sktime_mcp.server"
      ]
    }
  }
}
```

On Windows use the environment's `Scripts/python.exe`. The outer configuration
format varies by client; the command and arguments are the same. Restart or
reconnect the MCP server after changing its installed code.

## Run a small forecast

Ask your assistant:

> Find NaiveForecaster, fit it to the airline demo dataset, and forecast the next
> six months. Plot the forecast and tell me which data it used.

The essential calls are:

```json
{
  "name": "instantiate",
  "arguments": {
    "spec": "NaiveForecaster(strategy='last', sp=12)"
  }
}
```

Use the returned estimator handle in the next calls:

```json
{
  "name": "fit",
  "arguments": {
    "estimator_handle": "<model>",
    "y_dataset": "airline"
  }
}
```

```json
{
  "name": "predict",
  "arguments": {
    "estimator_handle": "<model>",
    "horizon": 6
  }
}
```

The result includes predictions and, for supported output containers, a
`prediction_handle`. Pass it to `plot_series` or `save_data`.

## Load your own data

```json
{
  "name": "load_data",
  "arguments": {
    "config": {
      "type": "file",
      "path": "/absolute/path/to/sales.csv",
      "time_column": "date",
      "target_column": "sales"
    }
  }
}
```

Read the returned `inspection` and `validation`. The loader preserves missing
values and duplicates. If repairs are needed, request them with `prepare_data`.
Use {doc}`data-guide` for policies and {doc}`user-guide` for complete workflows.

## If something fails

- **Command not found:** use the absolute Python path above.
- **Wrong tool names:** reconnect and confirm which checkout or release the client starts.
- **Missing dependency:** call `describe_component` and install only the required package.
- **Data validation issues:** inspect the report; declare the representation or preparation explicitly.
- **Background work:** poll `check_job_status`; completion is not pushed to the client.
