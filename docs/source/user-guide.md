# sktime-mcp Workshop: Conversational Time-Series Workflows with sktime

`sktime-mcp` connects an AI assistant to `sktime` through the Model Context Protocol (MCP). It lets the assistant discover sktime components, load data, fit models, forecast, evaluate, save artifacts, and export reproducible code through typed tools.

Credit: Shashank for the `sktime-mcp` implementation and demo material.

Useful links:

- Repository: https://github.com/sktime/sktime-mcp
- Documentation: https://sktime.github.io/sktime-mcp/
- MCP specification and docs: https://modelcontextprotocol.io/
- Docker image: https://hub.docker.com/r/sktime/sktime-mcp


## Agenda

1. What MCP is, and where `sktime-mcp` fits.
2. How to install `sktime-mcp` and connect it to an MCP client.
3. What tools are available.
4. Three core workflows, plus one optional classification example:
   - discover sktime capabilities,
   - run and save a retail-style forecast,
   - clean messy business data,
   - classify time-series sensor traces if time allows.
5. Current limitations and how to use the project well.


## 1. MCP + sktime-mcp

MCP is a standard way for an AI client to call external tools. The assistant does not need to guess code from memory. It can ask a local or remote MCP server for structured capabilities, call a named tool, and receive structured results.

`sktime-mcp` is an MCP server for `sktime`. In this workshop, it runs as a local stdio process started by the client.

Key properties of this setup:

- **No account is required** for the local server.
- **No API token is required** for local stdio usage.
- **No vendor lock-in**: the same server can be used from Claude, Codex, Cursor, VS Code, Antigravity, and other MCP clients.
- **Self setup**: the command runs in your environment or your Docker container.
- **Local trust boundary**: file access follows the permissions of the server process.

```text
User request
  -> MCP client
  -> sktime-mcp server command
  -> sktime registry, data, model, or persistence operation
  -> structured result back to the assistant
```


## 2. Install and Connect

Setup has two parts:

1. Choose one install method and get a server command.
2. Add that command to an MCP client.

The examples below use stdio transport. No HTTP or SSE server is needed for this workshop.


### Pick One Install Method

Each option shows the command shape that an MCP client needs.

<details>
<summary>Option A: uv and uvx</summary>

Install `uv`. `uvx` is included with `uv`.

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal and check:

```bash
uv --version
uvx --version
```

Run `sktime-mcp` directly:

```bash
uvx sktime-mcp
```

For optional extras:

```bash
uvx --from "sktime-mcp[forecasting,sql,files,mlflow]" sktime-mcp
```

MCP command shape:

```json
{
  "command": "uvx",
  "args": ["sktime-mcp"]
}
```

With extras:

```json
{
  "command": "uvx",
  "args": ["--from", "sktime-mcp[forecasting,sql,files,mlflow]", "sktime-mcp"]
}
```

</details>

<details>
<summary>Option B: pip</summary>

Create an environment and install from PyPI:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "sktime-mcp[all]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "sktime-mcp[all]"
```

Run the server from that environment:

```bash
python -m sktime_mcp.server
```

MCP command shape:

```json
{
  "command": "/absolute/path/to/.venv/bin/python",
  "args": ["-m", "sktime_mcp.server"]
}
```

If the console script is on `PATH`:

```json
{
  "command": "sktime-mcp",
  "args": []
}
```

</details>

<details>
<summary>Option C: source package</summary>

Use this for local development or a fork.

```bash
git clone https://github.com/sktime/sktime-mcp
cd sktime-mcp
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

Run from the source environment:

```bash
python -m sktime_mcp.server
```

MCP command shape:

```json
{
  "command": "/absolute/path/to/sktime-mcp/.venv/bin/python",
  "args": ["-m", "sktime_mcp.server"]
}
```

</details>

<details>
<summary>Option D: Docker Hub image</summary>

Use the published image when Docker is available and a local Python environment is not desired.

```bash
docker pull sktime/sktime-mcp:latest
docker run -i --rm sktime/sktime-mcp:latest
```

MCP command shape:

```json
{
  "command": "docker",
  "args": ["run", "-i", "--rm", "sktime/sktime-mcp:latest"]
}
```

Mount a local directory when the server needs file access:

```json
{
  "command": "docker",
  "args": [
    "run",
    "-i",
    "--rm",
    "-v",
    "/absolute/path/to/data:/data",
    "sktime/sktime-mcp:latest"
  ]
}
```

Inside the container, use the mounted path, for example `/data/sales.csv`.

</details>


### Add It to an MCP Client

The snippets use the `uvx` command. For pip, source, or Docker, replace `command` and `args` with the matching install option above.

<details>
<summary>Claude Desktop</summary>

Open Claude Desktop settings, go to the Developer section, and edit the MCP config file.

Common config paths:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/claude/claude_desktop_config.json`
- Windows: `%APPDATA%\\Claude\\claude_desktop_config.json`

Add:

```json
{
  "mcpServers": {
    "sktime": {
      "command": "uvx",
      "args": ["sktime-mcp"]
    }
  }
}
```

Restart Claude Desktop after saving the file.

</details>

<details>
<summary>Claude Code</summary>

Add the server from a terminal:

```bash
claude mcp add --transport stdio sktime -- uvx sktime-mcp
```

For a pip or source environment:

```bash
claude mcp add --transport stdio sktime -- /absolute/path/to/.venv/bin/python -m sktime_mcp.server
```

Check available servers:

```text
/mcp
```

</details>

<details>
<summary>Codex</summary>

Use the CLI:

```bash
codex mcp add sktime -- uvx sktime-mcp
```

Or edit `~/.codex/config.toml`:

```toml
[mcp_servers.sktime]
command = "uvx"
args = ["sktime-mcp"]
```

For pip or source:

```toml
[mcp_servers.sktime]
command = "/absolute/path/to/.venv/bin/python"
args = ["-m", "sktime_mcp.server"]
```

Use `/mcp` in the Codex TUI to confirm that the server is active.

</details>

<details>
<summary>Cursor</summary>

Open Cursor settings for MCP servers, or edit the MCP config file directly.

Common locations:

- Project: `.cursor/mcp.json`
- User/global: `~/.cursor/mcp.json`

Add:

```json
{
  "mcpServers": {
    "sktime": {
      "command": "uvx",
      "args": ["sktime-mcp"]
    }
  }
}
```

Restart Cursor or reload MCP servers from the settings UI.

</details>

<details>
<summary>VS Code</summary>

Open the Command Palette and run `MCP: Open User Configuration` or `MCP: Open Workspace Folder Configuration`.

Workspace config example at `.vscode/mcp.json`:

```json
{
  "servers": {
    "sktime": {
      "command": "uvx",
      "args": ["sktime-mcp"]
    }
  }
}
```

Command-line alternative:

```bash
code --add-mcp "{\"name\":\"sktime\",\"command\":\"uvx\",\"args\":[\"sktime-mcp\"]}"
```

Official VS Code MCP docs: https://code.visualstudio.com/docs/copilot/customization/mcp-servers

</details>

<details>
<summary>Antigravity</summary>

Open Antigravity's MCP or tools settings and add a local stdio server.

Use this command shape:

```json
{
  "name": "sktime",
  "command": "uvx",
  "args": ["sktime-mcp"]
}
```

If the configuration format follows Claude or Cursor, place the server under `mcpServers`. If it follows VS Code, place it under `servers`.

</details>


### First Check

After the client is configured, ask:

> What demo datasets are available through sktime-mcp?

Expected tool behavior:

- Call `list_available_data`.
- Filter to demo datasets.
- Return names, tasks, and short metadata.

<details>
<summary>Output placeholder</summary>

```text
• Available demo datasets through sktime-mcp:

  Forecasting: PBS_dataset, UCR_UEA_dataset, acsf1, airline, electric_devices_segmentation, forecastingdata, fpp3, from_arff_to_dataframe, from_long_to_dataframe, from_tsfile, from_tsfile_to_dataframe, from_ucr_tsv_to_dataframe, gun_point_segmentation, hierarchical_sales_toydata, japanese_vowels, longley, lynx, m5, macroeconomic, osuleaf, plaid, shampoo_sales, solar, tecator, tsf_to_dataframe, unit_test, unit_test_tsf, uschange

  Classification: arrow_head, basic_motions, gunpoint, italy_power_demand

  Regression: covid_3month

  Total: 33 demo datasets.
```

</details>


## 3. Tool Options

The full tool reference is in the project documentation: https://sktime.github.io/sktime-mcp/

| Need | Tool options | Rough explanation |
| --- | --- | --- |
| Discover what sktime can do | `list_available_data`, `query_registry`, `describe_component` | Find demo data, estimators, tags, and component details. |
| Bring data into the session | `load_data_source`, `inspect_data`, `transform_data`, `split_data`, `save_data` | Load files, inline data, SQL, or URLs into handles; inspect, clean, split, and persist them. |
| Build and run models | `instantiate_estimator`, `fit`, `predict`, `update`, `get_fitted_params`, `call_method` | Create sktime estimators or pipelines, fit them, forecast, update, or call native methods. |
| Evaluate and reproduce | `evaluate`, `export_code`, `save_model`, `load_model` | Cross-validate, generate Python code, and persist fitted models. |
| Manage runtime state | `list_handles`, `release_handle`, `release_data_handle`, `list_jobs`, `check_job_status`, `cancel_job` | See what is in memory, clean it up, and track async work. |

The practical mental model is simple: prompts create tool calls, tool calls create handles, and handles let later prompts continue the workflow.


## 4. Workshop Examples

These examples are written as prompts that can be pasted into an MCP client. Each starts with a real-world motivation, then shows the likely tool path and a placeholder for live output.


### Example 1: Discover sktime Coverage

**Motivation**

A data scientist is starting a time-series project and needs to know what `sktime` can handle before choosing an approach.

**Prompt**

> Use sktime-mcp to discover what sktime can do.
>
> - List available demo datasets by task.
> - Query forecasters, classifiers, regressors, transformers, detectors, splitters, and metrics.
> - For forecasting, identify models that support prediction intervals, multivariate targets, and missing-data handling.
> - Describe `NaiveForecaster`, `ThetaForecaster`, `ARIMA`, `TimeSeriesForestClassifier`, `MiniRocket`, `SlidingWindowSplitter`, and one detector.
> - Summarize which sktime tasks are best supported for actual execution through sktime-mcp today.

**Expected tool behavior**

`list_available_data` -> `query_registry` -> `describe_component`

**Expected outcome**

A map of available sktime tasks, useful estimators, and capability tags. The important point is that the assistant reads installed registry metadata instead of relying on memory.

<details>
<summary>Output placeholder</summary>

```text
• Demo Data

  Forecasting demos: PBS_dataset, UCR_UEA_dataset, acsf1, airline, electric_devices_segmentation, forecastingdata, fpp3, from_arff_to_dataframe,
  from_long_to_dataframe, from_tsfile, from_tsfile_to_dataframe, from_ucr_tsv_to_dataframe, gun_point_segmentation, hierarchical_sales_toydata, japanese_vowels,
  longley, lynx, m5, macroeconomic, osuleaf, plaid, shampoo_sales, solar, tecator, tsf_to_dataframe, unit_test, unit_test_tsf, uschange.

  Classification demos: arrow_head, basic_motions, gunpoint, italy_power_demand.

  Regression demos: covid_3month.

  Registry Coverage

  - Forecasters: 144, examples include ARIMA, AutoARIMA, AutoETS, NaiveForecaster, ThetaForecaster.
  - Classifiers: 77, examples include Arsenal, BOSSEnsemble, BaggingClassifier, TimeSeriesForestClassifier.
  - Regressors: 30, examples include DummyRegressor, KNeighborsTimeSeriesRegressor, CNNRegressor.
  - Transformers: 150, examples include BoxCoxTransformer, Catch22, MiniRocket.
  - Detectors: 28, examples include BinarySegmentation, CAPA, CROPS, HampelDetector.
  - Splitters: 15, examples include CutoffSplitter, ExpandingWindowSplitter, SlidingWindowSplitter.
  - Metrics: 39, examples include CRPS, EmpiricalCoverage, AUCalibration, GeometricMeanAbsoluteError.

  Forecasting Capabilities

  - Prediction intervals: 71 forecasters, including ARIMA, AutoARIMA, AutoETS, NaiveForecaster, ThetaForecaster, BaggingForecaster.
  - Multivariate targets: 43 forecasters, including AutoTS, BaggingForecaster, ColumnEnsembleForecaster, DynamicFactor.
  - Missing-data handling: 28 forecasters, including ARIMA, AutoARIMA, AutoETS, NaiveForecaster, BaggingForecaster, DynamicFactor.
  - All three together: 7 forecasters, including BaggingForecaster, DynamicFactor, FhPlexForecaster, ForecastX, ForecastingPipeline, Permute,
    TransformedTargetForecaster.

  Component Notes

  - NaiveForecaster: simple baseline forecaster with strategy, sp, and window_length; supports prediction intervals and missing values.
  - ThetaForecaster: theta-method forecaster with optional deseasonalization; supports prediction intervals but not missing values.
  - ARIMA: pmdarima-backed (S)ARIMA(X) forecaster; supports exogenous variables, prediction intervals, and missing values.
  - TimeSeriesForestClassifier: interval-based ensemble classifier using random intervals and tree ensembles; supports probability prediction and feature
    importance.

  - MiniRocket: convolutional feature transformer for univariate time series; fast feature extraction, no missing-value support.
  - SlidingWindowSplitter: temporal splitter with window_length, fh, step_length, and optional initial_window.
  - CAPA: collective and point anomaly detector; unsupervised segmentation/anomaly detection, supports multivariate data but not missing values.

  Tools Used

  I used list_available_data, query_registry, and describe_component.

  Execution is strongest for forecasting: the MCP has direct support for loading/formatting/splitting data, instantiating forecasters, fitting, predicting,
  prediction intervals, and evaluation. Classifiers, regressors, transformers, detectors, splitters, and metrics are well covered for discovery and component
  inspection; actual execution is possible mainly through generic instantiation plus call_method, but it is less streamlined than forecasting.
```

</details>


### Example 2: Retail Forecasting and Saving Results

**Motivation**

A retail analyst wants a quick demand forecast for the next planning cycle. The goal is not just a chat answer; the forecast must be saved so it can be shared with a planning spreadsheet or another workflow.

**Prompt**

> Run a retail-style forecasting experiment using the airline demo dataset as a stand-in for monthly demand.
>
> - Compare a simple seasonal baseline with a stronger seasonal or probabilistic forecaster.
> - Forecast the next 12 months.
> - Request prediction intervals or quantiles where supported.
> - Evaluate at least one model with cross-validation.
> - Explain the result in business language.
> - Save the forecast table to `outputs/airline_forecast.csv`.
> - Export reproducible Python code for the selected model.

**Expected tool behavior**

`list_available_data` -> `describe_component` -> `instantiate_estimator` -> `fit` -> `predict` -> `evaluate` -> `get_fitted_params` -> `export_code`

For persistence, use the most appropriate route:

- Use `save_data` when the result is held as a data handle.
- If the forecast is returned as a table in the tool response, write it to `outputs/airline_forecast.csv` from the client or exported code.
- Use `export_code` so the saving step can be reproduced outside the MCP session.

<details>
<summary>Output placeholder</summary>

Paste the forecast table, evaluation summary, saved file path, and exported code summary here.


```text
• Done. I inspected available demo data and used the forecasting demo dataset airline.

  Models chosen:

  - Baseline: NaiveForecaster(strategy='last', sp=12)
  - Stronger model: AutoETS(auto=True, sp=12, n_jobs=1)

  Forecasts

   Month       Naive    AutoETS    AutoETS 90% PI    AutoETS 95% PI
  ━━━━━━━━━  ━━━━━━━━  ━━━━━━━━━  ━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━
   1961-01    417.00     442.77     418.00-470.43     413.17-474.04
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-02    391.00     435.90     404.72-468.47     398.18-475.82
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-03    419.00     498.82     457.78-542.66     451.78-548.62
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-04    461.00     485.95     445.28-535.36     433.75-543.68
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-05    472.00     487.55     439.72-541.89     434.29-555.12
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-06    535.00     554.06     496.46-621.56     485.13-637.07
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-07    622.00     617.47     549.37-700.07     538.26-711.66
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-08    606.00     614.17     540.51-696.73     528.44-716.53
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-09    508.00     534.26     466.85-611.12     456.07-621.06
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-10    461.00     467.01     404.98-533.96     394.80-549.70
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-11    390.00     406.25     350.70-463.31     342.13-478.66
  ─────────  ────────  ─────────  ────────────────  ────────────────
   1961-12    432.00     457.38     395.24-527.61     382.99-540.75

  Both models also produced prediction intervals; I showed the stronger model’s intervals above.

  Cross-Validation

  3-fold CV on airline, metric: Mean Absolute Percentage Error.

   Model              Mean MAPE    Fit Time Mean
  ━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━
   NaiveForecaster       0.0838           0.006s
  ─────────────────  ───────────  ───────────────
   AutoETS               0.0255           1.593s

  Plain English: the seasonal naive model is extremely fast and reasonable as a baseline, but it mostly repeats last year’s seasonal pattern. AutoETS is much
  more accurate in CV, with about one-third of the baseline MAPE, and gives smoother seasonal forecasts with calibrated intervals. It is slower to fit, but still
  fast enough here.

  Fitted Parameters

  Naive:

  sp = 12
  window_length = 144

  AutoETS:

  aic  = 1080.9799
  aicc = 1086.4519
  bic  = 1134.4365
  hqic = 1102.7016

  Best Model Code

  from sktime.registry import craft
  from sktime.datasets import load_airline

  best_model = craft("AutoETS(auto=True, sp=12, n_jobs=1)")

  y = load_airline()
  best_model.fit(y)

  fh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  predictions = best_model.predict(fh=fh)
  intervals = best_model.predict_interval(fh=fh, coverage=[0.9, 0.95])

  print(predictions)
  print(intervals)
```

</details>


### Example 3: Messy Business Data

**Motivation**

A sales operations team has monthly revenue data exported from a business system. The time index is not perfect: there is a duplicate month, a missing month, and a missing revenue value.

**Prompt**

> Create or load an inline monthly sales time series with `date`, `revenue`, `promo_spend`, and `region`.
>
> Include:
>
> - one duplicate date,
> - one missing month,
> - one missing revenue value.
>
> Then use sktime-mcp to:
>
> - load the data with `date` as the time column and `revenue` as the target,
> - inspect it,
> - clean and format it,
> - inspect it again,
> - split the final 6 months as test data,
> - fit a forecaster on the train handle,
> - forecast the test horizon,
> - save the cleaned data to `/tmp/sktime_mcp_clean_sales.csv`,
> - explain the handles that were created and released.

**Expected tool behavior**

`load_data_source` -> `inspect_data` -> `transform_data` -> `save_data` -> `split_data` -> `instantiate_estimator` -> `fit` -> `predict` -> `release_data_handle` -> `release_handle`

**Expected outcome**

A cleaned time-series handle, a saved CSV file, train/test handles, forecast values for the holdout window, and a clear explanation of state cleanup.

<details>
<summary>Output placeholder</summary>

```text
• Completed, with one MCP limitation: the exact inline dataset containing a duplicate date was submitted first, but load_data_source rejected it before creating
  a handle:

  Duplicate time indices found: 1 duplicates

  To continue the experiment, I loaded the same dataset with the duplicate row removed; the missing month and missing revenue value were still present and were
  cleaned by the MCP loader/formatter.

  Loaded/cleaned data:

  - Date range: 2023-01-01 to 2024-06-01
  - Frequency: monthly, <MonthBegin>
  - Rows after cleaning: 18
  - Target: revenue
  - Exogenous columns retained: promo_spend, region
  - Missing target values after cleaning: 0
  - Saved CSV: /tmp/sktime_mcp_clean_sales.csv

  Split:

  - Train handle: data_ac691310, 12 months through 2023-12-01
  - Test handle: data_c0f25f37, final 6 months

  Modeling:

  - Tried NaiveForecaster(strategy='last', sp=12), fitted successfully, but prediction failed due an sktime/pandas month-start frequency issue: <MonthBegin> is
    not supported as period frequency.

  - Used replacement model: PolynomialTrendForecaster(degree=1)

  Forecast for test horizon:

  2024-01-01    165231.29
  2024-02-01    168917.92
  2024-03-01    172366.69
  2024-04-01    176053.32
  2024-05-01    179621.02
  2024-06-01    183307.65

  Handle lifecycle:

  - Failed duplicate-date load: no handle created.
  - data_ab48856a: loaded/inspected data handle; release later returned “not found”, likely superseded internally after transform.
  - data_47afaf61: cleaned/formatted handle; saved to CSV; released successfully.
  - data_ac691310: train split handle; released successfully.
  - data_c0f25f37: test split handle; released successfully.
  - est_426598d944cf: seasonal naive model; fitted but prediction failed; released successfully.
  - est_3b59fe5fe7fe: trend model; fitted and forecasted; released successfully.

```

</details>


### Example 4: Time-Series Classification

**Motivation**

A quality-monitoring team has sensor traces from a production process. Each trace needs a label, not a future forecast. This shows that `sktime-mcp` is not only a forecasting interface.

**Prompt**

> Use a time-series classification demo dataset such as `gunpoint` or `arrow_head`.
>
> - Query suitable classifiers.
> - Try `TimeSeriesForestClassifier` or `KNeighborsTimeSeriesClassifier`.
> - Fit the classifier on the demo dataset.
> - Predict labels and probabilities if supported.
> - Summarize the output shape and any limitations.

**Expected tool behavior**

`list_available_data` -> `query_registry` -> `instantiate_estimator` -> `fit` -> `predict` -> `get_fitted_params`

**Expected outcome**

A small classification workflow that demonstrates broader `sktime` coverage. Forecasting remains the smoothest path today, but discovery and generic execution are useful beyond forecasting.

<details>
<summary>Output placeholder</summary>

```text
• Used gunpoint with TimeSeriesForestClassifier(n_estimators=10, random_state=0).

  Registry check: TimeSeriesForestClassifier is a classifier and supports predict_proba.

  Fit/predict result:

  - Fitted on demo X_dataset="gunpoint", y_dataset="gunpoint"
  - Learned classes: ["1", "2"]
  - predict(X) returned 200 labels, so shape is (200,)
  - predict_proba(X) is supported and returned 200 rows with 2 class probabilities, so shape is (200, 2)

  The probability columns correspond to the learned class order ["1", "2"].
```

</details>


## 5. Current Limitations

These limitations matter before building expectations around the tool.

- The local stdio setup has no built-in auth layer. Treat it as a local trusted tool, and control access through the client, environment, and filesystem permissions.
- Handles live in memory. Save data, save models, or export code before restarting the server.
- File paths are resolved where the server runs. With Docker, mount files into the container and use container paths.
- Optional estimators may need optional dependencies. Missing dependency errors are expected for some advanced models.
- Forecasting workflows are the most polished. Classification, detection, splitters, metrics, and other scitypes are reachable, but may need `call_method` or more explicit prompts.
- Some prediction results are returned as structured text or JSON, not as persistent data handles. Save them through exported code or client-side file writing when needed.
- `evaluate` is strongest on demo datasets today. Custom train/test evaluation is a promising extension.


## 6. Wrap-Up

Use `sktime-mcp` when the task needs real `sktime` metadata or execution, not just a code suggestion.

Best current uses:

- discover estimators, datasets, tags, and capabilities;
- load, inspect, clean, split, and save time-series data;
- fit forecasters and produce forecasts;
- evaluate models on demo data;
- export reproducible Python code;
- save fitted models and important data artifacts.

A good workflow is:

1. Ask the assistant to inspect what is available.
2. Ask for a model choice with reasons.
3. Fit and forecast through tool calls.
4. Save the forecast or cleaned data.
5. Export code before moving out of the MCP session.

The main takeaway: MCP turns the assistant from a code generator into a tool-using interface. `sktime-mcp` makes that interface practical for time-series workflows in `sktime`.


