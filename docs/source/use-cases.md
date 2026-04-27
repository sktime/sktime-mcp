# 🎯 Use Cases & Personas

This page describes **who** uses sktime-mcp, **what** they want to accomplish, and **how** a typical session looks. For step-by-step walkthroughs, see the [User Guide](user-guide.md). For concrete prompt examples, see [Usage Examples](usage-examples.md).

---

## 👤 User Personas

### 1. Coder

**Goal:** Build, benchmark, and iterate on time-series models — and get reproducible Python code as output.

#### Use-Case 1: Benchmarking Time Series Models

**Scenario:** You want to compare several forecasting models (e.g., ARIMA, ExponentialSmoothing, Theta) on a dataset and pick the best one.

**What you do:**

1. Tell your AI assistant which dataset to use (a built-in demo or your own CSV/SQL data).
2. Ask it to compare several models using cross-validation.
3. Review the comparison table of metrics (MAE, RMSE, etc.).
4. Ask for the Python code that reproduces the best result.

**Example prompt:**

> *"Compare ARIMA, ExponentialSmoothing, and ThetaForecaster on the airline dataset with 5-fold cross-validation. Then export the code for the best model."*

**What you get:**

- A table of cross-validation metrics for each model.
- A recommendation for the best-performing model.
- A Python script that reproduces the experiment end-to-end.

---

#### Use-Case 2: Composing Pipelines

**Scenario:** You want to build a multi-step pipeline (e.g., deseasonalize → detrend → forecast) and evaluate it as a single unit.

**What you do:**

1. Describe the pipeline components and their order.
2. The assistant validates compatibility and creates the pipeline.
3. Ask the assistant to run it on your data.
4. Request the code to share with your team.

**Example prompt:**

> *"Build a pipeline: ConditionalDeseasonalizer, then Detrender, then ARIMA(1,1,1). Run it on the airline dataset for 12 months. Export the code."*

**What you get:**

- Confirmation that the pipeline is valid.
- Forecast results from the composed pipeline.
- A standalone Python script that constructs and runs the pipeline.

**Tips for designing pipelines:**

- Transformers (e.g., `Deseasonalizer`, `Detrender`, `BoxCoxTransformer`) go first.
- The final component should be a forecaster.
- If unsure which transformers to use, ask: *"What preprocessing transformers work well before ARIMA?"*

---

### 2. Business User

**Goal:** Get forecasts, analysis, and summary reports — without writing code.

#### Use-Case: Get Forecasts & Reports

**Scenario:** You have a CSV of monthly sales data and want a forecast for the next quarter, presented as a table you can share with stakeholders.

**What you do:**

1. Tell the assistant where your data file is and which columns to use.
2. Ask for a forecast (you can specify the model or let the assistant choose).
3. Ask the assistant to summarize the results.

**Example prompt:**

> *"Load my sales data from /home/user/quarterly_sales.csv (date column: 'quarter', target: 'revenue'). Forecast revenue for the next 4 quarters. Summarize the results."*

**What you get:**

- A table of forecasted values for each future period.
- A plain-language summary of the trend (e.g., *"Revenue is projected to grow ~5% per quarter"*).
- Optionally, you can ask for a chart or a CSV export of the results.

---

## 📂 Loading Data

### Demo Datasets

Built-in datasets are available for quick experimentation — no file paths needed:

> *"What demo datasets are available?"*
>
> *"Use the airline dataset."*

Demo datasets are useful for learning the workflows before bringing in your own data.

### Your Own Data

| Source | What to tell the assistant | Example |
|--------|---------------------------|---------|
| **Local CSV / Parquet / Excel** | Absolute file path, time column name, target column name. | *"Load /home/user/data.csv, time column 'date', target 'value'"* |
| **SQL database** | Connection string, query, time and target columns. | *"Load from my PostgreSQL database at localhost:5432/mydb, run 'SELECT date, sales FROM monthly', time column 'date', target 'sales'"* |

For detailed configuration options, see [Data Sources](data-sources.md).

---

## 📝 What You Can Get Out

| Output | How to ask for it | Best for |
|--------|-------------------|----------|
| **Forecast table** | *"Forecast X for N periods"* | Quick answers, sharing with stakeholders |
| **Comparison metrics** | *"Compare these models with cross-validation"* | Model selection |
| **Python code** | *"Export the code for this"* | Reproducibility, integration into production |
| **Saved model artifact** | *"Save this model to /path"* | Persistence across server restarts |

---

## 🔗 See Also

- [User Guide](user-guide.md) — Step-by-step instructions for every workflow.
- [Usage Examples](usage-examples.md) — Concrete prompt examples with expected outputs.
- [Data Sources](data-sources.md) — Detailed data loading configuration.
- [Background Jobs](background-jobs.md) — Running long operations asynchronously.
