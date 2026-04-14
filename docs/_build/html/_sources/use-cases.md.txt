# 🎯 Use Cases & Personas

This page provides clear, step-by-step workflows for both coders and business users, covering all major use-cases for sktime-mcp.

---

## 👤 User Personas

### 1. Coder
- **Goal:** Wants code files as outputs.

#### Use-Case 1: Benchmarking Time Series Models
- **Task:** Use a model (e.g., ARIMA) and benchmark it on a dataset (local or web).
- **What happens:**
    - Your dataset is automatically loaded (see [Loading Data](#loading-data)).
    - The selected model (e.g., ARIMA, Prophet) is instantiated and fitted.
    - Benchmarking is performed behind the scenes.
    - Forecast results and code files are generated for you. You just need to submit your query and wait for the results/status.

#### Use-Case 2: Composing Pipelines
- **Task:** Build and benchmark a pipeline (e.g., Deseasonalizer → Detrender → Estimator).
- **What happens:**
    - The pipeline components you specify are validated and composed automatically.
    - The pipeline is run on your dataset in the background.
    - Code files and benchmark results are produced for you. You only need to submit your pipeline query and monitor the status.

- **Designing Custom Pipelines:**
    - You can specify any sequence (e.g., deseasonalizer → detrender → estimator).
    - If unsure, see [pipeline design guide](usage-examples.md).

---

### 2. Business Person
- **Goal:** Wants forecasts, analysis, and reports as outputs.

#### Use-Case: Get Forecasts & Reports
- **What happens:**
    - Your data (CSV or demo) is loaded and processed automatically.
    - The requested analysis (forecast, report) is performed in the background.
    - Results (forecasts, charts, summary reports) are made available for you to view or download. Just submit your request and wait for completion.

---

## 📂 Loading Data

### 1. Demo Datasets
- Use built-in demo datasets for quick testing.

### 2. User-Defined Data
- **Local CSV:**  
    - Place your CSV in the project directory.
    - The system will automatically load your file when you reference it in your query—no manual code needed.
- **Web URL (Planned):**  
    - Soon, you’ll be able to load CSVs directly from a web URL by simply providing the link in your query.
    - _Action: Add web URL support and update docs when ready._

---

## 📝 Outputs
- **Code Files:**  
    - Scripts and notebooks for reproducibility.
- **Forecasts:**  
    - CSV, plots, and summary tables.

---

## 🔗 See Also
- [User Guide](user-guide.md)
- [Usage Examples](usage-examples.md)
- [Implementation Details](implementation.md)
