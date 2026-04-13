# Architecture Diagram

## New Data Flow

![New Data Flow](assets/mcp_data_flow.png)

## Component Interaction

![Component Interaction Sequence](assets/component_interaction.png)

## Data Adapter Pattern

![Data Adapter Pattern](assets/data_adapter_pattern.png)

## Data Validation Flow

```
Data Source
    │
    ▼
adapter.load()
    │
    ├─> Set time index
    ├─> Sort by time
    ├─> Infer/set frequency
    │
    ▼
adapter.validate()
    │
    ├─> Check DatetimeIndex ✓
    ├─> Check for duplicates ✓
    ├─> Check missing values ⚠
    ├─> Check monotonic ⚠
    ├─> Check frequency ⚠
    ├─> Check data size ⚠
    │
    ▼
Validation Report
    {
      "valid": true/false,
      "errors": [...],      # Critical issues
      "warnings": [...]     # Non-critical issues
    }
    │
    ▼
adapter.to_sktime_format()
    │
    ├─> Extract target (y)
    ├─> Extract exogenous (X)
    │
    ▼
(y, X) ready for sktime
```

## Handle Management

```
┌─────────────────────────────────────────────────────────────────┐
│                    Executor._data_handles                       │
│  {                                                              │
│    "data_abc123": {                                             │
│      "y": Series(...),                                          │
│      "X": DataFrame(...),                                       │
│      "metadata": {...},                                         │
│      "validation": {...},                                       │
│      "config": {...}                                            │
│    },                                                           │
│    "data_xyz789": {...}                                         │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         │ fit_predict_with_data()            │ release_data_handle()
         │                                    │
         ▼                                    ▼
    Retrieve data                        Delete handle
    Fit estimator                        Free memory
    Generate predictions
```

## Complete Workflow Example

```
1. Load Data
   ┌──────────────────────────────────────────────────────────┐
   │ load_data_source({                                       │
   │   "type": "sql",                                         │
   │   "connection_string": "postgresql://...",               │
   │   "query": "SELECT * FROM sales",                        │
   │   "time_column": "date",                                 │
   │   "target_column": "revenue"                             │
   │ })                                                       │
   └────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────────────────┐
   │ Returns:                                                 │
   │ {                                                        │
   │   "success": true,                                       │
   │   "data_handle": "data_abc123",                          │
   │   "metadata": {                                          │
   │     "rows": 1000,                                        │
   │     "columns": ["revenue", "temperature"],               │
   │     "frequency": "D",                                    │
   │     "start_date": "2020-01-01",                          │
   │     "end_date": "2022-09-27"                             │
   │   },                                                     │
   │   "validation": {                                        │
   │     "valid": true,                                       │
   │     "warnings": []                                       │
   │   }                                                      │
   │ }                                                        │
   └────────────────────┬─────────────────────────────────────┘
                        │
2. Instantiate Model    │
   ┌──────────────────────────────────────────────────────────┐
   │ instantiate_estimator("ARIMA", {"order": [1,1,1]})       │
   └────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────────────────┐
   │ Returns: {"handle": "est_xyz789"}                        │
   └────────────────────┬─────────────────────────────────────┘
                        │
3. Fit & Predict        │
   ┌──────────────────────────────────────────────────────────┐
   │ fit_predict_with_data(                                   │
   │   estimator_handle="est_xyz789",                         │
   │   data_handle="data_abc123",                             │
   │   horizon=7                                              │
   │ )                                                        │
   └────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────────────────┐
   │ Returns:                                                 │
   │ {                                                        │
   │   "success": true,                                       │
   │   "predictions": {                                       │
   │     "2022-09-28": 1250.5,                                │
   │     "2022-09-29": 1255.2,                                │
   │     ...                                                  │
   │   },                                                     │
   │   "horizon": 7                                           │
   │ }                                                        │
   └──────────────────────────────────────────────────────────┘
```
