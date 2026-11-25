# Scope and generalizability

Broad registry access is not the same as complete support for every sktime task.
This page separates data-container support from estimator execution and from
workflow features such as evaluation, plotting and persistence.

## Data tools

| Tool | Generality | Boundaries |
| --- | --- | --- |
| `load_data` | Shared source-independent mapping and validation; Series, Panel, Hierarchical, Table, Alignment and Proba contracts | JSON needs an explicit representation for complex containers. Optional/lazy backends need dependencies and are not all verified. |
| `inspect_data` | sktime type validation and bounded previews; array and nested-panel awareness | Some counts/statistics are unavailable for non-tabular containers. A valid container may still be unsuitable for a selected estimator. |
| `prepare_data`: format | Series, Panel, Hierarchical and Table; group-local missing-value handling | Temporal options require a time axis. No implicit resampling, learned imputer or automatic reinterpretation of a task. |
| `prepare_data`: convert | sktime's supported conversions within one scitype; preserves unselected X/y | Not every mtype pair is convertible or lossless. Array representations can omit labels. |
| `prepare_data`: split | Temporal splits per series; explicit whole-instance splits or Table row holdouts | No randomized/stratified classification split or automatic pairing of independent feature and label handles. |
| `save_data` | Canonical tabular export, hierarchy index columns and aligned exogenous data; native NPY arrays | File formats have representation/dtype limits. MultiIndex columns and pickle object arrays need a different explicit representation. |

Core tests exercise 16 representations, including pandas Series/DataFrames, ndarray
Series, numpy3D panels, unequal-length lists, nested panels, MultiIndex panels and
hierarchies, Table containers, alignments, intervals and quantiles. This is a tested
subset of sktime's registry, not a claim that every optional backend is equivalent.

## Model and workflow tools

| Tools | Current scope | Remaining limits |
| --- | --- | --- |
| `query_registry`, `describe_component` | Installed estimators, metrics, splitters and tags; dependency descriptions | Availability is environment-specific, not proof that a complete workflow works. |
| `instantiate` | Estimators and compositions built by sktime craft | Trusted expression evaluation; some incompatibilities appear only during fit. |
| `fit` | Forecasters, classifiers/regressors, transformers, detectors and clusterers through task-specific routing | Individual estimator tags still govern supported data. Failed/concurrent fits and releasing in-use handles need stronger lifecycle guarantees. |
| `predict` | Forecast modes; classification/regression predictions; transformer transform; detector/clusterer prediction | Modes vary by task. Output representation/handle coverage is less uniform outside pandas forecasting. |
| `update` | Incremental update where the estimator supports the exposed signature | Primarily forecasting-oriented; no universal incremental-learning contract across task types. |
| `get_fitted_params` | Fitted parameter inspection where provided by the estimator | Parameters may be large or contain complex native objects. |
| `call_method` | Public native methods, metrics, splitters and aligners; data injection | Executes synchronously; no universal lifecycle, streaming or return-handle contract. |
| `evaluate` | Numeric Series forecasting, including multivariate Series accepted by the selected forecaster/metric | No classification, regression, clustering, Panel or Hierarchical cross-validation through this tool. Native sktime has broader evaluation facilities. |
| `plot_series` | pandas time-series and forecast handles | Not a general panel, hierarchy, alignment, label or distribution visualization API. |
| `export_code` | Reconstruct estimator specs or saved-model loaders | Does not capture custom preprocessing; some task-specific fit examples and dependencies still need manual correction. |
| `save_model`, `load_model` | MLflow-backed model persistence; MLflow is a core dependency | Environment compatibility and path/URI handling remain relevant; model file-URI handling still needs correction. |
| Handle and job tools | Process-wide state, polling, release and cancellation | No per-client session isolation; running threads cannot be force-stopped safely. |
| `run_command` | General shell execution | Host privileges apply; it is not a controlled substitute for missing typed functionality. |

## What should improve next

1. **Estimator lifecycle:** reject conflicting fit/release operations and make failed
   refits leave accurate state.
2. **Task-specific evaluation and results:** define classification/regression metrics,
   stable result handles and deliberate panel/hierarchy evaluation geometry.
3. **Native method contracts:** explicit component selection, bounded outputs and
   state updates for mutating methods.
4. **Persistence and reproducibility:** exercise exported examples by task and make
   supported round trips explicit.

These are capability improvements. Adding more tool names alone would not resolve
them. Model comparison can be considered once evaluation produces consistent,
comparable results across the intended tasks.

For sktime's underlying contracts, see its
[data format specification](https://www.sktime.net/docs/api-reference/data-format/)
and [temporal splitting API](https://www.sktime.net/docs/api-reference/sktimesplittemporal-train-test-split/).
