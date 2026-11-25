# What sktime-mcp does

`sktime-mcp` connects an MCP client to the sktime installation running on your
machine or server. An assistant can discover estimators, construct models and
pipelines, load data, run supported methods, and return results to the conversation.

## A workflow you can continue

A loaded dataset gets a `data_…` handle; a model gets an `est_…` handle. Later calls
reuse these objects. You can fit a model, inspect its parameters, forecast, plot the
result, and save it without transferring the training data on every call.

The server exposes **25 tools**. Data work uses four entry points: `load_data`,
`inspect_data`, `prepare_data`, and `save_data`. Background loading, fitting,
prediction and evaluation return jobs that clients can poll.

## sktime defines the data contracts

The data layer uses sktime's datatype checks and converters. A scientific type
(*scitype*) describes meaning, such as Series or Panel. A machine type (*mtype*)
describes its representation, such as `pd.Series` or `numpy3D`.

Loading preserves values and row order. Cleaning, conversion and splitting happen
only through explicit preparation actions. A validation report describes issues;
a successful load does not mean every estimator can use that data.

## What is established, and what is limited

Forecasting has the broadest end-to-end coverage. Custom classification can use a
feature panel and a separate label container. Native method calls make additional
sktime components reachable, but do not give every task the same evaluation,
plotting or export support. Read {doc}`support-matrix` before choosing a path.

The server runs with its process account's permissions. Craft specifications and
shell commands are trusted execution, not a sandbox. HTTP authentication controls
access to the server; it does not isolate an authorized call from the filesystem.

Continue with {doc}`quickstart` or {doc}`data-guide`.
