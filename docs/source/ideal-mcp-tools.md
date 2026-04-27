# Ideal MCP Tools

This page captures the intended MCP tool surface for `sktime-mcp`.
It is a design-oriented reference for contributors and advanced users who want
to understand tool responsibilities and argument shapes at a high level.

For implementation details and concrete behavior, cross-reference:

- [API Reference](api.rst)
- [Usage Examples](usage-examples.md)
- [User Guide](user-guide.md)

## Tool Families

The toolset is organized into the following families:

- **Estimator discovery and inspection**
  - `list_estimators`
  - `describe_estimator`
- **Estimator and pipeline instantiation**
  - `instantiate_estimator`
  - `instantiate_pipeline`
  - `validate_pipeline`
  - `load_model`
- **Data loading and preparation**
  - `list_data_sources`
  - `load_data_source`
  - `format_data_handle`
  - `describe_data_handle`
- **Training, prediction, and evaluation**
  - `fit_predict`
  - `fit_predict_async`
  - `evaluate_estimator`
- **Lifecycle and export**
  - `save_model`
  - `export_code`
  - `release_handle`
  - `release_all_handles`
  - `list_active_handles`
- **Background jobs**
  - `check_job`
  - `list_jobs`
  - `cancel_job`

## Design Principles

- **Stable tool names:** keep externally visible names predictable for MCP clients.
- **Clear argument contracts:** validate inputs early and return actionable errors.
- **Handle-based state:** pass lightweight handles between tool calls.
- **Backward compatibility:** deprecate old tool aliases with explicit migration notes.
- **Observable behavior:** surface status, metadata, and warnings in structured payloads.

## Notes for Contributors

- Prefer adding new capability by extending an existing tool family before
  introducing a brand-new top-level tool.
- Keep docs, server dispatch, and tests synchronized whenever tool contracts
  change.
- If a tool is deprecated, update this page and related docs in the same PR.
