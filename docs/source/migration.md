# Migrating the data interface

This is an **unreleased source change after 0.1.3**. The package version has not
been bumped and these changes have not been published to PyPI. Restart the server
and refresh the client's tool list after installing this checkout.

## Tool names

| Earlier MCP call | Current MCP call |
| --- | --- |
| `load_data_source(config=...)` | `load_data(config=...)` |
| `transform_data(action="format", ...)` | `prepare_data(action="format", ...)` with explicit policies |
| `transform_data(action="convert", to_mtype=...)` | `prepare_data(action="convert", to_mtype=...)` |
| `split_data(test_size=... / fh=...)` | `prepare_data(action="split", test_size=... / fh=...)` |
| `inspect_data`, `save_data` | Names retained; results and representation support improved |

The advertised tool count changes from **26 to 25**. Old MCP names and hidden
`list_data_sources` / `auto_format_on_load` dispatch paths are rejected. They are not
silent aliases that bypass schema validation.

Python imports of `load_data_source_tool`, `Executor.load_data_source`,
`split_data_tool` and `transform_data_tool` remain compatibility entry points, but
use the new data contract. They do not retain the old automatic-repair behavior.
New Python code should use `load_data_tool`, `prepare_data_tool`, `Executor.load_data`
and `Executor.prepare_data`.

## Behavior changes

- Loading no longer sorts, fills, drops duplicates, guesses a target, or normalizes
  the stored time index to periods.
- `SKTIME_MCP_AUTO_FORMAT` is deprecated and ignored, with a warning when enabled.
- Without a target mapping, all columns remain primary variables. Use an explicit
  target mapping if other columns are features.
- Validation describes the current container. A repairable pandas input may load
  successfully while `validation.valid` is false; inspect it before fitting.
- `prepare_data.action` is required. Options from another action are rejected.
- Formatting defaults to preservation. For the common old repair behavior, request
  `sort_index=true`, `duplicates="first"` and `missing="ffill"` explicitly. This
  deliberately does not backward-fill leading gaps or infer a resampling grid.
- Use `frequency` plus `regularize=true` for deliberate gap insertion; use
  `index_type="period"` for explicit index conversion.
- Conversion preserves attached exogenous data and supports non-pandas outputs.
- Split outputs and their source must fit in the handle budget together.
- Load/prepare responses include bounded `inspection` and validation. Preparation
  reports structured changes by component.
- Tool failures set MCP's error flag consistently with the error body.

## A renamed call is not enough

Update clients that assumed the first column became y, that loading cleaned data,
or that every result was a pandas Series. Recheck custom preprocessing against the
new {doc}`data-guide`. Templates and cached tool schemas must be refreshed too.
