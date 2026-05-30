# Documentation Review Plan

## Overview
This document outlines 29 issues identified across the `sktime-mcp` Sphinx documentation. The issues range from broken image paths and duplicated content to missing features in the tool lists and incorrect tool usage in examples. 

## Identified Issues

### `intro.md`
1. **Hardcoded PyPI Version**: The header link points to `0.1.0` specifically instead of the general project URL.
2. **Installation Instructions**: The `pip install` command lacks a mention of the `[all]` optional dependencies for the quick start.
3. **Documentation Map**: The map table is missing an entry for `background-jobs.md`.

### `architecture.md`
4. **Broken Image Path 1**: `![New Data Flow](assets/mcp_data_flow.png)` uses `assets/` instead of `_static/`.
5. **Broken Image Path 2**: `![Component Interaction Sequence](assets/component_interaction.png)` uses `assets/`.
6. **Broken Image Path 3**: `![Data Adapter Pattern](assets/data_adapter_pattern.png)` uses `assets/`.

### `data-sources.md`
7. **SQL Troubleshooting**: "No module named 'sqlalchemy'" suggests `pip install sqlalchemy` rather than the preferred `pip install "sktime-mcp[sql]"`.
8. **Excel Troubleshooting**: "No module named 'openpyxl'" suggests `pip install openpyxl` instead of `pip install "sktime-mcp[files]"`.
9. **Parquet Troubleshooting**: "No module named 'pyarrow'" suggests `pip install pyarrow` instead of `pip install "sktime-mcp[files]"`.
10. **Formatting**: Boolean arguments (`true`/`false`) in the `list_available_data` section should be formatted as inline code for clarity.

### `implementation.md`
11. **Duplicated Content**: The `pyproject.toml` section is duplicated identically (lines 62-78).
12. **Incomplete Tool List**: The `server.py` section only lists 8 tools, missing the remaining 20.
13. **Missing Tool Modules**: The `src/sktime_mcp/tools/` section completely misses `evaluate.py`, `format_tools.py`, `job_tools.py`, `save_model.py`, and `list_available_data.py`.
14. **Missing Examples**: The `examples/` section misses 7 new example files (e.g., `background_training_example.py`, `sql_example.py`).
15. **Missing Tool in instantiate.py**: The description for `instantiate.py` misses the `load_model` tool.

### `usage-examples.md`
16. **Missing Background Jobs Example**: Lacks an end-to-end example demonstrating the major `fit_predict_async` feature.
17. **Missing Data Formatting Example**: Lacks an example showing how to use the `format_time_series` tool.

### `user-guide.md`
18. **Incomplete Core Capabilities 1**: The Core Capabilities table is missing `evaluate_estimator` (cross-validation).
19. **Incomplete Core Capabilities 2**: The table is missing background jobs (async execution).
20. **Incorrect Tool Name**: In the "Saving a Trained Model" section, the tool to load the model is incorrectly listed as `instantiate_estimator` instead of `load_model`.
21. **JSON Snippet Error**: The `load_model` arguments snippet is missing the `"tool": "load_model"` wrapper.
22. **Prerequisites Clarification**: Mentions "VS Code with Copilot", which lacks native MCP support without extensions like Cline.

### `use-cases.md`
23. **Missing Data Source Type**: The "Your Own Data" table examples omit the required `type` parameter (e.g., `type: file` or `type: sql`) in the prompt examples.
24. **Prompt Clarity**: The SQL prompt example should explicitly state the `target_column` mapping in a clearer format.

### `dev-guide.md`
25. **Outdated Formatter**: Mentions `python -m black --check .` but the project uses `ruff format`.
26. **Incorrect Doc Paths 1**: Mentions `docs/user-guide.md` instead of `docs/source/user-guide.md`.
27. **Incorrect Doc Paths 2**: Mentions `docs/data-sources.md` instead of `docs/source/data-sources.md`.

### `api.rst`
28. **Incomplete API Reference**: Currently only documents `sktime_mcp` and `sktime_mcp.server`. Should include `tools`, `registry`, `runtime`, and `composition` modules.

### `index.rst`
29. **Missing External Links**: The toctree could benefit from an explicit link to the GitHub repository for better navigation.

## Action Plan
1. Apply fixes to `intro.md` and `architecture.md`.
2. Update `data-sources.md` dependency recommendations.
3. Clean up and complete `implementation.md`.
4. Expand `usage-examples.md` and correct `user-guide.md`.
5. Refine `use-cases.md`, `dev-guide.md`, `api.rst`, and `index.rst`.
