# Contributing

## Set up

From the checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,files,sql]"
```

Use lightweight fixtures by default. No large model download or deep-learning
installation is required for the data-contract and MCP regression suites.

## Verify a change

```bash
ruff check .
ruff format --check .
pytest -q
sphinx-build -W --keep-going docs/source docs/_build/html
```

For a data change, start with:

```bash
pytest tests/test_general_data.py tests/test_url_limits.py tests/test_job_cancellation.py -q
```

Core tests use sktime's own example containers and compare behavior across sync and
async paths. Add tests for preserved values, indexes, X/y roles, group boundaries,
invalid action combinations, resource limits and real MCP calls. A test that only
checks an implementation detail is not evidence of a usable workflow.

## Extend the data layer

An adapter reads a source. Shared contracts handle explicit mappings, validation
and inspection; preparation handles changes to data. Do not infer task semantics
from the first column, silently resample, or add source-specific repair rules.

Use sktime's public datatype checks and converters. Keep optional-backend support
conditional on installed dependencies. Document where an operation cannot preserve
a representation, and return a meaningful error rather than flattening axes.

MCP tools are declared in `server.get_tools`. Loading/preparation schemas are in
`data/schemas.py`. Add action-specific input validation and update the
{doc}`../tool-reference`, {doc}`../data-guide` and {doc}`../support-matrix` together.

## Review documentation in the browser

Build with warnings treated as errors, then serve the rendered directory locally:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory docs/_build/html
```

Open `http://127.0.0.1:8765/`. Check every authored page at desktop and narrow widths:
headings, sidebar links, tables, code blocks, examples, local anchors and search.
Fix the source and rebuild; a successful Sphinx build alone does not verify the
rendered layout. CSS must target the configured Read the Docs theme.

## Submit a reviewable change

Explain the concrete behavior change, migration impact and checks run. Keep release
version/tag/PyPI changes separate unless a release is requested. Existing Python
compatibility helpers do not imply that retired MCP tool names remain supported.
