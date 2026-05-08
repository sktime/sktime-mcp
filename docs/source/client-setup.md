# mcp client setup

this page is for people who want to connect `sktime-mcp` to a real mcp client or a custom local agent project.

it focuses on practical setup, common mistakes, and a quick way to check whether the server is really working.

---

## before you start

you need:

- python 3.10 or newer
- a working install of `sktime-mcp`
- one mcp client such as claude desktop, cursor, or a custom local runner

if you are developing inside the repo, the safest setup is:

```bash
git clone https://github.com/sktime/sktime-mcp.git
cd sktime-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

if you mainly want to use the server instead of contributing, you can install from pypi:

```bash
pip install "sktime-mcp[all]"
```

---

## how to run the server

the normal command is:

```bash
sktime-mcp
```

if that command is not on your `PATH`, use:

```bash
python -m sktime_mcp.server
```

if you are using a virtual environment, it is often better to point your client directly at the venv python executable. that avoids confusion about which python installation is being used.

example:

```bash
/absolute/path/to/sktime-mcp/.venv/bin/python -m sktime_mcp.server
```

---

## claude desktop setup

add an mcp server entry to your `claude_desktop_config.json`.

common config locations:

- macos: `~/Library/Application Support/Claude/claude_desktop_config.json`
- linux: `~/.config/claude/claude_desktop_config.json`
- windows: `%APPDATA%\\Claude\\claude_desktop_config.json`

simple setup:

```json
{
  "mcpServers": {
    "sktime": {
      "command": "sktime-mcp"
    }
  }
}
```

more reliable setup when using a local repo and virtual environment:

```json
{
  "mcpServers": {
    "sktime": {
      "command": "/absolute/path/to/sktime-mcp/.venv/bin/python",
      "args": ["-m", "sktime_mcp.server"]
    }
  }
}
```

if claude desktop cannot find the command, this second version is usually the better choice.

---

## cursor or vscode style setup

different clients name the config file differently, but the important part is the same: define one stdio mcp server with a command and optional args.

example shape:

```json
{
  "mcpServers": {
    "sktime": {
      "command": "/absolute/path/to/sktime-mcp/.venv/bin/python",
      "args": ["-m", "sktime_mcp.server"]
    }
  }
}
```

if your client supports only a single command string, use the plain executable:

```bash
sktime-mcp
```

if you are unsure which form your client expects, start with the explicit python + `-m sktime_mcp.server` version.

---

## custom local agent setup

if you are wiring `sktime-mcp` into your own local agent project, prefer installing it into the same environment your agent uses:

```bash
pip install -e /absolute/path/to/sktime-mcp
```

if you are temporarily importing the source tree directly, a fallback is:

```bash
PYTHONPATH=/absolute/path/to/sktime-mcp/src python your_agent.py
```

that fallback is useful for experiments, but the editable install is cleaner and less fragile.

---

## quick sanity check

once your client is connected, try a tiny workflow:

1. ask for the available demo datasets
2. ask for 5 forecasting estimators
3. instantiate `NaiveForecaster`
4. forecast the `airline` dataset 3 steps ahead

under the hood this should map roughly to:

```json
{"tool": "list_available_data", "arguments": {"is_demo": true}}
```

```json
{"tool": "list_estimators", "arguments": {"task": "forecasting", "limit": 5}}
```

```json
{"tool": "instantiate_estimator", "arguments": {"estimator": "NaiveForecaster"}}
```

```json
{"tool": "fit_predict", "arguments": {"estimator_handle": "est_abc123", "dataset": "airline", "horizon": 3}}
```

if those work, the basic client wiring is in good shape.

---

## common problems

### `command not found: sktime-mcp`

this usually means the client is not using the python environment where `sktime-mcp` is installed.

fix:

- install into the correct environment
- or point the client directly at `/path/to/.venv/bin/python`

### `ModuleNotFoundError: No module named 'sktime_mcp'`

this usually means the package was not installed in the environment the client is using.

fix:

```bash
pip install -e .
```

or use the repo path explicitly:

```bash
pip install -e /absolute/path/to/sktime-mcp
```

### tools do not appear in the client

possible causes:

- the client config file is in the wrong location
- the json is invalid
- the command exits immediately
- the wrong python executable is being used

first try running the exact same command in a terminal yourself.

### file loading fails even though the file exists

for local files, use absolute paths.

good:

```text
/home/user/data/sales.csv
```

bad:

```text
data/sales.csv
```

### my custom data loads, but the model choice is bad

after `load_data_source`, inspect the returned metadata before choosing a model. fields like `rows`, `frequency`, `columns`, `exog_columns`, and `dtypes` help the agent decide whether the loaded handle matches the intended forecasting workflow.

---

## related pages

- [intro](intro.md)
- [user guide](user-guide.md)
- [data sources](data-sources.md)
- [usage examples](usage-examples.md)
