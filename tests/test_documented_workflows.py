"""Keep executable documentation examples and advertised tool schemas in agreement."""

import json
import re
from pathlib import Path

import jsonschema

from sktime_mcp.runtime.executor import Executor
from sktime_mcp.server import get_tools

DOCS = Path(__file__).resolve().parents[1] / "docs" / "source"


def test_documented_tool_calls_match_current_schemas():
    tools = {tool.name: tool for tool in get_tools()}
    checked = 0
    for page in DOCS.rglob("*.md"):
        for block in re.findall(r"```json\n(.*?)\n```", page.read_text(), flags=re.S):
            example = json.loads(block)
            if isinstance(example, dict) and "name" in example and "arguments" in example:
                assert example["name"] in tools, (page, example)
                jsonschema.validate(example["arguments"], tools[example["name"]].input_schema)
                checked += 1
    assert checked >= 10


def test_documented_python_workflows_execute(monkeypatch):
    executor = Executor()
    monkeypatch.setattr("sktime_mcp.runtime.executor._executor_instance", executor)
    before = set(executor._handle_manager._handles)
    try:
        examples = re.findall(
            r"```python\n(.*?)\n```", (DOCS / "user-guide.md").read_text(), flags=re.S
        )
        assert len(examples) == 2
        for example in examples:
            exec(compile(example, "documented-workflow", "exec"), {})
    finally:
        for handle in set(executor._handle_manager._handles) - before:
            executor._handle_manager.release_handle(handle)
