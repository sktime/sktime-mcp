import pytest
import json
import io
import asyncio
from contextlib import redirect_stdout, redirect_stderr
from mcp.types import TextContent
from mcp.shared.memory import create_connected_server_and_client_session
from sktime_mcp.server import server

CALL_TIMEOUT = 5.0


def validate_mcp_response(content, required_keys=None, strict=False):
    assert isinstance(content, list) and len(content) > 0
    block = content[0]
    assert isinstance(block, TextContent)

    try:
        data = json.loads(block.text)
    except json.JSONDecodeError:
        pytest.fail("Invalid JSON response")

    assert isinstance(data, dict)
    assert "success" in data

    if data["success"]:
        if required_keys:
            actual_keys = set(data.keys())
            missing = set(required_keys) - actual_keys
            assert not missing, f"Missing keys: {missing}"
            if strict:
                extra = actual_keys - set(required_keys) - {"success"}
                assert not extra, f"Unexpected keys: {extra}"
    else:
        assert "error" in data or "message" in data

    return data


async def test_transport_stream_cleanliness():
    f_out, f_err = io.StringIO(), io.StringIO()

    with redirect_stdout(f_out), redirect_stderr(f_err):
        async with create_connected_server_and_client_session(server) as session:
            await asyncio.wait_for(session.list_tools(), timeout=CALL_TIMEOUT)
            await asyncio.wait_for(
                session.call_tool("list_estimators", arguments={"limit": 1}),
                timeout=CALL_TIMEOUT,
            )

    assert f_out.getvalue() == ""
    assert f_err.getvalue() == ""


async def test_tool_list_not_empty():
    async with create_connected_server_and_client_session(server) as session:
        result = await asyncio.wait_for(session.list_tools(), timeout=CALL_TIMEOUT)

    assert isinstance(result.tools, list)
    assert len(result.tools) > 0


async def test_all_tools_have_descriptions():
    async with create_connected_server_and_client_session(server) as session:
        result = await asyncio.wait_for(session.list_tools(), timeout=CALL_TIMEOUT)

    assert len(result.tools) > 0
    for tool in result.tools:
        assert tool.description and tool.description.strip()


async def test_tool_error_response_is_structured():
    async with create_connected_server_and_client_session(server) as session:
        result = await asyncio.wait_for(
            session.call_tool(
                "instantiate_estimator",
                arguments={"estimator": "ThisEstimatorDefinitelyDoesNotExist_xyz123"},
            ),
            timeout=CALL_TIMEOUT,
        )

    data = validate_mcp_response(result.content)

    assert data["success"] is False
    assert "error" in data or "message" in data


async def test_list_estimators_returns_valid_json():
    async with create_connected_server_and_client_session(server) as session:
        result = await asyncio.wait_for(
            session.call_tool("list_estimators", arguments={}),
            timeout=CALL_TIMEOUT,
        )

    assert isinstance(result.content, list) and len(result.content) > 0
    block = result.content[0]
    assert isinstance(block, TextContent)

    try:
        parsed = json.loads(block.text)
    except json.JSONDecodeError as exc:
        pytest.fail(f"Invalid JSON: {exc}")

    try:
        json.dumps(parsed)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"Non-serializable data: {exc}")


async def test_concurrency_handle_uniqueness():
    async with create_connected_server_and_client_session(server) as session:
        results = await asyncio.gather(
            session.call_tool("instantiate_estimator", arguments={"estimator": "ARIMA"}),
            session.call_tool("instantiate_estimator", arguments={"estimator": "NaiveForecaster"}),
        )

        handles = []
        for res in results:
            data = validate_mcp_response(res.content, required_keys=["handle"])
            handles.append(data["handle"])

        assert len(set(handles)) == len(handles)

        await asyncio.gather(
            *(session.call_tool("release_handle", arguments={"handle": h}) for h in handles)
        )


async def test_minimal_tool_contracts():
    async with create_connected_server_and_client_session(server) as session:
        reg_result = await asyncio.wait_for(
            session.call_tool("list_estimators", arguments={"limit": 1}),
            timeout=CALL_TIMEOUT,
        )
        validate_mcp_response(reg_result.content, required_keys=["estimators"])

        inst_result = await asyncio.wait_for(
            session.call_tool(
                "instantiate_estimator", arguments={"estimator": "NaiveForecaster"}
            ),
            timeout=CALL_TIMEOUT,
        )
        validate_mcp_response(inst_result.content, required_keys=["handle"])