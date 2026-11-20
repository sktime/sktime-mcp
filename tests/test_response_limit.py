"""Tests for response token limiting in the MCP server."""

import json
import os

import pytest

from sktime_mcp.server import _apply_response_token_limit


def _text(result) -> str:
    """Serialize exactly as ``call_tool`` does."""
    return json.dumps(result, indent=2, default=str)


class TestResponseTokenLimit:
    """Test suite for _apply_response_token_limit function."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Ensure the environment variable is clean before and after each test."""
        old_val = os.environ.get("SKTIME_MCP_MAX_RESPONSE_TOKENS")
        if old_val is not None:
            del os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"]
        yield
        if old_val is not None:
            os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = old_val
        elif "SKTIME_MCP_MAX_RESPONSE_TOKENS" in os.environ:
            del os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"]

    def test_default_unlimited(self):
        """By default (variable unset), responses are not truncated."""
        result = {"success": True, "data": "A" * 1000}
        assert _apply_response_token_limit("test_tool", result) is result

    @pytest.mark.parametrize("invalid_value", ["0", "-1", "-100", "invalid", "1.5", ""])
    def test_invalid_or_zero_limit(self, invalid_value):
        """Zero, negative, empty or non-integer values default to unlimited (no truncation)."""
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = invalid_value
        result = {"success": True, "data": "A" * 1000}
        assert _apply_response_token_limit("test_tool", result) is result

    def test_no_truncation_when_under_budget(self):
        """Responses below the estimated character budget are returned unchanged."""
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = "20"  # 20 * 4 = 80 chars budget
        result = {"data": "A" * 50}
        assert len(_text(result)) < 80
        assert _apply_response_token_limit("test_tool", result) is result

    def test_no_truncation_when_exactly_at_budget(self):
        """Responses exactly at the estimated character budget are returned unchanged."""
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = "20"  # 20 * 4 = 80 chars budget
        result = {"data": "A" * 50}
        result["data"] += "A" * (80 - len(_text(result)))
        assert len(_text(result)) == 80
        assert _apply_response_token_limit("test_tool", result) is result

    def test_truncation_when_exceeding_budget(self):
        """Responses exceeding the budget are shrunk to fit and carry the notice."""
        max_tokens = 200
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = str(max_tokens)  # 800 chars budget
        result = {"success": True, "data": "A" * 3000}

        limited = _apply_response_token_limit("my_tool", result)
        text = _text(limited)

        # Serialized output fits the character budget (200 * 4 = 800 chars)
        assert len(text) <= 800
        # ... and is still valid JSON with the envelope intact.
        parsed = json.loads(text)
        assert parsed["success"] is True
        assert parsed["data"].startswith("AAAA")

        # The notice names the token limit and the tool.
        notice = parsed["truncated"]["notice"]
        assert "[sktime-mcp] Response truncated" in notice
        assert f"limit of {max_tokens} tokens" in notice
        assert "(tool: my_tool)" in notice
        assert notice.endswith("narrow your query for full results.")

    def test_very_small_budget_still_returns_valid_json(self):
        """Budgets smaller than the notice itself cannot be met, but the result stays parseable."""
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = "5"  # 20 chars budget
        result = {"success": True, "data": "A" * 100}

        limited = _apply_response_token_limit("test_tool", result)
        parsed = json.loads(_text(limited))

        assert parsed["success"] is True
        assert "[sktime-mcp] Response truncated" in parsed["truncated"]["notice"]
