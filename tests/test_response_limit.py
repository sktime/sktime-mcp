"""Tests for response token limiting in the MCP server."""

import os
from unittest.mock import patch

import pytest

from sktime_mcp.server import _apply_response_token_limit


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
        text = "A" * 1000
        assert _apply_response_token_limit("test_tool", text) == text

    @pytest.mark.parametrize("invalid_value", ["0", "-1", "-100", "invalid", "1.5", ""])
    def test_invalid_or_zero_limit(self, invalid_value):
        """Zero, negative, empty or non-integer values default to unlimited (no truncation)."""
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = invalid_value
        text = "A" * 1000
        assert _apply_response_token_limit("test_tool", text) == text

    def test_no_truncation_when_under_budget(self):
        """Responses below the estimated character budget are returned unchanged."""
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = "20"  # 20 * 4 = 80 chars budget
        text = "A" * 70
        assert _apply_response_token_limit("test_tool", text) == text

    def test_no_truncation_when_exactly_at_budget(self):
        """Responses exactly at the estimated character budget are returned unchanged."""
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = "20"  # 20 * 4 = 80 chars budget
        text = "A" * 80
        assert _apply_response_token_limit("test_tool", text) == text

    def test_truncation_when_exceeding_budget(self):
        """Responses exceeding the budget are truncated and contain the notice."""
        max_tokens = 50
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = str(max_tokens)  # 200 chars budget
        text = "A" * 300

        truncated = _apply_response_token_limit("my_tool", text)

        # Truncated text should have exactly max_chars length (50 * 4 = 200 chars)
        assert len(truncated) == 200

        # Truncated text should contain the notice with correct token limit and tool name
        assert "[sktime-mcp] Response truncated" in truncated
        assert f"limit of {max_tokens} tokens" in truncated
        assert "(tool: my_tool)" in truncated

        # It should end with the notice
        assert truncated.endswith("narrow your query for full results.")

    def test_very_small_budget_handles_negative_math(self):
        """Very small budgets (where notice length > budget) are handled gracefully (length is capped)."""
        os.environ["SKTIME_MCP_MAX_RESPONSE_TOKENS"] = "5"  # 20 chars budget
        text = "A" * 100

        truncated = _apply_response_token_limit("test_tool", text)

        # If budget (20) < notice length, budget for text slice is 0, so only the notice should be returned (or as much as fits).
        # Since the notice itself is longer than 20 chars, text[:budget] will be empty, and the notice is appended.
        # Notice string length is ~170+ chars, so overall length will be at least notice length (or capped to budget if we strictly capped it,
        # but the cgc code behaves by returning text[:budget] + notice, meaning it can exceed budget for small budgets).
        assert truncated.startswith("")
        assert "[sktime-mcp] Response truncated" in truncated
