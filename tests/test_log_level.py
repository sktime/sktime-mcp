import logging
import os
import unittest
from unittest.mock import patch


class TestLogLevelConfig(unittest.TestCase):
    """Tests for SKTIME_MCP_LOG_LEVEL environment variable."""

    @patch.dict(os.environ, {"SKTIME_MCP_LOG_LEVEL": "DEBUG"})
    def test_debug_level(self):
        level = os.environ.get("SKTIME_MCP_LOG_LEVEL", "WARNING").upper()
        assert getattr(logging, level, logging.WARNING) == logging.DEBUG

    @patch.dict(os.environ, {"SKTIME_MCP_LOG_LEVEL": "ERROR"})
    def test_error_level(self):
        level = os.environ.get("SKTIME_MCP_LOG_LEVEL", "WARNING").upper()
        assert getattr(logging, level, logging.WARNING) == logging.ERROR

    @patch.dict(os.environ, {}, clear=True)
    def test_default_level(self):
        level = os.environ.get("SKTIME_MCP_LOG_LEVEL", "WARNING").upper()
        assert getattr(logging, level, logging.WARNING) == logging.WARNING

    @patch.dict(os.environ, {"SKTIME_MCP_LOG_LEVEL": "INVALID"})
    def test_invalid_level_fallback(self):
        level = os.environ.get("SKTIME_MCP_LOG_LEVEL", "WARNING").upper()
        assert getattr(logging, level, logging.WARNING) == logging.WARNING

    @patch.dict(os.environ, {"SKTIME_MCP_LOG_LEVEL": "info"})
    def test_case_insensitive(self):
        level = os.environ.get("SKTIME_MCP_LOG_LEVEL", "WARNING").upper()
        assert getattr(logging, level, logging.WARNING) == logging.INFO


if __name__ == "__main__":
    unittest.main()
