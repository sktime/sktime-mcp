"""
Tests for analyze_data tool.
"""

import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, "src")


class TestAnalyzeDataTool(unittest.TestCase):
    def setUp(self):
        from sktime_mcp.runtime.executor import Executor, get_executor
        
        # Override to an isolated executor for tests
        self.executor = get_executor()
        
        # Setup dummy data handle with a clear trend, seasonality, and stationarity (non-stationary really due to trend)
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        
        # 1. Stationary series
        y_stationary = pd.Series(np.random.randn(100), index=dates)
        
        # 2. Trending non-stationary series
        y_trend = pd.Series(np.linspace(0, 10, 100) + np.random.randn(100) * 0.1, index=dates)
        
        # 3. Seasonal series (sin wave)
        y_seasonal = pd.Series(np.sin(np.linspace(0, 20*np.pi, 100)) + np.random.randn(100) * 0.1, index=dates)
        
        self.handle_stat = "data_stat_123"
        self.handle_trend = "data_trend_123"
        self.handle_seas = "data_seas_123"
        
        self.executor._data_handles[self.handle_stat] = {"y": y_stationary}
        self.executor._data_handles[self.handle_trend] = {"y": y_trend}
        self.executor._data_handles[self.handle_seas] = {"y": y_seasonal}

    def tearDown(self):
        # Cleanup
        if self.handle_stat in self.executor._data_handles:
            del self.executor._data_handles[self.handle_stat]
        if self.handle_trend in self.executor._data_handles:
            del self.executor._data_handles[self.handle_trend]
        if self.handle_seas in self.executor._data_handles:
            del self.executor._data_handles[self.handle_seas]

    def test_analyze_data_stationary(self):
        from sktime_mcp.tools.analyze_data import analyze_data_tool
        result = analyze_data_tool(self.handle_stat)
        
        self.assertTrue(result["success"], f"Failed: {result.get('error')}")
        self.assertEqual(result["summary"]["length"], 100)
        self.assertEqual(result["summary"]["frequency"], "D")
        self.assertEqual(result["summary"]["missing_values"], 0)
        
        # Should be stationary
        self.assertTrue(result["stationarity"]["is_stationary"])
        
        # No significant trend
        self.assertFalse(result["trend"]["has_trend"])

    def test_analyze_data_trend(self):
        from sktime_mcp.tools.analyze_data import analyze_data_tool
        result = analyze_data_tool(self.handle_trend)
        
        self.assertTrue(result["success"])
        # Should have a trend
        self.assertTrue(result["trend"]["has_trend"])
        self.assertGreater(result["trend"]["slope"], 0)
        
        # Likely non-stationary due to strong trend
        self.assertFalse(result["stationarity"]["is_stationary"])

    def test_analyze_data_seasonal(self):
        from sktime_mcp.tools.analyze_data import analyze_data_tool
        result = analyze_data_tool(self.handle_seas)
        
        self.assertTrue(result["success"])
        # Should detect seasonality
        self.assertTrue(result["seasonality"]["has_seasonality"])
        self.assertIsNotNone(result["seasonality"]["strongest_seasonal_period"])

    def test_analyze_data_invalid_handle(self):
        from sktime_mcp.tools.analyze_data import analyze_data_tool
        result = analyze_data_tool("invalid_handle_999")
        self.assertFalse(result["success"])
        self.assertIn("Unknown data handle", result["error"])

if __name__ == "__main__":
    unittest.main()
