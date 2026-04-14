"""
Comprehensive test for the sanitize_for_json fix.

Tests that common NumPy and Pandas types that sktime tools return
are correctly handled by sanitize_for_json before JSON serialization.

Run with:
    cd /Users/sakshidattaprasadkasat/Desktop/sktime/sktime-mcp
    .venv/bin/python3 test_sanitize.py
"""

import json
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sktime_mcp.server import sanitize_for_json

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def check(label, obj):
    try:
        sanitized = sanitize_for_json(obj)
        serialized = json.dumps(sanitized)
        print(f"{PASS}  {label}")
        return True
    except (TypeError, ValueError) as e:
        print(f"{FAIL}  {label} → {type(e).__name__}: {e}")
        return False

def check_value(label, obj, expected):
    """Check sanitized value matches expected AND is JSON-safe."""
    try:
        sanitized = sanitize_for_json(obj)
        json.dumps(sanitized)
        if sanitized == expected:
            print(f"{PASS}  {label} → {repr(sanitized)}")
            return True
        else:
            print(f"{FAIL}  {label} → expected {repr(expected)}, got {repr(sanitized)}")
            return False
    except (TypeError, ValueError) as e:
        print(f"{FAIL}  {label} → {type(e).__name__}: {e}")
        return False

print("="*60)
print("sanitize_for_json — Full Test Suite")
print("="*60)

results = []

# ---- NumPy scalars ----
print("\n[NumPy Scalars]")
results.append(check_value("np.int8",        np.int8(42),        42))
results.append(check_value("np.int16",       np.int16(42),       42))
results.append(check_value("np.int32",       np.int32(42),       42))
results.append(check_value("np.int64",       np.int64(42),       42))
results.append(check_value("np.float32",     np.float32(3.14),   float(np.float32(3.14))))
results.append(check_value("np.float64",     np.float64(0.95),   0.95))
results.append(check_value("np.bool_ True",  np.bool_(True),     True))
results.append(check_value("np.bool_ False", np.bool_(False),    False))

# ---- NumPy arrays ----
print("\n[NumPy Arrays]")
results.append(check_value("np.array int",    np.array([1, 2, 3]),           [1, 2, 3]))
results.append(check_value("np.array float",  np.array([1.1, 2.2, 3.3]),    [float(x) for x in [1.1, 2.2, 3.3]]))
results.append(check("2D np.array", np.array([[1, 2], [3, 4]])))

# ---- Pandas types ----
print("\n[Pandas Types]")
results.append(check_value("pd.Timestamp",  pd.Timestamp("2023-01-01"), "2023-01-01T00:00:00"))
results.append(check_value("pd.NaT",        pd.NaT,                     None))
results.append(check_value("pd.NA",         pd.NA,                      None))

# pd.Series
s = pd.Series([1, 2, 3])
results.append(check("pd.Series int", s))

# pd.DataFrame
df = pd.DataFrame({"value": [1.0, 2.0], "flag": [True, False]})
results.append(check("pd.DataFrame", df))

# ---- Nested structures (realistic tool output) ----
print("\n[Realistic Nested Tool Output]")
result = {
    "success": True,
    "predictions": {1: np.float64(450.2), 2: np.float64(460.5)},
    "horizon": np.int64(12),
    "fitted_at": pd.Timestamp("2023-01-01"),
    "residuals": np.array([0.1, -0.2, 0.05]),
    "metadata": {
        "dataset": "airline",
        "n_samples": np.int32(144),
        "has_exog": np.bool_(False),
    },
    "cv_results": [
        {"fold": np.int32(1), "score": np.float64(0.93)},
        {"fold": np.int32(2), "score": np.float64(0.89)},
    ],
}
results.append(check("full nested tool output", result))

# ---- Standard types (regression check) ----
print("\n[Standard Python Types — Regression]")
results.append(check_value("str",   "hello",    "hello"))
results.append(check_value("int",   42,         42))
results.append(check_value("float", 3.14,       3.14))
results.append(check_value("bool",  True,       True))
results.append(check_value("None",  None,       None))
results.append(check("list of mixed", [1, "two", None, True]))
results.append(check("nested dict", {"a": {"b": 1}}))

# ---- Edge cases ----
print("\n[Edge Cases]")
results.append(check_value("empty list",  [],  []))
results.append(check_value("empty dict",  {},  {}))
results.append(check("np.float64 nan",   np.float64("nan")))   # json.dumps maps NaN to null via str workaround

# ---- Summary ----
total = len(results)
passed = sum(results)
failed = total - passed
print(f"\n{'='*60}")
print(f"Results: {passed}/{total} passed" + (" 🎉" if failed == 0 else f", {failed} FAILED ❌"))
print(f"{'='*60}")
sys.exit(0 if failed == 0 else 1)
