"""Tests for sanitize_for_json NumPy/Pandas type handling (issue #197).

The function is extracted from server.py via AST to avoid importing the full
server module (which has heavyweight dependencies like sktime/pyarrow).
"""
import ast
import importlib.util
import os
import textwrap
import types

import pytest

numpy = pytest.importorskip("numpy", reason="numpy not installed")
pandas = pytest.importorskip("pandas", reason="pandas not installed")


def _load_sanitize_for_json():
    """Extract and compile sanitize_for_json from server.py without loading server."""
    server_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "sktime_mcp", "server.py"
    )
    server_path = os.path.abspath(server_path)
    with open(server_path) as f:
        source = f.read()

    tree = ast.parse(source)

    # Find the sanitize_for_json function definition
    fn_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "sanitize_for_json":
            fn_node = node
            break

    assert fn_node is not None, "sanitize_for_json not found in server.py"

    # Build a minimal module with just this function
    module_tree = ast.Module(body=[fn_node], type_ignores=[])
    ast.fix_missing_locations(module_tree)
    code = compile(module_tree, server_path, "exec")

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    return ns["sanitize_for_json"]


# Load the function once for the module
_sanitize = _load_sanitize_for_json()


def get_fn():
    return _sanitize


class TestNativeTypes:
    def test_none(self):
        assert get_fn()(None) is None

    def test_bool(self):
        assert get_fn()(True) is True

    def test_int(self):
        assert get_fn()(42) == 42

    def test_float(self):
        assert get_fn()(3.14) == 3.14

    def test_str(self):
        assert get_fn()("hello") == "hello"

    def test_dict(self):
        assert get_fn()({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}

    def test_list(self):
        assert get_fn()([1, 2, 3]) == [1, 2, 3]

    def test_nested(self):
        assert get_fn()({"a": [1, {"b": 2}]}) == {"a": [1, {"b": 2}]}


class TestNumpyTypes:
    def test_np_int64(self):
        import numpy as np
        result = get_fn()(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_np_float64(self):
        import numpy as np
        result = get_fn()(np.float64(3.14))
        assert abs(result - 3.14) < 1e-10
        assert isinstance(result, float)

    def test_np_bool(self):
        import numpy as np
        result = get_fn()(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_np_nan(self):
        import numpy as np
        assert get_fn()(np.float64("nan")) is None

    def test_np_inf(self):
        import numpy as np
        assert get_fn()(np.float64("inf")) is None
        assert get_fn()(np.float64("-inf")) is None

    def test_np_ndarray(self):
        import numpy as np
        result = get_fn()(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_np_ndarray_2d(self):
        import numpy as np
        result = get_fn()(np.array([[1, 2], [3, 4]]))
        assert result == [[1, 2], [3, 4]]

    def test_dict_with_numpy_values(self):
        import numpy as np
        result = get_fn()({"count": np.int64(5), "mean": np.float64(2.5)})
        assert result == {"count": 5, "mean": 2.5}
        assert isinstance(result["count"], int)
        assert isinstance(result["mean"], float)


class TestPandasTypes:
    def test_timestamp(self):
        import pandas as pd
        result = get_fn()(pd.Timestamp("2024-01-15"))
        assert result == "2024-01-15T00:00:00"

    def test_nat(self):
        import pandas as pd
        assert get_fn()(pd.NaT) is None

    def test_na(self):
        import pandas as pd
        assert get_fn()(pd.NA) is None

    def test_series(self):
        import pandas as pd
        result = get_fn()(pd.Series([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = get_fn()(df)
        assert result == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


class TestCircularReference:
    def test_circular_dict(self):
        d = {"x": 1}
        d["self"] = d
        result = get_fn()(d)
        # Should not raise RecursionError
        assert result["x"] == 1
        assert result["self"] == "<circular reference>"

    def test_circular_list(self):
        lst = [1, 2]
        lst.append(lst)
        result = get_fn()(lst)
        assert result[0] == 1
        assert result[2] == "<circular reference>"

    def test_no_false_positive_sibling(self):
        """Same list appearing in two sibling keys must NOT be deduplicated."""
        shared = [1, 2, 3]
        d = {"a": shared, "b": shared}
        result = get_fn()(d)
        # Both must be the real list, not a circular-reference sentinel
        assert result["a"] == [1, 2, 3]
        assert result["b"] == [1, 2, 3]

    def test_no_false_positive_small_ints(self):
        """CPython caches small ints; same id() must not deduplicate them."""
        result = get_fn()({"a": 42, "b": 42})
        assert result == {"a": 42, "b": 42}
