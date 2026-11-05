"""Fractional holdouts follow sktime's temporal split rounding (#556, F-52)."""

import pandas as pd
import pytest
from sktime.split import temporal_train_test_split

from sktime_mcp.runtime.executor import Executor
from sktime_mcp.tools.split_data import split_data_tool


def _register_series(
    monkeypatch: pytest.MonkeyPatch, n_rows: int, with_exog: bool
) -> tuple[Executor, pd.Series, pd.DataFrame | None]:
    """Register synthetic monthly data in an isolated executor."""
    executor = Executor()
    y = pd.Series(
        range(n_rows), index=pd.period_range("2026-01", periods=n_rows, freq="M"), name="sales"
    )
    X = pd.DataFrame({"promotion": range(n_rows)}, index=y.index) if with_exog else None
    executor._register_data_handle("source", {"y": y, "X": X})
    monkeypatch.setattr("sktime_mcp.tools.split_data.get_executor", lambda: executor)
    return executor, y, X


@pytest.mark.parametrize("with_exog", [False, True])
@pytest.mark.parametrize("n_rows,test_size", [(11, 0.2), (11, 0.5), (10, 0.2), (11, 0.01)])
def test_fractional_split_matches_sktime(
    monkeypatch: pytest.MonkeyPatch, n_rows: int, test_size: float, with_exog: bool
) -> None:
    """Match sktime's split rows and preserve aligned exogenous data."""
    executor, y, X = _register_series(monkeypatch, n_rows, with_exog)
    expected = temporal_train_test_split(y, X, test_size=test_size)

    result = split_data_tool("source", test_size=test_size)

    assert result["success"], result
    train = executor._data_handles[result["train_handle"]]
    test = executor._data_handles[result["test_handle"]]
    pd.testing.assert_series_equal(train["y"], expected[0])
    pd.testing.assert_series_equal(test["y"], expected[1])
    assert result["train_size"] == len(expected[0])
    assert result["n_test"] == len(expected[1])
    assert result["cutoff"] == str(expected[0].index[-1])
    assert train["metadata"]["rows"] == len(expected[0])
    assert test["metadata"]["rows"] == len(expected[1])
    assert executor._data_handles["source"]["y"] is y
    assert executor._data_handles["source"]["X"] is X
    if with_exog:
        pd.testing.assert_frame_equal(train["X"], expected[2])
        pd.testing.assert_frame_equal(test["X"], expected[3])
    else:
        assert train["X"] is None
        assert test["X"] is None


def test_fractional_split_rejects_empty_training_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a rounded holdout that consumes the series without adding handles."""
    executor, _, _ = _register_series(monkeypatch, n_rows=11, with_exog=False)

    result = split_data_tool("source", test_size=0.99)

    assert result["success"] is False
    assert "Test set would be 11 samples" in result["error"]
    assert list(executor._data_handles) == ["source"]
