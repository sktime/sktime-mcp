"""
compare_estimators tool for sktime MCP.

Evaluates multiple estimators on a dataset and ranks them
by cross-validation performance. This is the core building block for
agentic model selection workflows.
"""

import contextlib
import logging
from typing import Any

from sktime.forecasting.model_evaluation import evaluate

try:
    from sktime.split import ExpandingWindowSplitter
except ImportError:  # pragma: no cover - sktime < 0.29
    from sktime.forecasting.model_selection import ExpandingWindowSplitter

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager

logger = logging.getLogger(__name__)

# Estimators known to work reliably with standard demo datasets
_SAFE_BASELINE_ESTIMATORS = [
    "NaiveForecaster",
    "ExponentialSmoothing",
    "AutoETS",
    "ARIMA",
    "ThetaForecaster",
]


def compare_estimators_tool(
    dataset: str,
    estimator_names: list[str] | None = None,
    cv_folds: int = 3,
    metric: str = "test_MeanAbsolutePercentageError",
) -> dict[str, Any]:
    """
    Compare multiple estimators on a dataset and rank by cross-validation performance.

    Runs evaluate_estimator on each model in ``estimator_names`` using an
    ExpandingWindowSplitter, then ranks models by the requested metric.
    Returns the best model, a full leaderboard, and per-model metrics.

    This tool is the core building block for agentic model selection:
    an LLM agent can call ``list_estimators_tool`` to discover candidates,
    then call ``compare_estimators_tool`` to automatically identify the
    best performer — no manual iteration required.

    Args:
        dataset: Name of a demo dataset (e.g., "airline", "sunspots", "lynx").
        estimator_names: List of sktime estimator class names to compare.
            If None or empty, defaults to a curated set of fast baseline
            forecasters: NaiveForecaster, ExponentialSmoothing, ThetaForecaster,
            AutoETS, ARIMA.
        cv_folds: Number of expanding-window cross-validation folds (default: 3).
            Must be a positive integer.
        metric: Column name from the evaluate() output to rank by
            (default: "test_MeanAbsolutePercentageError").

    Returns:
        Dictionary with:
        - success: bool
        - best_model: name of the top-ranked estimator
        - best_score: metric value for the best model (lower is better)
        - leaderboard: list of dicts [{rank, estimator, <metric>, status}],
          sorted best-first
        - metric_used: the metric column used for ranking
        - cv_folds: number of folds used
        - failed: list of estimator names that raised errors during evaluation

    Example::

        >>> compare_estimators_tool(
        ...     dataset="airline",
        ...     estimator_names=["NaiveForecaster", "ThetaForecaster", "ARIMA"],
        ...     cv_folds=3,
        ... )
        {
            "success": True,
            "best_model": "ThetaForecaster",
            "best_score": 3.21,
            "leaderboard": [
                {"rank": 1, "estimator": "ThetaForecaster",
                 "test_MeanAbsolutePercentageError": 3.21, "status": "ok"},
                {"rank": 2, "estimator": "ARIMA",
                 "test_MeanAbsolutePercentageError": 4.87, "status": "ok"},
                {"rank": 3, "estimator": "NaiveForecaster",
                 "test_MeanAbsolutePercentageError": 9.14, "status": "ok"},
            ],
            "metric_used": "test_MeanAbsolutePercentageError",
            "cv_folds": 3,
            "failed": []
        }
    """
    # --- Input validation ---
    if not dataset or not str(dataset).strip():
        return {
            "success": False,
            "error": "dataset must be a non-empty string (e.g. 'airline').",
        }

    if cv_folds < 1:
        return {
            "success": False,
            "error": "cv_folds must be a positive integer.",
        }

    candidates = estimator_names if estimator_names else _SAFE_BASELINE_ESTIMATORS

    if not isinstance(candidates, list) or len(candidates) == 0:
        return {
            "success": False,
            "error": (
                "estimator_names must be a non-empty list of estimator class name "
                'strings, e.g. ["NaiveForecaster", "ThetaForecaster"].'
            ),
        }

    if not all(isinstance(n, str) and n.strip() for n in candidates):
        return {
            "success": False,
            "error": "All entries in estimator_names must be non-empty strings.",
        }

    # --- Load dataset once ---
    executor = get_executor()
    data_result = executor.load_dataset(dataset)
    if not data_result["success"]:
        return data_result

    y = data_result["data"]
    X = data_result.get("exog")
    n = len(y)

    folds = max(1, min(int(cv_folds), max(1, n - 1)))
    initial_window = max(1, n - folds)
    cv = ExpandingWindowSplitter(initial_window=initial_window, step_length=1, fh=[1])

    # --- Evaluate each estimator ---
    handle_manager = get_handle_manager()
    results = []
    failed = []

    for name in candidates:
        try:
            # Instantiate fresh copy for each evaluation
            handle_result = executor.instantiate(name, params=None)
            if not handle_result.get("success", False):
                raise RuntimeError(handle_result.get("error", "instantiation failed"))

            handle_id = handle_result["handle"]
            instance = handle_manager.get_instance(handle_id)

            # Run cross-validation
            cv_results = evaluate(forecaster=instance, y=y, X=X, cv=cv)

            if "estimator" in cv_results.columns:
                cv_results = cv_results.drop(columns=["estimator"])

            # Compute mean score across folds
            if metric in cv_results.columns:
                score = float(cv_results[metric].mean())
            else:
                # fallback: use first numeric column starting with "test_"
                test_cols = [c for c in cv_results.columns if c.startswith("test_")]
                if not test_cols:
                    raise ValueError(f"Metric '{metric}' not found and no test_ columns available.")
                score = float(cv_results[test_cols[0]].mean())

            results.append({"estimator": name, "score": score, "status": "ok"})

            # Release temporary handle to avoid memory bloat
            with contextlib.suppress(Exception):
                handle_manager.release(handle_id)

        except Exception as e:
            logger.warning(f"compare_estimators: {name} failed — {e}")
            failed.append({"estimator": name, "error": str(e)})

    if not results:
        return {
            "success": False,
            "error": (
                "All estimators failed evaluation. "
                f"Failed: {[f['estimator'] for f in failed]}. "
                "Check estimator names are valid sktime forecaster class names."
            ),
            "failed": failed,
        }

    # --- Rank by score (lower is better for error metrics) ---
    results.sort(key=lambda x: x["score"])

    leaderboard = [
        {
            "rank": i + 1,
            "estimator": r["estimator"],
            metric: round(r["score"], 6),
            "status": r["status"],
        }
        for i, r in enumerate(results)
    ]

    best = results[0]

    return {
        "success": True,
        "best_model": best["estimator"],
        "best_score": round(best["score"], 6),
        "leaderboard": leaderboard,
        "metric_used": metric,
        "cv_folds": folds,
        "dataset": dataset,
        "failed": [f["estimator"] for f in failed],
    }
