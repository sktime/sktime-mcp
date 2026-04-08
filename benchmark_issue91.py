"""Benchmark for _load_registry fix (issue #91).

Measures the per-call overhead of all_estimators with 8 typed calls vs 1 list call,
with sys.modules warm (sktime module tree already crawled on first import).

Usage: python benchmark_issue91.py
"""

import statistics
import time

import sktime
from sktime.registry import all_estimators

TASK_MAP = ["forecaster","transformer","classifier","regressor","clusterer","param_est","splitter","network"]

# Force module crawl once so both measurements start from the same warm state.
# This matches a running MCP server where sktime is already imported.
_ = all_estimators(return_names=True, as_dataframe=False)

# Before: 8-call loop (original _load_registry pattern)
times_loop = []
for _ in range(5):
    t0 = time.perf_counter()
    results = {}
    for t in TASK_MAP:
        results[t] = all_estimators(estimator_types=t, return_names=True, as_dataframe=False)
    times_loop.append(time.perf_counter() - t0)
total_loop = sum(len(v) for v in results.values())

# After: single call with estimator_types=list(TASK_MAP) (actual production call)
times_single = []
for _ in range(5):
    t0 = time.perf_counter()
    all_est = all_estimators(estimator_types=TASK_MAP, return_names=True, as_dataframe=False)
    times_single.append(time.perf_counter() - t0)

loop_med = statistics.median(times_loop) * 1000
single_med = statistics.median(times_single) * 1000

print(f"sktime         : {sktime.__version__}")
print(f"8-call loop    : {loop_med:.0f}ms  (median of 5) -> {sum(len(v) for v in results.values())} estimators")
print(f"1-call fix     : {single_med:.1f}ms  (median of 5) -> {len(all_est)} estimators")
print(f"Speedup        : {loop_med / single_med:.1f}x")
print()
print("Note: both measurements are warm-sys.modules (module tree already crawled).")
print("Cold first-run is dominated by sktime module loading (~5s on both paths).")
