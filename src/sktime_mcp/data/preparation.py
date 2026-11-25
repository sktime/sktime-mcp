"""Explicit, non-mutating data preparation using sktime's conversion contracts."""

from copy import deepcopy
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from sktime.datatypes import convert_to

from .contracts import MTYPE_SCITYPE, TEMPORAL_SCITYPES, describe_data, validate_data

_CANONICAL = {
    "Series": "pd.DataFrame",
    "Panel": "pd-multiindex",
    "Hierarchical": "pd_multiindex_hier",
    "Table": "pd_DataFrame_Table",
}


def canonical(data: Any, scitype: str, store: dict | None = None) -> pd.DataFrame:
    """Use sktime converters to obtain an operation-appropriate pandas view."""
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, pd.DataFrame) and not any(
        isinstance(v, (pd.Series, np.ndarray)) for v in data.head(1).to_numpy().flat
    ):
        return data.copy(deep=True)
    if scitype not in _CANONICAL:
        raise ValueError(
            f"This operation is not defined for {scitype}; use convert or an appropriate native method"
        )
    return convert_to(
        data, _CANONICAL[scitype], as_scitype=scitype, store=store, store_behaviour="update"
    )


def restore(frame: pd.DataFrame, original: Any, scitype: str, store: dict) -> Any:
    """Restore the original representation when it remains representable."""
    if isinstance(original, pd.Series):
        result = frame.iloc[:, 0]
        result.name = original.name
        return result
    if isinstance(original, pd.DataFrame) and not any(
        isinstance(v, (pd.Series, np.ndarray)) for v in original.head(1).to_numpy().flat
    ):
        return frame
    info = validate_data(original, scitype)["sktime"]
    mtype = info.get("mtype")
    if mtype is None:
        raise ValueError(
            "Cannot restore an invalid original representation; repair a pandas container first"
        )
    return convert_to(frame, mtype, as_scitype=scitype, store=store, store_behaviour="freeze")


def convert_component(data: Any, scitype: str, to_mtype: str) -> tuple[Any, dict]:
    if to_mtype not in MTYPE_SCITYPE:
        raise ValueError(f"Unknown sktime mtype {to_mtype!r}")
    if MTYPE_SCITYPE[to_mtype] != scitype:
        raise ValueError(
            f"Conversion preserves scitype {scitype}; {to_mtype} belongs to {MTYPE_SCITYPE[to_mtype]}"
        )
    before = describe_data(data, scitype)
    if not before["validation"]["valid"]:
        raise ValueError(f"Repair invalid data before converting: {before['validation']['errors']}")
    result = convert_to(deepcopy(data), to_mtype, as_scitype=scitype)
    after = describe_data(result, scitype)
    if not after["validation"]["valid"]:
        raise ValueError(
            f"Converted data failed sktime validation: {after['validation']['errors']}"
        )
    warnings = []
    if isinstance(data, (pd.Series, pd.DataFrame)) and isinstance(result, (np.ndarray, list)):
        warnings.append(
            "The requested representation may omit index/column labels; the original handle is retained."
        )
    return result, {"from_mtype": before["mtype"], "to_mtype": to_mtype, "warnings": warnings}


def format_component(
    data: Any,
    scitype: str,
    *,
    sort_index=False,
    duplicates="keep",
    missing="keep",
    fill_value=None,
    frequency=None,
    regularize=False,
    index_type="keep",
    max_rows=1_000_000,
) -> tuple[Any, dict]:
    """Repair only requested properties, separately within each individual series."""
    if scitype not in _CANONICAL:
        raise ValueError(f"format supports Series, Panel, Hierarchical and Table, not {scitype}")
    if duplicates not in {"keep", "first", "last", "drop", "error"}:
        raise ValueError("duplicates must be keep, first, last, drop or error")
    if missing not in {"keep", "ffill", "bfill", "interpolate", "constant"}:
        raise ValueError("missing must be keep, ffill, bfill, interpolate or constant")
    if index_type not in {"keep", "period"}:
        raise ValueError("index_type must be keep or period")
    if scitype == "Table" and (frequency or regularize or index_type != "keep"):
        raise ValueError("Time frequency/index options do not apply to Table data")
    if regularize and not frequency:
        raise ValueError("regularize requires an explicit frequency")
    if missing == "constant" and fill_value is None:
        raise ValueError("missing='constant' requires fill_value")
    store = {}
    frame = canonical(data, scitype, store)
    report = {
        "sorted": False,
        "duplicates_removed": 0,
        "missing_filled": 0,
        "gaps_filled": 0,
        "frequency_set": False,
        "index_converted": False,
    }
    if (
        not sort_index
        and duplicates == "keep"
        and missing == "keep"
        and not frequency
        and index_type == "keep"
    ):
        return deepcopy(data), report
    used_rows = 0

    def repair(group):
        nonlocal used_rows
        result = group.copy()
        if sort_index and not result.index.is_monotonic_increasing:
            result = result.sort_index(kind="stable")
            report["sorted"] = True
        if duplicates == "error" and result.index.has_duplicates:
            raise ValueError("Duplicate index entries found; choose an explicit removal policy")
        if duplicates not in {"keep", "error"}:
            mask = ~result.index.duplicated(keep=False if duplicates == "drop" else duplicates)
            report["duplicates_removed"] += int((~mask).sum())
            result = result.loc[mask]
        if frequency:
            from pandas.tseries.frequencies import to_offset

            from .base import check_frequency_alignment

            if not isinstance(result.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                raise ValueError("frequency requires a datetime or period time index")
            offset = (
                pd.Period("2000-01-01", freq=frequency).freq
                if isinstance(result.index, pd.PeriodIndex)
                else to_offset(frequency)
            )
            if offset.n <= 0:
                raise ValueError("frequency must be positive")
            if not result.index.is_unique or not result.index.is_monotonic_increasing:
                raise ValueError(
                    "frequency requires sorted, unique times; request sort_index and a duplicate policy"
                )
            check_frequency_alignment(result.index, frequency)
            if regularize and len(result):
                # Construct at most the remaining budget plus one, never an unbounded date_range.
                budget = max_rows - used_rows
                if budget < len(result):
                    raise ValueError(f"Preparation exceeds maximum of {max_rows} rows")
                start, end = result.index.min(), result.index.max()
                if isinstance(result.index, pd.PeriodIndex):
                    if start.freq != offset:
                        raise ValueError("Regularization must preserve the PeriodIndex frequency")
                    count = (end.ordinal - start.ordinal) // offset.n + 1
                    if count > budget:
                        raise ValueError(f"Frequency expansion exceeds maximum of {max_rows} rows")
                    full = pd.period_range(
                        start=start, periods=count, freq=frequency, name=result.index.name
                    )
                else:
                    try:
                        count_start, count_end = start, end
                        if start.tz is not None and isinstance(offset, pd.offsets.Day):
                            count_start, count_end = start.tz_localize(None), end.tz_localize(None)
                        count = (int(count_end.value) - int(count_start.value)) // offset.nanos + 1
                    except ValueError:
                        count, current = 0, start
                        while current <= end:
                            count += 1
                            if count > budget:
                                raise ValueError(
                                    f"Frequency expansion exceeds maximum of {max_rows} rows"
                                ) from None
                            following = current + offset
                            if following <= current:
                                raise ValueError("frequency must advance time") from None
                            current = following
                    if count > budget:
                        raise ValueError(f"Frequency expansion exceeds maximum of {max_rows} rows")
                    full = pd.date_range(
                        start=start, periods=count, freq=frequency, name=result.index.name
                    )
                if not result.index.isin(full).all():
                    raise ValueError("Frequency does not align with every observed timestamp")
                report["gaps_filled"] += len(full) - len(result)
                result = result.reindex(full)
            elif isinstance(result.index, pd.DatetimeIndex):
                result.index = pd.DatetimeIndex(result.index, freq=frequency)
            elif result.index.freq != offset:
                raise ValueError("Frequency differs from the existing PeriodIndex")
            report["frequency_set"] = True
        before = int(result.isna().sum().sum())
        if missing == "ffill":
            result = result.ffill()
        elif missing == "bfill":
            result = result.bfill()
        elif missing == "constant":
            result = result.fillna(fill_value)
        elif missing == "interpolate":
            numeric = result.select_dtypes(include="number").columns
            result[numeric] = result[numeric].interpolate(limit_area="inside")
        report["missing_filled"] += before - int(result.isna().sum().sum())
        if index_type == "period":
            if not isinstance(result.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                raise ValueError("index_type='period' requires a datetime or period time index")
            if isinstance(result.index, pd.DatetimeIndex):
                if result.index.tz is not None:
                    raise ValueError(
                        "PeriodIndex cannot preserve timezones; remove the timezone explicitly before converting"
                    )
                # pandas translates datetime aliases (e.g. MS/ME) to the
                # corresponding period alias (M) from the established grid.
                converted = result.index.to_period()
                if converted.has_duplicates and not result.index.has_duplicates:
                    raise ValueError("Period conversion would collapse distinct timestamps")
                result.index = converted
                report["index_converted"] = True
        used_rows += len(result)
        return result

    if isinstance(frame.index, pd.MultiIndex) and scitype in {"Panel", "Hierarchical"}:
        levels = list(range(frame.index.nlevels - 1))
        parts, keys, positions_after = [], [], []
        positions = pd.Series(np.arange(len(frame)), index=frame.index)
        for key, group_positions in positions.groupby(
            level=levels, sort=False, observed=True, dropna=False
        ):
            group = frame.iloc[group_positions.to_numpy()].droplevel(levels)
            keys.append(key)
            parts.append(repair(group))
            if not sort_index and not regularize:
                kept_positions = group_positions.to_numpy()
                if duplicates not in {"keep", "error"}:
                    kept_positions = kept_positions[
                        ~group.index.duplicated(keep=False if duplicates == "drop" else duplicates)
                    ]
                positions_after.extend(kept_positions)
        result = pd.concat(parts, keys=keys, names=frame.index.names) if parts else frame.copy()
        if positions_after:
            # Group-local filling must not silently regroup interleaved rows.
            result = result.iloc[np.argsort(positions_after, kind="stable")]
    else:
        result = repair(frame)
    return restore(result, data, scitype, store), report


def split_components(
    y: Any, X: Any, scitype: str, *, axis="time", test_size=None, fh=None
) -> tuple[list[tuple[Any, Any]], dict]:
    """Temporal holdout per series, or explicit whole-instance/row holdout."""
    if (test_size is None) == (fh is None):
        raise ValueError("Provide exactly one of test_size or fh")
    if test_size is not None and (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be a fraction strictly between 0 and 1")
    if fh is not None:
        steps = [fh] if isinstance(fh, int) else fh
        if (
            not isinstance(steps, list)
            or not steps
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in steps)
        ):
            raise ValueError("fh must be a positive integer or list of positive integers")
        horizon = max(steps)
        from sktime_mcp.config import settings

        if horizon > settings.max_horizon:
            raise ValueError(f"fh exceeds SKTIME_MCP_MAX_HORIZON ({settings.max_horizon})")
    if axis not in {"time", "instance"}:
        raise ValueError("axis must be time or instance")
    report = validate_data(y, scitype)
    if not report["valid"]:
        raise ValueError(f"Repair invalid data before splitting: {report['errors']}")
    if axis == "time":
        if scitype not in TEMPORAL_SCITYPES:
            raise ValueError(
                f"Temporal splitting does not apply to {scitype}; use axis='instance' for Table rows"
            )
        frame = canonical(y, scitype)
        if isinstance(frame.index, pd.MultiIndex):
            lengths = (
                frame.groupby(level=list(range(frame.index.nlevels - 1)), observed=True)
                .size()
                .tolist()
            )
        else:
            lengths = [len(frame)]
        counts = [ceil(n * test_size) if test_size is not None else horizon for n in lengths]
        if any(c >= n for c, n in zip(counts, lengths, strict=True)):
            raise ValueError("Split must leave at least one training observation in every series")
        from sktime.split import temporal_train_test_split

        size = test_size if test_size is not None else horizon
        values = temporal_train_test_split(y, X=X, test_size=size)
        y_train, y_test = values[:2]
        X_train, X_test = values[2:] if X is not None else (None, None)
        return [(y_train, X_train), (y_test, X_test)], {
            "axis": axis,
            "train_size": sum(lengths) - sum(counts),
            "n_test": sum(counts),
            "per_series_test_sizes": counts,
        }
    if fh is not None:
        raise ValueError("Instance splitting uses test_size, not fh")
    if scitype not in {"Table", "Panel", "Hierarchical"}:
        raise ValueError("Instance splitting requires Table, Panel or Hierarchical data")
    store = {}
    frame = canonical(y, scitype, store)
    multi = isinstance(frame.index, pd.MultiIndex) and scitype != "Table"
    keys = frame.index.droplevel(-1).unique() if multi else np.arange(len(frame))
    n_test = ceil(len(keys) * test_size)
    if n_test >= len(keys):
        raise ValueError("Split must leave at least one training instance")
    mask = (
        frame.index.droplevel(-1).isin(keys[-n_test:])
        if multi
        else np.arange(len(frame)) >= len(frame) - n_test
    )
    train = restore(frame.loc[~mask], y, scitype, store)
    test = restore(frame.loc[mask], y, scitype, store)
    X_train = X_test = None
    if X is not None:
        x_store = {}
        x_frame = canonical(X, scitype, x_store)
        if not frame.index.equals(x_frame.index):
            raise ValueError("X and primary data must have matching indexes for instance splitting")
        X_train = restore(x_frame.loc[~mask], X, scitype, x_store)
        X_test = restore(x_frame.loc[mask], X, scitype, x_store)
    return [(train, X_train), (test, X_test)], {
        "axis": axis,
        "train_size": len(keys) - n_test,
        "n_test": n_test,
    }
