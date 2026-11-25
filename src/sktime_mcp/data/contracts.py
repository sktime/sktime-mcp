"""Shared data contracts backed by sktime's public datatype registry.

Adapters read containers. This module describes and maps them without repairing
values or guessing task-specific column roles. Preparation is a separate step.
"""

from contextlib import suppress
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sktime.datatypes import MTYPE_REGISTER, check_is_mtype, check_is_scitype

MTYPE_SCITYPE = {name: scitype for name, scitype, _ in MTYPE_REGISTER}
SCITYPES = sorted(set(MTYPE_SCITYPE.values()))
TEMPORAL_SCITYPES = {"Series", "Panel", "Hierarchical"}


def scientific_type(data: Any, config: dict | None = None) -> str:
    """Resolve declared semantics first; only infer unambiguous container structure."""
    config = config or {}
    mtype = config.get("mtype")
    scitype = config.get("scitype")
    if mtype is not None:
        if mtype not in MTYPE_SCITYPE:
            raise ValueError(f"Unknown sktime mtype {mtype!r}; available: {sorted(MTYPE_SCITYPE)}")
        if scitype is not None and scitype != MTYPE_SCITYPE[mtype]:
            raise ValueError(f"mtype {mtype!r} belongs to {MTYPE_SCITYPE[mtype]}, not {scitype}")
        return MTYPE_SCITYPE[mtype]
    if scitype is not None:
        if scitype not in SCITYPES:
            raise ValueError(f"Unknown scitype {scitype!r}; choose from {SCITYPES}")
        return scitype
    if isinstance(data, (pd.Series, pd.DataFrame)) and isinstance(data.index, pd.MultiIndex):
        return "Panel" if data.index.nlevels == 2 else "Hierarchical"
    if isinstance(data, np.ndarray) and data.ndim == 3:
        return "Panel"
    if isinstance(data, list) and data and all(isinstance(x, pd.DataFrame) for x in data):
        return "Panel"
    if isinstance(data, pd.DataFrame) and any(
        isinstance(value, (pd.Series, np.ndarray)) for value in data.head(1).to_numpy().flat
    ):
        return "Panel"
    return "Series"


def materialize_inline(data: Any, config: dict) -> Any:
    """Decode JSON containers; preserve native Python containers and axis semantics."""
    mtype = config.get("mtype")
    if mtype in {"np.ndarray", "numpy1D", "numpy2D", "numpy3D", "numpyflat"}:
        return np.array(data, copy=True)
    if mtype == "df-list" and isinstance(data, list):
        return [
            item.copy(deep=True) if isinstance(item, pd.DataFrame) else pd.DataFrame(item)
            for item in data
        ]
    if mtype == "list_of_dict":
        return deepcopy(data)
    if (
        isinstance(data, dict)
        or isinstance(data, list)
        and not all(isinstance(x, pd.DataFrame) for x in data)
    ):
        data = pd.DataFrame(data)
    else:
        data = deepcopy(data)
    if mtype in {"pd.Series", "pd_Series_Table"} and isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(f"{mtype} requires one column")
        data = data.iloc[:, 0]
    if mtype == "nested_univ" and isinstance(data, pd.DataFrame):
        data = data.apply(
            lambda column: column.map(
                lambda value: pd.Series(value) if isinstance(value, (list, np.ndarray)) else value
            )
        )
    return data


def map_index(data: Any, config: dict) -> tuple[Any, list[str]]:
    """Apply only an explicitly requested index mapping and frequency declaration."""
    from .base import check_frequency_alignment, coerce_time_index

    time = config.get("time_column")
    instances = config.get("instance_columns", [])
    if isinstance(instances, str):
        instances = [instances]
    if not isinstance(instances, list) or not all(isinstance(c, str) for c in instances):
        raise ValueError("instance_columns must be a list of column names")
    if instances and (not time or isinstance(time, list)):
        raise ValueError("instance_columns requires one explicit time_column")
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        if time is not None or instances or config.get("frequency") is not None:
            raise ValueError(
                "Column/index mapping requires a pandas container; declare an mtype or convert first"
            )
        return data, []
    notes = []
    if time is not None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("time_column mapping requires a DataFrame")
        if isinstance(time, list):
            if not time:
                raise ValueError("time_column list must not be empty")
            instances, time = time[:-1], time[-1]
        if config.get("parse_dates", True) is False:
            data = data.set_index(instances + [time])
        else:
            data, notes = coerce_time_index(data, time)
            if instances:
                missing = [c for c in instances if c not in data.columns]
                if missing:
                    raise ValueError(f"Instance columns not found: {missing}")
                data = data.set_index(instances, append=True).reorder_levels(
                    list(range(1, len(instances) + 1)) + [0]
                )
    freq = config.get("frequency")
    if freq is not None:
        # A declaration validates the grid. It does not resample or insert rows.

        index = (
            data.index.get_level_values(-1) if isinstance(data.index, pd.MultiIndex) else data.index
        )
        if isinstance(index, pd.DatetimeIndex):
            check_frequency_alignment(index, freq)
            if (
                not isinstance(data.index, pd.MultiIndex)
                and index.is_unique
                and index.is_monotonic_increasing
            ):
                try:
                    data.index = pd.DatetimeIndex(index, freq=freq, name=index.name)
                except ValueError:
                    notes.append(
                        "Frequency grid contains gaps; rows were preserved. Use prepare_data to reindex explicitly."
                    )
        elif isinstance(index, pd.PeriodIndex):
            # Period aliases (M/Q/Y) differ from timestamp offsets (ME/QE/YE).
            offset = pd.Period("2000-01-01", freq=freq).freq
            if offset.n <= 0 or index.freq != offset:
                raise ValueError(
                    "Declared frequency differs from the existing PeriodIndex frequency"
                )
        else:
            raise ValueError("frequency requires a DatetimeIndex or PeriodIndex")
    return data, notes


def select_columns(data: Any, config: dict) -> tuple[Any, Any | None]:
    """Select explicit y/X columns; otherwise preserve all variables as the primary data."""
    if config.get("target_column") is not None and config.get("target_columns") is not None:
        raise ValueError("Use target_column or target_columns, not both")
    target = config.get("target_columns", config.get("target_column"))
    exog = config.get("exog_columns")
    role = config.get("role", "y")
    if role not in {"y", "X"}:
        raise ValueError("role must be 'y' or 'X'")
    if role == "X" and (target is not None or exog is not None):
        raise ValueError(
            "role='X' stores the complete feature container; omit target/exog column mappings"
        )
    if target is None and exog is None:
        return data, data if role == "X" else None
    if not isinstance(data, pd.DataFrame):
        raise ValueError("target/exog column mappings require a DataFrame")
    targets = [target] if isinstance(target, str) else target
    exog = [exog] if isinstance(exog, str) else exog
    for label, columns in [("Target", targets), ("Exogenous", exog)]:
        if columns is not None:
            if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns):
                raise ValueError(f"{label} columns must be a non-empty list of distinct names")
            missing = [c for c in columns if c not in data.columns]
            if missing and label == "Target" and len(missing) == 1:
                raise ValueError(
                    f"Target column {missing[0]!r} not found in data. Available columns: {list(data.columns)!r}"
                )
            if missing:
                raise ValueError(
                    f"{label} column(s) not found in data: {missing!r}. Available columns: {list(data.columns)!r}"
                )
    if targets is None:
        targets = [c for c in data.columns if c not in exog]
    if exog is None:
        exog = [c for c in data.columns if c not in targets]
    if not targets or set(targets) & set(exog):
        raise ValueError(
            "Target and exogenous columns must be non-overlapping, with at least one target"
        )
    unused = [c for c in data.columns if c not in targets + exog]
    if unused:
        raise ValueError(
            f"Unmapped columns {unused!r}; include them in target_columns or exog_columns"
        )
    y = data[targets]
    # Only an explicit scalar target mapping requests a Series. Panel/Hierarchy
    # mtypes are DataFrames even for one variable.
    if isinstance(target, str) and scientific_type(data, config) in {"Series", "Table"}:
        y = y.iloc[:, 0]
    return y, data[exog] if exog else None


def _json_metadata(value):
    if isinstance(value, dict):
        return {str(k): _json_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def validate_data(data: Any, scitype: str, mtype: str | None = None) -> dict:
    """Return sktime validation plus data-quality observations without modifying data."""
    if mtype is not None:
        valid, message, metadata = check_is_mtype(
            data, mtype=mtype, scitype=scitype, return_metadata=True
        )
    else:
        valid, message, metadata = check_is_scitype(data, scitype=scitype, return_metadata=True)
    notes = []
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if data.index.has_duplicates:
            notes.append("Duplicate index entries are present; no rows were removed.")
        if not data.index.is_monotonic_increasing and scitype in TEMPORAL_SCITYPES:
            notes.append("Index is not sorted; no rows were reordered.")
    if scitype == "Series" and isinstance(data, (pd.Series, pd.DataFrame)):
        frame = data.to_frame() if isinstance(data, pd.Series) else data
        non_numeric = [str(c) for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
        if non_numeric:
            notes.append(
                f"Non-numeric variables {non_numeric} were preserved; estimator tags determine whether they are supported."
            )
    if metadata and metadata.get("has_nans"):
        notes.append("Missing values are present; no values were filled.")
    if hasattr(data, "__len__") and len(data) == 0:
        notes.append("The container is empty; estimator support varies.")
    return {
        "valid": bool(valid),
        "errors": [] if valid else [str(message)],
        "warnings": notes,
        "sktime": _json_metadata(metadata or {}),
    }


def safe_value(value: Any) -> Any:
    """Bound nested previews and preserve labels as values, not dictionary keys."""
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray, list, tuple)):
        array = value.to_numpy() if isinstance(value, (pd.DataFrame, pd.Series)) else value
        return [safe_value(x) for x in list(array)[:5]]
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in list(value.items())[:20]}
    if isinstance(value, (pd.Timestamp, pd.Period, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if (
        value is None
        or (isinstance(value, (float, complex)) and not np.isfinite(value))
        or value is pd.NA
        or value is pd.NaT
    ):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def describe_data(data: Any, scitype: str | None = None) -> dict:
    """Describe a sktime container with bounded previews and validation metadata."""
    scitype = scitype or scientific_type(data)
    validation = validate_data(data, scitype)
    native = validation["sktime"]
    columns = (
        list(data.columns)
        if isinstance(data, pd.DataFrame)
        else [data.name if data.name is not None else "target"]
        if isinstance(data, pd.Series)
        else native.get("feature_names", [])
    )
    index = data.index if isinstance(data, (pd.DataFrame, pd.Series)) else None
    time_index = (
        (index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index)
        if scitype in TEMPORAL_SCITYPES
        else None
    )
    start_date = end_date = None
    if time_index is not None and len(time_index):
        # Invalid pandas input is still inspectable for explicit repair. Mixed
        # labels need not be comparable, and Table labels are not a time axis.
        with suppress(TypeError, ValueError):
            start_date, end_date = str(time_index.min()), str(time_index.max())
    freq = getattr(time_index, "freqstr", None)
    if (
        freq is None
        and isinstance(time_index, pd.DatetimeIndex)
        and not isinstance(index, pd.MultiIndex)
        and len(index) >= 3
    ):
        with suppress(ValueError, TypeError):
            freq = pd.infer_freq(index)
    if freq is None and time_index is not None and pd.api.types.is_integer_dtype(time_index.dtype):
        freq = "Integer"
    dtypes = (
        {str(c): str(t) for c, t in data.dtypes.items()}
        if isinstance(data, pd.DataFrame)
        else {str(columns[0]): str(data.dtype)}
        if isinstance(data, pd.Series)
        else {"values": str(data.dtype)}
        if isinstance(data, np.ndarray)
        else {}
    )
    shape = (
        list(data.shape)
        if hasattr(data, "shape")
        else [len(data)]
        if isinstance(data, list)
        else None
    )
    preview = {}
    missing = None
    if isinstance(data, (pd.Series, pd.DataFrame)):
        frame = data.to_frame() if isinstance(data, pd.Series) else data
        preview = {
            "columns": [str(c) for c in frame.columns[:20]],
            "index": [safe_value(x) for x in frame.index[:5]],
            "data": [[safe_value(x) for x in row] for row in frame.iloc[:5, :20].to_numpy()],
        }
        if scitype != "Panel" or native.get("mtype") != "nested_univ":
            missing = int(frame.isna().sum().sum())
    elif isinstance(data, (np.ndarray, list)):
        preview = {"data": safe_value(data)}
        if isinstance(data, np.ndarray):
            missing = int(pd.isna(data).sum())
    stats = {}
    if isinstance(data, (pd.DataFrame, pd.Series)) and native.get("mtype") != "nested_univ":
        try:
            frame = data.to_frame() if isinstance(data, pd.Series) else data
            stats = safe_value(frame.iloc[:, :20].describe(include="all").to_dict())
        except (TypeError, ValueError):
            pass
    return {
        "scitype": scitype,
        "mtype": native.get("mtype"),
        "shape": shape,
        "rows": len(data) if hasattr(data, "__len__") else None,
        "start_date": start_date,
        "end_date": end_date,
        "columns": [str(c) for c in columns],
        "dtypes": dtypes,
        "index_names": list(index.names) if index is not None else [],
        "freq": freq,
        "frequency": freq,
        "cutoff": end_date,
        "n_missing": missing,
        "head": preview,
        "summary_stats": stats,
        "validation": validation,
    }
