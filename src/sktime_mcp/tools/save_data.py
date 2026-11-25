"""Explicit persistence for sktime data containers, without implicit axis flattening."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from sktime_mcp.data.contracts import scientific_type
from sktime_mcp.data.preparation import canonical
from sktime_mcp.runtime.executor import get_executor


def save_data_tool(
    data_handle: str, path: str, format: str = "csv", overwrite: bool = False
) -> dict[str, Any]:
    executor = get_executor()
    if data_handle not in executor._data_handles:
        return {"success": False, **executor.data_handle_missing(data_handle)}
    fmt = format.lower()
    if fmt not in {"csv", "parquet", "json", "npy"}:
        return {"success": False, "error": "Unsupported format; choose csv, parquet, json or npy"}
    target = Path(path).expanduser().resolve()
    existed = target.exists()
    if existed and not overwrite:
        return {
            "success": False,
            "error": f"File already exists: '{target}'. Pass overwrite=true to replace it.",
            "saved_path": str(target),
        }
    record = executor._data_handles[data_handle]
    y, X = record["y"], record.get("X")
    scitype = record.get("scitype") or scientific_type(y)
    try:
        if record.get("role") == "X":
            X = None
        if fmt == "npy":
            if not isinstance(y, np.ndarray) or X is not None:
                raise ValueError(
                    "npy requires an ndarray without a separate exogenous component; convert explicitly or save components separately"
                )
            if y.dtype.hasobject:
                raise ValueError("npy object arrays require pickle and are not supported")
            frame = None
            load_config = {
                "type": "file",
                "path": str(target),
                "format": fmt,
                "scitype": scitype,
                "role": record.get("role", "y"),
            }
        else:
            frame = canonical(y, scitype)
            if isinstance(frame.columns, pd.MultiIndex):
                raise ValueError(
                    "This tabular export cannot preserve MultiIndex columns; convert to an explicit flat representation first"
                )
            if X is not None:
                features = canonical(X, scitype)
                if not frame.index.equals(features.index):
                    raise ValueError("Cannot save misaligned target and exogenous indexes")
                overlap = frame.columns.intersection(features.columns)
                if len(overlap):
                    raise ValueError(f"Target and feature column names overlap: {list(overlap)}")
                frame = pd.concat([frame, features], axis=1)
            if frame.columns.has_duplicates:
                raise ValueError("Export requires unique column names")
            # Explicit unique index columns preserve hierarchy levels and avoid
            # clobbering a real variable called 'time'. Integer times stay integers.
            names = []
            occupied = {str(c) for c in frame.columns}
            for i, name in enumerate(frame.index.names):
                base = "time" if i == frame.index.nlevels - 1 else name or f"instance_{i}"
                name = str(base)
                while name in occupied:
                    name = "_" + name
                occupied.add(name)
                names.append(name)
            index = frame.index
            if isinstance(index, pd.PeriodIndex):
                frame = frame.copy()
                frame.index = index.astype(str)
            elif isinstance(index, pd.MultiIndex) and any(
                isinstance(index.get_level_values(i), pd.PeriodIndex) for i in range(index.nlevels)
            ):
                frame = frame.copy()
                frame.index = pd.MultiIndex.from_arrays(
                    [
                        index.get_level_values(i).astype(str)
                        if isinstance(index.get_level_values(i), pd.PeriodIndex)
                        else index.get_level_values(i)
                        for i in range(index.nlevels)
                    ],
                    names=index.names,
                )
            frame = frame.reset_index(names=names)
            load_config = {
                "type": "file",
                "path": str(target),
                "format": fmt,
                "scitype": scitype,
                "time_column": names[-1],
                "role": record.get("role", "y"),
            }
            if len(names) > 1:
                load_config["instance_columns"] = names[:-1]
            if X is not None:
                load_config["target_columns"] = list(canonical(y, scitype).columns)
                load_config["exog_columns"] = list(canonical(X, scitype).columns)
            if scitype == "Table":
                load_config["parse_dates"] = False
        target.parent.mkdir(parents=True, exist_ok=True)
        # Serialize beside the destination before publishing it. Failed writers
        # cannot truncate an existing export or leave a partial new file.
        temporary = None
        try:
            with NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as output:
                temporary = Path(output.name)
                if fmt == "npy":
                    np.save(output, y, allow_pickle=False)
                elif fmt == "json":
                    output.write(
                        frame.to_json(orient="records", date_format="iso", indent=2).encode()
                    )
                elif fmt == "parquet":
                    frame.to_parquet(output, index=False)
                else:
                    output.write(frame.to_csv(index=False).encode())
            if overwrite:
                temporary.replace(target)
            else:
                # Linking a complete file is atomic and fails if another caller
                # created the destination, preserving overwrite=False.
                os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {
            "success": True,
            "saved_path": str(target),
            "format": fmt,
            "rows": len(y),
            "overwritten": existed,
            "load_config": load_config,
            "notes": [
                "Tabular files preserve values and index columns; dtype and exact mtype reconstruction may require explicit reader options."
            ]
            if fmt != "npy"
            else [],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "error_type": type(exc).__name__}
