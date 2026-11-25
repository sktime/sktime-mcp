"""JSON schemas for the public data tools, shared with schema regression tests."""

from .contracts import MTYPE_SCITYPE, SCITYPES

LOAD_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["memory", "pandas", "file", "sql", "url"]},
        "data": {
            "type": ["object", "array"],
            "description": "Inline records, column arrays, or an array with explicit mtype.",
        },
        "path": {"type": "string"},
        "url": {"type": "string"},
        "format": {"type": "string", "enum": ["csv", "excel", "parquet", "json", "npy"]},
        "scitype": {"type": "string", "enum": SCITYPES},
        "mtype": {"type": "string", "enum": sorted(MTYPE_SCITYPE)},
        "role": {
            "type": "string",
            "enum": ["y", "X"],
            "default": "y",
            "description": "X marks a standalone feature container for X_handle; y is the primary/target container.",
        },
        "time_column": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True},
            ]
        },
        "instance_columns": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True,
        },
        "target_column": {"type": "string"},
        "target_columns": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True,
        },
        "exog_columns": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True},
            ]
        },
        "frequency": {
            "type": "string",
            "description": "Validate an explicit grid without inserting or removing rows.",
        },
        "parse_dates": {
            "oneOf": [{"type": "boolean"}, {"type": "array", "items": {"type": "string"}}]
        },
        "csv_options": {"type": "object"},
        "excel_options": {"type": "object"},
        "json_options": {"type": "object"},
        "connection_string": {"type": "string"},
        "dialect": {"type": "string"},
        "host": {"type": "string"},
        "port": {"type": "integer"},
        "database": {"type": "string"},
        "username": {"type": "string"},
        "password": {"type": "string"},
        "query": {"type": "string"},
        "query_params": {"type": "object"},
        "table": {"type": "string"},
        "filters": {"type": "object"},
    },
    "required": ["type"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {"properties": {"type": {"enum": ["memory", "pandas"]}}},
            "then": {"required": ["data"]},
        },
        {"if": {"properties": {"type": {"const": "file"}}}, "then": {"required": ["path"]}},
        {"if": {"properties": {"type": {"const": "url"}}}, "then": {"required": ["url"]}},
        {"not": {"required": ["target_column", "target_columns"]}},
    ],
}

PREPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "data_handle": {"type": "string"},
        "action": {"type": "string", "enum": ["format", "convert", "split"]},
        "sort_index": {"type": "boolean", "default": False},
        "duplicates": {
            "type": "string",
            "enum": ["keep", "first", "last", "drop", "error"],
            "default": "keep",
        },
        "missing": {
            "type": "string",
            "enum": ["keep", "ffill", "bfill", "interpolate", "constant"],
            "default": "keep",
        },
        "fill_value": {"type": ["number", "string", "boolean"]},
        "frequency": {"type": "string"},
        "regularize": {"type": "boolean", "default": False},
        "index_type": {"type": "string", "enum": ["keep", "period"], "default": "keep"},
        "to_mtype": {"type": "string", "enum": sorted(MTYPE_SCITYPE)},
        "component": {"type": "string", "enum": ["y", "X", "both"], "default": "y"},
        "axis": {"type": "string", "enum": ["time", "instance"], "default": "time"},
        "test_size": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
        "fh": {
            "oneOf": [
                {"type": "integer", "minimum": 1},
                {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1},
            ]
        },
    },
    "required": ["data_handle", "action"],
    "additionalProperties": False,
    "oneOf": [],
}
for action, fields in {
    "format": [
        "sort_index",
        "duplicates",
        "missing",
        "fill_value",
        "frequency",
        "regularize",
        "index_type",
    ],
    "convert": ["to_mtype", "component"],
    "split": ["axis", "test_size", "fh"],
}.items():
    branch = {
        "properties": {"action": {"const": action}},
        "propertyNames": {"enum": ["data_handle", "action", *fields]},
    }
    if action == "convert":
        branch["required"] = ["to_mtype"]
    elif action == "split":
        branch["oneOf"] = [
            {"required": ["test_size"], "not": {"required": ["fh"]}},
            {"required": ["fh"], "not": {"required": ["test_size"]}},
        ]
    PREPARE_SCHEMA["oneOf"].append(branch)
