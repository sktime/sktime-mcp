"""Shared parameter coercion for tool arguments.

JSON has no integer type, so a client may legitimately send ``5.0`` for a
schema ``integer``. Slicing with a float then crashes with "slice indices must
be integers" (F-46); coerce integral floats instead and reject the rest with a
structured message.
"""

from typing import Any


def coerce_integer(value: Any, name: str) -> tuple[int | None, str | None]:
    """Return ``(int, None)`` for ints and integral floats, else ``(None, error)``.

    Bools are rejected: ``True`` is an ``int`` subclass but never a sensible
    count, horizon or offset.
    """
    if isinstance(value, bool):
        return None, f"'{name}' must be an integer, got bool."
    if isinstance(value, int):
        return value, None
    if isinstance(value, float):
        if value.is_integer():
            return int(value), None
        return None, f"'{name}' must be an integer, got non-integral float {value!r}."
    return None, f"'{name}' must be an integer, got {type(value).__name__}."
