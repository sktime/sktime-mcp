"""Redaction helpers that keep credentials out of tool responses and logs.

Three entry points:

* :func:`redact_url` masks the userinfo password and any credential-like
  query parameter of a URL (``postgresql://u:pw@h/db?password=x``).
* :func:`redact_connection_string` does the same for URL-style strings and
  additionally handles ``;``-separated key=value strings such as ODBC
  (``Driver={..};Server=h;UID=u;PWD=x``).
* :func:`redact_for_logging` walks an arbitrary JSON-like structure, masks
  values under credential-like keys, redacts strings, truncates long strings
  and summarises large containers so inline data and base64 payloads do not
  end up in the log file.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

MASK = "***"

# Substrings that mark a key (query parameter, ``key=value`` pair or dict key)
# as credential-bearing. Matched case-insensitively anywhere in the key.
_CREDENTIAL_KEY_PATTERN = re.compile(
    r"password|passwd|passphrase|pwd|(?<![a-z])pass(?![a-z])|token|secret|credential|key|"
    r"(?<![a-z])auth(?![a-z])|authorization|signature|odbc_connect|(?<![a-z])dsn(?![a-z])",
    re.IGNORECASE,
)

_PAIR_SEPARATORS = re.compile(r"([&;])")


def is_credential_key(key: Any) -> bool:
    """Return True when *key* names something that may hold a credential."""
    return isinstance(key, str) and bool(_CREDENTIAL_KEY_PATTERN.search(key))


def _redact_pairs(text: str) -> str:
    """Mask the value of every credential-like ``key=value`` pair in *text*.

    Pairs are separated by ``&`` or ``;``. The original encoding and
    ordering of untouched pairs is preserved.
    """
    parts = _PAIR_SEPARATORS.split(text)
    for i, part in enumerate(parts):
        if part in ("&", ";") or "=" not in part:
            continue
        key, _, _value = part.partition("=")
        if is_credential_key(key.strip()):
            parts[i] = f"{key}={MASK}"
    return "".join(parts)


def redact_url(url: str) -> str:
    """Mask the userinfo password and credential-like query parameters of *url*."""
    if not isinstance(url, str) or not url:
        return url
    try:
        parsed = urlsplit(url)
    except ValueError:
        return _redact_pairs(url)

    if not parsed.scheme and not parsed.netloc:
        # Not a URL at all; only key=value pairs can be masked.
        return _redact_pairs(url)

    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, _, hostport = netloc.rpartition("@")
        username, sep, _password = userinfo.partition(":")
        netloc = f"{username}:{MASK}@{hostport}" if sep else f"{MASK}@{hostport}"

    query = _redact_pairs(parsed.query) if parsed.query else parsed.query
    path = _redact_pairs(parsed.path) if "=" in parsed.path else parsed.path

    # Rebuild by hand: urlunsplit() would collapse "sqlite:///file" to "sqlite:/file".
    prefix = f"{parsed.scheme}://" if "://" in url else f"{parsed.scheme}:" if parsed.scheme else ""
    rebuilt = f"{prefix}{netloc}{path}"
    if query:
        rebuilt += f"?{query}"
    if parsed.fragment:
        rebuilt += f"#{parsed.fragment}"
    return rebuilt


def redact_connection_string(conn: str) -> str:
    """Mask credentials in a URL-style or ODBC-style connection string."""
    if not isinstance(conn, str) or not conn:
        return conn
    if "://" in conn:
        return redact_url(conn)
    return _redact_pairs(conn)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}…[truncated, {len(text)} chars total]"


def redact_for_logging(
    obj: Any,
    *,
    max_str_chars: int = 200,
    max_items: int = 20,
    _depth: int = 0,
) -> Any:
    """Return a copy of *obj* safe to write to a log file.

    - values under credential-like keys become ``"***"``
    - every string is passed through :func:`redact_connection_string` and
      truncated to *max_str_chars*
    - lists/tuples/dicts with more than *max_items* entries are replaced by a
      short summary so inline data payloads are not logged verbatim
    """
    if _depth > 16:
        return "<max depth>"

    if isinstance(obj, dict):
        if len(obj) > max_items:
            return f"<dict: {len(obj)} keys>"
        out: dict[Any, Any] = {}
        for key, value in obj.items():
            if is_credential_key(key) and value not in (None, ""):
                out[key] = MASK
            else:
                out[key] = redact_for_logging(
                    value, max_str_chars=max_str_chars, max_items=max_items, _depth=_depth + 1
                )
        return out

    if isinstance(obj, (list, tuple)):
        if len(obj) > max_items:
            return f"<{type(obj).__name__}: {len(obj)} items>"
        return [
            redact_for_logging(
                item, max_str_chars=max_str_chars, max_items=max_items, _depth=_depth + 1
            )
            for item in obj
        ]

    if isinstance(obj, str):
        return _truncate(redact_connection_string(obj), max_str_chars)

    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj

    return _truncate(redact_connection_string(str(obj)), max_str_chars)
