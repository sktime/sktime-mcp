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

import logging
import re
import traceback
from typing import Any
from urllib.parse import unquote_plus, urlsplit

MASK = "***"

# Substrings that mark a key (query parameter, ``key=value`` pair or dict key)
# as credential-bearing. Matched case-insensitively anywhere in the key.
_CREDENTIAL_KEY_PATTERN = re.compile(
    r"password|passwd|passphrase|pwd|(?<![a-z])pass(?![a-z])|token|secret|credential|key|"
    r"(?<![a-z])auth(?![a-z])|authorization|signature|odbc_connect|(?<![a-z])dsn(?![a-z])",
    re.IGNORECASE,
)

_PAIR_SEPARATORS = re.compile(r"([&;])")
_URL_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"']+")
_USERINFO_PATTERN = re.compile(r"(://)([^/?#\s]*@)")


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
        if is_credential_key(unquote_plus(key.strip())):
            parts[i] = f"{key}={MASK}"
    return "".join(parts)


def redact_url(url: str) -> str:
    """Mask the userinfo password and credential-like query parameters of *url*."""
    if not isinstance(url, str) or not url:
        return url

    # Mask userinfo before parsing, including malformed host/IPv6 URLs that
    # urlsplit rejects. Never return an unmasked password from the fallback.
    def mask_userinfo(match: re.Match) -> str:
        username, sep, _password = match[2][:-1].partition(":")
        return f"://{username}:{MASK}@" if sep else f"://{MASK}@"

    url = _USERINFO_PATTERN.sub(mask_userinfo, url)
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
        rebuilt += f"#{_redact_pairs(parsed.fragment)}"
    return rebuilt


def redact_connection_string(conn: str) -> str:
    """Mask credentials in a URL-style or ODBC-style connection string."""
    if not isinstance(conn, str) or not conn:
        return conn
    return redact_text(conn)


def redact_text(text: str) -> str:
    """Mask credential URLs and key=value pairs inside prose or tracebacks."""
    parts = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        start = 0
        for match in _URL_PATTERN.finditer(body):
            parts.append(_redact_pairs(body[start : match.start()]))
            parts.append(redact_url(match[0]))
            start = match.end()
        parts.append(_redact_pairs(body[start:]))
        parts.append(line[len(body) :])
    return "".join(parts)


def redact_errors(obj: Any) -> Any:
    """Redact error fields recursively without changing successful data values."""
    if isinstance(obj, dict):
        return {
            key: _redact_error_value(value) if key in {"error", "errors"} else redact_errors(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [redact_errors(value) for value in obj]
    return obj


def _redact_error_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_error_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_error_value(item) for key, item in value.items()}
    return value


class RedactingFilter(logging.Filter):
    """Redact formatted log messages and exception chains before any handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = redact_text("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact_text(record.exc_text)
        return True


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
