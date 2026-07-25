"""Tests for the ``sktime-mcp`` console-script command line interface."""

import subprocess
import sys

import pytest

import sktime_mcp
from sktime_mcp.server import build_arg_parser


def test_version_flag_exits_cleanly_and_prints_version(capsys):
    """``--version`` prints the version and exits 0 instead of starting the server."""
    parser = build_arg_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"sktime-mcp {sktime_mcp.__version__}"


def test_short_version_flag_matches_long_flag(capsys):
    """``-V`` is accepted as an alias for ``--version``."""
    parser = build_arg_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["-V"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"sktime-mcp {sktime_mcp.__version__}"


def test_no_arguments_parses_without_exiting():
    """Bare invocation must parse cleanly so the server can start on stdio."""
    assert build_arg_parser().parse_args([]) is not None


def test_version_reports_installed_package_metadata():
    """__version__ is single-sourced from package metadata, not hard-coded."""
    assert sktime_mcp.__version__ != "0.0.0+unknown"
    # A hard-coded literal would drift from pyproject.toml; metadata cannot.
    assert sktime_mcp.__version__[0].isdigit()


def test_version_flag_via_subprocess_does_not_block():
    """End-to-end: the real entry point must return promptly, not wait on stdin."""
    result = subprocess.run(
        [sys.executable, "-m", "sktime_mcp.server", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"sktime-mcp {sktime_mcp.__version__}"
