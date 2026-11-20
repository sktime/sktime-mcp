"""A dev install must provide the numba distance backend (#557, F-19/F-20).

``test_export_code_scitype.py``, ``test_nits_roundup.py`` and
``test_dataset_xy_convention.py`` all exercise distance-based classifiers.
Without ``numba`` sktime falls back to a pure-Python elastic-distance
implementation: a single full ``predict`` on ``arrow_head`` goes from about
ten seconds to over an hour, so the job stalls instead of failing and the
cause is invisible in the log.
"""

import importlib.util


def test_numba_backend_is_installed():
    """``pip install -e ".[dev]"`` must resolve numba, or the suite stalls."""
    assert importlib.util.find_spec("numba") is not None, (
        "numba is not installed, so distance-based estimators fall back to the "
        'pure-Python path and the suite takes hours. Install with: pip install -e ".[dev]"'
    )
