import os
import sys

# Path setup
sys.path.insert(0, os.path.abspath("../../src"))

# Project information
project = "sktime-mcp"
# Fixed start year: using the current year would make builds non-reproducible.
copyright = "2025, sktime-mcp contributors"
author = "sktime-mcp contributors"

# Single-sourced from the installed package metadata so the docs can never
# drift from pyproject.toml.
try:
    from importlib.metadata import version as _package_version

    release = _package_version("sktime-mcp")
except Exception:  # building from a source tree without an install
    release = "0.0.0+unknown"

version = ".".join(release.split(".")[:2])

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# Napoleon configuration
# Render docstring "Attributes" sections as :ivar: fields inside the class body
# rather than as standalone py:attribute objects. Without this, dataclass fields
# picked up by autodoc's undoc-members collide with the same names described in
# the docstring, producing "duplicate object description" warnings.
napoleon_use_ivar = True

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "extra.css",
]

# If you have a logo or favicon, add them here
html_logo = "_static/sktime-logo.png"
html_favicon = "_static/favicon.png"

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
