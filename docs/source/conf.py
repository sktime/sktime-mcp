import os
import sys
from datetime import datetime

# Path setup
sys.path.insert(0, os.path.abspath("../../src"))

# Project information
project = "sktime-mcp"
copyright = f"{datetime.now().year}, sktime-mcp contributors"
author = "sktime-mcp contributors"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

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
