# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------


project = "Segments.ai Python SDK"
copyright = f"{datetime.today().year}, Segments.ai"
author = "Bert De Brabandere & Arnout Hillen"

# The full version, including alpha/beta/rc tags
release = "1.0.3"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autodoc_pydantic",
]

add_module_names = False

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autodoc
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
# autodoc_class_signature = "separated"
autodoc_typehints_description_target = "documented"

# Autosummary
# autosummary_generate = True

# Intersphinx
intersphinx_mapping = {
    #     "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None)
    #     "pydantic": ("https://prompt-build--pydantic-docs.netlify.app/", None),
    #     "requests": ("https://docs.python-requests.org/en/stable/", None),
}

# Myst
myst_heading_anchors = 2

# Remove the prompt when copying examples
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# AutoAPI
# autoapi_type = "python"
# autoapi_dirs = ["../../src/segments/"]

# Autodoc pydantic
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config = False
autodoc_pydantic_model_signature_prefix = "class"
autodoc_pydantic_field_show_default = False
autodoc_pydantic_field_show_required = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -----------------------------------------------------------------------------
# Options for HTML output
# -----------------------------------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.

html_theme = "furo"
html_theme_options = {
    # Furo options (https://pradyunsg.me/furo/customisation/):
    # Furo will automatically add a small edit button to each document, when the documentation is generated on Read the Docs using a GitHub repository as the source.
    # "announcement": "<em>Important</em> announcement!",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        "font-stack": "gitbook-content-font, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
    },
}
pygments_style = "default"
html_logo = "_static/logo_blue_background.png"
html_title = "Python SDK"
html_favicon = "_static/favicon.ico"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# If false, no index is generated.
# html_use_index = False
