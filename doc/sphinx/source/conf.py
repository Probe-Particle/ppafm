# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Path setup
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))  # Package root relative to this file

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

_year = 2024
project = "ppafm"
copyright = f"{_year}, Probe-Particle team"
author = "Probe-Particle team"
version = "0.3.1"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Main library for html generation
    "sphinx.ext.napoleon",  # Understand Google-style doc-strings
    "sphinx.ext.viewcode",  # Add a [source] button to every function/class
    "myst_parser",  # For Markdown support
]
myst_enable_extensions = ["dollarmath"]

templates_path = ["_templates"]
exclude_patterns = []

# Enable writing multiple return values for Google-style docstrings
napoleon_custom_sections = [("Returns", "params_style")]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_static_path = ["_static"]
