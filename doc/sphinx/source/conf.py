# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Path setup
import os
import sys
sys.path.insert(0, os.path.abspath('../../..')) # Package root relative to this file

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Probe Particle Model'
copyright = '2022, Prokop Hapala, Aliaksandr Yakutovich, Ondřej Krejčí'
author = 'Prokop Hapala, Aliaksandr Yakutovich, Ondřej Krejčí'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Main library for html generation
    'sphinx.ext.napoleon',      # Understand Google-style doc-strings
    'sphinx.ext.viewcode'       # Add a [source] button to every function/class
]

templates_path = ['_templates']
exclude_patterns = []

# Enable writing multiple return values for Google-style docstrings
napoleon_custom_sections = [('Returns', 'params_style')]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
