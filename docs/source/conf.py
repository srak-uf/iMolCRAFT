# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

# Make the in-tree package importable so autodoc and the version lookup below
# work without installing iMolCRAFT first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imolcraft import __version__

project = 'iMolCRAFT'
copyright = '2025, Ryoma Sasaki'
author = 'Ryoma Sasaki'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    # 'myst_parser',
    'myst_nb',
]

# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.md': 'markdown',
# }

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo_transparent.png'
html_css_files = [
    'custom.css',
]

autodoc_default_flags = [
 'members',
 'private-members'
]
