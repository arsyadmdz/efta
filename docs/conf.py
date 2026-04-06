import os
import sys

# Make the efta package importable during doc build
sys.path.insert(0, os.path.abspath('..'))

project = 'efta'
author = 'arsyadmdz'
release = '1.0.0'
copyright = '2026, arsyadmdz'

extensions = [
    'sphinx.ext.autodoc',      # pulls docstrings automatically from source code
    'sphinx.ext.napoleon',     # supports NumPy-style and Google-style docstrings
    'sphinx.ext.viewcode',     # adds "View Source" links next to each function
    'sphinx.ext.autosummary',  # generates summary tables for modules
]

html_theme = 'sphinx_rtd_theme'

# Logo and branding
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_favicon = '_static/logo.png'
html_theme_options = {
    'logo_only': True,          # show logo image only, not the text project name
    'display_version': True,    # show version number next to logo
    'navigation_depth': 3,
    'collapse_navigation': False,
}

# autodoc settings — show members in the order they appear in source
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
