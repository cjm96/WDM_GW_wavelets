# -- Project information -----------------------------------------------------
project = 'WDM'
copyright = '2025, Christopher J. Moore'
author = 'Christopher J. Moore'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.mathjax',  
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    "sphinx.ext.napoleon",
]

math_number_all = True

# Enable figure numbering
numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
    'section': 'Section %s',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
