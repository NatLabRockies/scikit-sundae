# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://sphinx-doc.org/en/master/usage/configuration.html#project-information

import sksundae as sun

project = 'scikit-sundae'
copyright = 'Alliance for Energy Innovation, LLC'
author = 'Corey R. Randall'
version = sun.__version__
release = sun.__version__

json_url = 'https://scikit-sundae.readthedocs.io/en/latest/_static/switcher.json'


# -- General configuration ---------------------------------------------------
# https://sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autoapi.extension',
    'myst_nb',
    'sphinx_design',
    # 'sphinx_favicon',
    'sphinx_copybutton',
]

templates_path = ['_templates']

exclude_patterns = [
    'build',
    'Thumbs.db',
    '.DS_Store',
    '*.ipynb_checkpoints',
    '__pycache__',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

default_role = 'literal'
highlight_language = 'console'


# -- Options for HTML output -------------------------------------------------
# https://sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html

html_theme = 'pydata_sphinx_theme'

# html_favicon = 'static/favicon.ico'
html_context = {'default_mode': 'dark'}

html_static_path = ['_static']
html_js_files = ['custom.js']
html_css_files = ['custom.css']

html_sidebars = {'index': [], '**': ['sidebar-nav-bs']}

html_theme_options = {
    # 'logo': {
    #     'image_light': 'static/light.png',
    #     'image_dark': 'static/dark.png',
    # },
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/NatLabRockies/scikit-sundae',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/scikit-sundae',
            'icon': 'fa-solid fa-box',
        },
    ],
    'navbar_start': ['navbar-logo'],
    'navbar_align': 'content',
    'header_links_before_dropdown': 5,
    'footer_start': ['copyright'],
    'footer_end': ['sphinx-version'],
    'navbar_persistent': ['search-button-field'],
    'primary_sidebar_end': ['sidebar-ethical-ads'],
    'secondary_sidebar_items': ['page-toc'],
    'search_bar_text': 'Search...',
    'show_prev_next': False,
    'collapse_navigation': True,
    'show_toc_level': 0,
    'pygments_light_style': 'tango',
    'show_version_warning_banner': True,
    'switcher': {
        'json_url': json_url,
        'version_match': version,
    }
}


# -- Options for napoleon ----------------------------------------------------
# https://sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_use_rtype = False
napoleon_custom_sections = [
    'Accessing the documentation',
    'Acknowledgements',
]


# -- Options for autoapi -----------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

autoapi_type = 'python'
autoapi_ignore = ['*/__pycache__/*']
autoapi_dirs = ['../../src/sksundae']
autoapi_keep_files = True
autoapi_root = 'api'
autoapi_member_order = 'groupwise'
autodoc_typehints = 'none'
autoapi_python_class_content = 'both'
autoapi_options = [
    'members',
    'imported-members',
    'inherited-members',
    'show-module-summary',
]


# -- Options for myst --------------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

nb_execution_timeout = 300
nb_number_source_lines = True
myst_enable_extensions = ['amsmath', 'dollarmath']
