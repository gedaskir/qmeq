# -- Paths ------------------------------------------------
import sys
import os

sys.path.insert(0, os.path.abspath('../../qmeq/'))
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ------------------------------------------------
needs_sphinx = '1.3'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon'
             ]

templates_path = ['_templates']
#source_suffix = ['.rst', '.md']
source_suffix = '.rst'
#source_encoding = 'utf-8-sig'
master_doc = 'index'

project = u'qmeq'
copyright = u'2019, Gediminas Kirsanskas'
author = u'Gediminas Kirsanskas'

version = u'1.1'
release = u'1.1'

language = None
#today = ''
#today_fmt = '%B %d, %Y'
exclude_patterns = []
#default_role = None
#add_function_parentheses = True
#add_module_names = True
#show_authors = False
pygments_style = 'sphinx'
#modindex_common_prefix = []
#keep_warnings = False
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------
#html_theme = 'alabaster'
#html_theme = 'classic'
html_theme = 'sphinx_rtd_theme'
#html_theme = 'sphinxdoc'
#html_theme = 'epub'
#html_theme_options = {}
#html_theme_path = []
#html_title = u'qmeq v0'
#html_short_title = None
#html_logo = None
#html_favicon = None
#html_static_path = ['_static']
#html_extra_path = []
#html_last_updated_fmt = None
#html_use_smartypants = True
#html_sidebars = {}
#html_additional_pages = {}
#html_domain_indices = True
#html_use_index = True
#html_split_index = False
#html_show_sourcelink = True
#html_show_sphinx = True
#html_show_copyright = True
#html_use_opensearch = ''
#html_file_suffix = None
#html_search_language = 'en'
#html_search_options = {'type': 'default'}
#html_search_scorer = 'scorer.js'
htmlhelp_basename = 'qmeqdoc'


# -- Options for LaTeX output ---------------------------------------------
latex_elements = { }
latex_documents = [
    (master_doc, 'qmeq.tex', u'qmeq Documentation',
     u'Gediminas Kirsanskas', 'manual'),
]
#latex_logo = None
#latex_use_parts = False
#latex_show_pagerefs = False
#latex_show_urls = False
#latex_appendices = []
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------
man_pages = [
    (master_doc, 'qmeq', u'qmeq Documentation',
     [author], 1)
]
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------
texinfo_documents = [
    (master_doc, 'qmeq', u'qmeq Documentation',
     author, 'qmeq', 'One line description of project.',
     'Miscellaneous'),
]
#texinfo_appendices = []
#texinfo_domain_indices = True
#texinfo_show_urls = 'footnote'
#texinfo_no_detailmenu = False
