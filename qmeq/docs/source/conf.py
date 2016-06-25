# -- Paths ------------------------------------------------
import sys
import os
import shlex

sys.path.insert(0, os.path.abspath('../../'))


# -- General configuration ------------------------------------------------
needs_sphinx = '1.1'

extensions = ['sphinx.ext.autodoc',
              #'sphinx.ext.pngmath',
              'sphinx.ext.imgmath',
              'numpydoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.coverage',
              'sphinx.ext.autosummary',
              'matplotlib.sphinxext.plot_directive']

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'qmeq'
copyright = u'2016, Gediminas Kirsanskas'
author = u'Gediminas Kirsanskas'

version = '0.0'
release = '0.0'

exclude_patterns = ['_build']
pygments_style = 'sphinx'
todo_include_todos = False
numpydoc_show_class_members = False

# -- Options for HTML output ---------------------------------------------------
html_theme = 'scipy'
html_theme_path = ['_theme']
#html_logo = '_static/scipyshiny_small.png'
html_static_path = ['_static']
html_theme_options = {
    "edit_link": "true",
    "sidebar": "right",
    "scipy_org_logo": "true",
    "rootlinks": [("http://scipy.org/", "Scipy.org"),
                  ("http://docs.scipy.org/", "Docs")]
}

pngmath_latex_preamble = r"""
\usepackage{color}
\definecolor{textgray}{RGB}{51,51,51}
\color{textgray}
"""
pngmath_use_preview = True
pngmath_dvipng_args = ['-gamma 1.5', '-D 96', '-bg Transparent']

htmlhelp_basename = 'qmeqdoc'


#------------------------------------------------------------------------------
# Plot style
#------------------------------------------------------------------------------
plot_pre_code = """
import numpy as np
import scipy as sp
np.random.seed(123)
"""
plot_include_source = True
plot_formats = [('png', 96), 'pdf']
plot_html_show_formats = False

import math
phi = (math.sqrt(5) + 1)/2

font_size = 13*72/96.0  # 13 px

plot_rcparams = {
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'figure.figsize': (3*phi, 3),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
}


# -- Options for LaTeX output ---------------------------------------------
latex_elements = { }
latex_documents = [
  (master_doc, 'qmeq.tex', u'qmeq Documentation',
   u'Gediminas Kirsanskas', 'manual'),
]

# -- Options for manual page output ---------------------------------------
man_pages = [
    (master_doc, 'qmeq', u'qmeq Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------
texinfo_documents = [
  (master_doc, 'qmeq', u'qmeq Documentation',
   author, 'qmeq', 'One line description of project.',
   'Miscellaneous'),
]
