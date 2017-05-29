#from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function
#
try:
    from setuptools import setup, Extension
except:
    from distutils.core import setup, Extension
#
from Cython.Build import cythonize
import numpy as np

ext = [# Pauli
       Extension("qmeq.approach.c_pauli",
                 ["qmeq/approach/c_pauli.pyx"]),
       # Lindblad
       Extension("qmeq.approach.c_lindblad",
                 ["qmeq/approach/c_lindblad.pyx"]),
       # Redfield
       Extension("qmeq.approach.c_redfield",
                 ["qmeq/approach/c_redfield.pyx"]),
       # 1vN
       Extension("qmeq.approach.c_neumann1",
                 ["qmeq/approach/c_neumann1.pyx"]),
       # 2vN
       Extension("qmeq.approach.c_neumann2",
                 ["qmeq/approach/c_neumann2.pyx"]),
       # Special functions
       Extension("qmeq.specfuncc",
                 ["qmeq/specfuncc.pyx"])]

cext = cythonize(ext)

setup(name='qmeq',
      version='1.0',
      description='Package for transport calculations in quantum dots \
                   using approximate quantum master equations',
      url='http://github.com/gedaskir/qmeq',
      author='Gediminas Kirsanskas',
      author_email='qmeq.package@gmail.com',
      license='BSD 2-Clause',
      packages=['qmeq', 'qmeq/approach'],
      package_data={'qmeq': ['*.pyx', '*.c', '*.pyd', '*.o', '*.so']},
      zip_safe=False,
      install_requires=['numpy', 'scipy'],
      include_dirs = [np.get_include()],
      ext_modules = cext)
