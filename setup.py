#from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np

ext = [# Pauli, Redfield, 1vN approaches
       Extension("qmeq.neumannc",
                 ["qmeq/neumannc.pyx"],
                 #libraries = ["test"],
                 #extra_compile_args = ["-O3"],
                 extra_link_args=[]),
       # Lindblad approach
       Extension("qmeq.lindbladc",
                 ["qmeq/lindbladc.pyx"]),
       # 2vN approach
       Extension("qmeq.neumann2c",
                 ["qmeq/neumann2c.pyx"])]

cext = cythonize(ext)

setup(name='qmeq',
      version='0.0',
      description='Package for solving master equations',
      url='http://github.com/gedaskir/qmeq',
      author='Gediminas Kirsanskas',
      author_email='gediminas.kirsanskas@teorfys.lu.se',
      license='MIT',
      packages=['qmeq'],
      package_data={'qmeq': ['*.pyx', '*.c', '*.pyd', '*.o', '*.so']},
      zip_safe=False,
      install_requires=['numpy', 'scipy'],
      include_dirs = [np.get_include()],
      ext_modules = cext)
