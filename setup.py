#from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function
from setuptools import setup

import os
import sys
import subprocess

# Save the command line arguments, because sys.argv will be modified by 'qmeq/setupc.py'
argv = list(sys.argv)
# Build Cython extensions
cwd = os.path.abspath(os.path.dirname(__file__))
print("Entering", os.path.join(cwd, 'qmeq'))
os.chdir('qmeq')
sys.path.append("./")
import setupc
print("Going back to", cwd)
os.chdir('..')
# Restore the initial command line arguments
sys.argv = argv

setup(name='qmeq',
      version='0.0',
      description='Package for solving master equations',
      #url='http://github.com/xxxx/qmeq',
      author='Gediminas Kirsanskas',
      author_email='gediminas.kirsanskas@teorfys.lu.se',
      license='MIT',
      packages=['qmeq'],
      package_data={'qmeq': ['*.pyx', '*.c', '*.pyd', '*.o', '*.so']},
      zip_safe=False)
