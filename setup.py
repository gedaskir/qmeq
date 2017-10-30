from __future__ import print_function

import os
import sys
import numpy as np

try:
    from setuptools import setup, Extension
    #print('installing with setuptools')
except:
    from distutils.core import setup, Extension
    #print('installing with distutils')


def get_ext_modules():
    '''
    Generate C extensions

    1. By default already Cython generated *.c files are used.
    2. If *.c files are not present Cython generates them.
    3. If the option '--cython' is specified Cython generates new *.c files.
    '''

    # Check if *.c files are already there
    file_list = ['qmeq/approach/c_pauli.c',
                 'qmeq/approach/c_lindblad.c',
                 'qmeq/approach/c_redfield.c',
                 'qmeq/approach/c_neumann1.c',
                 'qmeq/approach/c_neumann2.c',
                 'qmeq/specfuncc.c']
    c_files_exist = all([os.path.isfile(f) for f in file_list])

    # Check if --cython option is specified
    if '--cython' in sys.argv:
        use_cython = True
        sys.argv.remove('--cython')
    else:
        use_cython = False

    if c_files_exist and not use_cython:
        cythonize = None
        file_ext = '.c'
        #print('using already Cython generated C files')
    else:
        from Cython.Build import cythonize
        file_ext = '.pyx'
        #print('using cythonize to generate C files')

    ext = [# Pauli
           Extension('qmeq.approach.c_pauli',
                     ['qmeq/approach/c_pauli'+file_ext]),
           # Lindblad
           Extension('qmeq.approach.c_lindblad',
                     ['qmeq/approach/c_lindblad'+file_ext]),
           # Redfield
           Extension('qmeq.approach.c_redfield',
                     ['qmeq/approach/c_redfield'+file_ext]),
           # 1vN
           Extension('qmeq.approach.c_neumann1',
                     ['qmeq/approach/c_neumann1'+file_ext]),
           # 2vN
           Extension('qmeq.approach.c_neumann2',
                     ['qmeq/approach/c_neumann2'+file_ext]),
           # Special functions
           Extension('qmeq.specfuncc',
                     ['qmeq/specfuncc'+file_ext])]

    cext = ext if cythonize is None else cythonize(ext)
    return cext

long_description = open('README.rst').read()

classifiers = ['Development Status :: 5 - Production/Stable',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: BSD License',
               'Operating System :: MacOS',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Programming Language :: Cython',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Physics']

setup(name='qmeq',
      version='1.0.1',
      description=('Package for transport calculations in quantum dots '
                  +'using approximate quantum master equations'),
      long_description=long_description,
      url='http://github.com/gedaskir/qmeq',
      author='Gediminas Kirsanskas',
      author_email='qmeq.package@gmail.com',
      license='BSD 2-Clause',
      classifiers=classifiers,
      packages=['qmeq', 'qmeq/approach', 'qmeq/tests'],
      package_data={'qmeq': ['*.pyx', '*.c', '*.pyd', '*.o', '*.so'],
                    'qmeq/approach': ['*.pyx', '*.c', '*.pyd', '*.o', '*.so']},
      zip_safe=False,
      install_requires=['numpy', 'scipy'],
      include_dirs=[np.get_include()],
      ext_modules=get_ext_modules())
