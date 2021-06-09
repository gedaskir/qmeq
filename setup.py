import os
import sys
import numpy as np

from setuptools import setup, Extension


def get_ext_modules():
    """"
    Generate C extensions

    1. By default already Cython generated *.c files are used.
    2. If *.c files are not present Cython generates them.
    3. If the option '--cython' is specified Cython generates new *.c files.
    """

    # Check if *.c files are already there
    file_list = ['qmeq/approach/c_aprclass.c',
                 'qmeq/approach/c_kernel_handler.c',
                 # base
                 'qmeq/approach/base/c_pauli.c',
                 'qmeq/approach/base/c_lindblad.c',
                 'qmeq/approach/base/c_redfield.c',
                 'qmeq/approach/base/c_neumann1.c',
                 'qmeq/approach/base/c_neumann2.c',
                 'qmeq/approach/base/c_RTD.c',
                 'qmeq/specfunc/c_specfunc.c',
                 # elph
                 'qmeq/approach/elph/c_pauli.c',
                 'qmeq/approach/elph/c_lindblad.c',
                 'qmeq/approach/elph/c_redfield.c',
                 'qmeq/approach/elph/c_neumann1.c',
                 'qmeq/specfunc/c_specfunc_elph.c',
                 # wrappers
                 'qmeq/wrappers/c_lapack.c',
                 'qmeq/wrappers/c_mytypes.c',]
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
        # print('using already Cython generated C files')
    else:
        from Cython.Build import cythonize
        file_ext = '.pyx'
        # print('using cythonize to generate C files')

    ext = []
    openmp_flag = '-fopenmp' if os.name == 'posix' else '/openmp'
    for file_no_ext in file_list:
        file_base = file_no_ext[:-2]
        file_name = file_base + file_ext
        module_name = file_base.replace('/', '.')
        ext.append(
            Extension(
                module_name,
                [file_name],
                extra_compile_args=[openmp_flag],
                extra_link_args=[openmp_flag],
            )
        )

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
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Physics']

setup(name='qmeq',
      version='1.1',
      description=('Package for transport calculations in quantum dots ' +
                   'using approximate quantum master equations'),
      long_description=long_description,
      url='http://github.com/gedaskir/qmeq',
      author='Gediminas Kirsanskas',
      author_email='qmeq.package@gmail.com',
      license='BSD 2-Clause',
      classifiers=classifiers,
      packages=['qmeq',
                'qmeq/approach',
                'qmeq/approach/base',
                'qmeq/approach/elph',
                'qmeq/builder',
                'qmeq/specfunc',
                'qmeq/tests',
                'qmeq/wrappers',],
      package_data={'qmeq/approach':      ['*.pyx', '*.c', '*.pyd', '*.o', '*.so'],
                    'qmeq/approach/base': ['*.pyx', '*.c', '*.pyd', '*.o', '*.so'],
                    'qmeq/approach/elph': ['*.pyx', '*.c', '*.pyd', '*.o', '*.so'],
                    'qmeq/specfunc':      ['*.pyx', '*.c', '*.pyd', '*.o', '*.so'],
                    'qmeq/wrappers':      ['*.pyx', '*.c', '*.pyd', '*.o', '*.so'], },
      zip_safe=False,
      install_requires=['numpy', 'scipy'],
      include_dirs=[np.get_include()],
      ext_modules=get_ext_modules())
