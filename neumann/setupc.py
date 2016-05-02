"""Module which build extensions written in cython."""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext = [Extension("neumannc",
                 ["neumannc.pyx"],
                 #libraries = ["test"],
                 #extra_compile_args = ["-O3"],
                 extra_link_args=[#"./fortran/libgfortran.a"
                                  "./fortran/digamma.o",
                                  "./fortran/cpsi.o",
                                  "./quadpack/dqc25c.o",
                                  "./quadpack/dqcheb.o",
                                  "./quadpack/dqk15w.o",
                                  "./quadpack/dqwgtc.o",
                                  "./quadpack/dqawc.o",
                                  "./quadpack/dqawce.o",
                                  "./quadpack/dqpsrt.o",
                                  "./fortran/d1mach.o",
                                  "./fortran/pvalint.o"
                                  ]),
      Extension("neumann2c",
                ["neumann2c.pyx"])]

cext = cythonize(ext)

setup(include_dirs = [np.get_include()],
      ext_modules = cext)
