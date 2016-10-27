"""Module which build extensions written in cython."""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext = [# Pauli, Redfield, 1vN approaches
       Extension("neumannc",
                 ["neumannc.pyx"],
                 #libraries = ["test"],
                 #extra_compile_args = ["-O3"],
                 extra_link_args=[]),
      # Lindblad approach
      Extension("lindbladc",
                 ["lindbladc.pyx"]),
      # 2vN approach
      Extension("neumann2c",
                ["neumann2c.pyx"])]

cext = cythonize(ext)

setup(include_dirs = [np.get_include()],
      ext_modules = cext)
