"""
Package that contains modules for various special functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .specfunc import fermi_func
from .specfunc import func_pauli
from .specfunc import func_1vN
from .specfunc import kernel_fredriksen
from .specfunc import hilbert_fredriksen
from .specfunc_elph import Func as pyFunc

try:
    from .c_specfunc import c_fermi_func
    from .c_specfunc import c_func_pauli
    from .c_specfunc import c_func_1vN
    from .c_specfunc_elph import Func
except ImportError:
    print("WARNING: Cannot import Cython compiled modules for the special functions (specfunc.__init__.py).")
    c_fermi_func = fermi_func
    c_func_pauli = func_pauli
    c_func_1vN = func_1vN
    Func = pyFunc
