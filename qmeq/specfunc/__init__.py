"""
Package that contains modules for various special functions.
"""

from .specfunc import fermi_func
from .specfunc import diff_fermi
from .specfunc import phi
from .specfunc import delta_phi
from .specfunc import diff_phi
from .specfunc import diff2_phi
from .specfunc import bose
from .specfunc import polygamma
from .specfunc import digamma
from .specfunc import integralD
from .specfunc import integralX
from .specfunc import BW_Ozaki
from .specfunc import func_pauli
from .specfunc import func_1vN
from .specfunc import kernel_fredriksen
from .specfunc import hilbert_fredriksen
from .specfunc_elph import Func as pyFunc

try:
    from .c_specfunc import c_fermi_func
    from .c_specfunc import c_diff_fermi
    from .c_specfunc import c_phi
    from .c_specfunc import c_delta_phi
    from .c_specfunc import c_diff_phi
    from .c_specfunc import c_diff2_phi
    from .c_specfunc import c_bose
    from .c_specfunc import c_polygamma
    from .c_specfunc import c_digamma
    from .c_specfunc import c_integralD
    from .c_specfunc import c_integralX
    from .c_specfunc import c_BW_Ozaki

    from .c_specfunc import c_func_pauli
    from .c_specfunc import c_func_1vN
    from .c_specfunc_elph import Func
except ImportError:
    print("WARNING: Cannot import Cython compiled modules for the special functions (specfunc.__init__.py).")
    c_fermi_func = fermi_func
    c_func_pauli = func_pauli
    c_func_1vN = func_1vN
    Func = pyFunc
