"""Module containing python functions, which generate first order Pauli kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...aprclass import ApproachElPh
from ...specfunc.specfunc_elph import FuncPauliElPh

from ...mytypes import doublenp

from ..base.pauli import generate_paulifct
from ..base.pauli import generate_kern_pauli
from ..base.pauli import generate_current_pauli
from ..base.pauli import generate_vec_pauli
from ..base.pauli import generate_norm_vec


def generate_paulifct_elph(self):
    (E, Vbbp, si) = (self.qd.Ea, self.baths.Vbbp, self.si_elph)
    func_pauli = FuncPauliElPh(self.baths.tlst_ph, self.baths.dlst_ph,
                               self.baths.bath_func, self.funcp.eps_elph)
    #
    paulifct = np.zeros((si.nbaths, si.ndm0), dtype=doublenp)
    for charge in range(si.ncharge):
        # The diagonal elements b=bp are excluded, because they do not contribute
        for b, bp in itertools.permutations(si.statesdm[charge], 2):
            bbp_bool = si.get_ind_dm0(b, bp, charge, maptype=2)
            if bbp_bool:
                bbp = si.get_ind_dm0(b, bp, charge)
                Ebbp = E[b]-E[bp]
                for l in range(si.nbaths):
                    xbbp = 0.5*(Vbbp[l, b, bp]*Vbbp[l, b, bp].conjugate() +
                                Vbbp[l, bp, b].conjugate()*Vbbp[l, bp, b]).real
                    func_pauli.eval(Ebbp, l)
                    paulifct[l, bbp] = xbbp*func_pauli.val

    self.paulifct_elph = paulifct
    return 0


# ---------------------------------------------------------------------------------------------------------
# Pauli master equation
# ---------------------------------------------------------------------------------------------------------
def generate_kern_pauli_elph(self):
    (paulifct, si, si_elph) = (self.paulifct_elph, self.si, self.si_elph)

    if self.kern is None:
        self.kern_ext = np.zeros((si.npauli+1, si.npauli), dtype=doublenp)
        self.kern = self.kern_ext[0:-1, :]
        generate_norm_vec(self, si.npauli)

    kern = self.kern
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            bb_bool = si.get_ind_dm0(b, b, charge, maptype=2)
            if bb_bool:
                for a in si.statesdm[charge]:
                    aa = si.get_ind_dm0(a, a, charge)
                    ab = si_elph.get_ind_dm0(a, b, charge)
                    ba = si_elph.get_ind_dm0(b, a, charge)
                    if aa != -1 and ba != -1:
                        for l in range(si.nbaths):
                            kern[bb, bb] -= paulifct[l, ab]
                            kern[bb, aa] += paulifct[l, ba]

    return 0


class ApproachPyPauli(ApproachElPh):

    kerntype = 'pyPauli'
    generate_fct = staticmethod(generate_paulifct)
    generate_kern = staticmethod(generate_kern_pauli)
    generate_current = staticmethod(generate_current_pauli)
    generate_vec = staticmethod(generate_vec_pauli)
    #
    generate_kern_elph = staticmethod(generate_kern_pauli_elph)
    generate_fct_elph = staticmethod(generate_paulifct_elph)
# ---------------------------------------------------------------------------------------------------------
