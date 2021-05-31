"""Module containing python functions, which generate first order Lindblad kernels."""

import numpy as np
import itertools

from ...wrappers.mytypes import doublenp

from ...specfunc.specfunc_elph import FuncPauliElPh

from ..aprclass import ApproachElPh
from ..base.lindblad import ApproachLindblad as ApproachLindbladBase


# ---------------------------------------------------------------------------------------------------------
# Lindblad approach
# ---------------------------------------------------------------------------------------------------------
class ApproachLindblad(ApproachElPh):

    kerntype = 'pyLindblad'

    def prepare_arrays(self):
        ApproachLindbladBase.prepare_arrays(self)
        Vbbp, mtype = self.baths.Vbbp, self.baths.mtype
        tLbbp_shape = Vbbp.shape + (2,)
        self.tLbbp = np.zeros(tLbbp_shape, dtype=mtype)

    def clean_arrays(self):
        ApproachLindbladBase.clean_arrays(self)
        self.tLbbp.fill(0.0)

    def generate_fct(self):
        ApproachLindbladBase.generate_fct(self)

        Vbbp, E = self.baths.Vbbp, self.qd.Ea,
        si = self.si
        ncharge, nbaths, statesdm = si.ncharge, si.nbaths, si.statesdm

        func_pauli = FuncPauliElPh(self.baths.tlst_ph, self.baths.dlst_ph,
                                   self.baths.bath_func, self.funcp.eps_elph)

        tLbbp = self.tLbbp
        # Diagonal elements
        for l in range(nbaths):
            func_pauli.eval(0., l)
            for charge in range(ncharge):
                for b in si.statesdm[charge]:
                    tLbbp[l, b, b, 0] = np.sqrt(0.5*func_pauli.val)*Vbbp[l, b, b]
                    tLbbp[l, b, b, 1] = tLbbp[l, b, b, 0].conjugate()
        # Off-diagonal elements
        for charge in range(ncharge):
            for b, bp in itertools.permutations(statesdm[charge], 2):
                Ebbp = E[b]-E[bp]
                for l in range(nbaths):
                    func_pauli.eval(Ebbp, l)
                    tLbbp[l, b, bp, 0] = np.sqrt(0.5*func_pauli.val)*Vbbp[l, b, bp]
                    tLbbp[l, b, bp, 1] = np.sqrt(0.5*func_pauli.val)*Vbbp[l, bp, b].conjugate()
        self.tLbbp = tLbbp

    def generate_coupling_terms(self, b, bp, bcharge):
        ApproachLindbladBase.generate_coupling_terms(self, b, bp, bcharge)

        tLbbp = self.tLbbp
        si, kh = self.si, self.kernel_handler
        nbaths, statesdm = si.nbaths, si.statesdm

        acharge = bcharge

        # --------------------------------------------------
        # Here letter convention is not used
        # For example, the label `a' has the same charge as the label `b'
        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                fct_aap = 0
                for (l, q) in itertools.product(range(si.nbaths), range(2)):
                    fct_aap += tLbbp[l, b, a, q]*tLbbp[l, bp, ap, q].conjugate()
                kh.set_matrix_element(1j*fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    for (l, q) in itertools.product(range(si.nbaths), range(2)):
                        fct_bppbp += -0.5*tLbbp[l, a, b, q].conjugate()*tLbbp[l, a, bpp, q]
                kh.set_matrix_element(1j*fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    for (l, q) in itertools.product(range(si.nbaths), range(2)):
                        fct_bbpp += -0.5*tLbbp[l, a, bpp, q].conjugate()*tLbbp[l, a, bp, q]
                kh.set_matrix_element(1j*fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------

    def generate_current(self):
        ApproachLindbladBase.generate_current(self)
