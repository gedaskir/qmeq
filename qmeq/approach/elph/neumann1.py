"""Module containing python functions, which generate first order 1vN kernel."""

import numpy as np
import itertools

from ...wrappers.mytypes import complexnp
from ...wrappers.mytypes import doublenp

from ...specfunc.specfunc_elph import Func1vNElPh

from ..aprclass import ApproachElPh
from ..base.neumann1 import Approach1vN as Approach1vNBase


# ---------------------------------------------------------------------------------------------------------
# 1 von Neumann approach
# ---------------------------------------------------------------------------------------------------------
class Approach1vN(ApproachElPh):

    kerntype = 'py1vN'

    def prepare_arrays(self):
        Approach1vNBase.prepare_arrays(self)
        nbaths, ndm0 = self.si_elph.nbaths, self.si_elph.ndm0
        self.w1fct = np.zeros((nbaths, ndm0, 2), dtype=complexnp)

    def clean_arrays(self):
        Approach1vNBase.clean_arrays(self)
        self.w1fct.fill(0.0)

    def generate_fct(self):
        Approach1vNBase.generate_fct(self)

        E, si = self.qd.Ea, self.si_elph
        ncharge, nbaths, statesdm = si.ncharge, si.nbaths, si.statesdm

        func_1vN_elph = Func1vNElPh(self.baths.tlst_ph, self.baths.dlst_ph,
                                    self.funcp.itype_ph, self.funcp.dqawc_limit,
                                    self.baths.bath_func,
                                    self.funcp.eps_elph)

        w1fct = self.w1fct
        # Diagonal elements
        for l in range(nbaths):
            func_1vN_elph.eval(0., l)
            for charge in range(ncharge):
                for b in statesdm[charge]:
                    bb = si.get_ind_dm0(b, b, charge)
                    bb_bool = si.get_ind_dm0(b, b, charge, maptype=2)
                    if bb != -1 and bb_bool:
                        w1fct[l, bb, 0] = func_1vN_elph.val0 - 0.5j*func_1vN_elph.val0.imag
                        w1fct[l, bb, 1] = func_1vN_elph.val1 - 0.5j*func_1vN_elph.val1.imag
        # Off-diagonal elements
        for charge in range(ncharge):
            for b, bp in itertools.permutations(statesdm[charge], 2):
                bbp = si.get_ind_dm0(b, bp, charge)
                bbp_bool = si.get_ind_dm0(b, bp, charge, maptype=2)
                if bbp != -1 and bbp_bool:
                    Ebbp = E[b]-E[bp]
                    for l in range(nbaths):
                        func_1vN_elph.eval(Ebbp, l)
                        w1fct[l, bbp, 0] = func_1vN_elph.val0
                        w1fct[l, bbp, 1] = func_1vN_elph.val1

    def generate_coupling_terms(self, b, bp, bcharge):
        Approach1vNBase.generate_coupling_terms(self, b, bp, bcharge)

        Vbbp, w1fct = self.baths.Vbbp, self.w1fct
        si, si_elph, kh = self.si, self.si_elph, self.kernel_handler
        nbaths, statesdm = si.nbaths, si.statesdm

        acharge = bcharge
        ccharge = bcharge

        # --------------------------------------------------
        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                bpa = si_elph.get_ind_dm0(bp, a, acharge)
                bap = si_elph.get_ind_dm0(b, ap, acharge)
                if bpa == -1 or bap == -1:
                    continue
                fct_aap = 0
                for l in range(nbaths):
                    gamma_ba_bpap = 0.5*(Vbbp[l, b, a]*Vbbp[l, bp, ap].conjugate() +
                                         Vbbp[l, a, b].conjugate()*Vbbp[l, ap, bp])
                    fct_aap += gamma_ba_bpap*(w1fct[l, bpa, 0].conjugate() - w1fct[l, bap, 0])
                kh.set_matrix_element(fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    bpa = si_elph.get_ind_dm0(bp, a, acharge)
                    if bpa == -1:
                        continue
                    for l in range(nbaths):
                        gamma_ba_bppa = 0.5*(Vbbp[l, b, a]*Vbbp[l, bpp, a].conjugate() +
                                             Vbbp[l, a, b].conjugate()*Vbbp[l, a, bpp])
                        fct_bppbp += gamma_ba_bppa*w1fct[l, bpa, 1].conjugate()
                for c in statesdm[ccharge]:
                    cbp = si_elph.get_ind_dm0(c, bp, bcharge)
                    if cbp == -1:
                        continue
                    for l in range(nbaths):
                        gamma_bc_bppc = 0.5*(Vbbp[l, b, c]*Vbbp[l, bpp, c].conjugate() +
                                             Vbbp[l, c, b].conjugate()*Vbbp[l, c, bpp])
                        fct_bppbp += gamma_bc_bppc*w1fct[l, cbp, 0]
                kh.set_matrix_element(fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    ba = si_elph.get_ind_dm0(b, a, acharge)
                    if ba == -1:
                        continue
                    for l in range(nbaths):
                        gamma_abpp_abp = 0.5*(Vbbp[l, a, bpp].conjugate()*Vbbp[l, a, bp] +
                                              Vbbp[l, bpp, a]*Vbbp[l, bp, a].conjugate())
                        fct_bbpp += -gamma_abpp_abp*w1fct[l, ba, 1]
                for c in statesdm[ccharge]:
                    cb = si_elph.get_ind_dm0(c, b, bcharge)
                    if cb == -1:
                        continue
                    for l in range(nbaths):
                        gamma_cbpp_cbp = 0.5*(Vbbp[l, c, bpp].conjugate()*Vbbp[l, c, bp] +
                                              Vbbp[l, bpp, c]*Vbbp[l, bp, c].conjugate())
                        fct_bbpp += -gamma_cbpp_cbp*w1fct[l, cb, 0].conjugate()
                kh.set_matrix_element(fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for c, cp in itertools.product(statesdm[ccharge], statesdm[ccharge]):
            if kh.is_included(c, cp, ccharge):
                cbp = si_elph.get_ind_dm0(c, bp, bcharge)
                cpb = si_elph.get_ind_dm0(cp, b, bcharge)
                if cbp == -1 or cpb == -1:
                    continue
                fct_ccp = 0
                for l in range(nbaths):
                    gamma_bc_bpcp = 0.5*(Vbbp[l, b, c]*Vbbp[l, bp, cp].conjugate() +
                                         Vbbp[l, c, b].conjugate()*Vbbp[l, cp, bp])
                    fct_ccp += gamma_bc_bpcp*(w1fct[l, cbp, 1] - w1fct[l, cpb, 1].conjugate())
                kh.set_matrix_element(fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    def generate_current(self):
        Approach1vNBase.generate_current(self)
