"""Module containing python functions, which generate first order Redfield kernel."""

import numpy as np
import itertools

from ...wrappers.mytypes import doublenp

from ..aprclass import ApproachElPh
from ..base.redfield import ApproachRedfield as ApproachRedfieldBase
from .neumann1 import Approach1vN


# ---------------------------------------------------------------------------------------------------------
# Redfield approach
# ---------------------------------------------------------------------------------------------------------
class ApproachRedfield(ApproachElPh):

    kerntype = 'pyRedfield'

    def prepare_arrays(self):
        Approach1vN.prepare_arrays(self)

    def clean_arrays(self):
        Approach1vN.clean_arrays(self)

    def generate_fct(self):
        Approach1vN.generate_fct(self)

    def generate_coupling_terms(self, b, bp, bcharge):
        ApproachRedfieldBase.generate_coupling_terms(self, b, bp, bcharge)

        Vbbp, w1fct = self.baths.Vbbp, self.w1fct
        si, si_elph, kh = self.si, self.si_elph, self.kernel_handler
        nbaths, statesdm = si.nbaths, si.statesdm

        acharge = bcharge
        ccharge = bcharge

        # --------------------------------------------------
        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                bpap = si_elph.get_ind_dm0(bp, ap, acharge)
                ba = si_elph.get_ind_dm0(b, a, acharge)
                if bpap == -1 or ba == -1:
                    continue
                fct_aap = 0
                for l in range(nbaths):
                    gamma_ba_bpap = 0.5*(Vbbp[l, b, a]*Vbbp[l, bp, ap].conjugate() +
                                         Vbbp[l, a, b].conjugate()*Vbbp[l, ap, bp])
                    fct_aap += gamma_ba_bpap*(w1fct[l, bpap, 0].conjugate() - w1fct[l, ba, 0])
                kh.set_matrix_element(fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    bppa = si_elph.get_ind_dm0(bpp, a, acharge)
                    if bppa == -1:
                        continue
                    for l in range(nbaths):
                        gamma_ba_bppa = 0.5*(Vbbp[l, b, a]*Vbbp[l, bpp, a].conjugate() +
                                             Vbbp[l, a, b].conjugate()*Vbbp[l, a, bpp])
                        fct_bppbp += +gamma_ba_bppa*w1fct[l, bppa, 1].conjugate()
                for c in statesdm[ccharge]:
                    cbpp = si_elph.get_ind_dm0(c, bpp, bcharge)
                    if cbpp == -1:
                        continue
                    for l in range(nbaths):
                        gamma_bc_bppc = 0.5*(Vbbp[l, b, c]*Vbbp[l, bpp, c].conjugate() +
                                             Vbbp[l, c, b].conjugate()*Vbbp[l, c, bpp])
                        fct_bppbp += +gamma_bc_bppc*w1fct[l, cbpp, 0]
                kh.set_matrix_element(fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    bppa = si_elph.get_ind_dm0(bpp, a, acharge)
                    if bppa == -1:
                        continue
                    for l in range(nbaths):
                        gamma_abpp_abp = 0.5*(Vbbp[l, a, bpp].conjugate()*Vbbp[l, a, bp] +
                                              Vbbp[l, bpp, a]*Vbbp[l, bp, a].conjugate())
                        fct_bbpp += -gamma_abpp_abp*w1fct[l, bppa, 1]
                for c in statesdm[ccharge]:
                    cbpp = si_elph.get_ind_dm0(c, bpp, bcharge)
                    if cbpp == -1:
                        continue
                    for l in range(nbaths):
                        gamma_cbpp_cbp = 0.5*(Vbbp[l, c, bpp].conjugate()*Vbbp[l, c, bp] +
                                              Vbbp[l, bpp, c]*Vbbp[l, bp, c].conjugate())
                        fct_bbpp += -gamma_cbpp_cbp*w1fct[l, cbpp, 0].conjugate()
                kh.set_matrix_element(fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for c, cp in itertools.product(statesdm[ccharge], statesdm[ccharge]):
            if kh.is_included(c, cp, ccharge):
                cpbp = si_elph.get_ind_dm0(cp, bp, bcharge)
                cb = si_elph.get_ind_dm0(c, b, bcharge)
                if cpbp == -1 or cb == -1:
                    continue
                fct_ccp = 0
                for l in range(nbaths):
                    gamma_bc_bpcp = 0.5*(Vbbp[l, b, c]*Vbbp[l, bp, cp].conjugate() +
                                         Vbbp[l, c, b].conjugate()*Vbbp[l, cp, bp])
                    fct_ccp += gamma_bc_bpcp*(w1fct[l, cpbp, 1] - w1fct[l, cb, 1].conjugate())
                kh.set_matrix_element(fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    def generate_current(self):
        ApproachRedfieldBase.generate_current(self)
