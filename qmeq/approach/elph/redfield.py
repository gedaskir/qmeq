"""Module containing python functions, which generate first order Redfield kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import doublenp

from ..aprclass import ApproachElPh
from ..base.redfield import ApproachRedfield as Approach
from .neumann1 import generate_w1fct_elph


# ---------------------------------------------------------------------------------------------------------
# Redfield approach
# ---------------------------------------------------------------------------------------------------------
def generate_kern_redfield_elph(self):
    (E, Vbbp, w1fct, kern) = (self.qd.Ea, self.baths.Vbbp, self.w1fct, self.kern)
    (si, si_elph) = (self.si, self.si_elph)

    # Here letter convention is not used
    # For example, the label `a' has the same charge as the label `b'
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(si.statesdm[charge], 2):
            bbp = si.get_ind_dm0(b, bp, charge)
            bbp_bool = si.get_ind_dm0(b, bp, charge, 2)
            if bbp != -1 and bbp_bool:
                bbpi = si.ndm0 + bbp - si.npauli
                bbpi_bool = True if bbpi >= si.ndm0 else False
                # --------------------------------------------------
                for a, ap in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                    aap = si.get_ind_dm0(a, ap, charge)
                    if aap != -1:
                        bpap = si_elph.get_ind_dm0(bp, ap, charge)
                        ba = si_elph.get_ind_dm0(b, a, charge)
                        fct_aap = 0
                        for l in range(si.nbaths):
                            gamma_ba_bpap = 0.5*(Vbbp[l, b, a]*Vbbp[l, bp, ap].conjugate() +
                                                 Vbbp[l, a, b].conjugate()*Vbbp[l, ap, bp])
                            fct_aap += gamma_ba_bpap*(w1fct[l, bpap, 0].conjugate() - w1fct[l, ba, 0])
                        aapi = si.ndm0 + aap - si.npauli
                        aap_sgn = +1 if si.get_ind_dm0(a, ap, charge, maptype=3) else -1
                        kern[bbp, aap] += fct_aap.imag
                        if aapi >= si.ndm0:
                            kern[bbp, aapi] += fct_aap.real*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] += fct_aap.imag*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] -= fct_aap.real
                # --------------------------------------------------
                for bpp in si.statesdm[charge]:
                    bppbp = si.get_ind_dm0(bpp, bp, charge)
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[charge]:
                            bppa = si_elph.get_ind_dm0(bpp, a, charge)
                            for l in range(si.nbaths):
                                gamma_ba_bppa = 0.5*(Vbbp[l, b, a]*Vbbp[l, bpp, a].conjugate() +
                                                     Vbbp[l, a, b].conjugate()*Vbbp[l, a, bpp])
                                fct_bppbp += +gamma_ba_bppa*w1fct[l, bppa, 1].conjugate()
                        for c in si.statesdm[charge]:
                            cbpp = si_elph.get_ind_dm0(c, bpp, charge)
                            for l in range(si.nbaths):
                                gamma_bc_bppc = 0.5*(Vbbp[l, b, c]*Vbbp[l, bpp, c].conjugate() +
                                                     Vbbp[l, c, b].conjugate()*Vbbp[l, c, bpp])
                                fct_bppbp += +gamma_bc_bppc*w1fct[l, cbpp, 0]
                        bppbpi = si.ndm0 + bppbp - si.npauli
                        bppbp_sgn = +1 if si.get_ind_dm0(bpp, bp, charge, maptype=3) else -1
                        kern[bbp, bppbp] += fct_bppbp.imag
                        if bppbpi >= si.ndm0:
                            kern[bbp, bppbpi] += fct_bppbp.real*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] += fct_bppbp.imag*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] -= fct_bppbp.real
                    # --------------------------------------------------
                    bbpp = si.get_ind_dm0(b, bpp, charge)
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[charge]:
                            bppa = si_elph.get_ind_dm0(bpp, a, charge)
                            for l in range(si.nbaths):
                                gamma_abpp_abp = 0.5*(Vbbp[l, a, bpp].conjugate()*Vbbp[l, a, bp] +
                                                      Vbbp[l, bpp, a]*Vbbp[l, bp, a].conjugate())
                                fct_bbpp += -gamma_abpp_abp*w1fct[l, bppa, 1]
                        for c in si.statesdm[charge]:
                            cbpp = si_elph.get_ind_dm0(c, bpp, charge)
                            for l in range(si.nbaths):
                                gamma_cbpp_cbp = 0.5*(Vbbp[l, c, bpp].conjugate()*Vbbp[l, c, bp] +
                                                      Vbbp[l, bpp, c]*Vbbp[l, bp, c].conjugate())
                                fct_bbpp += -gamma_cbpp_cbp*w1fct[l, cbpp, 0].conjugate()
                        bbppi = si.ndm0 + bbpp - si.npauli
                        bbpp_sgn = +1 if si.get_ind_dm0(b, bpp, charge, maptype=3) else -1
                        kern[bbp, bbpp] += fct_bbpp.imag
                        if bbppi >= si.ndm0:
                            kern[bbp, bbppi] += fct_bbpp.real*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] += fct_bbpp.imag*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] -= fct_bbpp.real
                # --------------------------------------------------
                for c, cp in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                    ccp = si.get_ind_dm0(c, cp, charge)
                    if ccp != -1:
                        cpbp = si_elph.get_ind_dm0(cp, bp, charge)
                        cb = si_elph.get_ind_dm0(c, b, charge)
                        fct_ccp = 0
                        for l in range(si.nbaths):
                            gamma_bc_bpcp = 0.5*(Vbbp[l, b, c]*Vbbp[l, bp, cp].conjugate() +
                                                 Vbbp[l, c, b].conjugate()*Vbbp[l, cp, bp])
                            fct_ccp += gamma_bc_bpcp*(w1fct[l, cpbp, 1] - w1fct[l, cb, 1].conjugate())
                        ccpi = si.ndm0 + ccp - si.npauli
                        ccp_sgn = +1 if si.get_ind_dm0(c, cp, charge, maptype=3) else -1
                        kern[bbp, ccp] += fct_ccp.imag
                        if ccpi >= si.ndm0:
                            kern[bbp, ccpi] += fct_ccp.real*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] += fct_ccp.imag*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] -= fct_ccp.real
                # --------------------------------------------------
    return 0


class ApproachRedfield(ApproachElPh):

    kerntype = 'pyRedfield'

    def generate_fct(self):
        Approach.generate_fct(self)
        generate_w1fct_elph(self)

    def generate_kern(self):
        Approach.generate_kern(self)
        generate_kern_redfield_elph(self)

    def generate_current(self):
        Approach.generate_current(self)

    def generate_vec(self, phi0):
        return Approach.generate_vec(self, phi0)
