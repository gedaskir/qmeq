"""Module containing python functions, which generate first order 1vN kernel.
   Functions in this module use StateIndexingDMc for indexing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ..mytypes import complexnp
from ..mytypes import doublenp

#---------------------------------------------------------------------------------------------------------
# 1 von Neumann approach
#---------------------------------------------------------------------------------------------------------
def generate_kern_1vNc(sys):
    (E, Tba, phi1fct) =(sys.qd.Ea, sys.leads.Tba, sys.phi1fct)
    (si, symq, norm_rowp) = (sys.si, sys.funcp.symq, sys.funcp.norm_row)
    norm_row = norm_rowp if symq else si.ndm0
    last_row = si.ndm0-1 if symq else si.ndm0
    kern = np.zeros((last_row+1, si.ndm0), dtype=complexnp)
    bvec = np.zeros(last_row+1, dtype=complexnp)
    bvec[norm_row] = 1
    for charge in range(si.ncharge):
        for b, bp in itertools.product(si.statesdm[charge], si.statesdm[charge]):
            bbp = si.get_ind_dm0(b, bp, charge)
            bbp_bool = si.get_ind_dm0(b, bp, charge, 2)
            if bbp != -1 and bbp_bool:
                kern[bbp, bbp] += E[b]-E[bp]
                #--------------------------------------------------
                for a, ap in itertools.product(si.statesdm[charge-1], si.statesdm[charge-1]):
                    aap = si.get_ind_dm0(a, ap, charge-1)
                    if aap != -1:
                        bpa = si.get_ind_dm1(bp, a, charge-1)
                        bap = si.get_ind_dm1(b, ap, charge-1)
                        fct_aap = 0
                        for l in range(si.nleads):
                            fct_aap += (+Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpa, 0].conjugate()
                                        -Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bap, 0])
                        kern[bbp, aap] += fct_aap
                #--------------------------------------------------
                for bpp in si.statesdm[charge]:
                    bppbp = si.get_ind_dm0(bpp, bp, charge)
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[charge-1]:
                            bpa = si.get_ind_dm1(bp, a, charge-1)
                            for l in range(si.nleads):
                                fct_bppbp += +Tba[l, b, a]*Tba[l, a, bpp]*phi1fct[l, bpa, 1].conjugate()
                        for c in si.statesdm[charge+1]:
                            cbp = si.get_ind_dm1(c, bp, charge)
                            for l in range(si.nleads):
                                fct_bppbp += +Tba[l, b, c]*Tba[l, c, bpp]*phi1fct[l, cbp, 0]
                        kern[bbp, bppbp] += fct_bppbp
                    #--------------------------------------------------
                    bbpp = si.get_ind_dm0(b, bpp, charge)
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[charge-1]:
                            ba = si.get_ind_dm1(b, a, charge-1)
                            for l in range(si.nleads):
                                fct_bbpp += -Tba[l, bpp, a]*Tba[l, a, bp]*phi1fct[l, ba, 1]
                        for c in si.statesdm[charge+1]:
                            cb = si.get_ind_dm1(c, b, charge)
                            for l in range(si.nleads):
                                fct_bbpp += -Tba[l, bpp, c]*Tba[l, c, bp]*phi1fct[l, cb, 0].conjugate()
                        kern[bbp, bbpp] += fct_bbpp
                #--------------------------------------------------
                for c, cp in itertools.product(si.statesdm[charge+1], si.statesdm[charge+1]):
                    ccp = si.get_ind_dm0(c, cp, charge+1)
                    if ccp != -1:
                        cbp = si.get_ind_dm1(c, bp, charge)
                        cpb = si.get_ind_dm1(cp, b, charge)
                        fct_ccp = 0
                        for l in range(si.nleads):
                            fct_ccp += (+Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cbp, 1]
                                        -Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpb, 1].conjugate())
                        kern[bbp, ccp] += fct_ccp
                #--------------------------------------------------
    # Normalisation condition
    kern[norm_row] = np.zeros(si.ndm0, dtype=complexnp)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            kern[norm_row, bb] += 1
    return kern, bvec

def generate_phi1_1vNc(sys):
    (phi0, E, Tba,) = (sys.phi0, sys.qd.Ea, sys.leads.Tba)
    (phi1fct, phi1fct_energy, si) = (sys.phi1fct, sys.phi1fct_energy, sys.si)
    phi1 = np.zeros((si.nleads, si.ndm1), dtype=complexnp)
    current = np.zeros(si.nleads, dtype=complexnp)
    energy_current = np.zeros(si.nleads, dtype=complexnp)
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = si.get_ind_dm1(c, b, bcharge)
            for l in range(si.nleads):
                fct1 = phi1fct[l, cb, 0]
                fct2 = phi1fct[l, cb, 1]
                fct1h = phi1fct_energy[l, cb, 0]
                fct2h = phi1fct_energy[l, cb, 1]
                for bp in si.statesdm[bcharge]:
                    bpb = si.get_ind_dm0(bp, b, bcharge)
                    if bpb != -1:
                        phi1[l, cb] += Tba[l, c, bp]*phi0[bpb]*fct1
                        current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0[bpb]*fct1
                        energy_current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0[bpb]*fct1h
                for cp in si.statesdm[ccharge]:
                    ccp = si.get_ind_dm0(c, cp, ccharge)
                    if ccp != -1:
                        phi1[l, cb] += Tba[l, cp, b]*phi0[ccp]*fct2
                        current[l] += Tba[l, b, c]*phi0[ccp]*Tba[l, cp, b]*fct2
                        energy_current[l] += Tba[l, b, c]*phi0[ccp]*Tba[l, cp, b]*fct2h
    for l in range(si.nleads):
        current[l] = -2*current[l].imag
        energy_current[l] = -2*energy_current[l].imag
    return phi1, current, energy_current
