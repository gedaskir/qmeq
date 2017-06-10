"""Module containing pcython functions, which generate first order Redfield kernel.
   For docstrings see documentation of module neumann1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ..mytypes import doublenp
from ..mytypes import complexnp

from ..aprclass import Approach
from .neumann1 import generate_phi1fct

#---------------------------------------------------------------------------------------------------
# Redfield approach
#---------------------------------------------------------------------------------------------------
def generate_kern_redfield(sys):
    (E, Tba, phi1fct, si, symq, norm_rowp) = (sys.qd.Ea, sys.leads.Tba, sys.phi1fct, sys.si,
                                              sys.funcp.symq, sys.funcp.norm_row)
    norm_row = norm_rowp if symq else si.ndm0r
    last_row = si.ndm0r-1 if symq else si.ndm0r
    kern = np.zeros((last_row+1, si.ndm0r), dtype=doublenp)
    bvec = np.zeros(last_row+1, dtype=doublenp)
    bvec[norm_row] = 1
    npauli, ndm0, nleads = si.npauli, si.ndm0, si.nleads
    for charge in range(si.ncharge):
        acharge = charge-1
        bcharge = charge
        ccharge = charge+1
        for b, bp in itertools.combinations_with_replacement(si.statesdm[bcharge], 2):
            bbp = si.get_ind_dm0(b, bp, bcharge)
            bbp_bool = si.get_ind_dm0(b, bp, bcharge, 2)
            if bbp != -1 and bbp_bool:
                bbpi = ndm0 + bbp - npauli
                bbpi_bool = True if bbpi >= ndm0 else False
                if bbpi_bool:
                    kern[bbp, bbpi] += E[b]-E[bp]
                    kern[bbpi, bbp] += E[bp]-E[b]
                #--------------------------------------------------
                for a, ap in itertools.product(si.statesdm[acharge], si.statesdm[acharge]):
                    aap = si.get_ind_dm0(a, ap, acharge)
                    if aap != -1:
                        bpap = si.get_ind_dm1(bp, ap, acharge)
                        ba = si.get_ind_dm1(b, a, acharge)
                        fct_aap = 0
                        for l in range(nleads):
                            fct_aap += (+Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpap, 0].conjugate()
                                        -Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, ba, 0])
                        aapi = ndm0 + aap - npauli
                        aap_sgn = +1 if si.get_ind_dm0(a, ap, acharge, maptype=3) else -1
                        kern[bbp, aap] += fct_aap.imag
                        if aapi >= ndm0:
                            kern[bbp, aapi] += fct_aap.real*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] += fct_aap.imag*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] -= fct_aap.real
                #--------------------------------------------------
                for bpp in si.statesdm[bcharge]:
                    bppbp = si.get_ind_dm0(bpp, bp, bcharge)
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[acharge]:
                            bppa = si.get_ind_dm1(bpp, a, acharge)
                            for l in range(nleads):
                                fct_bppbp += +Tba[l, b, a]*Tba[l, a, bpp]*phi1fct[l, bppa, 1].conjugate()
                        for c in si.statesdm[ccharge]:
                            cbpp = si.get_ind_dm1(c, bpp, bcharge)
                            for l in range(nleads):
                                fct_bppbp += +Tba[l, b, c]*Tba[l, c, bpp]*phi1fct[l, cbpp, 0]
                        bppbpi = ndm0 + bppbp - npauli
                        bppbp_sgn = +1 if si.get_ind_dm0(bpp, bp, bcharge, maptype=3) else -1
                        kern[bbp, bppbp] += fct_bppbp.imag
                        if bppbpi >= ndm0:
                            kern[bbp, bppbpi] += fct_bppbp.real*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] += fct_bppbp.imag*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] -= fct_bppbp.real
                    #--------------------------------------------------
                    bbpp = si.get_ind_dm0(b, bpp, bcharge)
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[acharge]:
                            bppa = si.get_ind_dm1(bpp, a, acharge)
                            for l in range(nleads):
                                fct_bbpp += -Tba[l, bpp, a]*Tba[l, a, bp]*phi1fct[l, bppa, 1]
                        for c in si.statesdm[ccharge]:
                            cbpp = si.get_ind_dm1(c, bpp, bcharge)
                            for l in range(nleads):
                                fct_bbpp += -Tba[l, bpp, c]*Tba[l, c, bp]*phi1fct[l, cbpp, 0].conjugate()
                        bbppi = ndm0 + bbpp - npauli
                        bbpp_sgn = +1 if si.get_ind_dm0(b, bpp, bcharge, maptype=3) else -1
                        kern[bbp, bbpp] += fct_bbpp.imag
                        if bbppi >= ndm0:
                            kern[bbp, bbppi] += fct_bbpp.real*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] += fct_bbpp.imag*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] -= fct_bbpp.real
                #--------------------------------------------------
                for c, cp in itertools.product(si.statesdm[ccharge], si.statesdm[ccharge]):
                    ccp = si.get_ind_dm0(c, cp, ccharge)
                    if ccp != -1:
                        cpbp = si.get_ind_dm1(cp, bp, bcharge)
                        cb = si.get_ind_dm1(c, b, bcharge)
                        fct_ccp = 0
                        for l in range(nleads):
                            fct_ccp += (+Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpbp, 1]
                                        -Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cb, 1].conjugate())
                        ccpi = ndm0 + ccp - npauli
                        ccp_sgn = +1 if si.get_ind_dm0(c, cp, ccharge, maptype=3) else -1
                        kern[bbp, ccp] += fct_ccp.imag
                        if ccpi >= ndm0:
                            kern[bbp, ccpi] += fct_ccp.real*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] += fct_ccp.imag*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] -= fct_ccp.real
                #--------------------------------------------------
    # Normalisation condition
    kern[norm_row] = np.zeros(si.ndm0r, dtype=doublenp)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            kern[norm_row, bb] += 1
    sys.kern = kern
    sys.bvec = bvec
    return 0

def generate_current_redfield(sys):
    (phi0p, E, Tba, phi1fct, phi1fct_energy, si) = (sys.phi0, sys.qd.Ea, sys.leads.Tba,
                                                    sys.phi1fct, sys.phi1fct_energy, sys.si)
    phi1 = np.zeros((si.nleads, si.ndm1), dtype=complexnp)
    current = np.zeros(si.nleads, dtype=complexnp)
    energy_current = np.zeros(si.nleads, dtype=complexnp)
    #
    phi0 = np.zeros(si.ndm0, dtype=complexnp)
    phi0[0:si.npauli] = phi0p[0:si.npauli]
    phi0[si.npauli:si.ndm0] = phi0p[si.npauli:si.ndm0] + 1j*phi0p[si.ndm0:]
    #
    nleads = si.nleads
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = si.get_ind_dm1(c, b, bcharge)
            for l in range(nleads):
                for bp in si.statesdm[bcharge]:
                    bpb = si.get_ind_dm0(bp, b, bcharge)
                    if bpb != -1:
                        cbp = si.get_ind_dm1(c, bp, bcharge)
                        fct1 = phi1fct[l, cbp, 0]
                        fct1h = phi1fct_energy[l, cbp, 0]
                        bpb_conj = si.get_ind_dm0(bp, b, bcharge, maptype=3)
                        phi0bpb = phi0[bpb] if bpb_conj else phi0[bpb].conjugate()
                        phi1[l, cb] += Tba[l, c, bp]*phi0bpb*fct1
                        current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1
                        energy_current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1h
                for cp in si.statesdm[ccharge]:
                    ccp = si.get_ind_dm0(c, cp, ccharge)
                    if ccp != -1:
                        cpb = si.get_ind_dm1(cp, b, bcharge)
                        fct2 = phi1fct[l, cpb, 1]
                        fct2h = phi1fct_energy[l, cpb, 1]
                        ccp_conj = si.get_ind_dm0(c, cp, ccharge, maptype=3)
                        phi0ccp = phi0[ccp] if ccp_conj else phi0[ccp].conjugate()
                        phi1[l, cb] += Tba[l, cp, b]*phi0ccp*fct2
                        current[l] += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2
                        energy_current[l] += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2h
        sys.phi1 = phi1
    sys.current = np.array(-2*current.imag, dtype=doublenp)
    sys.energy_current = np.array(-2*energy_current.imag, dtype=doublenp)
    sys.heat_current = sys.energy_current - sys.current*sys.leads.mulst
    return 0

def generate_vec_redfield(phi0p, sys):
    (E, Tba, phi1fct, si, norm_row) = (sys.qd.Ea, sys.leads.Tba, sys.phi1fct,
                                       sys.si, sys.funcp.norm_row)
    #
    phi0 = np.zeros(si.ndm0, dtype=complexnp)
    phi0[0:si.npauli] = phi0p[0:si.npauli]
    phi0[si.npauli:si.ndm0] = phi0p[si.npauli:si.ndm0] + 1j*phi0p[si.ndm0:]
    #
    i_dphi0_dt = np.zeros(si.ndm0, dtype=complexnp)
    norm = 0
    nleads = si.nleads
    for charge in range(si.ncharge):
        acharge = charge-1
        bcharge = charge
        ccharge = charge+1
        for b, bp in itertools.combinations_with_replacement(si.statesdm[bcharge], 2):
            bbp = si.get_ind_dm0(b, bp, bcharge)
            if bbp != -1:
                if b == bp: norm += phi0[bbp]
                bbp_bool = si.get_ind_dm0(b, bp, bcharge, maptype=2)
                if bbp_bool:
                    i_dphi0_dt[bbp] += (E[b]-E[bp])*phi0[bbp]
                    #--------------------------------------------------
                    for a, ap in itertools.product(si.statesdm[acharge], si.statesdm[acharge]):
                        aap = si.get_ind_dm0(a, ap, acharge)
                        if aap != -1:
                            bpap = si.get_ind_dm1(bp, ap, acharge)
                            ba = si.get_ind_dm1(b, a, acharge)
                            fct_aap = 0
                            for l in range(nleads):
                                fct_aap += (+Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpap, 0].conjugate()
                                            -Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, ba, 0])
                            phi0aap = ( phi0[aap] if si.get_ind_dm0(a, ap, acharge, maptype=3)
                                                  else phi0[aap].conjugate() )
                            i_dphi0_dt[bbp] += fct_aap*phi0aap
                    #--------------------------------------------------
                    for bpp in si.statesdm[bcharge]:
                        bppbp = si.get_ind_dm0(bpp, bp, bcharge)
                        if bppbp != -1:
                            fct_bppbp = 0
                            for a in si.statesdm[acharge]:
                                bppa = si.get_ind_dm1(bpp, a, acharge)
                                for l in range(nleads):
                                    fct_bppbp += +Tba[l, b, a]*Tba[l, a, bpp]*phi1fct[l, bppa, 1].conjugate()
                            for c in si.statesdm[ccharge]:
                                cbpp = si.get_ind_dm1(c, bpp, bcharge)
                                for l in range(nleads):
                                    fct_bppbp += +Tba[l, b, c]*Tba[l, c, bpp]*phi1fct[l, cbpp, 0]
                            phi0bppbp = ( phi0[bppbp] if si.get_ind_dm0(bpp, bp, bcharge, maptype=3)
                                                      else  phi0[bppbp].conjugate() )
                            i_dphi0_dt[bbp] += fct_bppbp*phi0bppbp
                        #--------------------------------------------------
                        bbpp = si.get_ind_dm0(b, bpp, charge)
                        if bbpp != -1:
                            fct_bbpp = 0
                            for a in si.statesdm[acharge]:
                                bppa = si.get_ind_dm1(bpp, a, acharge)
                                for l in range(nleads):
                                    fct_bbpp += -Tba[l, bpp, a]*Tba[l, a, bp]*phi1fct[l, bppa, 1]
                            for c in si.statesdm[ccharge]:
                                cbpp = si.get_ind_dm1(c, bpp, bcharge)
                                for l in range(nleads):
                                    fct_bbpp += -Tba[l, bpp, c]*Tba[l, c, bp]*phi1fct[l, cbpp, 0].conjugate()
                            phi0bbpp = ( phi0[bbpp] if si.get_ind_dm0(b, bpp, bcharge, maptype=3)
                                                    else phi0[bbpp].conjugate() )
                            i_dphi0_dt[bbp] += fct_bbpp*phi0bbpp
                    #--------------------------------------------------
                    for c, cp in itertools.product(si.statesdm[ccharge], si.statesdm[ccharge]):
                        ccp = si.get_ind_dm0(c, cp, ccharge)
                        if ccp != -1:
                            cpbp = si.get_ind_dm1(cp, bp, bcharge)
                            cb = si.get_ind_dm1(c, b, bcharge)
                            fct_ccp = 0
                            for l in range(nleads):
                                fct_ccp += (+Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpbp, 1]
                                            -Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cb, 1].conjugate())
                            phi0ccp = ( phi0[ccp] if si.get_ind_dm0(c, cp, ccharge, maptype=3)
                                                  else phi0[ccp].conjugate() )
                            i_dphi0_dt[bbp] += fct_ccp*phi0ccp
                    #--------------------------------------------------
    i_dphi0_dt[norm_row] = 1j*(norm-1)
    return np.concatenate((i_dphi0_dt.imag, i_dphi0_dt[si.npauli:si.ndm0].real))

class Approach_pyRedfield(Approach):

    kerntype = 'pyRedfield'
    generate_fct = staticmethod(generate_phi1fct)
    generate_kern = staticmethod(generate_kern_redfield)
    generate_current = staticmethod(generate_current_redfield)
    generate_vec = staticmethod(generate_vec_redfield)
#---------------------------------------------------------------------------------------------------
