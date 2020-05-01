"""Module containing python functions, which generate first order 1vN kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import complexnp
from ...mytypes import doublenp

from ...specfunc.specfunc import func_1vN
from ..aprclass import Approach
from .pauli import generate_norm_vec


def generate_phi1fct(self):
    """
    Make factors used for generating 1vN, Redfield master equation kernels.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.phi1fct : array
        (Modifies) Factors used for generating 1vN, Redfield master equation kernels.
    self.phi1fct_energy : array
        (Modifies) Factors used to calculate energy and heat currents in 1vN, Redfield approaches.
    """
    (E, si, mulst, tlst, dlst) = (self.qd.Ea, self.si,
                                  self.leads.mulst, self.leads.tlst, self.leads.dlst)
    (itype, limit) = (self.funcp.itype, self.funcp.dqawc_limit)
    phi1fct = np.zeros((si.nleads, si.ndm1, 2), dtype=complexnp)
    phi1fct_energy = np.zeros((si.nleads, si.ndm1, 2), dtype=complexnp)
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = si.get_ind_dm1(c, b, bcharge)
            Ecb = E[c]-E[b]
            for l in range(si.nleads):
                rez = func_1vN(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype, limit)
                phi1fct[l, cb, 0] = rez[0]
                phi1fct[l, cb, 1] = rez[1]
                phi1fct_energy[l, cb, 0] = rez[2]
                phi1fct_energy[l, cb, 1] = rez[3]
    self.phi1fct = phi1fct
    self.phi1fct_energy = phi1fct_energy
    return 0


# ---------------------------------------------------------------------------------------------------
# 1 von Neumann approach
# ---------------------------------------------------------------------------------------------------
def generate_kern_1vN(self):
    """
    Generates a kernel (Liouvillian) matrix corresponding to first order von Neumann approach (1vN).

    Parameters
    ----------
    self : Approach
        Approach object.

    self.kern : array
        (Modifies) Kernel matrix for 1vN approach.
    """
    (E, Tba, phi1fct, si) = (self.qd.Ea, self.leads.Tba, self.phi1fct, self.si)

    self.kern_ext = np.zeros((si.ndm0r+1, si.ndm0r), dtype=doublenp)
    self.kern = self.kern_ext[0:-1, :]

    generate_norm_vec(self, si.ndm0r)
    kern = self.kern
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(si.statesdm[charge], 2):
            bbp = si.get_ind_dm0(b, bp, charge)
            bbp_bool = si.get_ind_dm0(b, bp, charge, 2)
            if bbp != -1 and bbp_bool:
                bbpi = si.ndm0 + bbp - si.npauli
                bbpi_bool = True if bbpi >= si.ndm0 else False
                if bbpi_bool:
                    kern[bbp, bbpi] += E[b]-E[bp]
                    kern[bbpi, bbp] += E[bp]-E[b]
                # --------------------------------------------------
                for a, ap in itertools.product(si.statesdm[charge-1], si.statesdm[charge-1]):
                    aap = si.get_ind_dm0(a, ap, charge-1)
                    if aap != -1:
                        bpa = si.get_ind_dm1(bp, a, charge-1)
                        bap = si.get_ind_dm1(b, ap, charge-1)
                        fct_aap = 0
                        for l in range(si.nleads):
                            fct_aap += (+ Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpa, 0].conjugate()
                                        - Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bap, 0])
                        aapi = si.ndm0 + aap - si.npauli
                        aap_sgn = +1 if si.get_ind_dm0(a, ap, charge-1, maptype=3) else -1
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
                        for a in si.statesdm[charge-1]:
                            bpa = si.get_ind_dm1(bp, a, charge-1)
                            for l in range(si.nleads):
                                fct_bppbp += (+Tba[l, b, a]*Tba[l, a, bpp]
                                              * phi1fct[l, bpa, 1].conjugate())
                        for c in si.statesdm[charge+1]:
                            cbp = si.get_ind_dm1(c, bp, charge)
                            for l in range(si.nleads):
                                fct_bppbp += +Tba[l, b, c]*Tba[l, c, bpp]*phi1fct[l, cbp, 0]
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
                        for a in si.statesdm[charge-1]:
                            ba = si.get_ind_dm1(b, a, charge-1)
                            for l in range(si.nleads):
                                fct_bbpp += -Tba[l, bpp, a]*Tba[l, a, bp]*phi1fct[l, ba, 1]
                        for c in si.statesdm[charge+1]:
                            cb = si.get_ind_dm1(c, b, charge)
                            for l in range(si.nleads):
                                fct_bbpp += (-Tba[l, bpp, c]*Tba[l, c, bp]
                                             * phi1fct[l, cb, 0].conjugate())
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
                for c, cp in itertools.product(si.statesdm[charge+1], si.statesdm[charge+1]):
                    ccp = si.get_ind_dm0(c, cp, charge+1)
                    if ccp != -1:
                        cbp = si.get_ind_dm1(c, bp, charge)
                        cpb = si.get_ind_dm1(cp, b, charge)
                        fct_ccp = 0
                        for l in range(si.nleads):
                            fct_ccp += (+ Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cbp, 1]
                                        - Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpb, 1].conjugate())
                        ccpi = si.ndm0 + ccp - si.npauli
                        ccp_sgn = +1 if si.get_ind_dm0(c, cp, charge+1, maptype=3) else -1
                        kern[bbp, ccp] += fct_ccp.imag
                        if ccpi >= si.ndm0:
                            kern[bbp, ccpi] += fct_ccp.real*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] += fct_ccp.imag*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] -= fct_ccp.real
                # --------------------------------------------------
    return 0


def generate_current_1vN(self):
    """
    Calculates currents using 1vN approach.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.phi1 : array
        (Modifies) Values of first order density matrix elements
        stored in nleads by ndm1 numpy array.
    self.current : array
        (Modifies) Values of the current having nleads entries.
    self.energy_current : array
        (Modifies) Values of the energy current having nleads entries.
    self.heat_current : array
        (Modifies) Values of the heat current having nleads entries.
    """
    (phi0p, E, Tba, phi1fct, phi1fct_energy, si) = (self.phi0, self.qd.Ea, self.leads.Tba,
                                                    self.phi1fct, self.phi1fct_energy, self.si)
    phi1 = np.zeros((si.nleads, si.ndm1), dtype=complexnp)
    current = np.zeros(si.nleads, dtype=complexnp)
    energy_current = np.zeros(si.nleads, dtype=complexnp)
    #
    phi0 = np.zeros(si.ndm0, dtype=complexnp)
    phi0[0:si.npauli] = phi0p[0:si.npauli]
    phi0[si.npauli:si.ndm0] = phi0p[si.npauli:si.ndm0] + 1j*phi0p[si.ndm0:]
    #
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
                        bpb_conj = si.get_ind_dm0(bp, b, bcharge, maptype=3)
                        phi0bpb = phi0[bpb] if bpb_conj else phi0[bpb].conjugate()
                        phi1[l, cb] += Tba[l, c, bp]*phi0bpb*fct1
                        current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1
                        energy_current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1h
                for cp in si.statesdm[ccharge]:
                    ccp = si.get_ind_dm0(c, cp, ccharge)
                    if ccp != -1:
                        ccp_conj = si.get_ind_dm0(c, cp, ccharge, maptype=3)
                        phi0ccp = phi0[ccp] if ccp_conj else phi0[ccp].conjugate()
                        phi1[l, cb] += Tba[l, cp, b]*phi0ccp*fct2
                        current[l] += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2
                        energy_current[l] += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2h
    self.phi1 = phi1
    self.current = np.array(-2*current.imag, dtype=doublenp)
    self.energy_current = np.array(-2*energy_current.imag, dtype=doublenp)
    self.heat_current = self.energy_current - self.current*self.leads.mulst
    return 0


def generate_vec_1vN(self, phi0p):
    """
    Acts on given phi0p with Liouvillian of 1vN approach.

    Parameters
    ----------
    phi0p : array
        Some values of zeroth order density matrix elements.
    self : Approach
        Approach object.

    Returns
    -------
    phi0 : array
        Values of zeroth order density matrix elements
        after acting with Liouvillian, i.e., phi0=L(phi0p).
    """
    (E, Tba, phi1fct, si, norm_row) = (self.qd.Ea, self.leads.Tba, self.phi1fct,
                                       self.si, self.funcp.norm_row)
    #
    phi0 = np.zeros(si.ndm0, dtype=complexnp)
    phi0[0:si.npauli] = phi0p[0:si.npauli]
    phi0[si.npauli:si.ndm0] = phi0p[si.npauli:si.ndm0] + 1j*phi0p[si.ndm0:]
    #
    i_dphi0_dt = np.zeros(si.ndm0, dtype=complexnp)
    norm = 0
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(si.statesdm[charge], 2):
            bbp = si.get_ind_dm0(b, bp, charge)
            if bbp != -1:
                if b == bp:
                    norm += phi0[bbp]
                bbp_bool = si.get_ind_dm0(b, bp, charge, maptype=2)
                if bbp_bool:
                    i_dphi0_dt[bbp] += (E[b]-E[bp])*phi0[bbp]
                    # --------------------------------------------------
                    for a, ap in itertools.product(si.statesdm[charge-1], si.statesdm[charge-1]):
                        aap = si.get_ind_dm0(a, ap, charge-1)
                        if aap != -1:
                            bpa = si.get_ind_dm1(bp, a, charge-1)
                            bap = si.get_ind_dm1(b, ap, charge-1)
                            fct_aap = 0
                            for l in range(si.nleads):
                                fct_aap += (+ Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpa, 0].conjugate()
                                            - Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bap, 0])
                            phi0aap = (phi0[aap] if si.get_ind_dm0(a, ap, charge-1, maptype=3)
                                       else phi0[aap].conjugate())
                            i_dphi0_dt[bbp] += fct_aap*phi0aap
                    # --------------------------------------------------
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
                            phi0bppbp = (phi0[bppbp] if si.get_ind_dm0(bpp, bp, charge, maptype=3)
                                         else phi0[bppbp].conjugate())
                            i_dphi0_dt[bbp] += fct_bppbp*phi0bppbp
                        # --------------------------------------------------
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
                                    fct_bbpp += (-Tba[l, bpp, c]*Tba[l, c, bp]
                                                 * phi1fct[l, cb, 0].conjugate())
                            phi0bbpp = (phi0[bbpp] if si.get_ind_dm0(b, bpp, charge, maptype=3)
                                        else phi0[bbpp].conjugate())
                            i_dphi0_dt[bbp] += fct_bbpp*phi0bbpp
                    # --------------------------------------------------
                    for c, cp in itertools.product(si.statesdm[charge+1], si.statesdm[charge+1]):
                        ccp = si.get_ind_dm0(c, cp, charge+1)
                        if ccp != -1:
                            cbp = si.get_ind_dm1(c, bp, charge)
                            cpb = si.get_ind_dm1(cp, b, charge)
                            fct_ccp = 0
                            for l in range(si.nleads):
                                fct_ccp += (+ Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cbp, 1]
                                            - Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpb, 1].conjugate())
                            phi0ccp = (phi0[ccp] if si.get_ind_dm0(c, cp, charge+1, maptype=3)
                                       else phi0[ccp].conjugate())
                            i_dphi0_dt[bbp] += fct_ccp*phi0ccp
                    # --------------------------------------------------
    i_dphi0_dt[norm_row] = 1j*(norm-1)
    return np.concatenate((i_dphi0_dt.imag, i_dphi0_dt[si.npauli:si.ndm0].real))


class Approach1vN(Approach):

    kerntype = 'py1vN'
    generate_fct = generate_phi1fct
    generate_kern = generate_kern_1vN
    generate_current = generate_current_1vN
    generate_vec = generate_vec_1vN
# ---------------------------------------------------------------------------------------------------
