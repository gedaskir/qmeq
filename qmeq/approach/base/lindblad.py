"""Module containing python functions, which generate first order Lindblad kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import complexnp
from ...mytypes import doublenp

from ...specfunc.specfunc import func_pauli
from ..aprclass import Approach
from .pauli import generate_norm_vec


# ---------------------------------------------------------------------------------------------------
# Lindblad approach
# ---------------------------------------------------------------------------------------------------
def generate_tLba(self):
    """
    Make factors used for generating Lindblad master equation kernel.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.tLba : array
        (Modifies) Jump operator matrix in many-body basis.
    """
    (Tba, E, si) = (self.leads.Tba, self.qd.Ea, self.si)
    (mulst, tlst, dlst) = (self.leads.mulst, self.leads.tlst, self.leads.dlst)
    (mtype, itype) = (self.leads.mtype, self.funcp.itype)
    #
    tLba = np.zeros(Tba.shape, dtype=mtype)
    for charge in range(si.ncharge-1):
        bcharge = charge+1
        acharge = charge
        for b, a in itertools.product(si.statesdm[bcharge], si.statesdm[acharge]):
            Eba = E[b]-E[a]
            for l in range(si.nleads):
                fct1, fct2 = func_pauli(Eba, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                tLba[l, b, a] = np.sqrt(fct1)*Tba[l, b, a]
                tLba[l, a, b] = np.sqrt(fct2)*Tba[l, a, b]
    self.tLba = tLba
    return 0


def generate_kern_lindblad(self):
    """
    Generates a kernel (Liouvillian) matrix corresponding to Lindblad approach.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.kern : array
        (Modifies) Kernel matrix for first-order Lindblad approach.
    self.bvec : array
        (Modifies) Right hand side column vector for master equation.
        The entry funcp.norm_row is 1 representing normalization condition.
    """
    (E, tLba, si) = (self.qd.Ea, self.tLba, self.si)

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
                        fct_aap = 0
                        for l in range(si.nleads):
                            fct_aap += tLba[l, b, a]*tLba[l, bp, ap].conjugate()
                        aapi = si.ndm0 + aap - si.npauli
                        aap_sgn = +1 if si.get_ind_dm0(a, ap, charge-1, maptype=3) else -1
                        kern[bbp, aap] += fct_aap.real
                        if aapi >= si.ndm0:
                            kern[bbp, aapi] -= fct_aap.imag*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] += fct_aap.real*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] += fct_aap.imag
                # --------------------------------------------------
                for bpp in si.statesdm[charge]:
                    bppbp = si.get_ind_dm0(bpp, bp, charge)
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[charge-1]:
                            for l in range(si.nleads):
                                fct_bppbp += -0.5*tLba[l, a, b].conjugate()*tLba[l, a, bpp]
                        for c in si.statesdm[charge+1]:
                            for l in range(si.nleads):
                                fct_bppbp += -0.5*tLba[l, c, b].conjugate()*tLba[l, c, bpp]
                        bppbpi = si.ndm0 + bppbp - si.npauli
                        bppbp_sgn = +1 if si.get_ind_dm0(bpp, bp, charge, maptype=3) else -1
                        kern[bbp, bppbp] += fct_bppbp.real
                        if bppbpi >= si.ndm0:
                            kern[bbp, bppbpi] -= fct_bppbp.imag*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] += fct_bppbp.real*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] += fct_bppbp.imag
                    # --------------------------------------------------
                    bbpp = si.get_ind_dm0(b, bpp, charge)
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[charge-1]:
                            for l in range(si.nleads):
                                fct_bbpp += -0.5*tLba[l, a, bpp].conjugate()*tLba[l, a, bp]
                        for c in si.statesdm[charge+1]:
                            for l in range(si.nleads):
                                fct_bbpp += -0.5*tLba[l, c, bpp].conjugate()*tLba[l, c, bp]
                        bbppi = si.ndm0 + bbpp - si.npauli
                        bbpp_sgn = +1 if si.get_ind_dm0(b, bpp, charge, maptype=3) else -1
                        kern[bbp, bbpp] += fct_bbpp.real
                        if bbppi >= si.ndm0:
                            kern[bbp, bbppi] -= fct_bbpp.imag*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] += fct_bbpp.real*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] += fct_bbpp.imag
                # --------------------------------------------------
                for c, cp in itertools.product(si.statesdm[charge+1], si.statesdm[charge+1]):
                    ccp = si.get_ind_dm0(c, cp, charge+1)
                    if ccp != -1:
                        fct_ccp = 0
                        for l in range(si.nleads):
                            fct_ccp += tLba[l, b, c]*tLba[l, bp, cp].conjugate()
                        ccpi = si.ndm0 + ccp - si.npauli
                        ccp_sgn = +1 if si.get_ind_dm0(c, cp, charge+1, maptype=3) else -1
                        kern[bbp, ccp] += fct_ccp.real
                        if ccpi >= si.ndm0:
                            kern[bbp, ccpi] -= fct_ccp.imag*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] += fct_ccp.real*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] += fct_ccp.imag
                # --------------------------------------------------
    return 0


def generate_current_lindblad(self):
    """
    Calculates currents using Lindblad approach.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.current : array
        (Modifies) Values of the current having nleads entries.
    self.energy_current : array
        (Modifies) Values of the energy current having nleads entries.
    self.heat_current : array
        (Modifies) Values of the heat current having nleads entries.
    """
    (phi0p, E, tLba, si) = (self.phi0, self.qd.Ea, self.tLba, self.si)
    current = np.zeros(si.nleads, dtype=complexnp)
    energy_current = np.zeros(si.nleads, dtype=complexnp)
    #
    phi0 = np.zeros(si.ndm0, dtype=complexnp)
    phi0[0:si.npauli] = phi0p[0:si.npauli]
    phi0[si.npauli:si.ndm0] = phi0p[si.npauli:si.ndm0] + 1j*phi0p[si.ndm0:]
    #
    for charge in range(si.ncharge):
        ccharge = charge+1
        bcharge = charge
        acharge = charge-1
        for b, bp in itertools.product(si.statesdm[bcharge], si.statesdm[bcharge]):
            bbp = si.get_ind_dm0(b, bp, bcharge)
            if bbp != -1:
                bbp_conj = si.get_ind_dm0(b, bp, bcharge, maptype=3)
                phi0bbp = phi0[bbp] if bbp_conj else phi0[bbp].conjugate()
                for l in range(si.nleads):
                    for a in si.statesdm[acharge]:
                        fcta = tLba[l, a, b]*phi0bbp*tLba[l, a, bp].conjugate()
                        current[l] -= fcta
                        energy_current[l] += (E[a]-0.5*(E[b]+E[bp]))*fcta
                    for c in si.statesdm[ccharge]:
                        fctc = tLba[l, c, b]*phi0bbp*tLba[l, c, bp].conjugate()
                        current[l] += fctc
                        energy_current[l] += (E[c]-0.5*(E[b]+E[bp]))*fctc
    self.current = np.array(current.real, dtype=doublenp)
    self.energy_current = np.array(energy_current.real, dtype=doublenp)
    self.heat_current = self.energy_current - self.current*self.leads.mulst
    return 0


def generate_vec_lindblad(self, phi0p):
    """
    Acts on given phi0p with Liouvillian of Lindblad approach.

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
    (E, tLba, si, norm_row) = (self.qd.Ea, self.tLba, self.si, self.funcp.norm_row)
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
                            fct_aap = 0
                            for l in range(si.nleads):
                                fct_aap += tLba[l, b, a]*tLba[l, bp, ap].conjugate()
                            phi0aap = (phi0[aap] if si.get_ind_dm0(a, ap, charge-1, maptype=3)
                                       else phi0[aap].conjugate())
                            i_dphi0_dt[bbp] += 1.0j*fct_aap*phi0aap
                    # --------------------------------------------------
                    for bpp in si.statesdm[charge]:
                        bppbp = si.get_ind_dm0(bpp, bp, charge)
                        if bppbp != -1:
                            fct_bppbp = 0
                            for a in si.statesdm[charge-1]:
                                for l in range(si.nleads):
                                    fct_bppbp += -0.5*tLba[l, a, b].conjugate()*tLba[l, a, bpp]
                            for c in si.statesdm[charge+1]:
                                for l in range(si.nleads):
                                    fct_bppbp += -0.5*tLba[l, c, b].conjugate()*tLba[l, c, bpp]
                            phi0bppbp = (phi0[bppbp] if si.get_ind_dm0(bpp, bp, charge, maptype=3)
                                         else phi0[bppbp].conjugate())
                            i_dphi0_dt[bbp] += 1.0j*fct_bppbp*phi0bppbp
                        # --------------------------------------------------
                        bbpp = si.get_ind_dm0(b, bpp, charge)
                        if bbpp != -1:
                            fct_bbpp = 0
                            for a in si.statesdm[charge-1]:
                                for l in range(si.nleads):
                                    fct_bbpp += -0.5*tLba[l, a, bpp].conjugate()*tLba[l, a, bp]
                            for c in si.statesdm[charge+1]:
                                for l in range(si.nleads):
                                    fct_bbpp += -0.5*tLba[l, c, bpp].conjugate()*tLba[l, c, bp]
                            phi0bbpp = (phi0[bbpp] if si.get_ind_dm0(b, bpp, charge, maptype=3)
                                        else phi0[bbpp].conjugate())
                            i_dphi0_dt[bbp] += 1.0j*fct_bbpp*phi0bbpp
                    # --------------------------------------------------
                    for c, cp in itertools.product(si.statesdm[charge+1], si.statesdm[charge+1]):
                        ccp = si.get_ind_dm0(c, cp, charge+1)
                        if ccp != -1:
                            fct_ccp = 0
                            for l in range(si.nleads):
                                fct_ccp += tLba[l, b, c]*tLba[l, bp, cp].conjugate()
                            phi0ccp = (phi0[ccp] if si.get_ind_dm0(c, cp, charge+1, maptype=3)
                                       else phi0[ccp].conjugate())
                            i_dphi0_dt[bbp] += 1.0j*fct_ccp*phi0ccp
                    # --------------------------------------------------
    i_dphi0_dt[norm_row] = 1j*(norm-1)
    return np.concatenate((i_dphi0_dt.imag, i_dphi0_dt[si.npauli:si.ndm0].real))


class ApproachLindblad(Approach):

    kerntype = 'pyLindblad'
    generate_fct = generate_tLba
    generate_kern = generate_kern_lindblad
    generate_current = generate_current_lindblad
    generate_vec = generate_vec_lindblad
# ---------------------------------------------------------------------------------------------------
