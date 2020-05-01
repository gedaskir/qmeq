"""Module containing python functions, which generate first order Pauli kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import doublenp

from ...specfunc.specfunc import func_pauli
from ..aprclass import Approach


def generate_paulifct(self):
    """
    Make factors used for generating Pauli master equation kernel.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.paulifct : array
        (Modifies) Factors used for generating Pauli master equation kernel.
    """
    (E, Tba, si, mulst, tlst, dlst) = (self.qd.Ea, self.leads.Tba, self.si,
                                       self.leads.mulst, self.leads.tlst, self.leads.dlst)
    itype = self.funcp.itype
    paulifct = np.zeros((si.nleads, si.ndm1, 2), dtype=doublenp)
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = si.get_ind_dm1(c, b, bcharge)
            Ecb = E[c]-E[b]
            for l in range(si.nleads):
                xcb = (Tba[l, b, c]*Tba[l, c, b]).real
                rez = func_pauli(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                paulifct[l, cb, 0] = xcb*rez[0]
                paulifct[l, cb, 1] = xcb*rez[1]
    self.paulifct = paulifct
    return 0


# ---------------------------------------------------------------------------------------------------
# Pauli master equation
# ---------------------------------------------------------------------------------------------------
def generate_kern_pauli(self):
    """
    Generate Pauli master equation kernel.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.kern : array
        (Modifies) Kernel matrix for Pauli master equation.
    self.bvec : array
        (Modifies) Right hand side column vector for master equation.
        The entry funcp.norm_row is 1 representing normalization condition.
    """
    (paulifct, si, kern) = (self.paulifct, self.si, self.kern)

    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            bb_bool = si.get_ind_dm0(b, b, charge, 2)
            if bb_bool:
                for a in si.statesdm[charge-1]:
                    aa = si.get_ind_dm0(a, a, charge-1)
                    ba = si.get_ind_dm1(b, a, charge-1)
                    for l in range(si.nleads):
                        kern[bb, bb] -= paulifct[l, ba, 1]
                        kern[bb, aa] += paulifct[l, ba, 0]
                for c in si.statesdm[charge+1]:
                    cc = si.get_ind_dm0(c, c, charge+1)
                    cb = si.get_ind_dm1(c, b, charge)
                    for l in range(si.nleads):
                        kern[bb, bb] -= paulifct[l, cb, 0]
                        kern[bb, cc] += paulifct[l, cb, 1]
    return 0


def generate_current_pauli(self):
    """
    Calculates currents using Pauli master equation approach.

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
    (phi0, E, paulifct, si) = (self.phi0, self.qd.Ea, self.paulifct, self.si)
    current = np.zeros(si.nleads, dtype=doublenp)
    energy_current = np.zeros(si.nleads, dtype=doublenp)
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c in si.statesdm[ccharge]:
            cc = si.get_ind_dm0(c, c, ccharge)
            for b in si.statesdm[bcharge]:
                bb = si.get_ind_dm0(b, b, bcharge)
                cb = si.get_ind_dm1(c, b, bcharge)
                for l in range(si.nleads):
                    fct1 = +phi0[bb]*paulifct[l, cb, 0]
                    fct2 = -phi0[cc]*paulifct[l, cb, 1]
                    current[l] += fct1 + fct2
                    energy_current[l] += -(E[b]-E[c])*(fct1 + fct2)
    self.current = current
    self.energy_current = energy_current
    self.heat_current = energy_current - current*self.leads.mulst
    return 0


def generate_vec_pauli(self, phi0):
    """
    Acts on given phi0 with Liouvillian of Pauli approach.

    Parameters
    ----------
    phi0 : ndarray
        Some values of zeroth order density matrix elements.
    self : Approach
        Approach object.

    Returns
    -------
    dphi0_dt : array
        Values of zeroth order density matrix elements
        after acting with Liouvillian, i.e., dphi0_dt=L(phi0p).
    """
    (paulifct, si, norm_row) = (self.paulifct, self.si, self.funcp.norm_row)
    dphi0_dt = np.zeros(si.npauli, dtype=doublenp)
    norm = 0
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            bb_bool = si.get_ind_dm0(b, b, charge, 2)
            norm += phi0[bb]
            if bb_bool:
                for a in si.statesdm[charge-1]:
                    aa = si.get_ind_dm0(a, a, charge-1)
                    ba = si.get_ind_dm1(b, a, charge-1)
                    for l in range(si.nleads):
                        dphi0_dt[bb] -= paulifct[l, ba, 1]*phi0[bb]
                        dphi0_dt[bb] += paulifct[l, ba, 0]*phi0[aa]
                for c in si.statesdm[charge+1]:
                    cc = si.get_ind_dm0(c, c, charge+1)
                    cb = si.get_ind_dm1(c, b, charge)
                    for l in range(si.nleads):
                        dphi0_dt[bb] -= paulifct[l, cb, 0]*phi0[bb]
                        dphi0_dt[bb] += paulifct[l, cb, 1]*phi0[cc]
    dphi0_dt[norm_row] = norm-1
    return dphi0_dt


class ApproachPauli(Approach):

    kerntype = 'pyPauli'

    def get_kern_size(self):
        return self.si.npauli

    generate_fct = generate_paulifct
    generate_kern = generate_kern_pauli
    generate_current = generate_current_pauli
    generate_vec = generate_vec_pauli
# ---------------------------------------------------------------------------------------------------
