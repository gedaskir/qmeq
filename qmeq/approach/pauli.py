"""Module containing python functions, which generate first order Pauli kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ..mytypes import complexnp
from ..mytypes import doublenp

from ..specfunc import func_pauli
from ..aprclass import Approach

def generate_paulifct(sys):
    """
    Make factors used for generating Pauli master equation kernel.

    Parameters
    ----------
    sys : Approach
        Approach object.

    Modifies:
    sys.paulifct : array
        Factors used for generating Pauli master equation kernel.
    """
    (E, Tba, si, mulst, tlst, dlst) = (sys.qd.Ea, sys.leads.Tba, sys.si,
                                       sys.leads.mulst, sys.leads.tlst, sys.leads.dlst)
    itype = sys.funcp.itype
    paulifct = np.zeros((si.nleads, si.ndm1, 2), dtype=doublenp)
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = si.get_ind_dm1(c, b, bcharge)
            Ecb = E[c]-E[b]
            for l in range(si.nleads):
                xcb = (Tba[l, b, c]*Tba[l, c, b]).real
                rez = func_pauli(Ecb, mulst[l], tlst[l], dlst[l,0], dlst[l,1], itype)
                paulifct[l, cb, 0] = xcb*rez[0]
                paulifct[l, cb, 1] = xcb*rez[1]
    sys.paulifct = paulifct
    return 0

#---------------------------------------------------------------------------------------------------
# Pauli master equation
#---------------------------------------------------------------------------------------------------
def generate_kern_pauli(sys):
    """
    Generate Pauli master equation kernel.

    Parameters
    ----------
    sys : Approach
        Approach object.

    Modifies:
    sys.kern : array
        Kernel matrix for Pauli master equation.
    sys.bvec : array
        Right hand side column vector for master equation.
        The entry funcp.norm_row is 1 representing normalization condition.
    """
    (paulifct, si, symq, norm_rowp) = (sys.paulifct, sys.si, sys.funcp.symq, sys.funcp.norm_row)
    norm_row = norm_rowp if symq else si.npauli
    last_row = si.npauli-1 if symq else si.npauli
    kern = np.zeros((last_row+1, si.npauli), dtype=doublenp)
    bvec = np.zeros(last_row+1, dtype=doublenp)
    bvec[norm_row] = 1
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            bb_bool = si.get_ind_dm0(b, b, charge, 2)
            kern[norm_row, bb] += 1
            if not (symq and bb == norm_row) and bb_bool:
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
    sys.kern = kern
    sys.bvec = bvec
    return 0

def generate_current_pauli(sys):
    """
    Calculates currents using Pauli master equation approach.

    Parameters
    ----------
    sys : Approach
        Approach object.

    Modifies:
    sys.current : array
        Values of the current having nleads entries.
    sys.energy_current : array
        Values of the energy current having nleads entries.
    sys.heat_current : array
        Values of the heat current having nleads entries.
    """
    (phi0, E, paulifct, si) = (sys.phi0, sys.qd.Ea, sys.paulifct, sys.si)
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
    sys.current = current
    sys.energy_current = energy_current
    sys.heat_current = energy_current - current*sys.leads.mulst
    return 0

def generate_vec_pauli(phi0, sys):
    """
    Acts on given phi0 with Liouvillian of Pauli approach.

    Parameters
    ----------
    phi0 : array
        Some values of zeroth order density matrix elements.
    sys : Approach
        Approach object.

    Returns
    -------
    dphi0_dt : array
        Values of zeroth order density matrix elements
        after acting with Liouvillian, i.e., dphi0_dt=L(phi0p).
    """
    (paulifct, si, norm_row) = (sys.paulifct, sys.si, sys.funcp.norm_row)
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

class Approach_pyPauli(Approach):

    kerntype = 'pyPauli'
    generate_fct = staticmethod(generate_paulifct)
    generate_kern = staticmethod(generate_kern_pauli)
    generate_current = staticmethod(generate_current_pauli)
    generate_vec = staticmethod(generate_vec_pauli)
#---------------------------------------------------------------------------------------------------
