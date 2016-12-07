"""Module containing python functions, which generate first order Pauli kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ..mytypes import complexnp
from ..mytypes import doublenp

from ..specfunc import func_pauli

def generate_paulifct(sys):
    """
    Make factors used for generating Pauli master equation kernel.

    Parameters
    ----------
    sys : Transport
        Transport object.

    Returns
    -------
    paulifct : array
        Factors used for generating Pauli master equation kernel.
    """
    (E, Tba, si, mulst, tlst, dlst) = (sys.qd.Ea, sys.leads.Tba, sys.si, sys.leads.mulst, sys.leads.tlst, sys.leads.dlst)
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
    return paulifct

#---------------------------------------------------------------------------------------------------------
# Pauli master equation
#---------------------------------------------------------------------------------------------------------
def generate_kern_pauli(sys):
    """
    Generate Pauli master equation kernel.

    Parameters
    ----------
    sys : Transport
        Transport object.

    Returns
    -------
    kern : array
        Kernel matrix for Pauli master equation.
    bvec : array
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
    return kern, bvec

def generate_current_pauli(sys):
    """
    Calculates currents using Pauli master equation method.

    Parameters
    ----------
    sys : Transport
        Transport object.

    Returns
    -------
    current : array
        Values of the current having nleads entries.
    energy_current : array
        Values of the energy current having nleads entries.
    """
    (phi0, E, paulifct, si) = (sys.phi0, sys.qd.Ea, sys.paulifct, sys.si)
    current = np.zeros(si.nleads, dtype=complexnp)
    energy_current = np.zeros(si.nleads, dtype=complexnp)
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
    return current, energy_current
