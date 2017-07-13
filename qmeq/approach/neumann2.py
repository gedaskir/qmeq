"""Module containing python functions, which solve 2vN approach integral equations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy import pi
import itertools

from ..mytypes import complexnp
from ..mytypes import doublenp
from ..mytypes import intnp

from ..specfunc import kernel_fredriksen
from ..specfunc import hilbert_fredriksen
from ..aprclass import Approach2vN

def func_2vN(Ek, Ek_grid, l, eta, hfk):
    """
    Linearly interpolate the value of hfk on Ek_grid at point Ek.

    Parameters
    ----------
    Ek : float
        Energy value (not necessarily a grid point).
    Ek_grid : array
        Energy grid.
    l : int
        Lead label.
    eta : int
        Integer describing (+/-1) if infinitesimal eta is positive or negative.
    hfk : array
        Array containing Hilbert transform of Fermi function (or 1-Fermi).

    Returns
    -------
    float
        Interpolated value of hfk at Ek.
    """
    if Ek<Ek_grid[0] or Ek>Ek_grid[-1]:
        return 0
    #
    b_idx = int((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0]))+1
    if b_idx == len(Ek_grid): b_idx -= 1
    a_idx = b_idx - 1
    b, a = Ek_grid[b_idx], Ek_grid[a_idx]
    #
    fb = hfk[l, b_idx]
    fa = hfk[l, a_idx]
    rez = (fb-fa)/(b-a)*Ek + (b*fa-a*fb)/(b-a)
    return pi*rez if eta+1 else pi*rez.conjugate()

def get_at_k1(Ek, Ek_grid, l, cb, conj, phi1k, hphi1k):
    """
    Linearly interpolate the values of phi1k, hphi1k on Ek_grid at point Ek.

    Parameters
    ----------
    Ek : float
        Energy value (not necessarily a grid point).
    Ek_grid : array
        Energy grid.
    l : int
        Lead label.
    cb : int
        Index corresponding to Phi[1](k) matrix element.
    conj : bool
        If conj=True the term in the integral equation is conjugated.
    phi1k : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0),
        cotaining energy resolved first order density matrix elements Phi[1](k).
    hphi1k : array
        Numpy array with dimensions (len(Ek_grid_ext), nleads, ndm1, ndm0),
        containing Hilbert transform of phi1k.

    Returns
    -------
    array, array
        Interpolated values of phi1k and hphi1k at Ek.
    """
    if Ek<Ek_grid[0] or Ek>Ek_grid[-1]:
        return 0, 0
    #
    #b_idx = np.searchsorted(Ek_grid, Ek)
    b_idx = int((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0]))+1
    if b_idx == len(Ek_grid): b_idx -= 1
    a_idx = b_idx - 1
    b, a = Ek_grid[b_idx], Ek_grid[a_idx]
    #print(a_idx, b_idx, a, Ek, b)
    #
    fb = phi1k[b_idx, l, cb]
    fa = phi1k[a_idx, l, cb]
    u = (fb-fa)/(b-a)*Ek + (b*fa-a*fb)/(b-a)
    u = u.conjugate() if conj else u
    #
    fb = hphi1k[b_idx, l, cb]
    fa = hphi1k[a_idx, l, cb]
    hu = (fb-fa)/(b-a)*Ek + (b*fa-a*fb)/(b-a)
    hu = hu.conjugate() if conj else hu
    return pi*hu, pi*u

def phi1k_local_2vN(ind, Ek_grid, fk, hfkp, hfkm, E, Tba, si):
    """
    Constructs Phi[1](k) corresponding to local approximation at Ek_grid[ind].
    More precisely, generates local approximation kernels L1 and L0, Phi[1](k) = L0(k)Phi[0],
    L1(k)Phi[1](k) = L0p(k)Phi[0], L0 = L1^{-1}L0p.

    Parameters
    ----------
    ind : int
        Index of a point on a Ek_grid.
    Ek_grid : array
        Energy grid.
    fk : array
        nleads by len(Ek_grid) numpy array containing Fermi function.
    hfkp,hfkm : array
        Hilbert transforms of fk and 1-fk.
    E : array
        nmany by 1 array containing Hamiltonian eigenvalues.
    Tba : array
        nmany by nmany array, which contains many-body tunneling amplitude matrix.
    si : StateIndexingDM
        StateIndexingDM object.

    Returns
    -------
    kern0 : array
        Numpy array with dimensions (nleads, ndm1, ndm0).
        Gives local approximation kernel L0(k) at Ek_grid[ind].
        Shows how Phi[1](k) is expressed in terms of Phi[0].
    kern1 : array
        Numpy array with dimensions (nleads, ndm1, ndm1).
        Gives local approximation kernel L1(k) at Ek_grid[ind].
    """
    Ek = Ek_grid[ind]
    kern0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
    kern1 = np.zeros((si.nleads, si.ndm1, si.ndm1), dtype=complexnp)
    for charge in range(si.ncharge-1):
        dcharge = charge+2
        ccharge = charge+1
        bcharge = charge
        acharge = charge-1
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = si.get_ind_dm1(c, b, bcharge)
            for l in range(si.nleads):
                # Note that the bias is put in the distributions and not in the dispersion
                kern1[l, cb, cb] += Ek-E[c]+E[b]
                fp = fk[l, ind] #fermi_func(+(Ek-mulst[l])/tlst[l])
                fm = 1-fp       #fermi_func(-(Ek-mulst[l])/tlst[l])
                # Phi[0] terms
                for b1 in si.statesdm[bcharge]:
                    b1b = si.get_ind_dm0(b1, b, bcharge)
                    kern0[l, cb, b1b] += +Tba[l, c, b1]*fp
                for c1 in si.statesdm[ccharge]:
                    cc1 = si.get_ind_dm0(c, c1, ccharge)
                    kern0[l, cb, cc1] += -Tba[l, c1, b]*fm
                #---------------------------------------------------------------------------
                # Phi[1] terms
                # 2nd and 7th terms
                for b1, a1 in itertools.product(si.statesdm[bcharge], si.statesdm[acharge]):
                    b1a1 = si.get_ind_dm1(b1, a1, acharge)
                    for l1 in range(si.nleads):
                        kern1[l, cb, b1a1] -= +Tba[l1, c, b1]*Tba[l1, a1, b]*(
                                                +func_2vN(+(Ek-E[b1]+E[b]), Ek_grid, l1, +1, hfkp)
                                                -func_2vN(-(Ek-E[c]+E[a1]), Ek_grid, l1, -1, hfkp) )
                # 6th and 8th terms
                for b1 in si.statesdm[bcharge]:
                    cb1 = si.get_ind_dm1(c, b1, bcharge)
                    for l1 in range(si.nleads):
                        for c1 in si.statesdm[ccharge]:
                            kern1[l, cb, cb1] -= ( +Tba[l1, b1, c1]*Tba[l1, c1, b]
                                                *func_2vN(+(Ek-E[c]+E[c1]), Ek_grid, l1, +1, hfkp) )
                        for a1 in si.statesdm[acharge]:
                            kern1[l, cb, cb1] -= ( -Tba[l1, b1, a1]*Tba[l1, a1, b]
                                                *func_2vN(-(Ek-E[c]+E[a1]), Ek_grid, l1, -1, hfkm) )
                # 1st and 3rd terms
                for c1 in si.statesdm[ccharge]:
                    c1b = si.get_ind_dm1(c1, b, bcharge)
                    for l1 in range(si.nleads):
                        for b1 in si.statesdm[bcharge]:
                            kern1[l, cb, c1b] -= ( +Tba[l1, c, b1]*Tba[l1, b1, c1]
                                                *func_2vN(+(Ek-E[b1]+E[b]), Ek_grid, l1, +1, hfkm) )
                        for d1 in si.statesdm[dcharge]:
                            kern1[l, cb, c1b] -= ( -Tba[l1, c, d1]*Tba[l1, d1, c1]
                                                *func_2vN(-(Ek-E[d1]+E[b]), Ek_grid, l1, -1, hfkp) )
                # 5th and 4th terms
                for d1, c1 in itertools.product(si.statesdm[dcharge], si.statesdm[ccharge]):
                    d1c1 = si.get_ind_dm1(d1, c1, ccharge)
                    for l1 in range(si.nleads):
                        kern1[l, cb, d1c1] -= +Tba[l1, c, d1]*Tba[l1, c1, b]*(
                                                +func_2vN(+(Ek-E[c]+E[c1]), Ek_grid, l1, +1, hfkm)
                                                -func_2vN(-(Ek-E[d1]+E[b]), Ek_grid, l1, -1, hfkm) )
    for l in range(si.nleads):
        kern0[l] = np.dot(np.linalg.inv(kern1[l]), kern0[l])
    return kern0, kern1

def phi1k_iterate_2vN(ind, Ek_grid, phi1k, hphi1k, fk, kern1, E, Tba, si):
    """
    Iterates the 2vN integral equation.

    Parameters
    ----------
    ind : int
        Index of a point on a Ek_grid.
    Ek_grid : array
        Energy grid.
    phi1k : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0)
        cotaining difference from phi1k after performing one iteration.
    hphi1k : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0)
        Hilbert transform of phi1k.
    fk : array
        nleads by len(Ek_grid) numpy array containing Fermi function.
    kern1 : array
        Local approximation kernel L1(k) at Ek_grid[ind].
    E : array
        nmany by 1 array containing Hamiltonian eigenvalues.
    Tba : array
        nmany by nmany array, which contains many-body tunneling amplitude matrix.
    si : StateIndexingDM
        StateIndexingDM object.

    Returns
    -------
    kern0 : array
        Numpy array with dimensions (nleads, ndm1, ndm0)
        Gives a correction to Phi[1](k) after iteration of integral equation.
        Shows how delta[Phi[1](k)] is expressed in terms of Phi[0].
    """
    Ek = Ek_grid[ind]
    kern0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
    for charge in range(si.ncharge-1):
        dcharge = charge+2
        ccharge = charge+1
        bcharge = charge
        acharge = charge-1
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = si.get_ind_dm1(c, b, bcharge)
            for l in range(si.nleads):
                fp = fk[l, ind] #fermi_func((Ek-mulst[l])/tlst[l])
                fm = 1-fp
                term = np.zeros(si.ndm0, dtype=complexnp)
                # 1st term
                for a1 in si.statesdm[acharge]:
                    ba1 = si.get_ind_dm1(b, a1, acharge)
                    for b1, l1 in itertools.product(si.statesdm[bcharge], range(si.nleads)):
                        #print('1')
                        hu, u = get_at_k1(+(Ek-E[b1]+E[b]), Ek_grid, l1, ba1, True, phi1k, hphi1k)
                        term += -Tba[l1, c, b1]*Tba[l, b1, a1]*fp*(hu - 1j*u)
                # 2nd and 5th terms
                for b1, c1 in itertools.product(si.statesdm[bcharge], si.statesdm[ccharge]):
                    #print('2 and 5')
                    c1b1 = si.get_ind_dm1(c1, b1, bcharge)
                    for l1 in range(si.nleads):
                        # 2nd term
                        hu, u = get_at_k1(+(Ek-E[b1]+E[b]), Ek_grid, l1, c1b1, True, phi1k, hphi1k)
                        term += -Tba[l1, c, b1]*(hu - 1j*u)*fm*Tba[l, c1, b]
                        # 5th term
                        hu, u = get_at_k1(+(Ek-E[c]+E[c1]), Ek_grid, l1, c1b1, True, phi1k, hphi1k)
                        term += -Tba[l, c, b1]*fp*(hu - 1j*u)*Tba[l1, c1, b]
                # 3rd term
                for c1 in si.statesdm[ccharge]:
                    c1b = si.get_ind_dm1(c1, b, bcharge)
                    for d1, l1 in itertools.product(si.statesdm[dcharge], range(si.nleads)):
                        #print('3')
                        hu, u = get_at_k1(-(Ek-E[d1]+E[b]), Ek_grid, l1, c1b, False, phi1k, hphi1k)
                        term += +Tba[l1, c, d1]*Tba[l, d1, c1]*fp*(hu + 1j*u)
                # 4th term
                for d1, c1 in itertools.product(si.statesdm[dcharge], si.statesdm[ccharge]):
                    #print('4')
                    d1c1 = si.get_ind_dm1(d1, c1, ccharge)
                    for l1 in range(si.nleads):
                        hu, u = get_at_k1(-(Ek-E[d1]+E[b]), Ek_grid, l1, d1c1, False, phi1k, hphi1k)
                        term += +Tba[l1, c, d1]*(hu + 1j*u)*fm*Tba[l, c1, b]
                # 6th term
                for d1 in si.statesdm[dcharge]:
                    d1c = si.get_ind_dm1(d1, c, ccharge)
                    for c1, l1 in itertools.product(si.statesdm[ccharge], range(si.nleads)):
                        #print('6')
                        hu, u = get_at_k1(+(Ek-E[c]+E[c1]), Ek_grid, l1, d1c, True, phi1k, hphi1k)
                        term += -(hu - 1j*u)*fm*Tba[l, d1, c1]*Tba[l1, c1, b]
                # 7th term
                for b1, a1 in itertools.product(si.statesdm[bcharge], si.statesdm[acharge]):
                    #print('7')
                    b1a1 = si.get_ind_dm1(b1, a1, acharge)
                    for l1 in range(si.nleads):
                        hu, u = get_at_k1(-(Ek-E[c]+E[a1]), Ek_grid, l1, b1a1, False, phi1k, hphi1k)
                        term += +Tba[l, c, b1]*fp*(hu + 1j*u)*Tba[l1, a1, b]
                # 8th term
                for b1 in si.statesdm[bcharge]:
                    cb1 = si.get_ind_dm1(c, b1, bcharge)
                    for a1, l1 in itertools.product(si.statesdm[acharge], range(si.nleads)):
                        #print('8')
                        hu, u = get_at_k1(-(Ek-E[c]+E[a1]), Ek_grid, l1, cb1, False, phi1k, hphi1k)
                        term += +(hu + 1j*u)*fm*Tba[l, b1, a1]*Tba[l1, a1, b]
                kern0[l, cb] = term
    for l in range(si.nleads):
        kern0[l] = np.dot(np.linalg.inv(kern1[l]), kern0[l])
    return kern0

def get_phi1_phi0_2vN(sys):
    """
    Integrates phi1k over energy.

    Parameters
    ----------
    sys : Approach2vN
        Approach2vN object.

    Modifies:
    sys.phi1_phi0 : array
        Numpy array with dimensions (nleads, ndm1, ndm0),
        corresponding to energy integrated Phi[1](k).
    sys.e_phi1_phi0 : array
        Numpy array with dimensions (nleads, ndm1, ndm0),
        corresponding to energy integrated Ek*Phi[1](k).
    """
    #(phi1k, Ek_grid, si):
    (phi1k, Ek_grid, si) = (sys.phi1k, sys.Ek_grid, sys.si)
    # Get integrated Phi[1]_{cb} in terms of Phi[0]_{bb'}
    phi1_phi0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
    e_phi1_phi0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
    Eklen = len(Ek_grid)
    #
    for j1 in range(Eklen):
        # Trapezoidal rule
        if j1 == 0:         dx = Ek_grid[j1+1] - Ek_grid[j1]
        elif j1 == Eklen-1: dx = Ek_grid[j1]   - Ek_grid[j1-1]
        else:               dx = Ek_grid[j1+1] - Ek_grid[j1-1]
        phi1_phi0 += 1/2*dx*phi1k[j1]
        e_phi1_phi0 += 1/2*dx*Ek_grid[j1]*phi1k[j1]
    #
    sys.phi1_phi0 = phi1_phi0
    sys.e_phi1_phi0 = e_phi1_phi0
    return 0

def kern_phi0_2vN(sys):
    """
    From Phi[1](k) generate equations containing just Phi[0].

    Parameters
    ----------
    sys : Approach2vN
        Approach2vN object.

    Modifies:
    sys.kern : array
        Kernel for master equation involving just Phi[0].
    sys.bvec : array
        Right hand side column vector for master equation.
    """
    #(phi1_phi0, E, Tba, si, funcp, mulst, tlst, dlst)
    (phi1_phi0, E, Tba, si) = (sys.phi1_phi0, sys.qd.Ea, sys.leads.Tba, sys.si)
    symq = sys.funcp.symq
    norm_rowp = sys.funcp.norm_row
    # Integrated Phi[1]_{bc} in terms of phi1_phi0
    shuffle = np.zeros((si.ndm0, si.ndm0), dtype=intnp)
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(si.statesdm[charge], 2):
            bbp = si.get_ind_dm0(b, bp, charge)
            bpb = si.get_ind_dm0(bp, b, charge)
            shuffle[bbp, bpb] = 1
            shuffle[bpb, bbp] = 1
    phi1_phi0_conj = np.dot(np.conjugate(phi1_phi0), shuffle)
    # Set-up normalisation row
    norm_row = norm_rowp if symq else si.ndm0
    last_row = si.ndm0-1 if symq else si.ndm0
    bvec = np.zeros(last_row+1, dtype=complexnp)
    bvec[norm_row] = 1
    # Make equations for Phi[0]
    kern = np.zeros((last_row+1, si.ndm0), dtype=complexnp)
    norm_vec = np.zeros(si.ndm0, dtype=complexnp)
    for charge in range(si.ncharge):
        acharge = charge-1
        bcharge = charge
        ccharge = charge+1
        for b, bp in itertools.product(si.statesdm[bcharge], si.statesdm[bcharge]):
            bbp = si.get_ind_dm0(b, bp, bcharge)
            kern[bbp, bbp] = E[b]-E[bp]
            if b == bp:
                norm_vec[bbp] = 1
            #
            for a1 in si.statesdm[acharge]:
                bpa1 = si.get_ind_dm1(bp, a1, acharge)
                ba1 = si.get_ind_dm1(b, a1, acharge)
                for l in range(si.nleads):
                    kern[bbp] += +Tba[l, b, a1]*phi1_phi0_conj[l, bpa1]
                    kern[bbp] += -phi1_phi0[l, ba1]*Tba[l, a1, bp]
            #
            for c1 in si.statesdm[ccharge]:
                c1bp = si.get_ind_dm1(c1, bp, bcharge)
                c1b = si.get_ind_dm1(c1, b, bcharge)
                for l in range(si.nleads):
                    kern[bbp] += +Tba[l, b, c1]*phi1_phi0[l, c1bp]
                    kern[bbp] += -phi1_phi0_conj[l, c1b]*Tba[l, c1, bp]
    kern[norm_row] = norm_vec
    #
    sys.kern = kern
    sys.bvec = bvec
    return 0

def generate_current_2vN(sys):
    """
    Finds the currents for 2vN approach after an approximate
    solution for integral 2vN equations was found.

    Parameters
    ----------
    sys : Approach2vN
        Approach2vN object.

    Modifies:
    sys.phi1 : array
        nleads by ndm1 numpy array, which gives energy integrated
        Phi[1](k) after the values for Phi[0] were inserted.
    sys.current : array
        Values of the current having nleads entries.
    sys.energy_current : array
        Values of the energy current having nleads entries.
    sys.heat_current : array
        Values of the heat current having nleads entries.
    """
    #(phi1_phi0, e_phi1_phi0, phi0, Tba, si)
    (phi1_phi0, e_phi1_phi0) = (sys.phi1_phi0, sys.e_phi1_phi0)
    (phi0, Tba, si) = (sys.phi0, sys.leads.Tba, sys.si)
    #
    phi1 = np.zeros((si.nleads, si.ndm1), dtype=complexnp)
    h_phi1 = np.zeros((si.nleads, si.ndm1), dtype=complexnp)
    for l in range(si.nleads):
        phi1[l] = np.dot(phi1_phi0[l], phi0)
        h_phi1[l] = np.dot(e_phi1_phi0[l], phi0)
    #
    current = np.zeros(si.nleads, dtype=complexnp)
    energy_current = np.zeros(si.nleads, dtype=complexnp)
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = si.get_ind_dm1(c, b, bcharge)
            for l in range(si.nleads):
                current[l] += phi1[l, cb]*Tba[l, b, c]
                energy_current[l] = energy_current[l] + h_phi1[l, cb]*Tba[l, b, c]
    sys.phi1 = phi1
    sys.current = np.array(-2*current.imag, dtype=doublenp)
    sys.energy_current = np.array(-2*energy_current.imag, dtype=doublenp)
    sys.heat_current = sys.energy_current - sys.current*sys.leads.mulst
    return 0

def get_emin_emax(sys):
    """
    Finds how much the Ek_grid has to be expanded
    above the bandwidht D of the leads.

    Parameters
    ----------
    sys : Approach2vN
        Approach2vN object.

    Modifies:
    sys.funcp.emin : float
        Minimal energy in the updated Ek_grid.
    sys.funcp.emax : float
        Maximal energy in the updated Ek_grid.
    """
    #(E, si, dband) = (sys.qd.Ea, sys.si, sys.leads.dlst[0,1])
    (E, si, dmin, dmax) = (sys.qd.Ea, sys.si, sys.funcp.dmin, sys.funcp.dmax)
    lst = [dmin, dmax]
    for charge in range(si.ncharge):
        for b, bp in itertools.product(si.statesdm[charge], si.statesdm[charge]):
            lst.append(dmax-E[b]+E[bp])
            lst.append(dmin-E[b]+E[bp])
    for charge in range(si.ncharge-2):
        for d, b in itertools.product(si.statesdm[charge+2], si.statesdm[charge]):
            lst.append(dmin+E[d]-E[b])
            lst.append(dmax+E[d]-E[b])
    sys.funcp.emax = max(lst)
    sys.funcp.emin = min(lst)
    return 0

def get_grid_ext(sys):
    """
    Expands the Ek_grid from emin to emax.

    Parameters
    ----------
    sys : Approach2vN
        Approach2vN object.

    Modifies:
    sys.Ek_grid_ext : array
        Extended Ek_grid from emin to emax.
    sys.funcp.kpnt_left : int
        Number of points Ek_grid is extended to the left.
    sys.funcp.kpnt_right : int
        Number of points Ek_grid is extended to the right.
    """
    (emin_, emax_, dmin, dmax, ext_fct) = (sys.funcp.emin, sys.funcp.emax,
                                           sys.funcp.dmin, sys.funcp.dmax,
                                           sys.funcp.ext_fct)
    Ek_grid = sys.Ek_grid
    step = Ek_grid[1]-Ek_grid[0]
    emin = ext_fct*(emin_-dmin)+dmin
    emax = ext_fct*(emax_-dmax)+dmax
    ext_left = np.sort(-np.arange(-dmin+step, -emin+step, step))
    ext_right = np.arange(dmax+step, emax+step, step)
    sys.Ek_grid_ext = np.concatenate((ext_left, Ek_grid, ext_right))
    sys.funcp.kpnt_left, sys.funcp.kpnt_right = len(ext_left), len(ext_right)
    return 0

def get_htransf_phi1k(phi1k, funcp):
    """
    Performs Hilbert transform of phi1k.

    Parameters
    ----------
    phi1k : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0)
        cotaining energy resolved first order density matrix elements Phi[1](k).
    funcp : FunctionProperties
        FunctionProperties object.

    Modifies:
    funcp.ht_ker : array
        Kernel used when performing Hilbert transform using FFT.

    Returns
    -------
    phi1k : array
        Numpy array with dimensions (len(Ek_grid_ext), nleads, ndm1, ndm0).
        phi1k padded from the left and the right by zeros.
    hphi1k : array
        Numpy array with dimensions (len(Ek_grid_ext), nleads, ndm1, ndm0).
        Hilbert transform of phi1k on extended grid.

    """
    nleads, ndm1, ndm0 = phi1k.shape[1], phi1k.shape[2], phi1k.shape[3]
    Eklen_ext = phi1k.shape[0] + funcp.kpnt_left + funcp.kpnt_right
    # Create the kernels for Hilbert transformation
    if funcp.ht_ker is None or 2*Eklen_ext != len(funcp.ht_ker):
        funcp.ht_ker = kernel_fredriksen(Eklen_ext)
    # Make phi1k on extended grid Ek_grid_ext
    # Pad phi1k values with zeros from the left and the right
    phi1k = np.concatenate( (np.zeros((funcp.kpnt_left, nleads, ndm1, ndm0)),
                             phi1k,
                             np.zeros((funcp.kpnt_right, nleads, ndm1, ndm0))), axis=0)
    # Make the Hilbert transformation
    hphi1k = np.zeros(phi1k.shape, dtype=complexnp)
    for l, cb, bbp in itertools.product(range(nleads), range(ndm1), range(ndm0)):
        #print(l, cb, bbp)
        hphi1k[:, l, cb, bbp] = hilbert_fredriksen(phi1k[:, l, cb, bbp], funcp.ht_ker)
    return phi1k, hphi1k

def get_htransf_fk(fk, funcp):
    """
    Performs Hilbert transform of Fermi function.
    The energy is shifted by positive infinitesimal
    to the complex upper half-plane.

    Parameters
    ----------
    fk : array
        Numpy array with dimensions (nleads, len(Ek_grid)),
        cotaining Fermi function values on the grid Ek_grid.
    funcp : FunctionProperties
        FunctionProperties object.

    Modifies:
    funcp.ht_ker : array
        Kernel used when performing Hilbert transform using FFT.

    Returns
    -------
    fk : array
        Numpy array with dimensions (nleads, len(Ek_grid_ext)).
        fk padded from the left and the right by zeros.
    hfk : array
        Numpy array with dimensions (nleads, len(Ek_grid_ext)).
        Hilbert transform of fk on extended grid.
    """
    nleads = fk.shape[0]
    Eklen_ext = fk.shape[1] + funcp.kpnt_left + funcp.kpnt_right
    # Create the kernels for Hilbert transformation
    if funcp.ht_ker is None or 2*Eklen_ext != len(funcp.ht_ker):
        funcp.ht_ker = kernel_fredriksen(Eklen_ext)
    # Pad fk values with zeros from the left and the right
    fk = np.concatenate( (np.zeros((nleads, funcp.kpnt_left)),
                          fk,
                          np.zeros((nleads, funcp.kpnt_right))), axis=1)
    # Calculate the Hilbert transform with added positive infinitesimal of the Fermi functions
    hfk = np.zeros((nleads, Eklen_ext), dtype=doublenp)
    for l in range(nleads):
        hfk[l] = hilbert_fredriksen(fk[l], funcp.ht_ker).real
    # The energy is shifted by positive infinitesimal to the complex upper half-plane
    hfk = hfk - 1j*fk
    return fk, hfk

def iterate_2vN(sys):
    """
    Performs the iterative solution of the 2vN approach integral equations.

    Parameters
    ----------
    sys : Approach2vN
        Approach2vN object.

    Modifies:
    sys.phi1k_delta : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0).
        Correction to Phi[1](k) after an iteration.
    sys.hphi1k_delta : array
        Numpy array with dimensions (len(Ek_grid_ext), nleads, ndm1, ndm0).
        Hilbert transform of phi1k_delta on extended grid Ek_grid_ext.
    sys.kern1k : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm1)
        corresponding to energy resolved local kernel for Phi[1](k).
    sys.funcp.emin : float
        Minimal energy in the updated Ek_grid.
    sys.funcp.emax : float
        Maximal energy in the updated Ek_grid.
    sys.Ek_grid_ext : array
        Extended Ek_grid from emin to emax.
    sys.funcp.kpnt_left : int
        Number of points Ek_grid is extended to the left.
    sys.funcp.kpnt_right : int
        Number of points Ek_grid is extended to the right.
    sys.fkp : array
        nleads by len(Ek_grid_ext) numpy array containing Fermi function.
    sys.fkm : array
        nleads by len(Ek_grid_ext) numpy array containing 1-Fermi function.
    sys.hfkp : array
        Hilbert transform of fkp.
    sys.hfkm : array
        Hilbert transform of fkm.
    """
    (Ek_grid, Ek_grid_ext) = (sys.Ek_grid, sys.Ek_grid_ext)
    (E, Tba, si, funcp) = (sys.qd.Ea, sys.leads.Tba, sys.si, sys.funcp)
    (mulst, tlst) = (sys.leads.mulst, sys.leads.tlst)
    # Assign sys.phi1k_delta to phi1k_delta_old, because the new phi1k_delta
    # will be generated in this function
    (phi1k_delta_old, kern1k) = (sys.phi1k_delta, sys.kern1k)
    #
    Eklen = len(Ek_grid)
    if phi1k_delta_old is None:
        ## Define the extended grid Ek_grid_ext for calculations outside the bandwidth
        # Here sys.funcp.emin, sys.funcp.emax are defined
        get_emin_emax(sys)
        # Here sys.Ek_grid_ext, sys.funcp.kpnt_left, sys.funcp.kpnt_right are defined
        get_grid_ext(sys)
        Ek_grid_ext = sys.Ek_grid_ext
        Eklen_ext = len(Ek_grid_ext)
        # Generate the Fermi functions on the grid
        # This is necessary to generate only if Ek_grid, mulst, or tlst are changed
        sys.fkp = np.zeros((si.nleads, Eklen), dtype=doublenp)
        for l in range(si.nleads):
            sys.fkp[l] = 1/( np.exp((Ek_grid - mulst[l])/tlst[l]) + 1 )
        sys.fkm = 1-sys.fkp
        sys.fkp, sys.hfkp = get_htransf_fk(sys.fkp, funcp)
        sys.fkm, sys.hfkm = get_htransf_fk(sys.fkm, funcp)
        # Calculate the zeroth iteration of Phi[1](k)
        phi1k_delta = np.zeros((Eklen, si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
        kern1k = np.zeros((Eklen, si.nleads, si.ndm1, si.ndm1), dtype=complexnp)
        for j1 in range(Eklen):
            ind = j1 + funcp.kpnt_left
            phi1k_delta[j1], kern1k[j1] = phi1k_local_2vN(ind, Ek_grid_ext, sys.fkp,
                                                          sys.hfkp, sys.hfkm, E, Tba, si)
        hphi1k_delta = None
    elif kern1k is None:
        pass
    else:
        # Hilbert transform phi1k_delta_old on extended grid Ek_grid_ext
        #print('Hilbert transforming')
        phi1k_delta_old, hphi1k_delta = get_htransf_phi1k(phi1k_delta_old, funcp)
        #print('Making an iteration')
        phi1k_delta = np.zeros((Eklen, si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
        for j1 in range(Eklen):
            ind = j1 + funcp.kpnt_left
            phi1k_delta[j1] = phi1k_iterate_2vN(ind, Ek_grid_ext, phi1k_delta_old, hphi1k_delta,
                                                sys.fkp, kern1k[j1], E, Tba, si)
    #
    sys.phi1k_delta = phi1k_delta
    sys.hphi1k_delta = hphi1k_delta
    sys.kern1k = kern1k
    return 0

class Approach_py2vN(Approach2vN):

    kerntype = 'py2vN'
    iterate = staticmethod(iterate_2vN)
    get_phi1_phi0 = staticmethod(get_phi1_phi0_2vN)
    kern_phi0 = staticmethod(kern_phi0_2vN)
    generate_current = staticmethod(generate_current_2vN)
