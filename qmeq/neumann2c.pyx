"""Module containing cython functions, which solve 2vN approach integral equations.
   For docstrings see documentation of module neumann2py."""

# cython: profile=True

from __future__ import division
import numpy as np
import itertools

from specfunc import kernel_fredriksen
from specfunc import hilbert_fredriksen
from neumann2py import get_emin_emax
from neumann2py import get_grid_ext
from neumann2py import get_htransf_phi1k
from neumann2py import get_htransf_fk

cimport numpy as np
cimport cython

ctypedef np.uint8_t boolnp
#ctypedef bint boolnp
ctypedef np.int_t intnp
ctypedef np.long_t longnp
ctypedef np.double_t doublenp
#ctypedef double doublenp
ctypedef np.complex128_t complexnp
#ctypedef complex complexnp

#from scipy import pi as scipy_pi
#cdef doublenp pi = scipy_pi
cdef doublenp pi = 3.14159265358979323846

@cython.cdivision(True)
@cython.boundscheck(False)
cdef complexnp func_2vN(doublenp Ek,
                        np.ndarray[doublenp, ndim=1] Ek_grid,
                        intnp l,
                        intnp eta,
                        np.ndarray[complexnp, ndim=2] hfk):
    cdef longnp b_idx, a_idx
    cdef doublenp a, b
    cdef complexnp fa, fb, rez
    if Ek<Ek_grid[0] or Ek>Ek_grid[-1]:
        return 0
    #
    b_idx = (<longnp>((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0])))+1
    #if b_idx == len(Ek_grid): b_idx -= 1
    if b_idx == Ek_grid.shape[0]: b_idx -= 1
    a_idx = b_idx - 1
    b, a = Ek_grid[b_idx], Ek_grid[a_idx]
    #
    fb = hfk[l, b_idx]
    fa = hfk[l, a_idx]
    rez = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
    return pi*rez if eta+1 else pi*rez.conjugate()

@cython.cdivision(True)
@cython.boundscheck(False)
cdef intnp get_at_k1(doublenp Ek,
                     np.ndarray[doublenp, ndim=1] Ek_grid,
                     intnp l,
                     longnp cb,
                     bint conj,
                     np.ndarray[complexnp, ndim=4] phi1k,
                     np.ndarray[complexnp, ndim=4] hphi1k,
                     longnp ndm0,
                     complexnp fct,
                     intnp eta,
                     np.ndarray[complexnp, ndim=1] term):
    #
    cdef longnp b_idx, a_idx, bbp
    cdef doublenp a, b
    cdef complexnp fa, fb, u, hu
    if Ek<Ek_grid[0] or Ek>Ek_grid[-1]:
        return 0
    #
    b_idx = (<longnp>((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0])))+1
    #NOTE This line needs to be optimized
    #if b_idx == len(Ek_grid): b_idx -= 1
    if b_idx == Ek_grid.shape[0]: b_idx -= 1
    a_idx = b_idx - 1
    b, a = Ek_grid[b_idx], Ek_grid[a_idx]
    #
    for bbp in range(ndm0):
        #fa = phi1k[a_idx, l, cb, bbp].conjugate() if conj else phi1k[a_idx, l, cb, bbp]
        #fb = phi1k[b_idx, l, cb, bbp].conjugate() if conj else phi1k[b_idx, l, cb, bbp]
        #u = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
        fa = phi1k[a_idx, l, cb, bbp]
        fb = phi1k[b_idx, l, cb, bbp]
        u = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
        u = u.conjugate() if conj else u
        #
        #fa = hphi1k[a_idx, l, cb, bbp].conjugate() if conj else hphi1k[a_idx, l, cb, bbp]
        #fb = hphi1k[b_idx, l, cb, bbp].conjugate() if conj else hphi1k[b_idx, l, cb, bbp]
        #hu = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
        fa = hphi1k[a_idx, l, cb, bbp]
        fb = hphi1k[b_idx, l, cb, bbp]
        hu = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
        hu = hu.conjugate() if conj else hu
        #
        term[bbp] = term[bbp] + pi*fct*(hu+eta*1j*u)
    return 0

@cython.cdivision(True)
@cython.boundscheck(False)
def c_phi1k_local_2vN(longnp ind,
                      np.ndarray[doublenp, ndim=1] Ek_grid,
                      np.ndarray[doublenp, ndim=2] fk,
                      np.ndarray[complexnp, ndim=2] hfkp,
                      np.ndarray[complexnp, ndim=2] hfkm,
                      np.ndarray[doublenp, ndim=1] E,
                      np.ndarray[complexnp, ndim=3] Xba,
                      si):
    cdef intnp acharge, bcharge, ccharge, dcharge, charge, l, l1, nleads, itype, dqawc_limit
    cdef longnp c, b, cb, a1, b1, c1, d1, b1a1, b1b, cb1, c1b, cc1, d1c1
    cdef doublenp fp, fm
    cdef doublenp Ek = Ek_grid[ind]
    nleads = si.nleads
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
    #
    cdef np.ndarray[complexnp, ndim=3] kern0 = np.zeros((nleads, si.ndm1, si.ndm0), dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=3] kern1 = np.zeros((nleads, si.ndm1, si.ndm1), dtype=np.complex)
    for charge in range(si.ncharge-1):
        dcharge = charge+2
        ccharge = charge+1
        bcharge = charge
        acharge = charge-1
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
            for l in range(nleads):
                # Note that the bias is put in the distributions and not in the dispersion
                kern1[l, cb, cb] = kern1[l, cb, cb] + (Ek-E[c]+E[b]+0j)
                fp = fk[l, ind]  #fermi_func(+(Ek-mulst[l])/tlst[l])
                fm = 1.-fp       #fermi_func(-(Ek-mulst[l])/tlst[l])
                # Phi[0] terms
                for b1 in si.statesdm[bcharge]:
                    b1b = mapdm0[lenlst[bcharge]*dictdm[b1] + dictdm[b] + shiftlst0[bcharge]]
                    kern0[l, cb, b1b] = kern0[l, cb, b1b] + fp*Xba[l, c, b1]
                for c1 in si.statesdm[ccharge]:
                    cc1 = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[c1] + shiftlst0[ccharge]]
                    kern0[l, cb, cc1] = kern0[l, cb, cc1] - fm*Xba[l, c1, b]
                #---------------------------------------------------------------------------
                # Phi[1] terms
                # 2nd and 7th terms
                for b1, a1 in itertools.product(si.statesdm[bcharge], si.statesdm[acharge]):
                    b1a1 = lenlst[acharge]*dictdm[b1] + dictdm[a1] + shiftlst1[acharge]
                    for l1 in range(nleads):
                        kern1[l, cb, b1a1] = kern1[l, cb, b1a1] - Xba[l1, c, b1]*Xba[l1, a1, b]*(+func_2vN(+(Ek-E[b1]+E[b]), Ek_grid, l1, +1, hfkp)
                                                                                                 -func_2vN(-(Ek-E[c]+E[a1]), Ek_grid, l1, -1, hfkp) )
                # 6th and 8th terms
                for b1 in si.statesdm[bcharge]:
                    cb1 = lenlst[bcharge]*dictdm[c] + dictdm[b1] + shiftlst1[bcharge]
                    for l1 in range(nleads):
                        for c1 in si.statesdm[ccharge]:
                            kern1[l, cb, cb1] = kern1[l, cb, cb1] - Xba[l1, b1, c1]*Xba[l1, c1, b]*func_2vN(+(Ek-E[c]+E[c1]), Ek_grid, l1, +1, hfkp)
                        for a1 in si.statesdm[acharge]:
                            kern1[l, cb, cb1] = kern1[l, cb, cb1] + Xba[l1, b1, a1]*Xba[l1, a1, b]*func_2vN(-(Ek-E[c]+E[a1]), Ek_grid, l1, -1, hfkm)
                # 1st and 3rd terms
                for c1 in si.statesdm[ccharge]:
                    c1b = lenlst[bcharge]*dictdm[c1] + dictdm[b] + shiftlst1[bcharge]
                    for l1 in range(nleads):
                        for b1 in si.statesdm[bcharge]:
                            kern1[l, cb, c1b] = kern1[l, cb, c1b] - Xba[l1, c, b1]*Xba[l1, b1, c1]*func_2vN(+(Ek-E[b1]+E[b]), Ek_grid, l1, +1, hfkm)
                        for d1 in si.statesdm[dcharge]:
                            kern1[l, cb, c1b] = kern1[l, cb, c1b] + Xba[l1, c, d1]*Xba[l1, d1, c1]*func_2vN(-(Ek-E[d1]+E[b]), Ek_grid, l1, -1, hfkp)
                #
                for d1, c1 in itertools.product(si.statesdm[dcharge], si.statesdm[ccharge]):
                    d1c1 = lenlst[ccharge]*dictdm[d1] + dictdm[c1] + shiftlst1[ccharge]
                    for l1 in range(nleads):
                        kern1[l, cb, d1c1] = kern1[l, cb, d1c1] - Xba[l1, c, d1]*Xba[l1, c1, b]*(+func_2vN(+(Ek-E[c]+E[c1]), Ek_grid, l1, +1, hfkm)
                                                                                                 -func_2vN(-(Ek-E[d1]+E[b]), Ek_grid, l1, -1, hfkm) )
    for l in range(nleads):
        kern0[l] = np.dot(np.linalg.inv(kern1[l]), kern0[l])
    return kern0, kern1

@cython.cdivision(True)
@cython.boundscheck(False)
def c_phi1k_iterate_2vN(longnp ind,
                        np.ndarray[doublenp, ndim=1] Ek_grid,
                        np.ndarray[complexnp, ndim=4] phi1k,
                        np.ndarray[complexnp, ndim=4] hphi1k,
                        np.ndarray[doublenp, ndim=2] fk,
                        np.ndarray[complexnp, ndim=3] kern1,
                        np.ndarray[doublenp, ndim=1] E,
                        np.ndarray[complexnp, ndim=3] Xba,
                        si):
    #
    cdef intnp acharge, bcharge, ccharge, dcharge, charge, l, l1, nleads
    cdef longnp ndm0, c, b, cb, a1, b1, c1, d1, ba1, c1b1, c1b, d1c1, d1c, b1a1, cb1, bbp
    cdef doublenp fp, fm
    cdef doublenp Ek = Ek_grid[ind]
    ndm0 = si.ndm0
    nleads = si.nleads
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
    #
    cdef np.ndarray[complexnp, ndim=3] kern0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=1] term = np.zeros(ndm0, dtype=np.complex)
    #
    for charge in range(si.ncharge-1):
        dcharge = charge+2
        ccharge = charge+1
        bcharge = charge
        acharge = charge-1
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
            for l in range(nleads):
                fp = fk[l, ind] #fermi_func(Ek/tlst[l])
                fm = 1.-fp
                # 1st term
                for a1 in si.statesdm[acharge]:
                    ba1 = lenlst[acharge]*dictdm[b] + dictdm[a1] + shiftlst1[acharge]
                    for b1, l1 in itertools.product(si.statesdm[bcharge], range(nleads)):
                        #print '1'
                        get_at_k1(+(Ek-E[b1]+E[b]), Ek_grid, l1, ba1, True, phi1k, hphi1k,
                                  ndm0, -fp*Xba[l1, c, b1]*Xba[l, b1, a1], -1, term) #*
                # 2nd and 5th terms
                for b1, c1 in itertools.product(si.statesdm[bcharge], si.statesdm[ccharge]):
                    #print '2 and 5'
                    c1b1 = lenlst[bcharge]*dictdm[c1] + dictdm[b1] + shiftlst1[bcharge]
                    for l1 in range(nleads):
                        # 2nd term
                        get_at_k1(+(Ek-E[b1]+E[b]), Ek_grid, l1, c1b1, True, phi1k, hphi1k,
                                  ndm0, -fm*Xba[l1, c, b1]*Xba[l, c1, b], -1, term)
                        # 5th term
                        get_at_k1(+(Ek-E[c]+E[c1]), Ek_grid, l1, c1b1, True, phi1k, hphi1k,
                                  ndm0, -fp*Xba[l, c, b1]*Xba[l1, c1, b], -1, term)
                # 3rd term
                for c1 in si.statesdm[ccharge]:
                    c1b = lenlst[bcharge]*dictdm[c1] + dictdm[b] + shiftlst1[bcharge]
                    for d1, l1 in itertools.product(si.statesdm[dcharge], range(nleads)):
                        #print '3'
                        get_at_k1(-(Ek-E[d1]+E[b]), Ek_grid, l1, c1b, False, phi1k, hphi1k,
                                  ndm0, +fp*Xba[l1, c, d1]*Xba[l, d1, c1], +1, term)
                # 4th term
                for d1, c1 in itertools.product(si.statesdm[dcharge], si.statesdm[ccharge]):
                    #print '4'
                    d1c1 = lenlst[ccharge]*dictdm[d1] + dictdm[c1] + shiftlst1[ccharge]
                    for l1 in range(nleads):
                        get_at_k1(-(Ek-E[d1]+E[b]), Ek_grid, l1, d1c1, False, phi1k, hphi1k,
                                  ndm0, +fm*Xba[l1, c, d1]*Xba[l, c1, b], +1, term)
                # 6th term
                for d1 in si.statesdm[dcharge]:
                    d1c = lenlst[ccharge]*dictdm[d1] + dictdm[c] + shiftlst1[ccharge]
                    for c1, l1 in itertools.product(si.statesdm[ccharge], range(nleads)):
                        #print '6'
                        get_at_k1(+(Ek-E[c]+E[c1]), Ek_grid, l1, d1c, True, phi1k, hphi1k,
                                  ndm0, -fm*Xba[l, d1, c1]*Xba[l1, c1, b], -1, term)
                # 7th term
                for b1, a1 in itertools.product(si.statesdm[bcharge], si.statesdm[acharge]):
                    #print '7'
                    b1a1 = lenlst[acharge]*dictdm[b1] + dictdm[a1] + shiftlst1[acharge]
                    for l1 in range(nleads):
                        get_at_k1(-(Ek-E[c]+E[a1]), Ek_grid, l1, b1a1, False, phi1k, hphi1k,
                                  ndm0, +fp*Xba[l, c, b1]*Xba[l1, a1, b], +1, term)
                # 8th term
                for b1 in si.statesdm[bcharge]:
                    cb1 = lenlst[bcharge]*dictdm[c] + dictdm[b1] + shiftlst1[bcharge]
                    for a1, l1 in itertools.product(si.statesdm[acharge], range(nleads)):
                        #print '8'
                        get_at_k1(-(Ek-E[c]+E[a1]), Ek_grid, l1, cb1, False, phi1k, hphi1k,
                                  ndm0, +fm*Xba[l, b1, a1]*Xba[l1, a1, b], +1, term)
                for bbp in range(ndm0):
                    kern0[l, cb, bbp] = term[bbp]
                    term[bbp] = 0
    for l in range(nleads):
        kern0[l] = np.dot(np.linalg.inv(kern1[l]), kern0[l])
    return kern0

@cython.boundscheck(False)
def c_get_phi1_phi0_2vN(sys):
    cdef longnp j1, Eklen
    cdef doublenp dx
    cdef np.ndarray[doublenp, ndim=1] Ek_grid = sys.Ek_grid
    #
    (phi1k, si) = (sys.phi1k, sys.si)
    # Get integrated Phi[1]_{cb} in terms of Phi[0]_{bb'}
    #cdef np.ndarray[complexnp, ndim=3]
    phi1_phi0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=np.complex)
    e_phi1_phi0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=np.complex)
    Eklen = Ek_grid.shape[0] #len(Ek_grid)
    #
    for j1 in range(Eklen):
        # Trapezoidal rule
        if j1 == 0:         dx = Ek_grid[j1+1] - Ek_grid[j1]
        elif j1 == Eklen-1: dx = Ek_grid[j1]   - Ek_grid[j1-1]
        else:               dx = Ek_grid[j1+1] - Ek_grid[j1-1]
        phi1_phi0 += 0.5*dx*phi1k[j1]
        e_phi1_phi0 += 0.5*dx*Ek_grid[j1]*phi1k[j1]
    return phi1_phi0, e_phi1_phi0

@cython.boundscheck(False)
def c_iterate_2vN(sys):
    cdef longnp j1, Eklen, ind, Ek_left
    cdef np.ndarray[doublenp, ndim=1] Ek_grid = sys.Ek_grid
    #cdef np.ndarray[doublenp, ndim=1] Ek_grid_ext = sys.Ek_grid_ext
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complexnp, ndim=3] Xba = sys.leads.Xba
    #
    (Ek_grid_ext) = (sys.Ek_grid_ext)
    (si, funcp) = (sys.si, sys.funcp)
    (mulst, tlst) = (sys.leads.mulst, sys.leads.tlst)
    # Assign sys.phi1k_delta to phi1k_delta_old, because the new phi1k_delta
    # will be generated in this function
    (phi1k_delta_old, kern1k) = (sys.phi1k_delta, sys.kern1k)
    #
    Eklen = Ek_grid.shape[0] #len(Ek_grid)
    if phi1k_delta_old is None:
        ## Define the extended grid Ek_grid_ext for calculations outside the bandwidth
        # Here sys.funcp.emin, sys.funcp.emax are defined
        get_emin_emax(sys)
        # Here sys.Ek_grid_ext, sys.funcp.Ek_left, sys.funcp.Ek_right are defined
        get_grid_ext(sys)
        Ek_grid_ext = sys.Ek_grid_ext
        Eklen_ext = Ek_grid_ext.shape[0] #len(Ek_grid_ext)
        # Generate the Fermi functions on the grid
        # This is necessary to generate only if Ek_grid, mulst, or tlst are changed
        sys.fkp = np.zeros((si.nleads, Eklen), dtype=np.double)
        for l in range(si.nleads):
            sys.fkp[l] = 1/( np.exp((Ek_grid - mulst[l])/tlst[l]) + 1 )
        sys.fkm = 1-sys.fkp
        sys.fkp, sys.hfkp = get_htransf_fk(sys.fkp, funcp)
        sys.fkm, sys.hfkm = get_htransf_fk(sys.fkm, funcp)
        # Calculate the zeroth iteration of Phi[1](k)
        phi1k_delta = np.zeros((Eklen, si.nleads, si.ndm1, si.ndm0), dtype=np.complex)
        kern1k = np.zeros((Eklen, si.nleads, si.ndm1, si.ndm1), dtype=np.complex)
        Ek_left = funcp.Ek_left
        for j1 in range(Eklen):
            ind = j1 + Ek_left
            phi1k_delta[j1], kern1k[j1] = c_phi1k_local_2vN(ind, Ek_grid_ext, sys.fkp, sys.hfkp, sys.hfkm, E, Xba, si)
        hphi1k_delta = None
    elif kern1k is None:
        pass
    else:
        # Hilbert transform phi1k_delta_old on extended grid Ek_grid_ext
        #print 'Hilbert transforming'
        phi1k_delta_old, hphi1k_delta = get_htransf_phi1k(phi1k_delta_old, funcp)
        #print 'Making an iteration'
        phi1k_delta = np.zeros((Eklen, si.nleads, si.ndm1, si.ndm0), dtype=np.complex)
        Ek_left = funcp.Ek_left
        for j1 in range(Eklen):
            ind = j1 + Ek_left
            phi1k_delta[j1] = c_phi1k_iterate_2vN(ind, Ek_grid_ext, phi1k_delta_old, hphi1k_delta, sys.fkp, kern1k[j1], E, Xba, si)
    return phi1k_delta, hphi1k_delta, kern1k
