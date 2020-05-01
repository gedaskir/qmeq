"""Module containing cython functions, which solve 2vN approach integral equations.
   For docstrings see documentation of module neumann2."""

# Python imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...specfunc.specfunc import kernel_fredriksen
from ...specfunc.specfunc import hilbert_fredriksen
from .neumann2 import get_emin_emax
from .neumann2 import get_grid_ext
from .neumann2 import get_htransf_phi1k
from .neumann2 import get_htransf_fk
from .neumann2 import kern_phi0_2vN
from .neumann2 import generate_current_2vN

from ...mytypes import doublenp
from ...mytypes import complexnp

from ..aprclass import ApproachBase2vN

# Cython imports

cimport numpy as np
cimport cython

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

# from scipy import pi as scipy_pi
# cdef double_t pi = scipy_pi
cdef double_t pi = 3.14159265358979323846


@cython.cdivision(True)
@cython.boundscheck(False)
cdef complex_t func_2vN(double_t Ek,
                        np.ndarray[double_t, ndim=1] Ek_grid,
                        int_t l,
                        int_t eta,
                        np.ndarray[complex_t, ndim=2] hfk):
    cdef long_t b_idx, a_idx
    cdef double_t a, b
    cdef complex_t fa, fb, rez
    if Ek<Ek_grid[0] or Ek>Ek_grid[-1]:
        return 0
    #
    b_idx = (<long_t>((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0])))+1
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
cdef int_t get_at_k1(double_t Ek,
                     np.ndarray[double_t, ndim=1] Ek_grid,
                     int_t l,
                     long_t cb,
                     bint conj,
                     np.ndarray[complex_t, ndim=4] phi1k,
                     np.ndarray[complex_t, ndim=4] hphi1k,
                     long_t ndm0,
                     complex_t fct,
                     int_t eta,
                     np.ndarray[complex_t, ndim=1] term):
    #
    cdef long_t b_idx, a_idx, bbp
    cdef double_t a, b
    cdef complex_t fa, fb, u, hu
    if Ek<Ek_grid[0] or Ek>Ek_grid[-1]:
        return 0
    #
    b_idx = (<long_t>((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0])))+1
    # NOTE This line needs to be optimized
    # if b_idx == len(Ek_grid): b_idx -= 1
    if b_idx == Ek_grid.shape[0]: b_idx -= 1
    a_idx = b_idx - 1
    b, a = Ek_grid[b_idx], Ek_grid[a_idx]
    #
    for bbp in range(ndm0):
        # fa = phi1k[a_idx, l, cb, bbp].conjugate() if conj else phi1k[a_idx, l, cb, bbp]
        # fb = phi1k[b_idx, l, cb, bbp].conjugate() if conj else phi1k[b_idx, l, cb, bbp]
        # u = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
        fa = phi1k[a_idx, l, cb, bbp]
        fb = phi1k[b_idx, l, cb, bbp]
        u = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
        u = u.conjugate() if conj else u
        #
        # fa = hphi1k[a_idx, l, cb, bbp].conjugate() if conj else hphi1k[a_idx, l, cb, bbp]
        # fb = hphi1k[b_idx, l, cb, bbp].conjugate() if conj else hphi1k[b_idx, l, cb, bbp]
        # hu = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
        fa = hphi1k[a_idx, l, cb, bbp]
        fb = hphi1k[b_idx, l, cb, bbp]
        hu = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
        hu = hu.conjugate() if conj else hu
        #
        term[bbp] = term[bbp] + pi*fct*(hu+eta*1j*u)
    return 0


@cython.cdivision(True)
@cython.boundscheck(False)
def phi1k_local_2vN(long_t ind,
                    np.ndarray[double_t, ndim=1] Ek_grid,
                    np.ndarray[double_t, ndim=2] fk,
                    np.ndarray[complex_t, ndim=2] hfkp,
                    np.ndarray[complex_t, ndim=2] hfkm,
                    np.ndarray[double_t, ndim=1] E,
                    np.ndarray[complex_t, ndim=3] Tba,
                    si):
    cdef int_t acharge, bcharge, ccharge, dcharge, charge, l, l1, nleads, itype, dqawc_limit
    cdef long_t c, b, cb, a1, b1, c1, d1, b1a1, b1b, cb1, c1b, cc1, d1c1
    cdef double_t fp, fm
    cdef double_t Ek = Ek_grid[ind]
    nleads = si.nleads
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    #
    cdef np.ndarray[complex_t, ndim=3] kern0 = np.zeros((nleads, si.ndm1, si.ndm0), dtype=complexnp)
    cdef np.ndarray[complex_t, ndim=3] kern1 = np.zeros((nleads, si.ndm1, si.ndm1), dtype=complexnp)
    cdef np.ndarray[complex_t, ndim=3] kern1_inv = np.zeros((nleads, si.ndm1, si.ndm1), dtype=complexnp)
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
                fp = fk[l, ind]  # fermi_func(+(Ek-mulst[l])/tlst[l])
                fm = 1.-fp       # fermi_func(-(Ek-mulst[l])/tlst[l])
                # Phi[0] terms
                for b1 in si.statesdm[bcharge]:
                    b1b = mapdm0[lenlst[bcharge]*dictdm[b1] + dictdm[b] + shiftlst0[bcharge]]
                    kern0[l, cb, b1b] = kern0[l, cb, b1b] + fp*Tba[l, c, b1]
                for c1 in si.statesdm[ccharge]:
                    cc1 = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[c1] + shiftlst0[ccharge]]
                    kern0[l, cb, cc1] = kern0[l, cb, cc1] - fm*Tba[l, c1, b]
                # ---------------------------------------------------------------------------
                # Phi[1] terms
                # 2nd and 7th terms
                for b1, a1 in itertools.product(si.statesdm[bcharge], si.statesdm[acharge]):
                    b1a1 = lenlst[acharge]*dictdm[b1] + dictdm[a1] + shiftlst1[acharge]
                    for l1 in range(nleads):
                        kern1[l, cb, b1a1] = kern1[l, cb, b1a1] - Tba[l1, c, b1]*Tba[l1, a1, b]*(
                                              + func_2vN(+(Ek-E[b1]+E[b]), Ek_grid, l1, +1, hfkp)
                                              - func_2vN(-(Ek-E[c]+E[a1]), Ek_grid, l1, -1, hfkp) )
                # 6th and 8th terms
                for b1 in si.statesdm[bcharge]:
                    cb1 = lenlst[bcharge]*dictdm[c] + dictdm[b1] + shiftlst1[bcharge]
                    for l1 in range(nleads):
                        for c1 in si.statesdm[ccharge]:
                            kern1[l, cb, cb1] = kern1[l, cb, cb1] - (Tba[l1, b1, c1]*Tba[l1, c1, b]
                                                 * func_2vN(+(Ek-E[c]+E[c1]), Ek_grid, l1, +1, hfkp))
                        for a1 in si.statesdm[acharge]:
                            kern1[l, cb, cb1] = kern1[l, cb, cb1] + (Tba[l1, b1, a1]*Tba[l1, a1, b]
                                                 * func_2vN(-(Ek-E[c]+E[a1]), Ek_grid, l1, -1, hfkm))
                # 1st and 3rd terms
                for c1 in si.statesdm[ccharge]:
                    c1b = lenlst[bcharge]*dictdm[c1] + dictdm[b] + shiftlst1[bcharge]
                    for l1 in range(nleads):
                        for b1 in si.statesdm[bcharge]:
                            kern1[l, cb, c1b] = kern1[l, cb, c1b] - (Tba[l1, c, b1]*Tba[l1, b1, c1]
                                                 * func_2vN(+(Ek-E[b1]+E[b]), Ek_grid, l1, +1, hfkm))
                        for d1 in si.statesdm[dcharge]:
                            kern1[l, cb, c1b] = kern1[l, cb, c1b] + (Tba[l1, c, d1]*Tba[l1, d1, c1]
                                                 * func_2vN(-(Ek-E[d1]+E[b]), Ek_grid, l1, -1, hfkp))
                #
                for d1, c1 in itertools.product(si.statesdm[dcharge], si.statesdm[ccharge]):
                    d1c1 = lenlst[ccharge]*dictdm[d1] + dictdm[c1] + shiftlst1[ccharge]
                    for l1 in range(nleads):
                        kern1[l, cb, d1c1] = kern1[l, cb, d1c1] - Tba[l1, c, d1]*Tba[l1, c1, b]*(
                                              + func_2vN(+(Ek-E[c]+E[c1]), Ek_grid, l1, +1, hfkm)
                                              - func_2vN(-(Ek-E[d1]+E[b]), Ek_grid, l1, -1, hfkm) )
    for l in range(nleads):
        kern1_inv[l] = np.linalg.inv(kern1[l])
        kern0[l] = np.dot(kern1_inv[l], kern0[l])
    return kern0, kern1_inv


@cython.cdivision(True)
@cython.boundscheck(False)
def phi1k_iterate_2vN(long_t ind,
                      np.ndarray[double_t, ndim=1] Ek_grid,
                      np.ndarray[complex_t, ndim=4] phi1k,
                      np.ndarray[complex_t, ndim=4] hphi1k,
                      np.ndarray[double_t, ndim=2] fk,
                      np.ndarray[complex_t, ndim=3] kern1_inv,
                      np.ndarray[double_t, ndim=1] E,
                      np.ndarray[complex_t, ndim=3] Tba,
                      si):
    #
    cdef int_t acharge, bcharge, ccharge, dcharge, charge, l, l1, nleads
    cdef long_t ndm0, c, b, cb, a1, b1, c1, d1, ba1, c1b1, c1b, d1c1, d1c, b1a1, cb1, bbp
    cdef double_t fp, fm
    cdef double_t Ek = Ek_grid[ind]
    ndm0 = si.ndm0
    nleads = si.nleads
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    #
    cdef np.ndarray[complex_t, ndim=3] kern0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
    cdef np.ndarray[complex_t, ndim=1] term = np.zeros(ndm0, dtype=complexnp)
    #
    for charge in range(si.ncharge-1):
        dcharge = charge+2
        ccharge = charge+1
        bcharge = charge
        acharge = charge-1
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
            for l in range(nleads):
                fp = fk[l, ind]  # fermi_func(Ek/tlst[l])
                fm = 1.-fp
                # 1st term
                for a1 in si.statesdm[acharge]:
                    ba1 = lenlst[acharge]*dictdm[b] + dictdm[a1] + shiftlst1[acharge]
                    for b1, l1 in itertools.product(si.statesdm[bcharge], range(nleads)):
                        # print('1')
                        get_at_k1(+(Ek-E[b1]+E[b]), Ek_grid, l1, ba1, True, phi1k, hphi1k,
                                  ndm0, -fp*Tba[l1, c, b1]*Tba[l, b1, a1], -1, term)
                # 2nd and 5th terms
                for b1, c1 in itertools.product(si.statesdm[bcharge], si.statesdm[ccharge]):
                    # print('2 and 5')
                    c1b1 = lenlst[bcharge]*dictdm[c1] + dictdm[b1] + shiftlst1[bcharge]
                    for l1 in range(nleads):
                        # 2nd term
                        get_at_k1(+(Ek-E[b1]+E[b]), Ek_grid, l1, c1b1, True, phi1k, hphi1k,
                                  ndm0, -fm*Tba[l1, c, b1]*Tba[l, c1, b], -1, term)
                        # 5th term
                        get_at_k1(+(Ek-E[c]+E[c1]), Ek_grid, l1, c1b1, True, phi1k, hphi1k,
                                  ndm0, -fp*Tba[l, c, b1]*Tba[l1, c1, b], -1, term)
                # 3rd term
                for c1 in si.statesdm[ccharge]:
                    c1b = lenlst[bcharge]*dictdm[c1] + dictdm[b] + shiftlst1[bcharge]
                    for d1, l1 in itertools.product(si.statesdm[dcharge], range(nleads)):
                        # print('3')
                        get_at_k1(-(Ek-E[d1]+E[b]), Ek_grid, l1, c1b, False, phi1k, hphi1k,
                                  ndm0, +fp*Tba[l1, c, d1]*Tba[l, d1, c1], +1, term)
                # 4th term
                for d1, c1 in itertools.product(si.statesdm[dcharge], si.statesdm[ccharge]):
                    # print('4')
                    d1c1 = lenlst[ccharge]*dictdm[d1] + dictdm[c1] + shiftlst1[ccharge]
                    for l1 in range(nleads):
                        get_at_k1(-(Ek-E[d1]+E[b]), Ek_grid, l1, d1c1, False, phi1k, hphi1k,
                                  ndm0, +fm*Tba[l1, c, d1]*Tba[l, c1, b], +1, term)
                # 6th term
                for d1 in si.statesdm[dcharge]:
                    d1c = lenlst[ccharge]*dictdm[d1] + dictdm[c] + shiftlst1[ccharge]
                    for c1, l1 in itertools.product(si.statesdm[ccharge], range(nleads)):
                        # print('6')
                        get_at_k1(+(Ek-E[c]+E[c1]), Ek_grid, l1, d1c, True, phi1k, hphi1k,
                                  ndm0, -fm*Tba[l, d1, c1]*Tba[l1, c1, b], -1, term)
                # 7th term
                for b1, a1 in itertools.product(si.statesdm[bcharge], si.statesdm[acharge]):
                    # print('7')
                    b1a1 = lenlst[acharge]*dictdm[b1] + dictdm[a1] + shiftlst1[acharge]
                    for l1 in range(nleads):
                        get_at_k1(-(Ek-E[c]+E[a1]), Ek_grid, l1, b1a1, False, phi1k, hphi1k,
                                  ndm0, +fp*Tba[l, c, b1]*Tba[l1, a1, b], +1, term)
                # 8th term
                for b1 in si.statesdm[bcharge]:
                    cb1 = lenlst[bcharge]*dictdm[c] + dictdm[b1] + shiftlst1[bcharge]
                    for a1, l1 in itertools.product(si.statesdm[acharge], range(nleads)):
                        # print('8')
                        get_at_k1(-(Ek-E[c]+E[a1]), Ek_grid, l1, cb1, False, phi1k, hphi1k,
                                  ndm0, +fm*Tba[l, b1, a1]*Tba[l1, a1, b], +1, term)
                for bbp in range(ndm0):
                    kern0[l, cb, bbp] = term[bbp]
                    term[bbp] = 0
    for l in range(nleads):
        kern0[l] = np.dot(kern1_inv[l], kern0[l])
    return kern0


@cython.boundscheck(False)
def get_phi1_phi0_2vN(self):
    cdef long_t j1, Eklen
    cdef double_t dx
    cdef np.ndarray[double_t, ndim=1] Ek_grid = self.Ek_grid
    #
    (phi1k, si) = (self.phi1k, self.si)
    # Get integrated Phi[1]_{cb} in terms of Phi[0]_{bb'}
    # cdef np.ndarray[complex_t, ndim=3]
    phi1_phi0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
    e_phi1_phi0 = np.zeros((si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
    Eklen = Ek_grid.shape[0]  # len(Ek_grid)
    #
    for j1 in range(Eklen):
        # Trapezoidal rule
        if j1 == 0:
          dx = Ek_grid[j1+1] - Ek_grid[j1]
        elif j1 == Eklen-1:
          dx = Ek_grid[j1] - Ek_grid[j1-1]
        else:
          dx = Ek_grid[j1+1] - Ek_grid[j1-1]
        phi1_phi0 += 0.5*dx*phi1k[j1]
        e_phi1_phi0 += 0.5*dx*Ek_grid[j1]*phi1k[j1]
    self.phi1_phi0 = phi1_phi0
    self.e_phi1_phi0 = e_phi1_phi0
    return 0


@cython.boundscheck(False)
def iterate_2vN(self):
    cdef long_t j1, Eklen, ind, kpnt_left
    cdef np.ndarray[double_t, ndim=1] Ek_grid = self.Ek_grid
    # cdef np.ndarray[double_t, ndim=1] Ek_grid_ext = self.Ek_grid_ext
    cdef np.ndarray[double_t, ndim=1] E = self.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] Tba = self.leads.Tba
    #
    (Ek_grid_ext) = (self.Ek_grid_ext)
    (si, funcp) = (self.si, self.funcp)
    (mulst, tlst) = (self.leads.mulst, self.leads.tlst)
    # Assign self.phi1k_delta to phi1k_delta_old, because the new phi1k_delta
    # will be generated in this function
    (phi1k_delta_old, kern1k_inv) = (self.phi1k_delta, self.kern1k_inv)
    #
    Eklen = Ek_grid.shape[0]  # len(Ek_grid)
    if phi1k_delta_old is None:
        # Define the extended grid Ek_grid_ext for calculations outside the bandwidth
        # Here self.funcp.emin, self.funcp.emax are defined
        get_emin_emax(self)
        # Here self.Ek_grid_ext, self.funcp.kpnt_left, self.funcp.kpnt_right are defined
        get_grid_ext(self)
        Ek_grid_ext = self.Ek_grid_ext
        Eklen_ext = Ek_grid_ext.shape[0]  # len(Ek_grid_ext)
        # Generate the Fermi functions on the grid
        # This is necessary to generate only if Ek_grid, mulst, or tlst are changed
        self.fkp = np.zeros((si.nleads, Eklen), dtype=doublenp)
        for l in range(si.nleads):
            self.fkp[l] = 1/( np.exp((Ek_grid - mulst[l])/tlst[l]) + 1 )
        self.fkm = 1-self.fkp
        self.fkp, self.hfkp = get_htransf_fk(self.fkp, funcp)
        self.fkm, self.hfkm = get_htransf_fk(self.fkm, funcp)
        # Calculate the zeroth iteration of Phi[1](k)
        phi1k_delta = np.zeros((Eklen, si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
        kern1k_inv = np.zeros((Eklen, si.nleads, si.ndm1, si.ndm1), dtype=complexnp)
        kpnt_left = funcp.kpnt_left
        for j1 in range(Eklen):
            ind = j1 + kpnt_left
            phi1k_delta[j1], kern1k_inv[j1] = phi1k_local_2vN(ind, Ek_grid_ext, self.fkp,
                                                              self.hfkp, self.hfkm, E, Tba, si)
        hphi1k_delta = None
    elif kern1k_inv is None:
        pass
    else:
        # Hilbert transform phi1k_delta_old on extended grid Ek_grid_ext
        # print('Hilbert transforming')
        phi1k_delta_old, hphi1k_delta = get_htransf_phi1k(phi1k_delta_old, funcp)
        # print('Making an iteration')
        phi1k_delta = np.zeros((Eklen, si.nleads, si.ndm1, si.ndm0), dtype=complexnp)
        kpnt_left = funcp.kpnt_left
        for j1 in range(Eklen):
            ind = j1 + kpnt_left
            phi1k_delta[j1] = phi1k_iterate_2vN(ind, Ek_grid_ext, phi1k_delta_old, hphi1k_delta,
                                                self.fkp, kern1k_inv[j1], E, Tba, si)
    self.phi1k_delta = phi1k_delta
    self.hphi1k_delta = hphi1k_delta
    self.kern1k_inv = kern1k_inv
    return 0


class Approach2vN(ApproachBase2vN):

    kerntype = '2vN'

    def iterate(self):
        iterate_2vN(self)

    def get_phi1_phi0(self):
        get_phi1_phi0_2vN(self)

    kern_phi0 = kern_phi0_2vN
    generate_current = generate_current_2vN
