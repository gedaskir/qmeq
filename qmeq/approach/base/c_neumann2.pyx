# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

"""Module containing cython functions, which solve 2vN approach integral equations.
   For docstrings see documentation of module neumann2."""

# Python imports

import numpy as np
import itertools

from .neumann2 import get_htransf_phi1k
from .neumann2 import get_htransf_fk
from .neumann2 import Approach2vN as Approach2vNPy

from ..aprclass import ApproachBase2vN

from ...specfunc.specfunc import kernel_fredriksen
from ...specfunc.specfunc import hilbert_fredriksen

from ...wrappers.mytypes import doublenp
from ...wrappers.mytypes import complexnp

# Cython imports

cimport numpy as np
cimport cython

cdef double_t pi = 3.14159265358979323846


class Approach2vN(Approach2vNPy):

    kerntype = '2vN'
    no_coherences = False

    def prepare_kernel_handler(self):
        self.kernel_handler = KernelHandler(self.si, self.no_coherences)

    def iterate(self):
        terms_calculator = TermsCalculator2vN(self)
        terms_calculator.iterate()

    def determine_phi1_phi0(self):
        cdef long_t i, Eklen
        cdef double_t dx
        cdef np.ndarray[double_t, ndim=1] Ek_grid = self.Ek_grid

        phi1k = self.phi1k
        # Get integrated Phi[1]_{cb} in terms of Phi[0]_{bb'}
        phi1_phi0 = self.phi1_phi0
        e_phi1_phi0 = self.e_phi1_phi0

        Eklen = Ek_grid.shape[0]
        for i in range(Eklen):
            # Trapezoidal rule
            if i == 0:
              dx = Ek_grid[i+1] - Ek_grid[i]
            elif i == Eklen-1:
              dx = Ek_grid[i] - Ek_grid[i-1]
            else:
              dx = Ek_grid[i+1] - Ek_grid[i-1]
            phi1_phi0 += 0.5*dx*phi1k[i]
            e_phi1_phi0 += 0.5*dx*Ek_grid[i]*phi1k[i]


cdef class TermsCalculator2vN:

    def __init__(self, appr):
        self.appr = appr

    def iterate(self):
        self.retrieve_approach_variables()

        cdef KernelHandler kh = self.kernel_handler
        cdef long_t k, l
        cdef long_t Eklen = len(self.Ek_grid)
        cdef long_t nleads = kh.nleads

        if self.appr.is_zeroth_iteration:
            # Calculate the zeroth iteration of Phi[1](k)
            for k in range(Eklen):
                for l in range(nleads):
                    self.phi1k_local(k, l, kh)

            self.appr.is_zeroth_iteration = False
        else:
            # Hilbert transform phi1k_delta on extended grid Ek_grid_ext
            # Here phi1k_delta_old is current phi1k_delta state, but on extended grid
            # print('Hilbert transforming')
            self.phi1k_delta_old, self.hphi1k_delta = get_htransf_phi1k(self.phi1k_delta, self.appr.funcp)
            # print('Making an iteration')
            for k in range(Eklen):
                for l in range(nleads):
                    self.phi1k_iterate(k, l, kh)

    cdef void retrieve_approach_variables(self):
        self.kernel_handler = self.appr.kernel_handler

        self.kpnt_left = self.appr.funcp.kpnt_left

        self.Ek_grid = self.appr.Ek_grid
        self.Ek_grid_ext = self.appr.Ek_grid_ext

        self.Ea = self.appr.qd.Ea
        self.Tba = self.appr.leads.Tba

        self.fkp = self.appr.fkp
        self.fkm = self.appr.fkm
        self.hfkp = self.appr.hfkp
        self.hfkm = self.appr.hfkm

        self.phi1k_delta = self.appr.phi1k_delta
        self.kern1k_inv = self.appr.kern1k_inv

    cdef void phi1k_local(self, long_t k, long_t l, KernelHandler kh):

        cdef long_t i, j, j1, j2, l1
        cdef long_t acharge, bcharge, ccharge, dcharge, charge
        cdef long_t acount, bcount, ccount, dcount
        cdef long_t c, b, cb, a1, b1, c1, d1, b1a1, b1b, cb1, c1b, cc1, d1c1

        cdef double_t [:] E = self.Ea
        cdef complex_t [:, :, :] Tba = self.Tba

        cdef double_t [:, :] fk = self.fkp
        cdef complex_t [:, :] hfkp = self.hfkp
        cdef complex_t [:, :] hfkm = self.hfkm

        cdef double_t [:] Ek_grid = self.Ek_grid_ext

        cdef long_t ncharge = kh.ncharge
        cdef long_t nleads = kh.nleads
        cdef long_t ndm0 = kh.ndm0
        cdef long_t ndm1 = kh.ndm1
        cdef long_t [:, :] statesdm = kh.statesdm

        cdef complex_t [:, :] kern0 = np.zeros((ndm1, ndm0), dtype=complexnp)
        cdef complex_t [:, :] kern1 = np.zeros((ndm1, ndm1), dtype=complexnp)
        cdef complex_t [:, :] kern1_inv = np.zeros((ndm1, ndm1), dtype=complexnp)

        cdef long_t ind = k + self.kpnt_left
        cdef double_t Ek = Ek_grid[ind]

        # Note that the bias is put in the distributions and not in the dispersion
        cdef double_t fp = fk[l, ind]  # fermi_func(+(Ek-mulst[l])/tlst[l])
        cdef double_t fm = 1.-fp       # fermi_func(-(Ek-mulst[l])/tlst[l])

        for i in range(kh.ndm1):
            c = kh.all_ba[i, 0]
            b = kh.all_ba[i, 1]

            charge = kh.all_ba[i, 2]
            dcharge = charge+2
            ccharge = charge+1
            bcharge = charge
            acharge = charge-1

            acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
            bcount = kh.statesdm_count[bcharge]
            ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0
            dcount = kh.statesdm_count[dcharge] if dcharge <= kh.ncharge else 0

            cb = kh.get_ind_dm1(c, b, bcharge)

            kern1[cb, cb] = kern1[cb, cb] + (Ek-E[c]+E[b]+0j)
            # Phi[0] terms
            for j in range(bcount):
                b1 = statesdm[bcharge, j]
                b1b = kh.get_ind_dm0(b1, b, bcharge)
                kern0[cb, b1b] = kern0[cb, b1b] + fp*Tba[l, c, b1]
            for j in range(ccount):
                c1 = statesdm[ccharge, j]
                cc1 = kh.get_ind_dm0(c, c1, ccharge)
                kern0[cb, cc1] = kern0[cb, cc1] - fm*Tba[l, c1, b]
            # ---------------------------------------------------------------------------
            # Phi[1] terms
            # 2nd and 7th terms
            for j1 in range(bcount):
                for j2 in range(acount):
                    b1 = statesdm[bcharge, j1]
                    a1 = statesdm[acharge, j2]
                    b1a1 = kh.get_ind_dm1(b1, a1, acharge)
                    for l1 in range(nleads):
                        kern1[cb, b1a1] = kern1[cb, b1a1] - Tba[l1, c, b1]*Tba[l1, a1, b]*(
                                           + self.func_2vN(+(Ek-E[b1]+E[b]), l1, +1, hfkp)
                                           - self.func_2vN(-(Ek-E[c]+E[a1]), l1, -1, hfkp) )
            # 6th and 8th terms
            for j1 in range(bcount):
                b1 = statesdm[bcharge, j1]
                cb1 = kh.get_ind_dm1(c, b1, bcharge)
                for l1 in range(nleads):
                    for j2 in range(ccount):
                        c1 = statesdm[ccharge, j2]
                        kern1[cb, cb1] = kern1[cb, cb1] - (Tba[l1, b1, c1]*Tba[l1, c1, b]
                                          * self.func_2vN(+(Ek-E[c]+E[c1]), l1, +1, hfkp))
                    for j2 in range(acount):
                        a1 = statesdm[acharge, j2]
                        kern1[cb, cb1] = kern1[cb, cb1] + (Tba[l1, b1, a1]*Tba[l1, a1, b]
                                          * self.func_2vN(-(Ek-E[c]+E[a1]), l1, -1, hfkm))
            # 1st and 3rd terms
            for j1 in range(ccount):
                c1 = statesdm[ccharge, j1]
                c1b = kh.get_ind_dm1(c1, b, bcharge)
                for l1 in range(nleads):
                    for j2 in range(bcount):
                        b1 = statesdm[bcharge, j2]
                        kern1[cb, c1b] = kern1[cb, c1b] - (Tba[l1, c, b1]*Tba[l1, b1, c1]
                                          * self.func_2vN(+(Ek-E[b1]+E[b]), l1, +1, hfkm))
                    for j2 in range(dcount):
                        d1 = statesdm[dcharge, j2]
                        kern1[cb, c1b] = kern1[cb, c1b] + (Tba[l1, c, d1]*Tba[l1, d1, c1]
                                          * self.func_2vN(-(Ek-E[d1]+E[b]), l1, -1, hfkp))
            # 4th and 5th terms
            for j1 in range(dcount):
                for j2 in range(ccount):
                    d1 = statesdm[dcharge, j1]
                    c1 = statesdm[ccharge, j2]
                    d1c1 = kh.get_ind_dm1(d1, c1, ccharge)
                    for l1 in range(nleads):
                        kern1[cb, d1c1] = kern1[cb, d1c1] - Tba[l1, c, d1]*Tba[l1, c1, b]*(
                                           + self.func_2vN(+(Ek-E[c]+E[c1]), l1, +1, hfkm)
                                           - self.func_2vN(-(Ek-E[d1]+E[b]), l1, -1, hfkm) )

        kern1_inv = np.linalg.inv(kern1)
        kern0 = np.dot(kern1_inv, kern0)

        self.kern1k_inv[k, l, :] = kern1_inv
        self.phi1k_delta[k, l, :] = kern0

    cdef void phi1k_iterate(self, long_t k, long_t l, KernelHandler kh):

        cdef long_t i, j, j1, j2, l1
        cdef long_t acharge, bcharge, ccharge, dcharge, charge
        cdef long_t c, b, cb, a1, b1, c1, d1, ba1, c1b1, c1b, d1c1, d1c, b1a1, cb1, bbp

        cdef double_t [:] E = self.Ea
        cdef complex_t [:, :, :] Tba = self.Tba

        cdef complex_t [:, :, :, :] phi1k = self.phi1k_delta_old
        cdef complex_t [:, :, :, :] kern1k_inv = self.kern1k_inv
        cdef complex_t [:, :, : ,:] hphi1k = self.hphi1k_delta

        cdef double_t [:, :] fk = self.fkp
        cdef double_t [:] Ek_grid = self.Ek_grid_ext

        cdef long_t ncharge = kh.ncharge
        cdef long_t nleads = kh.nleads
        cdef long_t ndm0 = kh.ndm0
        cdef long_t ndm1 = kh.ndm1
        cdef long_t [:, :] statesdm = kh.statesdm

        cdef complex_t [:, :] kern1_inv = self.kern1k_inv[k, l]
        cdef complex_t [:, :] kern0 = np.zeros((ndm1, ndm0), dtype=complexnp)
        cdef complex_t [:] term = np.zeros(ndm0, dtype=complexnp)

        cdef long_t kpnt_left = self.kpnt_left
        cdef long_t ind = k + kpnt_left
        cdef double_t Ek = Ek_grid[ind]

        cdef double_t fp = fk[l, ind]
        cdef double_t fm = 1.-fp

        for i in range(kh.ndm1):
            c = kh.all_ba[i, 0]
            b = kh.all_ba[i, 1]

            charge = kh.all_ba[i, 2]
            dcharge = charge+2
            ccharge = charge+1
            bcharge = charge
            acharge = charge-1

            acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
            bcount = kh.statesdm_count[bcharge]
            ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0
            dcount = kh.statesdm_count[dcharge] if dcharge <= kh.ncharge else 0

            cb = kh.get_ind_dm1(c, b, bcharge)

            # 1st term
            for j1 in range(acount):
                a1 = statesdm[acharge, j1]
                ba1 = kh.get_ind_dm1(b, a1, acharge)
                for j2 in range(bcount):
                    b1 = statesdm[bcharge, j2]
                    for l1 in range(nleads):
                        # print('1')
                        self.get_at_k1(+(Ek-E[b1]+E[b]), l1, ba1, True,
                                       -fp*Tba[l1, c, b1]*Tba[l, b1, a1], -1, term)
            # 2nd and 5th terms
            for j1 in range(bcount):
                for j2 in range(ccount):
                    b1 = statesdm[bcharge, j1]
                    c1 = statesdm[ccharge, j2]
                    # print('2 and 5')
                    c1b1 = kh.get_ind_dm1(c1, b1, bcharge)
                    for l1 in range(nleads):
                        # 2nd term
                        self.get_at_k1(+(Ek-E[b1]+E[b]), l1, c1b1, True,
                                       -fm*Tba[l1, c, b1]*Tba[l, c1, b], -1, term)
                        # 5th term
                        self.get_at_k1(+(Ek-E[c]+E[c1]), l1, c1b1, True,
                                       -fp*Tba[l, c, b1]*Tba[l1, c1, b], -1, term)
            # 3rd term
            for j1 in range(ccount):
                c1 = statesdm[ccharge, j1]
                c1b = kh.get_ind_dm1(c1, b, bcharge)
                for j2 in range(dcount):
                    d1 = statesdm[dcharge, j2]
                    for l1 in range(nleads):
                        # print('3')
                        self.get_at_k1(-(Ek-E[d1]+E[b]), l1, c1b, False,
                                       +fp*Tba[l1, c, d1]*Tba[l, d1, c1], +1, term)
            # 4th term
            for j1 in range(dcount):
                for j2 in range(ccount):
                    d1 = statesdm[dcharge, j1]
                    c1 = statesdm[ccharge, j2]
                    # print('4')
                    d1c1 = kh.get_ind_dm1(d1, c1, ccharge)
                    for l1 in range(nleads):
                        self.get_at_k1(-(Ek-E[d1]+E[b]), l1, d1c1, False,
                                       +fm*Tba[l1, c, d1]*Tba[l, c1, b], +1, term)
            # 6th term
            for j1 in range(dcount):
                d1 = statesdm[dcharge, j1]
                d1c = kh.get_ind_dm1(d1, c, ccharge)
                for j2 in range(ccount):
                    c1 = statesdm[ccharge, j2]
                    for l1 in range(nleads):
                        # print('6')
                        self.get_at_k1(+(Ek-E[c]+E[c1]), l1, d1c, True,
                                       -fm*Tba[l, d1, c1]*Tba[l1, c1, b], -1, term)
            # 7th term
            for j1 in range(bcount):
                for j2 in range(acount):
                    b1 = statesdm[bcharge, j1]
                    a1 = statesdm[acharge, j2]
                    # print('7')
                    b1a1 = kh.get_ind_dm1(b1, a1, acharge)
                    for l1 in range(nleads):
                        self.get_at_k1(-(Ek-E[c]+E[a1]), l1, b1a1, False,
                                       +fp*Tba[l, c, b1]*Tba[l1, a1, b], +1, term)
            # 8th term
            for j1 in range(bcount):
                b1 = statesdm[bcharge, j1]
                cb1 = kh.get_ind_dm1(c, b1, bcharge)
                for j2 in range(acount):
                    a1 = statesdm[acharge, j2]
                    for l1 in range(nleads):
                        # print('8')
                        self.get_at_k1(-(Ek-E[c]+E[a1]), l1, cb1, False,
                                       +fm*Tba[l, b1, a1]*Tba[l1, a1, b], +1, term)

            for bbp in range(ndm0):
                kern0[cb, bbp] = term[bbp]
                term[bbp] = 0

        kern0 = np.dot(kern1_inv, kern0)
        self.phi1k_delta[k, l, :] = kern0

    cdef complex_t func_2vN(self,
                            double_t Ek,
                            long_t l,
                            int_t eta,
                            complex_t [:, :] hfk):

        cdef long_t b_idx, a_idx
        cdef double_t a, b
        cdef complex_t fa, fb, rez

        cdef double_t [:] Ek_grid = self.Ek_grid_ext
        cdef long_t Eklen = Ek_grid.shape[0]
        if Ek<Ek_grid[0] or Ek>Ek_grid[Eklen-1]:
            return 0

        b_idx = (<long_t>((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0])))+1
        if b_idx == Eklen: b_idx -= 1
        a_idx = b_idx - 1
        b, a = Ek_grid[b_idx], Ek_grid[a_idx]

        fb = hfk[l, b_idx]
        fa = hfk[l, a_idx]
        rez = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)

        return pi*rez if eta+1 else pi*rez.conjugate()

    cdef void get_at_k1(self,
                        double_t Ek,
                        long_t l,
                        long_t cb,
                        bint conj,
                        complex_t fct,
                        int_t eta,
                        complex_t [:] term):

        cdef long_t b_idx, a_idx, bbp
        cdef double_t a, b
        cdef complex_t fa, fb, u, hu

        cdef long_t ndm0 = term.shape[0]
        cdef complex_t [:, :, :, :] phi1k = self.phi1k_delta_old
        cdef complex_t [:, :, : ,:] hphi1k = self.hphi1k_delta

        cdef double_t [:] Ek_grid = self.Ek_grid_ext
        cdef long_t Eklen = Ek_grid.shape[0]
        if Ek<Ek_grid[0] or Ek>Ek_grid[Eklen-1]:
            return

        b_idx = (<long_t>((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0])))+1
        # NOTE This line needs to be optimized
        # if b_idx == len(Ek_grid): b_idx -= 1
        if b_idx == Eklen: b_idx -= 1
        a_idx = b_idx - 1
        b, a = Ek_grid[b_idx], Ek_grid[a_idx]

        for bbp in range(ndm0):
            # fa = phi1k[a_idx, l, cb, bbp].conjugate() if conj else phi1k[a_idx, l, cb, bbp]
            # fb = phi1k[b_idx, l, cb, bbp].conjugate() if conj else phi1k[b_idx, l, cb, bbp]
            # u = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
            fa = phi1k[a_idx, l, cb, bbp]
            fb = phi1k[b_idx, l, cb, bbp]
            u = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
            u = u.conjugate() if conj else u

            # fa = hphi1k[a_idx, l, cb, bbp].conjugate() if conj else hphi1k[a_idx, l, cb, bbp]
            # fb = hphi1k[b_idx, l, cb, bbp].conjugate() if conj else hphi1k[b_idx, l, cb, bbp]
            # hu = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
            fa = hphi1k[a_idx, l, cb, bbp]
            fb = hphi1k[b_idx, l, cb, bbp]
            hu = Ek/(b-a)*(fb-fa) + 1/(b-a)*(b*fa-a*fb)
            hu = hu.conjugate() if conj else hu

            term[bbp] = term[bbp] + pi*fct*(hu+eta*1j*u)
