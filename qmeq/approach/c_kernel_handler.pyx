# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

# Python imports

import numpy as np

from ..wrappers.mytypes import longnp

from ..indexing import StateIndexingDMc

# Cython imports

cimport numpy as np
cimport cython


cdef class KernelHandler:

    def __init__(self, si, no_coherences=False):
        self.nmany = si.nmany
        self.ndm0 = si.ndm0
        self.ndm0r = si.ndm0r
        self.ndm1 = si.ndm1
        self.npauli = si.npauli
        self.nleads = si.nleads
        self.nbaths = si.nbaths
        self.ncharge = si.ncharge

        self.lenlst = si.lenlst
        self.dictdm = si.dictdm
        self.shiftlst0 = si.shiftlst0
        self.shiftlst1 = si.shiftlst1
        self.mapdm0 = si.mapdm0
        self.booldm0 = si.booldm0
        self.conjdm0 = si.conjdm0

        self.kern = None
        self.phi0 = None
        self.statesdm = None
        self.statesdm_count = None
        self.all_bbp = None
        self.all_ba = None

        self.no_coherences = no_coherences
        self.no_conjugates = not isinstance(si, StateIndexingDMc)
        self.nelements = self.npauli if self.no_coherences else self.ndm0

        self.set_statesdm(si)
        self.set_all_bbp()
        self.set_all_ba()

    cdef void set_statesdm(self, si):
        cdef int_t max_states = 0
        cdef int_t i, j
        statesdm = si.statesdm

        for states in statesdm:
            max_states = max(max_states, len(states))

        statesdm_len = len(statesdm)
        self.statesdm_count = np.zeros(statesdm_len, dtype=longnp)
        self.statesdm = np.zeros((statesdm_len, max_states), dtype=longnp)

        for i in range(statesdm_len):
            self.statesdm_count[i] = len(statesdm[i])
            for j in range(self.statesdm_count[i]):
                self.statesdm[i, j] = statesdm[i][j]

    cdef void set_all_bbp(self):
        self.all_bbp = np.zeros((self.nelements, 3), dtype=longnp)
        cdef long_t bcharge, bcount, i, j, j_lower, b, bp, ind
        ind = 0
        for bcharge in range(self.ncharge):
            bcount = self.statesdm_count[bcharge]
            for i in range(bcount):
                j_lower = i if self.no_conjugates else 0
                for j in range(j_lower, bcount):
                    if self.no_coherences and i != j:
                        continue
                    b = self.statesdm[bcharge, i]
                    bp = self.statesdm[bcharge, j]
                    if self.is_unique(b, bp, bcharge):
                        self.all_bbp[ind, 0] = b
                        self.all_bbp[ind, 1] = bp
                        self.all_bbp[ind, 2] = bcharge
                        ind += 1

    cdef void set_all_ba(self):
        self.all_ba = np.zeros((self.ndm1, 3), dtype=longnp)
        cdef long_t bcharge, acharge, i, j, b, a, ind

        ind = 0
        for bcharge in range(1, self.ncharge):
            acharge = bcharge-1
            for i in range(self.statesdm_count[bcharge]):
                for j in range(self.statesdm_count[acharge]):
                    b = self.statesdm[bcharge, i]
                    a = self.statesdm[acharge, j]
                    self.all_ba[ind, 0] = b
                    self.all_ba[ind, 1] = a
                    self.all_ba[ind, 2] = acharge
                    ind += 1

    cpdef void set_kern(self, kern):
        if self.no_conjugates:
            self.kern = kern

    cpdef void set_phi0(self, phi0):
        if self.no_conjugates:
            self.phi0 = phi0

    cdef bool_t is_included(self, long_t b, long_t bp, long_t bcharge) nogil:
        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        if bbp == -1:
            return False

        return True

    cdef bool_t is_unique(self, long_t b, long_t bp, long_t bcharge) nogil:
        cdef bool_t bbp_bool = self.get_ind_dm0_bool(b, bp, bcharge)
        return bbp_bool

    cdef void set_energy(self,
                double_t energy,
                long_t b, long_t bp, long_t bcharge) nogil:

        if b == bp:
            return

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        cdef long_t bbpi = self.ndm0 + bbp - self.npauli

        self.kern[bbp, bbpi] = self.kern[bbp, bbpi] + energy
        self.kern[bbpi, bbp] = self.kern[bbpi, bbp] - energy

    cdef void set_matrix_element(self,
                complex_t fct,
                long_t b, long_t bp, long_t bcharge,
                long_t a, long_t ap, long_t acharge) nogil:

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        cdef long_t bbpi = self.ndm0 + bbp - self.npauli
        cdef bool_t bbpi_bool = True if bbpi >= self.ndm0 else False

        cdef long_t aap = self.get_ind_dm0(a, ap, acharge)
        cdef long_t aapi = self.ndm0 + aap - self.npauli
        cdef int_t aap_sgn = +1 if self.get_ind_dm0_conj(a, ap, acharge) else -1

        cdef double_t fct_imag = fct.imag
        cdef double_t fct_real = fct.real

        self.kern[bbp, aap] = self.kern[bbp, aap] + fct_imag
        if aapi >= self.ndm0:
            self.kern[bbp, aapi] = self.kern[bbp, aapi] + fct_real*aap_sgn
            if bbpi_bool:
                self.kern[bbpi, aapi] = self.kern[bbpi, aapi] + fct_imag*aap_sgn
        if bbpi_bool:
            self.kern[bbpi, aap] = self.kern[bbpi, aap] - fct_real

    cdef void set_matrix_element_pauli(self,
                double_t fctm, double_t fctp,
                long_t bb, long_t aa) nogil:

        self.kern[bb, bb] += fctm
        self.kern[bb, aa] += fctp

    cdef complex_t get_phi0_element(self, long_t b, long_t bp, long_t bcharge) nogil:

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        if bbp == -1:
            return 0.0

        cdef long_t bbpi = self.ndm0 + bbp - self.npauli
        cdef bool_t bbpi_bool = True if bbpi >= self.ndm0 else False

        cdef double_t phi0_real = self.phi0[bbp]
        cdef double_t phi0_imag = 0
        if bbpi_bool:
            phi0_imag = self.phi0[bbpi] if self.get_ind_dm0_conj(b, bp, bcharge) else -self.phi0[bbpi]

        return phi0_real + 1j*phi0_imag

    cdef long_t get_ind_dm0(self, long_t b, long_t bp, long_t bcharge) nogil:
        return self.mapdm0[self.lenlst[bcharge]*self.dictdm[b] + self.dictdm[bp] + self.shiftlst0[bcharge]]

    cdef bool_t get_ind_dm0_conj(self, long_t b, long_t bp, long_t bcharge) nogil:
        return self.conjdm0[self.lenlst[bcharge]*self.dictdm[b] + self.dictdm[bp] + self.shiftlst0[bcharge]]

    cdef bool_t get_ind_dm0_bool(self, long_t b, long_t bp, long_t bcharge) nogil:
        return self.booldm0[self.lenlst[bcharge]*self.dictdm[b] + self.dictdm[bp] + self.shiftlst0[bcharge]]

    cdef long_t get_ind_dm1(self, long_t b, long_t a, long_t acharge) nogil:
        return self.lenlst[acharge]*self.dictdm[b] + self.dictdm[a] + self.shiftlst1[acharge]


cdef class KernelHandlerMatrixFree(KernelHandler):

    def __init__(self, si, no_coherences=False):
        KernelHandler.__init__(self, si, no_coherences)
        self.dphi0_dt = None

    cpdef void set_dphi0_dt(self, double_t [:] dphi0_dt):
        self.dphi0_dt = dphi0_dt

    cdef void set_energy(self,
                double_t energy,
                long_t b, long_t bp, long_t bcharge) nogil:

        if b == bp:
            return

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        cdef long_t bbpi = self.ndm0 + bbp - self.npauli

        cdef complex_t phi0bbp = self.get_phi0_element(b, bp, bcharge)
        cdef complex_t dphi0_dt_bbp = -1j*energy*phi0bbp

        self.dphi0_dt[bbp] = self.dphi0_dt[bbp] + dphi0_dt_bbp.real
        self.dphi0_dt[bbpi] = self.dphi0_dt[bbpi] - dphi0_dt_bbp.imag

    cdef void set_matrix_element(self,
                complex_t fct,
                long_t b, long_t bp, long_t bcharge,
                long_t a, long_t ap, long_t acharge) nogil:

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        cdef long_t bbpi = self.ndm0 + bbp - self.npauli
        cdef bool_t bbpi_bool = True if bbpi >= self.ndm0 else False
        cdef long_t aap = self.get_ind_dm0(a, ap, acharge)

        cdef complex_t phi0aap = self.get_phi0_element(a, ap, acharge)
        cdef complex_t dphi0_dt_bbp = -1j*fct*phi0aap

        self.dphi0_dt[bbp] = self.dphi0_dt[bbp] + dphi0_dt_bbp.real
        if bbpi_bool:
            self.dphi0_dt[bbpi] = self.dphi0_dt[bbpi] - dphi0_dt_bbp.imag

    cdef void set_matrix_element_pauli(self,
                double_t fctm, double_t fctp,
                long_t bb, long_t aa) nogil:

        self.dphi0_dt[bb] = self.dphi0_dt[bb] + fctm*self.phi0[bb] + fctp*self.phi0[aa]

    cdef double_t get_phi0_norm(self):

        cdef long_t bcharge, bcount, b, bb, i
        cdef double_t norm = 0.0

        for bcharge in range(self.ncharge):
            bcount = self.statesdm_count[bcharge]
            for i in range(bcount):
                b = self.statesdm[bcharge, i]
                bb = self.get_ind_dm0(b, b, bcharge)
                norm += self.phi0[bb]

        return norm


cdef class KernelHandlerRTD(KernelHandler):

    def __init__(self, si, no_coherences=False):
        KernelHandler.__init__(self, si, no_coherences)
        self.nsingle = si.nsingle

    cdef void add_matrix_element(self, double_t fct, long_t l, long_t b, long_t bp,
                    long_t bcharge, long_t a, long_t ap, long_t acharge, int_t mi) nogil:
        cdef long_t indx1 = self.get_ind_dm0(b, bp, bcharge)
        cdef long_t indx2 = self.get_ind_dm0(a, ap, acharge)
        if b != bp:
            indx1 -= self.npauli
            if b > bp:
                indx1 += self.ndm0 - self.npauli
        if a != ap:
            indx2 -= self.npauli
            if a > ap:
                indx2 += self.ndm0 - self.npauli

        if mi == 3:
            self.ReWdn[l, indx1, indx2] += fct
        elif mi == 4:
            self.ImWdn[l, indx1, indx2] += fct
        elif mi == 5:
            self.ReWnd[indx1, indx2] += fct
        elif mi == 6:
            self.ImWnd[indx1, indx2] += fct
        elif mi == 7:
            self.Lnn[indx2] += fct

    cdef void set_matrix_element_dd(self, long_t l, double_t fctm,
                        double_t fctp, long_t bb, long_t aa, long_t mi) nogil:
        if mi == 0:
            self.Wdd[l, bb, bb] += fctm
            self.Wdd[l, bb, aa] += fctp
        elif mi == 1:
            self.WE1[l, bb, bb] += fctm
            self.WE1[l, bb, aa] += fctp
        elif mi == 2:
            self.WE2[l, bb, bb] += fctm
            self.WE2[l, bb, aa] += fctp

    cdef void add_element_2nd_order(self, long_t t_id, long_t l, double_t fct, long_t indx0, long_t indx1,
                            long_t a3, long_t charge3, long_t a4, long_t charge4) nogil:
        cdef long_t indx3 = self.get_ind_dm0(a3, a3, charge3)
        cdef long_t indx4 = self.get_ind_dm0(a4, a4, charge4)
        fct = 2 * fct

        self.Wdd2[t_id, l, indx4 , indx0] += fct
        # Flipping left-most vertex p3 = -p3
        self.Wdd2[t_id, l, indx3, indx0] += -fct
        # Flipping right-most vertex p0 = -p0
        self.Wdd2[t_id, l, indx4, indx1] += fct
        # Flipping left-most and right-most vertices p0 = -p0 and p3 = -p3
        self.Wdd2[t_id, l, indx3, indx1] += -fct

    cdef void set_matrix_list(self):
        pass