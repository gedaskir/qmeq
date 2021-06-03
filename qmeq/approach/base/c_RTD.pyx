# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

"""Module containing cython functions, which generate first order Pauli kernel.
   For docstrings see documentation of module pauli."""

# Python imports

import numpy as np

from ...approach.base.RTD import ApproachPyRTD as ApproachPyRTD
from ...approach.aprclass import Approach as ApproachPy

from ...wrappers.mytypes import doublenp
from ...wrappers.mytypes import complexnp

# Cython imports
from cython.parallel cimport prange
from cython.parallel cimport parallel
from libc.math cimport fabs
from ...specfunc.c_specfunc cimport phi
from ...specfunc.c_specfunc cimport integralD
from ...specfunc.c_specfunc cimport integralX
from ...specfunc.c_specfunc cimport fermi_func
from ...specfunc.c_specfunc cimport delta_phi
from ...specfunc.c_specfunc cimport BW_Ozaki
from ...specfunc.c_specfunc cimport func_pauli
from ...specfunc.c_specfunc cimport pi
from ...specfunc.c_specfunc cimport cabs
from ...specfunc.c_specfunc cimport diag_matrix_multiply

cimport openmp
cimport numpy as np
cimport cython

from ..c_aprclass cimport Approach
from ..c_kernel_handler cimport KernelHandlerRTD

# ---------------------------------------------------------------------------------------------------
# RTD Approach
# ---------------------------------------------------------------------------------------------------
cdef class ApproachRTD(Approach):

    kerntype = 'RTD'
    no_coherences = True

    def __init__(self, *args):
        Approach.__init__(self, *args)
        self.BW_Ozaki_expansion = 0
        self.Ozaki_poles_and_residues = np.zeros((2,2), doublenp)
        self.ImGamma = False
        self.printed_warning_ImGamma = False
        # For parallel
        self.nbr_Wdd2_copies = min(self.si.npauli, openmp.omp_get_max_threads())

    cpdef long_t get_kern_size(self):
        return self._kernel_handler.npauli


    def prepare_kernel_handler(self):
        self.kernel_handler = KernelHandlerRTD(self.si, self.no_coherences)
        self._kernel_handler = self.kernel_handler


    def restart(self):
        ApproachPyRTD.restart(self)


    cdef void prepare_arrays(self):
        ApproachPy.prepare_arrays(self)
        cdef KernelHandlerRTD kh = self._kernel_handler

        cdef long_t nleads = kh.nleads
        cdef long_t ndm1 = kh.ndm1
        cdef long_t kern_size = self.get_kern_size()
        cdef long_t kern_size2

        self.paulifct = np.zeros((nleads, ndm1, 2), dtype=doublenp)
        self.rez = np.zeros(2, dtype=doublenp)
        self.Wdd2 = np.zeros((self.nbr_Wdd2_copies, nleads, kern_size, kern_size), dtype=self.dtype, order='F')
        self.WE1 = np.zeros((nleads, kern_size, kern_size), dtype=self.dtype, order='F')
        self.WE2 = np.zeros((nleads, kern_size, kern_size), dtype=self.dtype, order='F')

        self.generate_LN()

        if self.funcp.off_diag_corrections:
            kern_size2 = 2 * kh.ndm0 - 2 * kh.npauli
            self.ReWnd = np.zeros((kern_size2, kern_size), dtype=self.dtype, order='F')
            self.ImWnd = np.zeros((kern_size2, kern_size), dtype=self.dtype, order='F')
            self.ReWdn = np.zeros((nleads, kern_size, kern_size2), dtype=self.dtype, order='F')
            self.ImWdn = np.zeros((nleads, kern_size, kern_size2), dtype=self.dtype, order='F')
            self.Lnn = np.zeros(kern_size2, dtype=self.dtype)

            self._ReWnd = self.ReWnd
            self._ImWnd = self.ImWnd
            self._ReWdn = self.ReWdn
            self._ImWdn = self.ImWdn
            self._Lnn = self.Lnn

            kh.ReWnd = self.ReWnd
            kh.ImWnd = self.ImWnd
            kh.ReWdn = self.ReWdn
            kh.ImWdn = self.ImWdn
            kh.Lnn = self.Lnn

        self._kern = self.kern
        self._bvec = self.bvec
        self._norm_vec = self.norm_vec
        self._paulifct = self.paulifct

        self._phi0 = self.phi0
        self._dphi0_dt = self.dphi0_dt
        self._current = self.current
        self._energy_current = self.energy_current
        self._heat_current = self.heat_current

        self._tlst = self.leads.tlst
        self._tleads_array = self.leads.tleads_array

        self._mulst = self.leads.mulst
        self._dlst = self.leads.dlst

        self._Ea = self.qd.Ea
        self._Tba = self.leads.Tba

        self._Wdd = self.Wdd2[0,...]
        self._WE1 = self.WE1
        self._WE2 = self.WE2
        self._Wdd2 = self.Wdd2

        kh.Wdd = self._Wdd
        kh.Wdd2 = self._Wdd2
        kh.WE1 = self._WE1
        kh.WE2 = self._WE2


    cdef void clean_arrays(self):
        self.ImGamma = False

        if not self._mfreeq:
            self._kern[::1] = 0.0
            self._bvec[::1] = 0.0

        self._phi0[::1] = 0.0
        self._current[::1] = 0.0
        self._energy_current[::1] = 0.0
        self._heat_current[::1] = 0.0

        self._paulifct[::1]  = 0.0
        self._WE1[::1] = 0.0
        self._WE2[::1] = 0.0
        self._Wdd2[::1] = 0.0

        if self.funcp.off_diag_corrections:
            self._ReWnd[::1] = 0.0
            self._ImWnd[::1] = 0.0
            self._ReWdn[::1] = 0.0
            self._ImWdn[::1] = 0.0
            self._Lnn[::1] = 0.0


    cpdef void generate_kern(self):
        cdef long_t bcharge, b, bp, bcount, i, j
        cdef KernelHandlerRTD kh = self.kernel_handler
        cdef long_t ncharge = kh.ncharge
        cdef long_t kern_size = self.get_kern_size()
        cdef long_t[:,:] statesdm = kh.statesdm
        cdef double_t[:] tlst = self._tlst
        cdef bool_t off_diag_corrections = self.funcp.off_diag_corrections

        if np.any(abs(self.leads.Tba.imag)>0):
            self.set_Ozaki_params()
        else:
            for i in range(1, kh.nleads):
                if tlst[i] != tlst[0]:
                    self.set_Ozaki_params()
                    break

        # Calcualte Wdd^2 first to be able to resuse memory (Wdd1 & Wdd1 write to the same memory).
        for i in prange(kern_size, nogil=True):
            b = kh.all_bbp[i, 0]
            bcharge = kh.all_bbp[i, 2]
            self.generate_matrix_element_2nd_order(b, bcharge, kh)

        for i in range(1, self.nbr_Wdd2_copies):
            self._Wdd2[0,...] += self._Wdd2[i,...]


        # Loop over diagonal states and build kernels
        for i in prange(kern_size, nogil=True):
            b = kh.all_bbp[i, 0]
            bcharge = kh.all_bbp[i, 2]
            self.generate_row_1st_order_kernel(b, bcharge, kh)
            self.generate_row_1st_energy_kernel(b, bcharge, kh)
            self.generate_row_2nd_energy_kernel(b, bcharge, kh)

            if off_diag_corrections:
                self.generate_col_nondiag_kern_1st_order_nd(b, bcharge, kh)

        self.kern[:kern_size, :kern_size] += np.sum(self._Wdd, 0)

        # Loop over non-diagonal states and build kernels
        if off_diag_corrections:
            for bcharge in prange(ncharge, nogil=True):
                bcount = kh.statesdm_count[bcharge]
                for i in range(bcount):
                    b = statesdm[bcharge, i]
                    for j in range(bcount):
                        bp = statesdm[bcharge, j]
                        if b == bp:
                            continue
                        self.generate_col_nondiag_kern_1st_order_dn(b, bp, bcharge, kh)
                        self.generate_row_inverse_Liouvillian(b, bp, bcharge, kh)
            self.add_off_diag_corrections(kh)


    cdef void add_off_diag_corrections(self, KernelHandlerRTD kh):
        cdef double_t[:,:,:] Wcorr = np.zeros(self._Wdd.shape, dtype=doublenp)
        cdef long_t l

        diag_matrix_multiply(kh.Lnn, kh.ImWnd)
        diag_matrix_multiply(kh.Lnn, kh.ReWnd)
        for l in range(kh.nleads):
            Wcorr.base[l,:,:] = np.matmul(kh.ReWdn.base[l,:,:], kh.ImWnd)
            Wcorr.base[l,:,:] = Wcorr.base[l,:,:] + np.matmul(kh.ImWdn.base[l,:,:], kh.ReWnd)

        self._Wdd += Wcorr
        cdef long_t kern_size = self.get_kern_size()
        self.kern[:kern_size, :kern_size] += np.sum(Wcorr, 0)


    cdef void generate_LN(self):
        cdef KernelHandlerRTD kh = self._kernel_handler
        cdef double_t[:] charge_lst = np.zeros(kh.npauli, doublenp)
        cdef long_t i, charge

        for i in range(kh.npauli):
            charge = kh.all_bbp[i, 2]
            charge_lst[i] = 2 * float(charge)

        self.LN = charge_lst


    cpdef void generate_current(self):
        cdef long_t i,l, nleads
        cdef double_t[:] Ea, LE, mulst, current, energy_current, heat_current
        cdef KernelHandlerRTD kh = self._kernel_handler

        current = self._current
        energy_current = self._energy_current
        heat_current = self._heat_current
        mulst = self.leads.mulst
        nleads = kh.nleads
        Ea = self._Ea
        LE = np.zeros(kh.npauli, dtype=doublenp)

        for i in range(kh.npauli):
            LE[i] = 2.0 * Ea[i]

        for l in range(nleads):
            current[l] = 0.5 * np.dot(self.LN, np.dot(kh.Wdd.base[l, :, :], self.phi0))
            energy_current[l] = 0.5 * np.dot(LE, np.dot(kh.Wdd.base[l, :, :], self.phi0))
            energy_current[l] += -0.5 * np.sum(np.dot(kh.WE1.base[l, :, :], self.phi0))
            energy_current[l] += 0.5 * np.sum(np.dot(kh.WE2.base[l, :, :], self.phi0))
            heat_current[l] = energy_current[l] - current[l] * mulst[l]

        if self.ImGamma:
            self.energy_current.fill(np.nan)
            self.heat_current.fill(np.nan)
            if not self.printed_warning_ImGamma:
                print('Warning! Complex matrix elements detected, which are not supported ' +
                      'when calculating the energy current.')
                self.printed_warning_ImGamma = True



    cpdef void generate_fct(self):
        cdef int_t itype
        cdef long_t c, b, bcharge, cb, l, nleads
        cdef double_t Ecb, xcb
        cdef double_t[:] E, mulst, tlst, rez
        cdef double_t[:,:] dlst
        cdef double_t [:,:,:] paulifct
        cdef complex_t[:,:,:] Tba
        cdef KernelHandlerRTD kh = self._kernel_handler

        E = self._Ea
        Tba = self._Tba
        mulst = self._mulst
        tlst = self._tlst
        dlst = self._dlst
        nleads = kh.nleads
        itype = self.funcp.itype
        paulifct = self._paulifct
        rez = self.rez

        for i in range(kh.ndm1):
            c = kh.all_ba[i, 0]
            b = kh.all_ba[i, 1]
            bcharge = kh.all_ba[i, 2]
            cb = kh.get_ind_dm1(c, b, bcharge)
            Ecb = E[c]-E[b]
            for l in range(nleads):
                xcb = (Tba[l, b, c]*Tba[l, c, b]).real
                func_pauli(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype, rez)
                paulifct[l, cb, 0] = xcb*rez[0]
                paulifct[l, cb, 1] = xcb*rez[1]


    cdef void generate_row_1st_order_kernel(self,
        long_t b, long_t bcharge, KernelHandlerRTD kh) nogil:
        cdef long_t a, c, aa, bb, cc, ba, cb, i, l, acharge, ccharge, acount, ccount, nleads
        cdef double_t fctm, fctp
        cdef long_t [:, :] statesdm
        cdef double_t[:,:,:] paulifct

        nleads = kh.nleads
        statesdm = kh.statesdm
        paulifct = self._paulifct

        acharge = bcharge-1
        ccharge = bcharge+1
        acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
        ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0

        bb = kh.get_ind_dm0(b, b, bcharge)

        for i in range(acount):
            a = statesdm[acharge, i]
            aa = kh.get_ind_dm0(a, a, acharge)
            ba = kh.get_ind_dm1(b, a, acharge)
            for l in range(nleads):
                fctm = -paulifct[l, ba, 1]
                fctp = paulifct[l, ba, 0]
                kh.set_matrix_element_dd(l, fctm, fctp, bb, aa, 0)

        for i in range(ccount):
            c = statesdm[ccharge, i]
            cc = kh.get_ind_dm0(c, c, ccharge)
            cb = kh.get_ind_dm1(c, b, bcharge)
            for l in range(nleads):
                fctm = -paulifct[l, cb, 0]
                fctp = paulifct[l, cb, 1]
                kh.set_matrix_element_dd(l, fctm, fctp, bb, cc, 0)


    cdef void generate_row_1st_energy_kernel(self, long_t b, long_t bcharge, KernelHandlerRTD kh) nogil:
        cdef long_t i, a, aa, l, lp, n1, c, cc, acount, ccount, acharge, ccharge, nleads, nsingle
        cdef long_t [:, :] statesdm
        cdef double_t mu, Tr, dE, temp, PI, maxTemp, t_cutoff
        cdef double_t [:] E, mulst, tlst
        cdef double_t [:, :] dlst
        cdef complex_t gamma
        cdef complex_t[:,:] tleads_array
        cdef complex_t[:,:,:] Tba

        E = self._Ea
        Tba = self._Tba
        tleads_array = self._tleads_array
        mulst = self._mulst
        tlst = self._tlst
        dlst = self._dlst
        nleads = kh.nleads
        statesdm = kh.statesdm
        nsingle = kh.nsingle
        PI = pi

        maxTemp = 0.0
        for i in range(nleads):
            if tlst[i] > maxTemp:
                maxTemp = tlst[i]
        t_cutoff = 1e-15*maxTemp*maxTemp

        acharge = bcharge-1
        ccharge = bcharge+1
        acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
        ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0

        bb = kh.get_ind_dm0(b, b, bcharge)
        for i in range(acount):
            a = statesdm[acharge, i]
            aa = kh.get_ind_dm0(a, a, acharge)
            for l in range(nleads):
                mu, Tr, gamma = mulst[l], tlst[l], 0.0 + 0.0j
                dE = E[b] - E[a]
                for lp in range(nleads):
                    if lp == l: continue
                    for n1 in range(nsingle):
                        gamma += Tba[l, a, b] * Tba[lp, a, b].conjugate() * tleads_array[l, n1] * tleads_array[lp, n1].conjugate()
                
                if fabs(gamma.imag) > t_cutoff:
                    self.ImGamma = True

                temp = gamma.real * phi((dE - mu) / Tr, dlst[l, 0] / Tr, dlst[l, 1] / Tr)
                temp += gamma.real * phi(-(dE - mu) / Tr, dlst[l, 0] / Tr, dlst[l, 1] / Tr)            
                temp *= PI
                kh.set_matrix_element_dd(l, temp, temp, bb, aa, 1)

        for i in range(ccount):
            c =  statesdm[ccharge, i]
            cc = kh.get_ind_dm0(c, c, ccharge)
            for l in range(nleads):
                mu, Tr, gamma = mulst[l], tlst[l], 0.0 + 0.0j
                dE = E[c] - E[b]
                for lp in range(nleads):
                    if lp == l: continue
                    for n1 in range(nsingle):
                        gamma += Tba[l, b, c] * Tba[lp, b, c].conjugate() * tleads_array[l, n1] * tleads_array[lp, n1].conjugate()

                if fabs(gamma.imag) > t_cutoff:
                    self.ImGamma = True

                temp = gamma.real * phi((dE - mu) / Tr, dlst[l, 0] / Tr, dlst[l, 1] / Tr)
                temp += gamma.real * phi(-(dE - mu) / Tr, dlst[l, 0] / Tr, dlst[l, 1] / Tr)
                temp *= PI
                kh.set_matrix_element_dd(l, temp, temp, bb, cc, 1)


    cdef void generate_row_2nd_energy_kernel(self, long_t b, long_t bcharge, KernelHandlerRTD kh) nogil:

        cdef long_t i, a, aa, l, lp, n1, c, cc, acount, ccount, acharge, ccharge, nleads, nsingle
        cdef long_t [:, :] statesdm
        cdef double_t mu, Tr, dE, temp, PI, maxTemp, t_cutoff
        cdef double_t [:] E, mulst, tlst
        cdef double_t [:, :] dlst
        cdef complex_t gamma
        cdef complex_t[:,:] tleads_array
        cdef complex_t[:,:,:] Tba

        E = self._Ea
        Tba = self._Tba
        tleads_array = self._tleads_array
        mulst = self._mulst
        tlst = self._tlst
        dlst = self._dlst
        nleads = kh.nleads
        statesdm = kh.statesdm
        nsingle = kh.nsingle
        PI = pi

        maxTemp = 0.0
        for i in range(nleads):
            if tlst[i] > maxTemp:
                maxTemp = tlst[i]
        t_cutoff = 1e-20*maxTemp*maxTemp

        acharge = bcharge-1
        ccharge = bcharge+1
        acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
        ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0

        bb = kh.get_ind_dm0(b, b, bcharge)
        for i in range(acount):
            a = statesdm[acharge, i]
            aa = kh.get_ind_dm0(a, a, acharge)
            for l in range(nleads):
                temp = 0.0
                for lp in range(nleads):
                    if lp != l:
                        mu, Tr, gamma = mulst[lp], tlst[lp], 0.0 + 0.0j
                        for n1 in range(nsingle):
                            gamma += Tba[l, a, b] * Tba[lp, a, b].conjugate() * tleads_array[l, n1] * tleads_array[lp, n1].conjugate()
                        dE = E[b] - E[a]                        
                        temp += gamma.real * phi((dE - mu) / Tr, dlst[lp, 0] / Tr, dlst[lp, 1] / Tr)
                        temp += gamma.real * phi(-(dE - mu) / Tr, dlst[lp, 0] / Tr, dlst[lp, 1] / Tr)
                        if fabs(gamma.imag) > t_cutoff:
                            self.ImGamma = True
                temp *= PI
                kh.set_matrix_element_dd(l, temp, temp, bb, aa, 2)

        for i in range(ccount):
            c = statesdm[ccharge, i]
            cc = kh.get_ind_dm0(c, c, ccharge)
            for l in range(nleads):
                temp = 0.0
                for lp in range(nleads):
                    if lp != l:
                        mu, Tr, gamma = mulst[lp], tlst[lp], 0.0 + 0.0j
                        for n1 in range(nsingle):
                            gamma += Tba[l, b, c] * Tba[lp, b, c].conjugate() * tleads_array[l, n1] * tleads_array[lp, n1].conjugate()
                        dE = E[c] - E[b]
                        temp += gamma.real * phi((dE - mu) / Tr, dlst[lp, 0] / Tr, dlst[lp, 1] / Tr)
                        temp += gamma.real * phi(-(dE - mu) / Tr, dlst[lp, 0] / Tr, dlst[lp, 1] / Tr)
                        if fabs(gamma.imag) > t_cutoff:
                            self.ImGamma = True

                temp *= PI
                kh.set_matrix_element_dd(l, temp, temp, bb, cc, 2)


    cdef void generate_matrix_element_2nd_order(self, long_t a0, long_t charge, KernelHandlerRTD kh) nogil:
        cdef long_t nleads, acharge, bcharge, ccharge, dcharge, acount, bcount, ccount, dcount
        cdef long_t indx0, indx1, r0, r1, a1p, a2p, a2m, a3p, a3m, i, j, k, t_id
        cdef long_t [:, :] statesdm
        cdef double_t T1, T2, mu1, mu2, D, temp, E1, E2, E3, t_cutoff1, t_cutoff2, t_cutoff3, maxTemp
        cdef double_t[:] E, mulst, tlst
        cdef double_t[:,:] b_and_R, dlst
        cdef complex_t t, t1, t2D, t2X, tempD, tempX
        cdef complex_t[:,:,:] Tba
        cdef bint ImGamma

        # For parallel
        t_id = openmp.omp_get_thread_num()

        E = self._Ea
        Tba = self._Tba
        mulst = self._mulst
        dlst = self._dlst
        tlst = self._tlst
        b_and_R = self.Ozaki_poles_and_residues
        nleads = kh.nleads

        maxTemp = 0.0
        for i in range(nleads):
            if tlst[i] > maxTemp:
                maxTemp = tlst[i]

        t_cutoff1 = 0.0
        t_cutoff2 = 1e-10*maxTemp
        t_cutoff3 = 1e-20*maxTemp*maxTemp

        statesdm = kh.statesdm

        acharge = charge-1
        bcharge = charge
        ccharge = charge+1
        dcharge = charge+2
        acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
        bcount = kh.statesdm_count[bcharge]
        ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0
        dcount = kh.statesdm_count[dcharge] if dcharge <= kh.ncharge else 0

        indx0 = kh.get_ind_dm0(a0, a0, charge)
        for r0 in range(nleads):
            T1 = tlst[r0]
            mu1 = mulst[r0]
            for r1 in range(nleads):
                T2 = tlst[r1]
                mu2 = mulst[r1]
                D = fabs(dlst[r0, 1]) + fabs(dlst[r0, 0])
                for i in range(ccount):
                    a1p = statesdm[ccharge, i]
                    t = Tba[r0, a0, a1p]
                    if cabs(t) == t_cutoff1:
                        continue
                    indx1 = kh.get_ind_dm0(a1p, a1p, ccharge)
                    E1 = E[a1p] - E[a0]
                    #eta1 = 1
                    #p1 = 1
                    for j in range(dcount):
                        a2p = statesdm[dcharge, j]
                        t1 = t * Tba[r1, a1p, a2p]
                        if cabs(t1) <= t_cutoff2:
                            continue
                        E2 = E[a2p] - E[a0]
                        #p2 = 1
                        for k in range(ccount):
                            a3p = statesdm[ccharge, k]
                            t2D = t1 * Tba[r1, a3p, a2p].conjugate() * Tba[r0, a0, a3p].conjugate()
                            t2X = t1 * Tba[r0, a3p, a2p].conjugate() * Tba[r1, a0, a3p].conjugate()
                            E3 = E[a3p] - E[a0]
                            if cabs(t2D) > t_cutoff3:
                                ImGamma = fabs(t2D.imag) > t_cutoff3
                                tempD = t2D * integralD(1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r0, tempD.real, indx0, indx1, a3p, ccharge, a0, bcharge)
                            if cabs(t2X) > t_cutoff3:
                                ImGamma = fabs(t2X.imag) > t_cutoff3
                                tempX = -t2X * integralX(1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r1, tempX.real, indx0, indx1, a3p, ccharge, a0, bcharge)
                        #p2 = -1
                        for k in range(ccount):
                            a3m = statesdm[ccharge, k]
                            t2D = t1 * Tba[r1, a0, a3m].conjugate() * Tba[r0, a3m, a2p].conjugate()
                            t2X = t1 * Tba[r0, a0, a3m].conjugate() * Tba[r1, a3m, a2p].conjugate()
                            E3 = E[a2p] - E[a3m]
                            if cabs(t2D) > t_cutoff3:
                                ImGamma = fabs(t2D.imag) > t_cutoff3
                                tempD = t2D * integralD(1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r0, tempD.real, indx0, indx1, a2p, dcharge, a3m, ccharge)
                            if cabs(t2X) > t_cutoff3:
                                ImGamma = fabs(t2X.imag) > t_cutoff3
                                tempX = -t2X * integralX(1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r1, tempX.real, indx0, indx1, a2p, dcharge, a3m, ccharge)
                    #p1 = -1
                    for j in range(acount):
                        a2m = statesdm[acharge, j]
                        t1 = t * Tba[r1, a2m, a0]
                        if cabs(t1) <= t_cutoff2:
                            continue
                        E2 = E[a1p] - E[a2m]
                        #p2 = 1
                        for k in range(bcount):
                            a3p = statesdm[bcharge, k]
                            t2D = t1 * Tba[r1, a3p, a1p].conjugate() * Tba[r0, a2m, a3p].conjugate()
                            t2X = t1 * Tba[r0, a3p, a1p].conjugate() * Tba[r1, a2m, a3p].conjugate()
                            E3 = E[a3p] - E[a2m]
                            if cabs(t2D) > t_cutoff3:
                                ImGamma = fabs(t2D.imag) > t_cutoff3
                                tempD = t2D * integralD(-1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r0, tempD.real, indx0, indx1, a3p, bcharge, a2m, acharge)
                            if cabs(t2X) > t_cutoff3:
                                ImGamma = fabs(t2X.imag) > t_cutoff3
                                tempX = -t2X * integralX(-1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r1, tempX.real, indx0, indx1, a3p, bcharge, a2m, acharge)
                        #p2 = -1
                        for k in range(bcount):
                            a3m = statesdm[bcharge, k]
                            t2D = t1 * Tba[r1, a2m, a3m].conjugate() * Tba[r0, a3m, a1p].conjugate()
                            t2X = t1 * Tba[r0, a2m, a3m].conjugate() * Tba[r1, a3m, a1p].conjugate()
                            E3 = E[a1p] - E[a3m]
                            if cabs(t2D) > t_cutoff3:
                                ImGamma = fabs(t2D.imag) > t_cutoff3
                                tempD = t2D * integralD(-1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r0, tempD.real, indx0, indx1, a1p, ccharge, a3m, bcharge)
                            if cabs(t2X) > t_cutoff3:
                                ImGamma = fabs(t2X.imag) > t_cutoff3
                                tempX = -t2X * integralX(-1, 1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r1, tempX.real, indx0, indx1, a1p, ccharge, a3m, bcharge)
                    #eta1 = -1
                    for j in range(bcount):
                        a2p = statesdm[bcharge, j]
                        E2 = E[a2p] - E[a0]
                        t1 = t * Tba[r1, a2p, a1p].conjugate()
                        if cabs(t1) <= t_cutoff2:
                            continue
                        #p2 = 1
                        for k in range(ccount):
                            a3p = statesdm[ccharge, k]
                            t2D = t1 * Tba[r1, a2p, a3p] * Tba[r0, a0, a3p].conjugate()
                            E3 = E[a3p] - E[a0]
                            if cabs(t2D) > t_cutoff3:
                                ImGamma = fabs(t2D.imag) > t_cutoff3
                                tempD = t2D * integralD(1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r0, tempD.real, indx0, indx1, a3p, ccharge, a0, bcharge)
                        for k in range(acount):
                            a3p = statesdm[acharge, k]
                            t2X = t1 * Tba[r0, a3p, a2p].conjugate() * Tba[r1, a3p, a0]
                            E3 = E[a3p] - E[a0]
                            if cabs(t2X) > t_cutoff3:
                                ImGamma = fabs(t2X.imag) > t_cutoff3
                                tempX = -t2X * integralX(1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r1, tempX.real, indx0, indx1, a3p, acharge, a0, bcharge)
                        #p2 = -1
                        for k in range(acount):
                            a3m = statesdm[acharge, k]
                            t2D = t1 * Tba[r1, a3m, a0] * Tba[r0, a3m, a2p].conjugate()
                            E3 = E[a2p] - E[a3m]
                            if cabs(t2D) > t_cutoff3:
                                ImGamma = fabs(t2D.imag) > t_cutoff3
                                tempD = t2D * integralD(1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r0, tempD.real, indx0, indx1, a2p, bcharge, a3m, acharge)
                        for k in range(ccount):
                            a3m = statesdm[ccharge, k]
                            t2X = t1 * Tba[r0, a0, a3m].conjugate() * Tba[r1, a2p, a3m]
                            E3 = E[a2p] - E[a3m]
                            if cabs(t2X) > t_cutoff3:
                                ImGamma = fabs(t2X.imag) > t_cutoff3
                                tempX = -t2X * integralX(1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r1, tempX.real, indx0, indx1, a2p, bcharge, a3m, ccharge)
                    #p1 = -1
                    for j in range(ccount):
                        a2m = statesdm[ccharge, j]
                        E2 = E[a1p] - E[a2m]
                        t1 = t * Tba[r1, a0, a2m].conjugate()
                        if cabs(t1) <= t_cutoff2:
                            continue
                        #p2 = 1
                        for k in range(dcount):
                            a3p = statesdm[dcharge, k]
                            t2D = t1 * Tba[r1, a1p , a3p] * Tba[r0, a2m,  a3p].conjugate()
                            E3 = E[a3p] - E[a2m]
                            if cabs(t2D) > t_cutoff3:
                                ImGamma = fabs(t2D.imag) > t_cutoff3
                                tempD = t2D * integralD(-1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r0, tempD.real, indx0, indx1, a3p, dcharge, a2m, ccharge)
                        for k in range(bcount):
                            a3p = statesdm[bcharge, k]
                            t2X = t1 * Tba[r0, a3p, a1p].conjugate() * Tba[r1, a3p, a2m]
                            E3 = E[a3p] - E[a2m]
                            if cabs(t2X) > t_cutoff3:
                                ImGamma = fabs(t2X.imag) > t_cutoff3
                                tempX = -t2X * integralX(-1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r1, tempX.real, indx0, indx1, a3p, bcharge, a2m, ccharge)
                        #p2 = -1
                        for k in range(bcount):
                            a3m = statesdm[bcharge, k]
                            t2D = t1 * Tba[r1, a3m, a2m] * Tba[r0, a3m, a1p].conjugate()
                            E3 = E[a1p] - E[a3m]
                            if cabs(t2D) > t_cutoff3:
                                ImGamma = fabs(t2D.imag) > t_cutoff3
                                tempD = t2D * integralD(-1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r0, tempD.real, indx0, indx1, a1p, ccharge, a3m, bcharge)
                        for k in range(dcount):
                            a3m = statesdm[dcharge, k]
                            t2X = t1 * Tba[r0, a2m, a3m].conjugate() * Tba[r1, a1p, a3m]
                            E3 = E[a1p] - E[a3m]
                            if cabs(t2X) > t_cutoff3:
                                ImGamma = fabs(t2X.imag) > t_cutoff3
                                tempX = -t2X * integralX(-1, -1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)
                                kh.add_element_2nd_order(t_id, r1, tempX.real, indx0, indx1, a1p, ccharge, a3m, dcharge)


    cdef void generate_col_nondiag_kern_1st_order_dn(self, long_t a1, long_t b1, long_t charge, KernelHandlerRTD kh) nogil:
        cdef long_t a2, c, i, j, l, acharge, bcharge, ccharge, acount, bcount, ccount, nleads
        cdef long_t[:,:] statesdm
        cdef double_t E1, E2, f, phi0, temp1, temp2, PI
        cdef double_t[:] E, mulst, tlst
        cdef double_t[:,:] dlst
        cdef complex_t t2
        cdef complex_t[:,:,:] Tba

        E = self._Ea
        Tba = self._Tba
        mulst = self._mulst
        tlst = self._tlst
        dlst = self._dlst
        PI = pi
        statesdm = kh.statesdm
        nleads = kh.nleads

        acharge = charge - 1
        bcharge = charge
        ccharge = charge + 1
        acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
        bcount = kh.statesdm_count[bcharge]
        ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0

        # final state in higher charge state
        if charge != kh.ncharge - 1:
            # Loop over diagonal final state
            for i in range(ccount):
                a2 = statesdm[ccharge, i]
                E1 = E[a2] - E[a1]
                E2 = E[a2] - E[b1]
                for l in range(nleads):
                    t2 = Tba[l, a1, a2] * Tba[l, b1, a2].conjugate()
                    f = fermi_func((E1 - mulst[l]) / tlst[l]) + fermi_func((E2 - mulst[l]) / tlst[l])
                    phi0 = delta_phi((E1 - mulst[l]) / tlst[l], (E2 - mulst[l]) / tlst[l],
                                     dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l])
                    temp1 = PI * t2.real * f - t2.imag * phi0
                    temp2 = t2.real * phi0 + PI * t2.imag * f
                    kh.add_matrix_element(temp1, l, a2, a2, ccharge, a1, b1, bcharge, 3)
                    kh.add_matrix_element(temp2, l, a2, a2, ccharge, a1, b1, bcharge, 4)
        # Final state in lower charge state
        if bcharge != 0:
            # Loop through diagonal final states
            for i in range(acount):
                a2 = statesdm[acharge, i]
                E1 = E[b1] - E[a2]
                E2 = E[a1] - E[a2]
                for l in range(nleads):
                    t2 = Tba[l, a2, b1] * Tba[l, a2, a1].conjugate()
                    f = fermi_func(-(E1 - mulst[l]) / tlst[l]) + fermi_func(-(E2 - mulst[l]) / tlst[l])
                    phi0 = delta_phi((E1 - mulst[l]) / tlst[l], (E2 - mulst[l]) / tlst[l],
                                     dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l], sign=-1)
                    temp1 = PI * t2.real * f - t2.imag * phi0
                    temp2 = t2.real * phi0 + PI * t2.imag * f
                    kh.add_matrix_element(temp1, l, a2, a2, acharge, a1, b1, bcharge, 3)
                    kh.add_matrix_element(temp2, l, a2, a2, acharge, a1, b1, bcharge, 4)
        # Loop over final state, conserving charge
        for i in range(bcount):
            a2 = statesdm[bcharge, i]
            if bcharge != kh.ncharge - 1:
                # Intermediate state in higher charge state
                for j in range(ccount):
                    c = statesdm[ccharge, j]
                    if b1 == a2:  # vertices on upper prop -> state on lower prop cannot change
                        E1 = E[c] - E[a2]
                        for l in range(nleads):
                            t2 = Tba[l, a1, c] * Tba[l, a2, c].conjugate()
                            f = fermi_func((E1 - mulst[l]) / tlst[l])
                            phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l],
                                       dlst[l, 1] / tlst[l], sign=1)
                            temp1 = -PI * t2.real * f - t2.imag * phi0
                            temp2 = -PI * t2.imag * f + t2.real * phi0
                            kh.add_matrix_element(temp1, l, a2, a2, bcharge, a1, b1, bcharge, 3)
                            kh.add_matrix_element(temp2, l, a2, a2, bcharge, a1, b1, bcharge, 4)
                    if a1 == a2:  # vertices on lower prop -> state on upper prop cannot change
                        E1 = E[c] - E[a2]
                        for l in range(nleads):
                            t2 = Tba[l, a2, c] * Tba[l, b1, c].conjugate()
                            f = fermi_func((E1 - mulst[l]) / tlst[l])
                            phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l],
                                       dlst[l, 1] / tlst[l], sign=1)
                            temp1 = - PI * t2.real * f + t2.imag * phi0
                            temp2 = - PI * t2.imag * f - t2.real * phi0
                            kh.add_matrix_element(temp1, l, a2, a2, bcharge, a1, b1, bcharge, 3)
                            kh.add_matrix_element(temp2, l, a2, a2, bcharge, a1, b1, bcharge, 4)
            if bcharge != 0:
                # Intermediate state in lower charge state
                for j in range(acount):
                    c = statesdm[acharge, j]
                    if b1 == a2:  # vertices on upper prop -> state on lower prop cannot change
                        E1 = E[a2] - E[c]
                        for l in range(nleads):
                            t2 = Tba[l, c, a2] * Tba[l, c, a1].conjugate()
                            f = fermi_func(-(E1 - mulst[l]) / tlst[l])
                            phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l],
                                       dlst[l, 1] / tlst[l], sign=-1)
                            temp1 = - PI * t2.real * f + t2.imag * phi0
                            temp2 = - PI * t2.imag * f - t2.real * phi0
                            kh.add_matrix_element(temp1, l, a2, a2, bcharge, a1, b1, bcharge, 3)
                            kh.add_matrix_element(temp2, l, a2, a2, bcharge, a1, b1, bcharge, 4)
                    if a1 == a2:  # vertices on lower prop -> state on upper prop cannot change
                        E1 = E[a2] - E[c]
                        for l in range(nleads):
                            t2 = Tba[l, c, b1] * Tba[l, c, a2].conjugate()
                            f = fermi_func(-(E1 - mulst[l]) / tlst[l])
                            phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l],
                                       dlst[l, 1] / tlst[l], sign=-1)
                            temp1 = - PI * t2.real * f - t2.imag * phi0
                            temp2 = - PI * t2.imag * f + t2.real * phi0
                            kh.add_matrix_element(temp1, l, a2, a2, bcharge, a1, b1, bcharge, 3)
                            kh.add_matrix_element(temp2, l, a2, a2, bcharge, a1, b1, bcharge, 4)


    cdef void generate_col_nondiag_kern_1st_order_nd(self, long_t a1, long_t charge, KernelHandlerRTD kh) nogil:
        cdef long_t a2, b2, c, i, j, k, l, acharge, bcharge, ccharge, acount, bcount, ccount, nleads
        cdef long_t[:,:] statesdm
        cdef double_t E1, E2, f, phi0, temp1, temp2, PI
        cdef double_t[:] E, mulst, tlst
        cdef double_t[:,:] dlst
        cdef complex_t t2
        cdef complex_t[:,:,:] Tba

        E = self._Ea
        Tba = self._Tba
        mulst = self._mulst
        tlst = self._tlst
        dlst = self._dlst
        PI = pi
        statesdm = kh.statesdm
        nleads = kh.nleads

        acharge = charge - 1
        bcharge = charge
        ccharge = charge + 1
        acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
        bcount = kh.statesdm_count[bcharge]
        ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0

        if charge != kh.ncharge - 1:
            # Loop over final state, adding electron to the QD
            for i in range(ccount):
                a2 = statesdm[ccharge, i]
                E2 = E[a2] - E[a1]
                for j in range(ccount):
                    b2 = statesdm[ccharge, j]
                    if a2 == b2:  # Final state must be off-diagonal
                        continue
                    E1 = E[b2] - E[a1]
                    for l in range(nleads):
                        t2 = Tba[l, a1, a2] * Tba[l, a1, b2].conjugate()
                        f = fermi_func((E1 - mulst[l]) / tlst[l]) + fermi_func((E2 - mulst[l]) / tlst[l])
                        phi0 = delta_phi((E1 - mulst[l]) / tlst[l], (E2 - mulst[l]) / tlst[l],
                                         dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l])
                        temp1 = PI * t2.real * f - t2.imag * phi0
                        temp2 = t2.real * phi0 + PI * t2.imag * f
                        kh.add_matrix_element(temp1, l, a2, b2, bcharge + 1, a1, a1, bcharge, 5)
                        kh.add_matrix_element(temp2, l, a2, b2, bcharge + 1, a1, a1, bcharge, 6)
        if bcharge != 0:
            # Loop over final states, removing electron from the QD
            for i in range(acount):
                a2 = statesdm[acharge, i]
                E1 = E[a1] - E[a2]
                for j in range(acount):
                    b2 = statesdm[acharge, j]
                    if b2 == a2:  # Final state must be off-diagonal
                        continue
                    E2 = E[a1] - E[b2]
                    for l in range(nleads):
                        t2 = Tba[l, b2, a1] * Tba[l, a2, a1].conjugate()
                        f = fermi_func(-(E1 - mulst[l]) / tlst[l]) + fermi_func(-(E2 - mulst[l]) / tlst[l])
                        phi0 = delta_phi((E1 - mulst[l]) / tlst[l], (E2 - mulst[l]) / tlst[l],
                                         dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l], sign=-1)
                        temp1 = PI * t2.real * f - t2.imag * phi0
                        temp2 = t2.real * phi0 + PI * t2.imag * f
                        kh.add_matrix_element(temp1, l, a2, b2, acharge, a1, a1, bcharge, 5)
                        kh.add_matrix_element(temp2, l, a2, b2, acharge, a1, a1, bcharge, 6)
        # Loop over final state conserving charge
        for i in range(bcount):
            a2 = statesdm[bcharge, i]
            for j in range(bcount):
                b2 = statesdm[bcharge, j]
                if a2 == b2:  # Final state must be off-diagonal
                    continue
                if bcharge != kh.ncharge - 1:
                    # Intermediate state in higher charge state
                    for k in range(ccount):
                        c = statesdm[ccharge, k]
                        if a1 == b2:
                            E1 = E[c] - E[b2]
                            for l in range(nleads):
                                t2 = Tba[l, a1, c] * Tba[l, a2, c].conjugate()
                                f = fermi_func((E1 - mulst[l]) / tlst[l])
                                phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l])
                                temp1 = - PI * t2.real * f - t2.imag * phi0
                                temp2 = - PI * t2.imag * f + t2.real * phi0
                                kh.add_matrix_element(temp1, l, a2, b2, bcharge, a1, a1, bcharge, 5)
                                kh.add_matrix_element(temp2, l, a2, b2, bcharge, a1, a1, bcharge, 6)
                        if a1 == a2:
                            E1 = E[c] - E[a2]
                            for l in range(nleads):
                                t2 = Tba[l, b2, c] * Tba[l, a1, c].conjugate()
                                f = fermi_func((E1 - mulst[l]) / tlst[l])
                                phi0 = phi((E1 - mulst[l]) / tlst[l], dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l])
                                temp1 = - PI * t2.real * f + t2.imag * phi0
                                temp2 = - PI * t2.imag * f - t2.real * phi0
                                kh.add_matrix_element(temp1, l, a2, b2, bcharge, a1, a1, bcharge, 5)
                                kh.add_matrix_element(temp2, l, a2, b2, bcharge, a1, a1, bcharge, 6)
                if bcharge != 0:
                    # Intermediate state in lower charge state
                    for k in range(acount):
                        c = statesdm[acharge, k]
                        if a1 == b2:
                            E1 = E[b2] - E[c]
                            for l in range(nleads):
                                t2 = Tba[l, c, a2] * Tba[l, c, a1].conjugate()
                                f = fermi_func(-(E1 - mulst[l]) / tlst[l])
                                phi0 = phi((E1 - mulst[l])/tlst[l], dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l], sign=-1)
                                temp1 = - PI * t2.real * f + t2.imag * phi0
                                temp2 = - PI * t2.imag * f - t2.real * phi0
                                kh.add_matrix_element(temp1, l, a2, b2, bcharge, a1, a1, bcharge, 5)
                                kh.add_matrix_element(temp2, l, a2, b2, bcharge, a1, a1, bcharge, 6)
                        if a1 == a2:
                            E1 = E[a2] - E[c]
                            for l in range(nleads):
                                t2 = Tba[l, c, a1] * Tba[l, c, b2].conjugate()
                                f = fermi_func(-(E1 - mulst[l]) / tlst[l])
                                phi0 = phi((E1 - mulst[l])/tlst[l], dlst[l, 0] / tlst[l], dlst[l, 1] / tlst[l], sign=-1)
                                temp1 = - PI * t2.real * f - t2.imag * phi0
                                temp2 = - PI * t2.imag * f + t2.real * phi0
                                kh.add_matrix_element(temp1, l, a2, b2, bcharge, a1, a1, bcharge, 5)
                                kh.add_matrix_element(temp2, l, a2, b2, bcharge, a1, a1, bcharge, 6)


    cdef void generate_row_inverse_Liouvillian(self, long_t a1, long_t b1, long_t charge, KernelHandlerRTD kh) nogil:
        cdef double_t minE, E1
        cdef double_t [:] E = self._Ea

        minE = 1e-10
        E1 = E[a1] - E[b1]
        if minE > E1 >= 0:
            E1 = minE
        elif -minE < E1 <= 0:
            E1 = -minE

        kh.add_matrix_element(1.0/E1, 0, a1, b1, charge, a1, b1, charge, 7)


    cdef void set_Ozaki_params(self):
        cdef double_t[:,:] Ozaki_poles_and_residues

        cdef double_t BW_T = (abs(self.leads.dlst[0][0]) + abs(self.leads.dlst[0][1])) / 2.0 / min(self.leads.tlst)

        if self.BW_Ozaki_expansion < BW_T:
            self.Ozaki_poles_and_residues = BW_Ozaki(BW_T)
            self.BW_Ozaki_expansion = BW_T