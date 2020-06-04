from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class KernelHandler(object):

    def __init__(self, si):
        self.si = si
        self.ndm0 = si.ndm0
        self.ndm0r = si.ndm0r
        self.npauli = si.npauli
        self.phi0 = None
        self.kern = None

    def set_kern(self, kern):
        self.kern = kern

    def set_phi0(self, phi0):
        self.phi0 = phi0

    def is_included(self, b, bp, bcharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        if bbp == -1:
            return False

        return True

    def is_unique(self, b, bp, bcharge):
        bbp_bool = self.si.get_ind_dm0(b, bp, bcharge, maptype=2)
        return bbp_bool

    def set_energy(self, energy, b, bp, bcharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = self.ndm0 + bbp - self.npauli
        bbpi_bool = True if bbpi >= self.ndm0 else False

        if bbpi_bool:
            self.kern[bbp, bbpi] = self.kern[bbp, bbpi] + energy
            self.kern[bbpi, bbp] = self.kern[bbpi, bbp] - energy

    def set_matrix_element(self, fct, b, bp, bcharge, a, ap, acharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = self.ndm0 + bbp - self.npauli
        bbpi_bool = True if bbpi >= self.ndm0 else False

        aap = self.si.get_ind_dm0(a, ap, acharge)
        aapi = self.ndm0 + aap - self.npauli
        aap_sgn = +1 if self.si.get_ind_dm0(a, ap, acharge, maptype=3) else -1

        fct_imag = fct.imag
        fct_real = fct.real

        self.kern[bbp, aap] += fct_imag
        if aapi >= self.ndm0:
            self.kern[bbp, aapi] += fct_real*aap_sgn
            if bbpi_bool:
                self.kern[bbpi, aapi] += fct_imag*aap_sgn
        if bbpi_bool:
            self.kern[bbpi, aap] += -fct_real

    def set_matrix_element_pauli(self, fctm, fctp, bb, aa):
        self.kern[bb, bb] += fctm
        self.kern[bb, aa] += fctp

    def get_phi0_element(self, b, bp, bcharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        if bbp == -1:
            return 0.0

        bbpi = self.ndm0 + bbp - self.npauli
        bbpi_bool = True if bbpi >= self.ndm0 else False

        phi0_real = self.phi0[bbp]
        phi0_imag = 0
        if bbpi_bool:
            bbp_conj = self.si.get_ind_dm0(b, bp, bcharge, maptype=3)
            phi0_imag = self.phi0[bbpi] if bbp_conj else -self.phi0[bbpi]

        return phi0_real + 1j*phi0_imag

class KernelHandlerMatrixFree(KernelHandler):

    def __init__(self, si):
        KernelHandler.__init__(self, si)
        self.dphi0_dt = None

    def set_dphi0_dt(self, dphi0_dt):
        self.dphi0_dt = dphi0_dt

    def set_energy(self, energy, b, bp, bcharge):
        if b == bp:
            return

        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = self.ndm0 + bbp - self.npauli

        phi0bbp = self.get_phi0_element(b, bp, bcharge)
        dphi0_dt_bbp = -1j*energy*phi0bbp

        self.dphi0_dt[bbp] += dphi0_dt_bbp.real
        self.dphi0_dt[bbpi] -= dphi0_dt_bbp.imag

    def set_matrix_element(self, fct, b, bp, bcharge, a, ap, acharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = self.ndm0 + bbp - self.npauli
        bbpi_bool = True if bbpi >= self.ndm0 else False
        aap = self.si.get_ind_dm0(a, ap, acharge)

        phi0aap = self.get_phi0_element(a, ap, acharge)
        dphi0_dt_bbp = -1j*fct*phi0aap

        self.dphi0_dt[bbp] += dphi0_dt_bbp.real
        if bbpi_bool:
            self.dphi0_dt[bbpi] -= dphi0_dt_bbp.imag

    def set_matrix_element_pauli(self, fctm, fctp, bb, aa):
        self.dphi0_dt[bb] += fctm*self.phi0[bb] + fctp*self.phi0[aa]

    def get_phi0_norm(self):
        ncharge, statesdm = self.si.ncharge, self.si.statesdm

        norm = 0.0
        for bcharge in range(ncharge):
            for b in statesdm[bcharge]:
                bb = self.si.get_ind_dm0(b, b, bcharge)
                norm += self.phi0[bb]

        return norm
