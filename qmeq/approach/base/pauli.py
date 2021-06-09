"""Module containing python functions, which generate first order Pauli kernel."""

import numpy as np
import itertools

from ...wrappers.mytypes import doublenp

from ...specfunc.specfunc import func_pauli
from ..aprclass import Approach


# ---------------------------------------------------------------------------------------------------
# Pauli master equation
# ---------------------------------------------------------------------------------------------------
class ApproachPauli(Approach):

    kerntype = 'pyPauli'

    def get_kern_size(self):
        return self.si.npauli

    def prepare_arrays(self):
        Approach.prepare_arrays(self)
        nleads, ndm1 = self.si.nleads, self.si.ndm1
        self.paulifct = np.zeros((nleads, ndm1, 2), dtype=doublenp)

    def clean_arrays(self):
        Approach.clean_arrays(self)
        self.paulifct.fill(0.0)

    def generate_fct(self):
        """
        Make factors used for generating Pauli master equation kernel.

        Parameters
        ----------
        paulifct : array
            (Modifies) Factors used for generating Pauli master equation kernel.
        """
        E, Tba, si = self.qd.Ea, self.leads.Tba, self.si
        mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        itype = self.funcp.itype
        paulifct = self.paulifct
        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)
                Ecb = E[c]-E[b]
                for l in range(nleads):
                    xcb = (Tba[l, b, c]*Tba[l, c, b]).real
                    rez = func_pauli(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                    paulifct[l, cb, 0] = xcb*rez[0]
                    paulifct[l, cb, 1] = xcb*rez[1]

    def generate_kern(self):
        """
        Generate Pauli master equation kernel.

        Parameters
        ----------
        kern : array
            (Modifies) Kernel matrix for Pauli master equation.
        """
        si, kh = self.si, self.kernel_handler
        ncharge, statesdm = si.ncharge, si.statesdm

        for bcharge in range(ncharge):
            for b in statesdm[bcharge]:
                if not kh.is_unique(b, b, bcharge):
                    continue
                self.generate_coupling_terms(b, b, bcharge)

    def generate_coupling_terms(self, b, bp, bcharge):
        paulifct = self.paulifct
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm

        acharge = bcharge-1
        ccharge = bcharge+1

        bb = si.get_ind_dm0(b, b, bcharge)
        for a in statesdm[acharge]:
            aa = si.get_ind_dm0(a, a, acharge)
            ba = si.get_ind_dm1(b, a, acharge)
            fctm, fctp = 0, 0
            for l in range(nleads):
                fctm -= paulifct[l, ba, 1]
                fctp += paulifct[l, ba, 0]
            kh.set_matrix_element_pauli(fctm, fctp, bb, aa)
        for c in statesdm[ccharge]:
            cc = si.get_ind_dm0(c, c, ccharge)
            cb = si.get_ind_dm1(c, b, bcharge)
            fctm, fctp = 0, 0
            for l in range(nleads):
                fctm -= paulifct[l, cb, 0]
                fctp += paulifct[l, cb, 1]
            kh.set_matrix_element_pauli(fctm, fctp, bb, cc)

    def generate_current(self):
        """
        Calculates currents using Pauli master equation approach.

        Parameters
        ----------
        current : array
            (Modifies) Values of the current having nleads entries.
        energy_current : array
            (Modifies) Values of the energy current having nleads entries.
        heat_current : array
            (Modifies) Values of the heat current having nleads entries.
        """
        phi0, E, paulifct, si = self.phi0, self.qd.Ea, self.paulifct, self.si
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        current = self.current
        energy_current = self.energy_current

        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c in statesdm[ccharge]:
                cc = si.get_ind_dm0(c, c, ccharge)
                for b in statesdm[bcharge]:
                    bb = si.get_ind_dm0(b, b, bcharge)
                    cb = si.get_ind_dm1(c, b, bcharge)
                    for l in range(nleads):
                        fct1 = +phi0[bb]*paulifct[l, cb, 0]
                        fct2 = -phi0[cc]*paulifct[l, cb, 1]
                        current[l] += fct1 + fct2
                        energy_current[l] += -(E[b]-E[c])*(fct1 + fct2)

        self.heat_current[:] = energy_current - current*self.leads.mulst
# ---------------------------------------------------------------------------------------------------
