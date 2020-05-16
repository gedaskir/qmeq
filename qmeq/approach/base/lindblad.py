"""Module containing python functions, which generate first order Lindblad kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import complexnp
from ...mytypes import doublenp

from ...specfunc.specfunc import func_pauli
from ..aprclass import Approach


# ---------------------------------------------------------------------------------------------------
# Lindblad approach
# ---------------------------------------------------------------------------------------------------
class ApproachLindblad(Approach):

    kerntype = 'pyLindblad'

    def generate_fct(self):
        """
        Make factors used for generating Lindblad master equation kernel.

        Parameters
        ----------
        self.tLba : array
            (Modifies) Jump operator matrix in many-body basis.
        """
        Tba, E, si = self.leads.Tba, self.qd.Ea, self.si
        mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst
        mtype, itype = self.leads.mtype, self.funcp.itype
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm
        #
        tLba = np.zeros(Tba.shape, dtype=mtype)
        for charge in range(ncharge-1):
            bcharge = charge+1
            acharge = charge
            for b, a in itertools.product(statesdm[bcharge], statesdm[acharge]):
                Eba = E[b]-E[a]
                for l in range(nleads):
                    fct1, fct2 = func_pauli(Eba, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                    tLba[l, b, a] = np.sqrt(fct1)*Tba[l, b, a]
                    tLba[l, a, b] = np.sqrt(fct2)*Tba[l, a, b]
        self.tLba = tLba

    def generate_coupling_terms(self, b, bp, bcharge):
        tLba = self.tLba
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm

        acharge = bcharge-1
        ccharge = bcharge+1

        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                fct_aap = 0
                for l in range(nleads):
                    fct_aap += tLba[l, b, a]*tLba[l, bp, ap].conjugate()
                kh.set_matrix_element(1j*fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    for l in range(nleads):
                        fct_bppbp += -0.5*tLba[l, a, b].conjugate()*tLba[l, a, bpp]
                for c in statesdm[ccharge]:
                    for l in range(nleads):
                        fct_bppbp += -0.5*tLba[l, c, b].conjugate()*tLba[l, c, bpp]
                kh.set_matrix_element(1j*fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    for l in range(nleads):
                        fct_bbpp += -0.5*tLba[l, a, bpp].conjugate()*tLba[l, a, bp]
                for c in statesdm[ccharge]:
                    for l in range(nleads):
                        fct_bbpp += -0.5*tLba[l, c, bpp].conjugate()*tLba[l, c, bp]
                kh.set_matrix_element(1j*fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for c, cp in itertools.product(statesdm[ccharge], statesdm[ccharge]):
            if kh.is_included(c, cp, ccharge):
                fct_ccp = 0
                for l in range(nleads):
                    fct_ccp += tLba[l, b, c]*tLba[l, bp, cp].conjugate()
                kh.set_matrix_element(1j*fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    def generate_current(self):
        """
        Calculates currents using Lindblad approach.

        Parameters
        ----------
        self : Approach
            Approach object.

        self.current : array
            (Modifies) Values of the current having nleads entries.
        self.energy_current : array
            (Modifies) Values of the energy current having nleads entries.
        self.heat_current : array
            (Modifies) Values of the heat current having nleads entries.
        """
        phi0p, E, tLba, si = self.phi0, self.qd.Ea, self.tLba, self.si
        ndm0, npauli = si.ndm0, si.npauli
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        kh = self.kernel_handler
        kh.set_phi0(self.phi0)

        current = np.zeros(nleads, dtype=complexnp)
        energy_current = np.zeros(nleads, dtype=complexnp)

        for charge in range(ncharge):
            ccharge = charge+1
            bcharge = charge
            acharge = charge-1

            for b, bp in itertools.product(statesdm[bcharge], statesdm[bcharge]):
                if not kh.is_included(b, bp, bcharge):
                    continue
                phi0bbp = kh.get_phi0_element(b, bp, bcharge)

                for l in range(nleads):
                    for a in statesdm[acharge]:
                        fcta = tLba[l, a, b]*phi0bbp*tLba[l, a, bp].conjugate()
                        current[l] -= fcta
                        energy_current[l] += (E[a]-0.5*(E[b]+E[bp]))*fcta
                    for c in statesdm[ccharge]:
                        fctc = tLba[l, c, b]*phi0bbp*tLba[l, c, bp].conjugate()
                        current[l] += fctc
                        energy_current[l] += (E[c]-0.5*(E[b]+E[bp]))*fctc

        self.current = np.array(current.real, dtype=doublenp)
        self.energy_current = np.array(energy_current.real, dtype=doublenp)
        self.heat_current = self.energy_current - self.current*self.leads.mulst
# ---------------------------------------------------------------------------------------------------
