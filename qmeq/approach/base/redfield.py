"""Module containing python functions, which generate first order Redfield kernel.
   For docstrings see documentation of module neumann1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import doublenp
from ...mytypes import complexnp

from ..aprclass import Approach
from .neumann1 import Approach1vN


# ---------------------------------------------------------------------------------------------------
# Redfield approach
# ---------------------------------------------------------------------------------------------------
class ApproachRedfield(Approach):

    kerntype = 'pyRedfield'

    def generate_fct(self):
        Approach1vN.generate_fct(self)

    def generate_coupling_terms(self, b, bp, bcharge):
        Tba, phi1fct = self.leads.Tba, self.phi1fct
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm

        acharge = bcharge-1
        ccharge = bcharge+1

        # --------------------------------------------------
        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                bpap = si.get_ind_dm1(bp, ap, acharge)
                ba = si.get_ind_dm1(b, a, acharge)
                fct_aap = 0
                for l in range(nleads):
                    fct_aap += (+ Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpap, 0].conjugate()
                                - Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, ba, 0])
                kh.set_matrix_element(fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    bppa = si.get_ind_dm1(bpp, a, acharge)
                    for l in range(nleads):
                        fct_bppbp += +Tba[l, b, a]*Tba[l, a, bpp]*phi1fct[l, bppa, 1].conjugate()
                for c in statesdm[ccharge]:
                    cbpp = si.get_ind_dm1(c, bpp, bcharge)
                    for l in range(nleads):
                        fct_bppbp += +Tba[l, b, c]*Tba[l, c, bpp]*phi1fct[l, cbpp, 0]
                kh.set_matrix_element(fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    bppa = si.get_ind_dm1(bpp, a, acharge)
                    for l in range(nleads):
                        fct_bbpp += -Tba[l, bpp, a]*Tba[l, a, bp]*phi1fct[l, bppa, 1]
                for c in statesdm[ccharge]:
                    cbpp = si.get_ind_dm1(c, bpp, bcharge)
                    for l in range(nleads):
                        fct_bbpp += -Tba[l, bpp, c]*Tba[l, c, bp]*phi1fct[l, cbpp, 0].conjugate()
                kh.set_matrix_element(fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for c, cp in itertools.product(statesdm[ccharge], statesdm[ccharge]):
            if kh.is_included(c, cp, ccharge):
                cpbp = si.get_ind_dm1(cp, bp, bcharge)
                cb = si.get_ind_dm1(c, b, bcharge)
                fct_ccp = 0
                for l in range(nleads):
                    fct_ccp += (+ Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpbp, 1]
                                - Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cb, 1].conjugate())
                kh.set_matrix_element(fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    def generate_current(self):
        E, Tba = self.qd.Ea, self.leads.Tba
        phi1fct, phi1fct_energy = self.phi1fct, self.phi1fct_energy

        si = self.si
        ndm0, ndm1, npauli = si.ndm0, si.ndm1, si.npauli
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        kh = self.kernel_handler
        kh.set_phi0(self.phi0)

        phi1 = np.zeros((nleads, ndm1), dtype=complexnp)
        current = np.zeros(nleads, dtype=complexnp)
        energy_current = np.zeros(nleads, dtype=complexnp)

        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)

                for l in range(nleads):
                    for bp in statesdm[bcharge]:
                        if not kh.is_included(bp, b, bcharge):
                            continue
                        phi0bpb = kh.get_phi0_element(bp, b, bcharge)

                        cbp = si.get_ind_dm1(c, bp, bcharge)
                        fct1 = phi1fct[l, cbp, 0]
                        fct1h = phi1fct_energy[l, cbp, 0]

                        phi1[l, cb] += Tba[l, c, bp]*phi0bpb*fct1
                        current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1
                        energy_current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1h

                    for cp in statesdm[ccharge]:
                        if not kh.is_included(c, cp, ccharge):
                            continue
                        phi0ccp = kh.get_phi0_element(c, cp, ccharge)

                        cpb = si.get_ind_dm1(cp, b, bcharge)
                        fct2 = phi1fct[l, cpb, 1]
                        fct2h = phi1fct_energy[l, cpb, 1]

                        phi1[l, cb] += Tba[l, cp, b]*phi0ccp*fct2
                        current[l] += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2
                        energy_current[l] += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2h

        self.phi1 = phi1
        self.current = np.array(-2*current.imag, dtype=doublenp)
        self.energy_current = np.array(-2*energy_current.imag, dtype=doublenp)
        self.heat_current = self.energy_current - self.current*self.leads.mulst
# ---------------------------------------------------------------------------------------------------
