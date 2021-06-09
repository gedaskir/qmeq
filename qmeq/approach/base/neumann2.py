"""Module containing python functions, which solve 2vN approach integral equations."""

import numpy as np
from scipy import pi
import itertools

from ...wrappers.mytypes import complexnp
from ...wrappers.mytypes import doublenp
from ...wrappers.mytypes import intnp

from ...specfunc.specfunc import kernel_fredriksen
from ...specfunc.specfunc import hilbert_fredriksen
from ..aprclass import ApproachBase2vN


def get_htransf_phi1k(phi1k, funcp):
    """
    Performs Hilbert transform of phi1k.

    Parameters
    ----------
    phi1k : ndarray
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0)
        containing energy resolved first order density matrix elements Phi[1](k).
    funcp : FunctionProperties
        FunctionProperties object.

    funcp.ht_ker : ndarray
        (Modifies) Kernel used when performing Hilbert transform using FFT.

    Returns
    -------
    phi1k : ndarray
        Numpy array with dimensions (len(Ek_grid_ext), nleads, ndm1, ndm0).
        phi1k padded from the left and the right by zeros.
    hphi1k : ndarray
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
    phi1k = np.concatenate((np.zeros((funcp.kpnt_left, nleads, ndm1, ndm0)),
                            phi1k,
                            np.zeros((funcp.kpnt_right, nleads, ndm1, ndm0))), axis=0)
    # Make the Hilbert transformation
    hphi1k = np.zeros(phi1k.shape, dtype=complexnp)
    for l, cb, bbp in itertools.product(range(nleads), range(ndm1), range(ndm0)):
        # print(l, cb, bbp)
        hphi1k[:, l, cb, bbp] = hilbert_fredriksen(phi1k[:, l, cb, bbp], funcp.ht_ker)
    return phi1k, hphi1k


def get_htransf_fk(fk, funcp):
    """
    Performs Hilbert transform of Fermi function.
    The energy is shifted by positive infinitesimal
    to the complex upper half-plane.

    Parameters
    ----------
    fk : ndarray
        Numpy array with dimensions (nleads, len(Ek_grid)),
        containing Fermi function values on the grid Ek_grid.
    funcp : FunctionProperties
        FunctionProperties object.

    funcp.ht_ker : ndarray
        (Modifies) Kernel used when performing Hilbert transform using FFT.

    Returns
    -------
    fk : ndarray
        Numpy array with dimensions (nleads, len(Ek_grid_ext)).
        fk padded from the left and the right by zeros.
    hfk : ndarray
        Numpy array with dimensions (nleads, len(Ek_grid_ext)).
        Hilbert transform of fk on extended grid.
    """
    nleads = fk.shape[0]
    Eklen_ext = fk.shape[1] + funcp.kpnt_left + funcp.kpnt_right
    # Create the kernels for Hilbert transformation
    if funcp.ht_ker is None or 2*Eklen_ext != len(funcp.ht_ker):
        funcp.ht_ker = kernel_fredriksen(Eklen_ext)
    # Pad fk values with zeros from the left and the right
    fk = np.concatenate((np.zeros((nleads, funcp.kpnt_left)),
                         fk,
                         np.zeros((nleads, funcp.kpnt_right))), axis=1)
    # Calculate the Hilbert transform with added positive infinitesimal of the Fermi functions
    hfk = np.zeros((nleads, Eklen_ext), dtype=doublenp)
    for l in range(nleads):
        hfk[l] = hilbert_fredriksen(fk[l], funcp.ht_ker).real
    # The energy is shifted by positive infinitesimal to the complex upper half-plane
    hfk = hfk - 1j*fk
    return fk, hfk


class Approach2vN(ApproachBase2vN):

    kerntype = 'py2vN'

    #region Preparation

    def prepare_kern(self):
        if self.is_zeroth_iteration:
            self.prepare_energy_grid()
            self.prepare_distribution()

        ApproachBase2vN.prepare_kern(self)

    def prepare_energy_grid(self):
        """"Prepares the energy grid."""
        self.make_Ek_grid()

        # Define the extended grid Ek_grid_ext for calculations outside the bandwidth
        # Here funcp.emin, funcp.emax are defined
        self.determine_emin_emax()

        # Here Ek_grid_ext, funcp.kpnt_left, funcp.kpnt_right are defined
        self.determine_grid_ext()

    def make_Ek_grid(self):
        """Make an energy grid on which 2vN equations are solved. """
        if self.funcp.kpnt is None:
            raise ValueError('kpnt needs to be specified.')
        if self.si.nleads > 0:
            dmin = np.min(self.leads.dlst)
            dmax = np.max(self.leads.dlst)
            Ek_grid, kpnt = self.Ek_grid, self.funcp.kpnt
            if Ek_grid[0] != dmin or Ek_grid[-1] != dmax or Ek_grid.shape[0] != kpnt:
                self.funcp.dmin = dmin
                self.funcp.dmax = dmax
                self.Ek_grid = np.linspace(dmin, dmax, kpnt)
                #
                if self.niter != -1:
                    print("WARNING: Ek_grid has changed. Restarting the calculation.")
                    self.restart()
                #
                if ((dmin * np.ones(self.si.nleads)).tolist() != self.leads.dlst.T[0].tolist() or
                    (dmax * np.ones(self.si.nleads)).tolist() != self.leads.dlst.T[1].tolist()):
                    print("WARNING: The bandwidth and Ek_grid for all leads will be the same: from " +
                          "dmin=" + str(dmin) + " to dmax=" + str(dmax) + ".")

    def determine_emin_emax(self):
        """
        Finds how much the Ek_grid has to be expanded
        above the bandwidth D of the leads.

        Parameters
        ----------
        funcp.emin : float
            (Modifies) Minimal energy in the updated Ek_grid.
        funcp.emax : float
            (Modifies) Maximal energy in the updated Ek_grid.
        """
        E, si = self.qd.Ea, self.si
        dmin, dmax = self.funcp.dmin, self.funcp.dmax

        lst = [dmin, dmax]
        for charge in range(si.ncharge):
            for b, bp in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                lst.append(dmax-E[b]+E[bp])
                lst.append(dmin-E[b]+E[bp])
        for charge in range(si.ncharge-2):
            for d, b in itertools.product(si.statesdm[charge+2], si.statesdm[charge]):
                lst.append(dmin+E[d]-E[b])
                lst.append(dmax+E[d]-E[b])

        self.funcp.emax = max(lst)
        self.funcp.emin = min(lst)

    def determine_grid_ext(self):
        """
        Expands the Ek_grid from emin to emax.

        Parameters
        ----------
        Ek_grid_ext : ndarray
            (Modifies) Extended Ek_grid from emin to emax.
        funcp.kpnt_left : int
            (Modifies) Number of points Ek_grid is extended to the left.
        funcp.kpnt_right : int
            (Modifies) Number of points Ek_grid is extended to the right.
        """
        emin_, emax_ = self.funcp.emin, self.funcp.emax
        dmin, dmax = self.funcp.dmin, self.funcp.dmax
        ext_fct = self.funcp.ext_fct
        Ek_grid = self.Ek_grid

        step = Ek_grid[1]-Ek_grid[0]
        emin = ext_fct*(emin_-dmin)+dmin
        emax = ext_fct*(emax_-dmax)+dmax
        ext_left = np.sort(-np.arange(-dmin+step, -emin+step, step))
        ext_right = np.arange(dmax+step, emax+step, step)

        self.Ek_grid_ext = np.concatenate((ext_left, Ek_grid, ext_right))
        self.funcp.kpnt_left, self.funcp.kpnt_right = len(ext_left), len(ext_right)

    def prepare_distribution(self):
        """"
        Prepares the Fermi distribution functions
        and their Hilbert transform on the energy grid.

        Parameters
        ----------
        fkp : ndarray
            (Modifies) nleads by len(Ek_grid_ext) numpy array containing Fermi function.
        fkm : ndarray
            (Modifies) nleads by len(Ek_grid_ext) numpy array containing 1-Fermi function.
        hfkp : ndarray
            (Modifies) Hilbert transform of fkp.
        hfkm : ndarray
            (Modifies) Hilbert transform of fkm.
        """
        # Generate the Fermi functions on the grid
        # This is necessary to generate only if Ek_grid, mulst, or tlst are changed
        nleads = self.si.nleads
        mulst, tlst = self.leads.mulst, self.leads.tlst
        Ek_grid = self.Ek_grid
        Eklen = len(Ek_grid)

        self.fkp = np.zeros((nleads, Eklen), dtype=doublenp)
        for l in range(nleads):
            self.fkp[l] = 1/(np.exp((Ek_grid - mulst[l])/tlst[l]) + 1)
        self.fkm = 1-self.fkp
        self.fkp, self.hfkp = get_htransf_fk(self.fkp, self.funcp)
        self.fkm, self.hfkm = get_htransf_fk(self.fkm, self.funcp)

    #endregion Preparation

    #region Generation

    def iterate(self):
        terms_calculator = TermsCalculator2vN(self)
        terms_calculator.iterate()

    def determine_phi1_phi0(self):
        """
        Integrates phi1k over energy.

        Parameters
        ----------
        phi1_phi0 : ndarray
            (Modifies) Numpy array with dimensions (nleads, ndm1, ndm0),
            corresponding to energy integrated Phi[1](k).
        e_phi1_phi0 : ndarray
            (Modifies) Numpy array with dimensions (nleads, ndm1, ndm0),
            corresponding to energy integrated Ek*Phi[1](k).
        """
        phi1k, Ek_grid = self.phi1k, self.Ek_grid
        # Get integrated Phi[1]_{cb} in terms of Phi[0]_{bb'}
        phi1_phi0 = self.phi1_phi0
        e_phi1_phi0 = self.e_phi1_phi0

        Eklen = len(Ek_grid)
        for i in range(Eklen):
            # Trapezoidal rule
            if i == 0:
                dx = Ek_grid[i+1] - Ek_grid[i]
            elif i == Eklen-1:
                dx = Ek_grid[i] - Ek_grid[i-1]
            else:
                dx = Ek_grid[i+1] - Ek_grid[i-1]
            phi1_phi0 += 1/2*dx*phi1k[i]
            e_phi1_phi0 += 1/2*dx*Ek_grid[i]*phi1k[i]

    def generate_kern(self):
        """
        From Phi[1](k) generate equations containing just Phi[0].

        Parameters
        ----------
        kern : ndarray
            (Modifies) Kernel for master equation involving just Phi[0].
        bvec : ndarray
            (Modifies) Right hand side column vector for master equation.
        """
        phi1_phi0, E, Tba, si = self.phi1_phi0, self.qd.Ea, self.leads.Tba, self.si
        ncharge, nleads, ndm0, statesdm = si.ncharge, si.nleads, si.ndm0, si.statesdm

        # Integrated Phi[1]_{bc} in terms of phi1_phi0
        shuffle = np.zeros((ndm0, ndm0), dtype=intnp)
        for charge in range(ncharge):
            for b, bp in itertools.combinations_with_replacement(statesdm[charge], 2):
                bbp = si.get_ind_dm0(b, bp, charge)
                bpb = si.get_ind_dm0(bp, b, charge)
                shuffle[bbp, bpb] = 1
                shuffle[bpb, bbp] = 1
        phi1_phi0_conj = np.dot(np.conjugate(phi1_phi0), shuffle)

        kern = self.kern
        for charge in range(ncharge):
            acharge = charge-1
            bcharge = charge
            ccharge = charge+1
            for b, bp in itertools.product(statesdm[bcharge], statesdm[bcharge]):
                bbp = si.get_ind_dm0(b, bp, bcharge)
                kern[bbp, bbp] = E[b]-E[bp]

                for a1 in statesdm[acharge]:
                    bpa1 = si.get_ind_dm1(bp, a1, acharge)
                    ba1 = si.get_ind_dm1(b, a1, acharge)
                    for l in range(nleads):
                        kern[bbp] += +Tba[l, b, a1]*phi1_phi0_conj[l, bpa1]
                        kern[bbp] += -phi1_phi0[l, ba1]*Tba[l, a1, bp]

                for c1 in statesdm[ccharge]:
                    c1bp = si.get_ind_dm1(c1, bp, bcharge)
                    c1b = si.get_ind_dm1(c1, b, bcharge)
                    for l in range(nleads):
                        kern[bbp] += +Tba[l, b, c1]*phi1_phi0[l, c1bp]
                        kern[bbp] += -phi1_phi0_conj[l, c1b]*Tba[l, c1, bp]

    def generate_current(self):
        """
        Finds the currents for 2vN approach after an approximate
        solution for integral 2vN equations was found.

        Parameters
        ----------
        phi1 : ndarray
            (Modifies) nleads by ndm1 numpy array, which gives energy integrated
            Phi[1](k) after the values for Phi[0] were inserted.
        current : ndarray
            (Modifies) Values of the current having nleads entries.
        energy_current : ndarray
            (Modifies) Values of the energy current having nleads entries.
        heat_current : ndarray
            (Modifies) Values of the heat current having nleads entries.
        """
        phi1_phi0, e_phi1_phi0 = self.phi1_phi0, self.e_phi1_phi0
        phi0, Tba, si = self.phi0, self.leads.Tba, self.si
        ncharge, nleads, ndm1, statesdm = si.ncharge, si.nleads, si.ndm1, si.statesdm

        phi1 = self.phi1
        h_phi1 = self.h_phi1
        current = self.current
        energy_current = self.energy_current

        for l in range(nleads):
            phi1[l] = np.dot(phi1_phi0[l], phi0)
            h_phi1[l] = np.dot(e_phi1_phi0[l], phi0)

        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)

                for l in range(nleads):
                    current_l = phi1[l, cb]*Tba[l, b, c]
                    energy_current_l = h_phi1[l, cb]*Tba[l, b, c]

                    current[l] += -2*current_l.imag
                    energy_current[l] += -2*energy_current_l.imag

        self.heat_current[:] = energy_current - current*self.leads.mulst

    #endregion Generation


class TermsCalculator2vN(object):

    def __init__(self, appr):
        self.appr = appr

    def iterate(self):
        self.retrieve_approach_variables()


    def retrieve_approach_variables(self):
        self.si = self.appr.si
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


    def iterate(self):
        """
        Performs the iterative solution of the 2vN approach integral equations.

        Parameters
        ----------
        phi1k_delta : ndarray
            (Modifies) Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0).
            Correction to Phi[1](k) after an iteration.
        hphi1k_delta : ndarray
            (Modifies) Numpy array with dimensions (len(Ek_grid_ext), nleads, ndm1, ndm0).
            Hilbert transform of phi1k_delta on extended grid Ek_grid_ext.
        kern1k_inv : ndarray
            (Modifies) Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm1)
            corresponding to inverse of energy resolved local kernel for Phi[1](k).
        """
        self.retrieve_approach_variables()

        Eklen = len(self.Ek_grid)
        nleads = self.si.nleads

        if self.appr.is_zeroth_iteration:
            # Calculate the zeroth iteration of Phi[1](k)
            for k in range(Eklen):
                for l in range(nleads):
                    self.phi1k_local(k, l)

            self.appr.is_zeroth_iteration = False
        else:
            # Hilbert transform phi1k_delta on extended grid Ek_grid_ext
            # Here phi1k_delta_old is current phi1k_delta state, but on extended grid
            # print('Hilbert transforming')
            self.phi1k_delta_old, self.hphi1k_delta = get_htransf_phi1k(self.phi1k_delta, self.appr.funcp)
            # print('Making an iteration')
            for k in range(Eklen):
                for l in range(nleads):
                    self.phi1k_iterate(k, l)

    def phi1k_local(self, k, l):
        """
        Constructs Phi[1](k) corresponding to local approximation at Ek_grid[ind].
        More precisely, generates local approximation kernels L1 and L0, Phi[1](k) = L0(k)Phi[0],
        L1(k)Phi[1](k) = L0p(k)Phi[0], L0 = L1^{-1}L0p.

        Parameters
        ----------
        k : int
            Index of a point on a Ek_grid_ext.
        l : int
            Lead index.

        kern0 : ndarray
            (Modifies) Numpy array with dimensions (nleads, ndm1, ndm0).
            Gives local approximation kernel L0(k) at Ek_grid[ind].
            Shows how Phi[1](k) is expressed in terms of Phi[0].
        kern1_inv : ndarray
            (Modifies) Numpy array with dimensions (nleads, ndm1, ndm1).
            Gives inverse of local approximation kernel L1(k) at Ek_grid[ind].
        """
        E, Tba, si = self.Ea, self.Tba, self.si
        fk, hfkp, hfkm = self.fkp, self.hfkp, self.hfkm
        Ek_grid = self.Ek_grid_ext

        ncharge, nleads, ndm0, ndm1 = si.ncharge, si.nleads, si.ndm0, si.ndm1
        statesdm = si.statesdm

        kern0 = np.zeros((ndm1, ndm0), dtype=complexnp)
        kern1 = np.zeros((ndm1, ndm1), dtype=complexnp)
        kern1_inv = np.zeros((ndm1, ndm1), dtype=complexnp)

        func_2vN = self.func_2vN

        ind = k + self.kpnt_left
        Ek = Ek_grid[ind]

        # Note that the bias is put in the distributions and not in the dispersion
        fp = fk[l, ind]  # fermi_func(+(Ek-mulst[l])/tlst[l])
        fm = 1-fp        # fermi_func(-(Ek-mulst[l])/tlst[l])

        for charge in range(ncharge-1):
            dcharge = charge+2
            ccharge = charge+1
            bcharge = charge
            acharge = charge-1
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)
                kern1[cb, cb] += Ek-E[c]+E[b]
                # Phi[0] terms
                for b1 in statesdm[bcharge]:
                    b1b = si.get_ind_dm0(b1, b, bcharge)
                    kern0[cb, b1b] += +Tba[l, c, b1]*fp
                for c1 in statesdm[ccharge]:
                    cc1 = si.get_ind_dm0(c, c1, ccharge)
                    kern0[cb, cc1] += -Tba[l, c1, b]*fm
                # ---------------------------------------------------------------------------
                # Phi[1] terms
                # 2nd and 7th terms
                for b1, a1 in itertools.product(statesdm[bcharge], statesdm[acharge]):
                    b1a1 = si.get_ind_dm1(b1, a1, acharge)
                    for l1 in range(nleads):
                        kern1[cb, b1a1] -= +Tba[l1, c, b1]*Tba[l1, a1, b]*(
                                             + func_2vN(+(Ek-E[b1]+E[b]), l1, +1, hfkp)
                                             - func_2vN(-(Ek-E[c]+E[a1]), l1, -1, hfkp))
                # 6th and 8th terms
                for b1 in statesdm[bcharge]:
                    cb1 = si.get_ind_dm1(c, b1, bcharge)
                    for l1 in range(nleads):
                        for c1 in statesdm[ccharge]:
                            kern1[cb, cb1] -= (+Tba[l1, b1, c1]*Tba[l1, c1, b]
                                               * func_2vN(+(Ek-E[c]+E[c1]), l1, +1, hfkp))
                        for a1 in statesdm[acharge]:
                            kern1[cb, cb1] -= (-Tba[l1, b1, a1]*Tba[l1, a1, b]
                                               * func_2vN(-(Ek-E[c]+E[a1]), l1, -1, hfkm))
                # 1st and 3rd terms
                for c1 in statesdm[ccharge]:
                    c1b = si.get_ind_dm1(c1, b, bcharge)
                    for l1 in range(nleads):
                        for b1 in statesdm[bcharge]:
                            kern1[cb, c1b] -= (+Tba[l1, c, b1]*Tba[l1, b1, c1]
                                               * func_2vN(+(Ek-E[b1]+E[b]), l1, +1, hfkm))
                        for d1 in statesdm[dcharge]:
                            kern1[cb, c1b] -= (-Tba[l1, c, d1]*Tba[l1, d1, c1]
                                               * func_2vN(-(Ek-E[d1]+E[b]), l1, -1, hfkp))
                # 5th and 4th terms
                for d1, c1 in itertools.product(statesdm[dcharge], statesdm[ccharge]):
                    d1c1 = si.get_ind_dm1(d1, c1, ccharge)
                    for l1 in range(nleads):
                        kern1[cb, d1c1] -= +Tba[l1, c, d1]*Tba[l1, c1, b]*(
                                             + func_2vN(+(Ek-E[c]+E[c1]), l1, +1, hfkm)
                                             - func_2vN(-(Ek-E[d1]+E[b]), l1, -1, hfkm))

        kern1_inv = np.linalg.inv(kern1)
        kern0 = np.dot(kern1_inv, kern0)

        self.kern1k_inv[k, l, :] = kern1_inv
        self.phi1k_delta[k, l, :] = kern0

    def phi1k_iterate(self, k, l):
        """
        Iterates the 2vN integral equation.

        Parameters
        ----------
        k : int
            Index of a point on a Ek_grid.
        l : int
            Lead index.

        kern0 : ndarray
            (Modifies) Numpy array with dimensions (nleads, ndm1, ndm0)
            Gives a correction to Phi[1](k) after iteration of integral equation.
            Shows how delta[Phi[1](k)] is expressed in terms of Phi[0].
        """
        E, Tba, si = self.Ea, self.Tba, self.si
        fk = self.fkp
        Ek_grid = self.Ek_grid_ext

        ncharge, nleads, ndm0, ndm1 = si.ncharge, si.nleads, si.ndm0, si.ndm1
        statesdm = si.statesdm

        kern1_inv = self.kern1k_inv[k, l]
        kern0 = np.zeros((ndm1, ndm0), dtype=complexnp)
        term = np.zeros(ndm0, dtype=complexnp)

        get_at_k1 = self.get_at_k1

        ind = k + self.kpnt_left
        Ek = Ek_grid[ind]

        fp = fk[l, ind]  # fermi_func((Ek-mulst[l])/tlst[l])
        fm = 1-fp

        for charge in range(ncharge-1):
            dcharge = charge+2
            ccharge = charge+1
            bcharge = charge
            acharge = charge-1
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)
                term.fill(0.0)
                # 1st term
                for a1 in statesdm[acharge]:
                    ba1 = si.get_ind_dm1(b, a1, acharge)
                    for b1, l1 in itertools.product(statesdm[bcharge], range(nleads)):
                        # print('1')
                        hu, u = get_at_k1(+(Ek-E[b1]+E[b]), l1, ba1, True)
                        term += -Tba[l1, c, b1]*Tba[l, b1, a1]*fp*(hu - 1j*u)
                # 2nd and 5th terms
                for b1, c1 in itertools.product(statesdm[bcharge], statesdm[ccharge]):
                    # print('2 and 5')
                    c1b1 = si.get_ind_dm1(c1, b1, bcharge)
                    for l1 in range(nleads):
                        # 2nd term
                        hu, u = get_at_k1(+(Ek-E[b1]+E[b]), l1, c1b1, True)
                        term += -Tba[l1, c, b1]*(hu - 1j*u)*fm*Tba[l, c1, b]
                        # 5th term
                        hu, u = get_at_k1(+(Ek-E[c]+E[c1]), l1, c1b1, True)
                        term += -Tba[l, c, b1]*fp*(hu - 1j*u)*Tba[l1, c1, b]
                # 3rd term
                for c1 in statesdm[ccharge]:
                    c1b = si.get_ind_dm1(c1, b, bcharge)
                    for d1, l1 in itertools.product(statesdm[dcharge], range(nleads)):
                        # print('3')
                        hu, u = get_at_k1(-(Ek-E[d1]+E[b]), l1, c1b, False)
                        term += +Tba[l1, c, d1]*Tba[l, d1, c1]*fp*(hu + 1j*u)
                # 4th term
                for d1, c1 in itertools.product(statesdm[dcharge], statesdm[ccharge]):
                    # print('4')
                    d1c1 = si.get_ind_dm1(d1, c1, ccharge)
                    for l1 in range(nleads):
                        hu, u = get_at_k1(-(Ek-E[d1]+E[b]), l1, d1c1, False)
                        term += +Tba[l1, c, d1]*(hu + 1j*u)*fm*Tba[l, c1, b]
                # 6th term
                for d1 in statesdm[dcharge]:
                    d1c = si.get_ind_dm1(d1, c, ccharge)
                    for c1, l1 in itertools.product(statesdm[ccharge], range(nleads)):
                        # print('6')
                        hu, u = get_at_k1(+(Ek-E[c]+E[c1]), l1, d1c, True)
                        term += -(hu - 1j*u)*fm*Tba[l, d1, c1]*Tba[l1, c1, b]
                # 7th term
                for b1, a1 in itertools.product(statesdm[bcharge], statesdm[acharge]):
                    # print('7')
                    b1a1 = si.get_ind_dm1(b1, a1, acharge)
                    for l1 in range(nleads):
                        hu, u = get_at_k1(-(Ek-E[c]+E[a1]), l1, b1a1, False)
                        term += +Tba[l, c, b1]*fp*(hu + 1j*u)*Tba[l1, a1, b]
                # 8th term
                for b1 in statesdm[bcharge]:
                    cb1 = si.get_ind_dm1(c, b1, bcharge)
                    for a1, l1 in itertools.product(statesdm[acharge], range(nleads)):
                        # print('8')
                        hu, u = get_at_k1(-(Ek-E[c]+E[a1]), l1, cb1, False)
                        term += +(hu + 1j*u)*fm*Tba[l, b1, a1]*Tba[l1, a1, b]
                kern0[cb] = term

        kern0 = np.dot(kern1_inv, kern0)
        self.phi1k_delta[k, l, :] = kern0

    def func_2vN(self, Ek, l, eta, hfk):
        """
        Linearly interpolate the value of hfk on Ek_grid at point Ek.

        Parameters
        ----------
        Ek : float
            Energy value (not necessarily a grid point).
        l : int
            Lead label.
        eta : int
            Integer describing (+/-1) if infinitesimal eta is positive or negative.
        hfk : ndarray
            Array containing Hilbert transform of Fermi function (or 1-Fermi).

        Returns
        -------
        float
            Interpolated value of hfk at Ek.
        """
        Ek_grid = self.Ek_grid_ext
        if Ek < Ek_grid[0] or Ek > Ek_grid[-1]:
            return 0
        #
        b_idx = int((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0]))+1
        if b_idx == len(Ek_grid):
            b_idx -= 1
        a_idx = b_idx - 1
        b, a = Ek_grid[b_idx], Ek_grid[a_idx]
        #
        fb = hfk[l, b_idx]
        fa = hfk[l, a_idx]
        rez = (fb-fa)/(b-a)*Ek + (b*fa-a*fb)/(b-a)
        return pi*rez if eta+1 else pi*rez.conjugate()

    def get_at_k1(self, Ek, l, cb, conj):
        """
        Linearly interpolate the values of phi1k, hphi1k on Ek_grid at point Ek.

        Parameters
        ----------
        Ek : float
            Energy value (not necessarily a grid point).
        l : int
            Lead label.
        cb : int
            Index corresponding to Phi[1](k) matrix element.
        conj : bool
            If conj=True the term in the integral equation is conjugated.

        Returns
        -------
        ndarray, ndarray
            Interpolated values of phi1k and hphi1k at Ek.
        """
        Ek_grid = self.Ek_grid_ext
        phi1k = self.phi1k_delta_old
        hphi1k = self.hphi1k_delta

        if Ek < Ek_grid[0] or Ek > Ek_grid[-1]:
            return 0, 0
        #
        # b_idx = np.searchsorted(Ek_grid, Ek)
        b_idx = int((Ek-Ek_grid[0])/(Ek_grid[1]-Ek_grid[0]))+1
        if b_idx == len(Ek_grid):
            b_idx -= 1
        a_idx = b_idx - 1
        b, a = Ek_grid[b_idx], Ek_grid[a_idx]
        # print(a_idx, b_idx, a, Ek, b)
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
