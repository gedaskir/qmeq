"""Module containing python functions, which generate second order RTD kernels."""


class KernelHandler(object):
    """Class responsible for inserting matrix elements into the various matrices used."""
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
        """ Checks if the density matrix entry :math:`|b><bp|` is included in the calculations.

        Parameters
        ----------
        b : int
            first state
        bp : int
            second state
        bcharge : int
            charge of the states b and bp

        Returns
        -------
        bool
            true if it's included
        """
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        if bbp == -1:
            return False

        return True

    def is_unique(self, b, bp, bcharge):
        """ Check if the entry :math:`|b><bp|` is unique.

        Parameters
        ----------
        b  : int
            first state
        bp : int
            second state
        bcharge : int
            charge of the states b and bp

        Returns
        -------
        bool
            true if unique
        """
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
        """ Adds a complex value to the matrix element connecting :math:`|a><ap|` and :math:`|b><bp|` in the kernel.

        Parameters
        ----------
        fct : complex
            value to be added
        b : int
            first state of :math:`|b><bp|`
        bp : int
            second state of :math:`|b><bp|`
        bcharge : int
            charge of states b and bp
        a : int
            first state of :math:`|a><ap|`
        ap : int
            second state of :math:`|a><ap|`
        acharge : int
            charge of the states a and ap
        self.kern : ndarray
            (modifies) the kernel
        """
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
        """ Adds a real value (fctp) to the the matrix element connecting the states
        bb and aa in the Pauli kernel. In addition, adds another another real value (fctm)
        to the diagonal kern[bb, bb].

        Parameters
        ----------
        fctm : double
            value to be added to kern[bb, aa]
        fctp : double
            value to be added to kern[bb, bb]
        bb : int
            first state/index
        aa : int
            second state/index
        self.kern : ndarray
            (modifies) the kernel
        """
        self.kern[bb, bb] += fctm
        self.kern[bb, aa] += fctp

    def get_phi0_element(self, b, bp, bcharge):
        r""" Gets the entry of the density matrix given by :math:`|b><bp|`.

        Parameters
        ----------
        b : int
            first state
        bp : int
            second state
        bcharge : int
            charge of the states b and bp

        Returns
        -------
        complex
            the value :math:`<b|\phi_0|bp>`
        """
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
    """Class used for inserting matrix elements into vectors when using the matrix free
        solution method."""

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
        r""" Adds a contribution to :math:`d\phi_o /dt` that stems from the matrix element
        connecting :math:`|b><bp|` and :math:`|a><ap|` in the full off-diagonal in the kernel.

        Parameters
        ----------
        fct : complex
            value to be added
        b : int
            first state of :math:`|b><bp|`
        bp : int
            second state of :math:`|b><bp|`
        bcharge : int
            charge for the states b and bp
        a : int
            first state of :math:`|a><ap|`
        ap : int
            second state of :math:`|a><ap|`
        acharge : int
            charge of the states a and ap
        self.dphi0_dt : ndarray
            (modifies) time derivative of the density matrix
        """
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
        r""" Adds a contribution to :math:`d\phi_o /dt` that stems from the matrix element
        connecting :math:`|b><b|` and :math:`|a><a|` in the Pauli kernel.

        Parameters
        ----------
        fctm : double
            value from the diagonal of the kernel kern[bb, bb]
        fctp : double
            value from the off-diagonal of the kernel kern[bb, aa]
        b : int
            first state
        a : int
            second state
        self.dphi0_dt : ndarray
            (modifies) time derivative of the density matrix
        """
        self.dphi0_dt[bb] += fctm*self.phi0[bb] + fctp*self.phi0[aa]

    def get_phi0_norm(self):
        ncharge, statesdm = self.si.ncharge, self.si.statesdm

        norm = 0.0
        for bcharge in range(ncharge):
            for b in statesdm[bcharge]:
                bb = self.si.get_ind_dm0(b, b, bcharge)
                norm += self.phi0[bb]

        return norm

class KernelHandlerRTD(KernelHandler):
    """Class used for inserting matrix elements into the matrices used in the RTD approach."""

    def set_matrix_list(self):
        self.mats = [self.Wdd, self.WE1, self.WE2, self.ReWdn, self.ImWdn, self.ReWnd, self.ImWnd, self.Lnn]

    def add_matrix_element(self, fct, l, b, bp, bcharge, a, ap, acharge, mi):
        r"""
        Adds a value to the lead-resolved ndarray (kernel) given by index mi. The indices are set by the entries
        :math:`|b><bp|` and :math:`|a><ap|` in the density matrix. Which matrix to add the value to is determined by
        mi  as 0 = :math:`W_{dd}`, 1 = :math:`W_{E,1}`, 2 = :math:`W_{E,2}`, 3 = :math:`\Re(W_{dn}^{(1)})`,
        4 =   :math:`\Im (W_{dn}^{(1)})`, 5 =  :math:`\Re (W_{nd}^{(1)})`, 6 =  :math:`\Im(W_{nd}^{(1)})`,
        7 = :math:`L_{nn}`.

        Parameters
        ----------
        fct : float
           the value to be added
        l : int
           lead index
        b : int
            first index for state 1
        bp : int
            second index for state 1
        bcharge : int
            charge of state 1
        a : int
            first index for state 2
        ap : int
            second index for state 2
        acharge : int
            charge of state 2
        mi : int
            index for selecting which matrix to insert into
        self.mats[mi] : ndarray
            (Modifies) the matrix selected by mi
        """


        indx1 = self.si.get_ind_dm0(b, bp, bcharge)
        indx2 = self.si.get_ind_dm0(a, ap, acharge)
        if b != bp:
            indx1 -= self.npauli
            if b > bp:
                indx1 += self.ndm0 - self.npauli
        if a != ap:
            indx2 -= self.npauli
            if a > ap:
                indx2 += self.ndm0 - self.npauli

        mat = self.mats[mi]
        mat[l, indx1, indx2] += fct

    def set_matrix_element_dd(self, l, fctm, fctp, bb, aa, mi):
        """
        Adds a value to the lead-resolved kernel connecting :math:`|b><b|` to :math:`|a><a|`,
        and uses the conservation law to add a second value to the diagonal (connecting :math:`|b><b|`
        to itself).

        Parameters
        ----------
        l : int
            lead index
        fctm : float
            value to be added to the diagonal (tunneling out)
        fctp : float
            value to be added to the off-diagonal (tunneling in)
        bb : int
            index for the entry :math:`|b><b|`
        aa :  int
            index for the entry :math:`|a><a|`
        mi : int
            index for selecting which matrix to insert into
        self.mats[mi] : ndarray
            (Modifies) the matrix selected by mi
        """
        mat = self.mats[mi]
        mat[l, bb, bb] += fctm
        mat[l, bb, aa] += fctp

    def add_element_2nd_order(self, r, fct, indx0, indx1, a3, charge3, a4, charge4):
        """
        Adds a value to the lead-resolved kernel for the diagonal density matrix. Uses symmetries
        between second order diagrams in the RTD approach to add the value to four places in the matrix.


        Parameters
        ----------
        r : int
            lead index
        fct : float
            value to be added
        indx0 : int
            index for inital state
        indx1 : int
            index for intermidiate state 1
        a3 : int
            intermediate state 3 is given by :math:`|a3><a3|`
        charge3 : int
            charge of intermediate state 3
        a4 : int
            final state is given by :math:`|a4><a4|`
        charge4 : int
            charge of the final state
        self.Wdd : ndarray
            (Modifies) the lead-resolved kernel for the diagonal density matrix.

        """
        si = self.si
        indx3 = si.get_ind_dm0(a3, a3, charge3)
        indx4 = si.get_ind_dm0(a4, a4, charge4)

        fct = 2 * fct
        self.Wdd[r, indx4, indx0] += fct
        # Flipping left-most vertex p3 = -p3
        self.Wdd[r, indx3, indx0] += -fct
        # Flipping right-most vertex p0 = -p0
        self.Wdd[r, indx4, indx1] += fct
        # Flipping left-most and right-most vertices p0 = -p0 and p3 = -p3
        self.Wdd[r, indx3, indx1] += -fct


    def add_element_Lnn(self, a1, b1, charge, fct):
        """
        Adds a value to the part of :math:`L_{N,+}` connecting an off-diagonal component of the density matrix to
        itself.

        Parameters
        ----------
        a1 : int
            first part of the component :math:`|a1><b1|`
        b1 :  int
            second part of the component :math:`|a1><b1|`
        charge : int
            charge of the states a1 and b1
        fct : float
            the value to be added
        self.Lnn : ndarray
            (Modifies) the anti-commutator Liouvillian connecting non-diagonal elements
        """
        indx = self.si.get_ind_dm0(a1, b1, charge) - self.npauli
        if a1 > b1:
            indx += self.ndm0 - self.npauli
        self.Lnn[indx, indx] += fct
