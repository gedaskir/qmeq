"""Module containing various functions."""

import numpy as np

from ..indexing import szrange
from ..indexing import ssqrange
from ..indexing import sz_to_ind
from ..indexing import ssq_to_ind


def get_charge(self, b):
    """
    Get charge of the many-body state b

    Parameters
    ----------
    self : Builder or Approach.
        The system given as Builder or Approach object.
    b : int
        Label of the many-body state.

    Returns
    -------
    int
        Charge of the many-body state b.
    """
    return sum(self.si.get_state(b))


def multiarray_sort(arr, srt=[0]):
    """
    Sort rows of a two-dimensional array for a given
    hierarchy of rows.

    Parameters
    ----------
    arr : ndarray
        A two-dimensional numpy array.
    srt : list
        List specifying in which order of rows to sort.

    Returns
    -------
    ndarray
        A sorted array.
    """
    ind = np.lexsort([arr[i] for i in reversed(srt)])
    return (arr.T[ind]).T


def sort_eigenstates(self, srt=None):
    """
    Sort many-body states of the system by given order of properties.

    Parameters
    ----------
    self : Builder or Approach.
        The system given as Approach, Approach2vN, or Builder object.
    srt : list
        List specifying in which order of properties to sort.
        For example, in the case of 'ssq' indexing we have  such convention:
        0 - energy
        1 - charge
        2 - spin projection :math:`S_{z}`
        3 - total spin :math:`S^{2}`
        The default sorting order for 'ssq' indexing is srt=[1, 2, 3, 0]

    self.qd.Ea_ext : ndarray
        (Modifies) A two-dimensional numpy array containing in the zeroth row energies,
        first row charge, and etc.
    self.si.states_order : ndarray
        (Modifies) A numpy row containing a new order of many-body states.
    """
    if srt is not None:
        if self.qd.Ea_ext is None:
            self.qd.Ea_ext = construct_Ea_extended(self)
        else:
            self.qd.Ea_ext[0] = self.qd.Ea
        srt.append(-1)
        self.si.states_order = np.array(multiarray_sort(self.qd.Ea_ext, srt)[-1], dtype=int)
    else:
        self.si.states_order = range(self.si.nmany)


def get_phi0(self, b_, bp_):
    """
    Get the reduced density matrix element corresponding to
    many-body states b and bp.

    Parameters
    ----------
    self : Builder or Approach
        The system given as Builder or Approach object.
    b_,bp_ : int
        Labels of the many-body states.

    Returns
    --------
    phi0bbp : complex
        A matrix element of the reduced density matrix (complex number).
    """
    b = self.si.states_order[b_]
    bp = self.si.states_order[bp_]
    bcharge = sum(self.si.get_state(b))
    bpcharge = sum(self.si.get_state(bp))
    phi0bbp = 0.0
    if self.funcp.kerntype == 'Pauli':
        if b == bp:
            ind = self.si.get_ind_dm0(b, b, bcharge, maptype=1)
            phi0bbp = self.phi0[ind]
    elif bcharge == bpcharge:
        ind = self.si.get_ind_dm0(b, bp, bcharge, maptype=1)
        conj = self.si.get_ind_dm0(b, bp, bcharge, maptype=3)
        if ind != -1:
            if type(self.si).__name__ == 'StateIndexingDMc':
                phi0bbp = self.phi0[ind]
            else:
                ndm0, npauli = self.si.ndm0, self.si.npauli
                phi0bbp = (self.phi0[ind] + 1j*self.phi0[ndm0-npauli+ind]
                           * (+1 if conj else -1)
                           * (0 if ind < npauli else 1))
    return phi0bbp


def get_phi1(self, l, c_, b_):
    """
    Get the energy integrated current amplitudes corresponding to
    lead l and many-body states c and b.

    Parameters
    ----------
    self : Builder or Approach
        The system given as Builder or Approach object.
    l : int
        Label of the lead channel.
    c_,b_ : int
        Labels of the many-body states.

    Returns
    --------
    phi0bbp : complex
        A matrix element of the reduced density matrix (complex number).
    """
    if self.funcp.kerntype == 'Pauli':
        return None
    else:
        c = self.si.states_order[c_]
        b = self.si.states_order[b_]
        ccharge = sum(self.si.get_state(c))
        bcharge = sum(self.si.get_state(b))
        phi1cb = 0.0
        if ccharge == bcharge+1:
            ind = self.si.get_ind_dm1(c, b, bcharge)
            phi1cb = self.phi1[l, ind]
        elif ccharge+1 == bcharge:
            ind = self.si.get_ind_dm1(b, c, ccharge)
            phi1cb = self.phi1[l, ind].conjugate()
        return phi1cb


def construct_Ea_extended(self):
    """
    Constructs an array containing properties of the many-body states,
    like energy, charge, spin-projection :math:`S_{z}`, etc.


    Parameters
    ----------
    self : Builder or Approach
        The system given as Builder or Approach object.

    Returns
    --------
    Ea_ext : ndarray
        | A two-dimensional numpy array containing in the zeroth row energies,
          first row charge, and etc. We have such convention:
          Ea_ext[property number, state number]
        |
        | For example, for 'charge' indexing we have such properties:
        | 0 - energy
        | 1 - charge
        | 2 - state number
        |
        | and for 'ssq' indexing we have:
        | 0 - energy
        | 1 - charge
        | 2 - spin projection :math:`S_{z}`
        | 3 - total spin :math:`S^{2}`
        | 4 - state number
    """
    (si, Ea) = (self.si, self.qd.Ea)
    if si.indexing == 'sz':
        Ea_ext = np.zeros((4, len(Ea)), dtype=float)
        Ea_ext[0] = self.qd.Ea
        Ea_ext[3] = np.arange(si.nmany)
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, si.nsingle):
                # Iterate over many-body states for given charge and sz
                szind = sz_to_ind(sz, charge, si.nsingle)
                for ind in range(len(si.szlst[charge][szind])):
                    # The mapping of many-body states is according to szlst
                    sn = si.szlst[charge][szind][ind]
                    Ea_ext[1, sn] = charge
                    Ea_ext[2, sn] = sz
    elif si.indexing == 'ssq':
        Ea_ext = np.zeros((5, len(Ea)), dtype=float)
        Ea_ext[0] = self.qd.Ea
        Ea_ext[4] = np.arange(si.nmany)
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, si.nsingle):
                szind = sz_to_ind(sz, charge, si.nsingle)
                # Iterate over total spin ssq
                for ssq in ssqrange(charge, sz, si.nsingle):
                    ssqind = ssq_to_ind(ssq, sz)
                    # Iterate over many-body states for given charge, sz, and ssq
                    for ind in range(len(si.ssqlst[charge][szind][ssqind])):
                        # The mapping of many-body states is according to ssqlst
                        sn = si.ssqlst[charge][szind][ssqind][ind]
                        Ea_ext[1, sn] = charge
                        Ea_ext[2, sn] = sz
                        Ea_ext[3, sn] = ssq
    else:
        # Iterate over charges
        Ea_ext = np.zeros((3, len(Ea)), dtype=float)
        Ea_ext[0] = self.qd.Ea
        Ea_ext[2] = np.arange(si.nmany)
        for charge in range(si.ncharge):
            # Iterate over many-body states for given charge
            for ind in range(len(si.chargelst[charge])):
                # The mapping of many-body states is according to chargelst
                sn = si.chargelst[charge][ind]
                Ea_ext[1, sn] = charge
    return Ea_ext


def remove_coherences(self, dE):
    """
    Remove the coherences with energy difference larger than dE.

    Parameters
    ----------
    self : Builder or Approach
        The system given as Builder or Approach.
    dE : float
        Energy difference.

    self.si.mapdm0 : list
        (Modifies) List showing which density matrix elements are mapped to each other due to symmetries
        and which density matrix elements are neglected (entries with values -1).
    """
    # Find which coherences to remove
    si, E = self.si, self.qd.Ea
    ilst, indlst = [], []
    for i in range(si.ndm0_):
        if si.mapdm0[i] != -1:
            ind = si.mapdm0[i]
            b, bp = si.inddm0[ind]
            if abs(E[b]-E[bp]) > dE:
                si.mapdm0[i] = -1
            else:
                ilst.append(i)
                indlst.append(ind)
    # Count the new number of density matrix elements
    srt = multiarray_sort(np.array([ilst, indlst]), [1, 0])
    ilst, indlst = srt[0], srt[1]
    ind, count = 0, 0
    for i in range(len(indlst)):
        if indlst[i] == ind:
            indlst[i] = count
        else:
            count += 1
            ind = indlst[i]
            indlst[i] = count
    # Relabel the density matrix elements
    si.ndm0 = count+1
    si.ndm0r = 2*si.ndm0-si.npauli
    for i in range(len(indlst)):
        si.mapdm0[ilst[i]] = indlst[i]


def remove_states(self, dE):
    """
    Remove the states with energy dE larger than the ground state
    for the transport calculations.

    Parameters
    ----------
    self : Builder or Approach
        The system given as Builder or Approach object.
    dE : float
        Energy above the ground state.

    self.si.statesdm : list
        (Modifies) List containing indices of many-body state under consideration.
    """
    Emax = min(self.qd.Ea)+dE
    statesdm = [[] for _ in range(self.si.nsingle+1)]
    for charge in range(self.si.ncharge):
        for b in self.si.chargelst[charge]:
            if self.qd.Ea[b] < Emax:
                statesdm[charge].append(b)
    self.si.set_statesdm(statesdm)


def use_all_states(self):
    """
    Use all states for the transport calculations.

    Parameters
    ----------
    self : Builder or Approach
        The system given as Builder or Approach.

    self.si.statesdm : list
        (Modifies) List containing indices of many-body state under consideration.
    """
    self.si.set_statesdm(self.si.chargelst)


def print_state(self, b_, eps=0.0, prntq=True, filename=None, separator=''):
    """
    Prints properties of given many-body eigenstate of the quantum dot Hamiltonian

    Parameters
    ----------
    self : Builder or Approach
        The system given as Builder or Approach object.
    b_ : int
        Index of the many-body eigenstate.
    eps : float
        Value at which expansion coefficients of the many-body eigenstate in Fock basis
        are neglected.
    prntq : bool
        If true then eigenstate properties are printed into command line.
    filename : string
        File to which to print eigenstate properties. The output is appended to this file.
    separator : string
        String added at the beginning of the output.
    """
    si = self.si
    b = si.states_order[b_]
    Eb = self.qd.Ea[b]
    sout = ('Energy: ' + str(Eb)+'\n' +
            'Eigenstate coefficients:')
    #
    if si.indexing == 'sz':
        charge, sz, alpha = si.ind_qn[b]
        sout = 'S^z [in units of 1/2]: ' + str(sz) + '\n' + sout
        szind = sz_to_ind(sz, charge, si.nsingle)
        ind = si.szlst[charge][szind].index(b)
        coeffs = np.array([abs(self.qd.vecslst[charge][szind][:, ind]),
                           range(len(si.szlst[charge][szind]))])
        coeffs = np.array(multiarray_sort(coeffs)[1], dtype=int)
        # for j1 in range(len(si.szlst[charge][szind])):
        for j1 in reversed(coeffs):
            sn = si.szlst[charge][szind][j1]
            val = self.qd.vecslst[charge][szind][j1, ind]
            if abs(val) >= eps:
                sout = sout + '\n'+'    |'+si.get_state(sn, strq=True)+'>: ' + str(val)
    elif si.indexing == 'ssq':
        charge, sz, ssq, alpha = si.ind_qn[b]
        sout = ('S^z [in units of 1/2]: ' + str(sz) + '\n' +
                'S^2 [in units of 1/2]: ' + str(ssq) + '\n' + sout)
        szind = sz_to_ind(sz, charge, si.nsingle)
        ssqind = ssq_to_ind(ssq, sz)
        ind = si.ssqlst[charge][szind][ssqind].index(b)
        coeffs = np.array([abs(self.qd.vecslst[charge][szind][ssqind][:, ind]),
                           range(len(si.szlst[charge][szind]))])
        coeffs = np.array(multiarray_sort(coeffs)[1], dtype=int)
        # for j1 in range(len(si.szlst[charge][szind])):
        for j1 in reversed(coeffs):
            sn = si.szlst[charge][szind][j1]
            val = self.qd.vecslst[charge][szind][ssqind][j1, ind]
            if abs(val) >= eps:
                sout = sout + '\n'+'    |'+si.get_state(sn, strq=True)+'>: ' + str(val)
    else:
        charge, alpha = si.ind_qn[b]
        # ind = si.dictdm[b]
        ind = si.chargelst[charge].index(b)
        coeffs = np.array([abs(self.qd.vecslst[charge][:, ind]),
                           range(len(si.chargelst[charge]))])
        coeffs = np.array(multiarray_sort(coeffs)[1], dtype=int)
        # for j1 in range(len(si.chargelst[charge])):
        for j1 in reversed(coeffs):
            sn = si.chargelst[charge][j1]
            val = self.qd.vecslst[charge][j1, ind]
            if abs(val) >= eps:
                sout = sout + '\n'+'    |'+si.get_state(sn, strq=True)+'>: ' + str(val)
    #
    sout = (separator +
            'Sorted label: ' + str(b_) + '\n' +
            'Original label: ' + str(b) + '\n' +
            'Charge: ' + str(charge) + '\n' + sout)
    #
    if filename is not None:
        with open(filename, 'a') as f:
            f.writelines(sout+'\n')
    if prntq:
        print(sout)
    pass


def print_all_states(self, filename, eps=0.0, separator='', mode='w'):
    """
    Prints properties of all many-body eigenstates to a file.

    Parameters
    ----------
    self : Builder or Approach
        The system given as Builder or Approach object.
    filename : string
        File to which to print eigenstate properties. The output is appended to this file.
    eps : float
        Value at which expansion coefficients of the many-body eigenstate in Fock basis
        are neglected.
    separator : string
        String added at the beginning of the output.
    mode : string
        Mode in which the file will be treated. For example, to append use 'a' or
        to write into clea file use 'w'.
    """
    with open(filename, mode) as f:
        f.close()
    for ind in range(self.si.nmany):
        print_state(self, ind, eps, False, filename, separator)
    pass
