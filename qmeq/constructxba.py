"""Module for defining tunneling amplitudes from quantum dot to leads."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from .indexing import szrange
from .indexing import ssqrange
from .indexing import sz_to_ind
from .indexing import ssq_to_ind

def construct_Xba(tleads, stateind, mtype=complex, Xba_=None):
    """
    Constructs many-body tunneling amplitude matrix Xba from single particle
    tunneling amplitudes.

    Parameters
    ----------
    tleads : dict
        Dictionary containing single particle tunneling amplitudes.
        tleads[(lead, state)] = tunneling amplitude.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of Xba matrix. For example, float, complex, etc.
    Xba_ : None or array
        nbaths by nmany by nmany numpy array containing old values of Xba.
        The values in tleads are added to Xba\_.

    Returns
    -------
    Xba : array
        nleads by nmany by nmany numpy array containing many-body tunneling amplitudes.
        The returned Xba corresponds to Fock basis.
    """
    if Xba_ is None:
        Xba = np.zeros((stateind.nleads, stateind.nmany, stateind.nmany), dtype=mtype)
    else:
        Xba = Xba_
    # Iterate over many-body states
    for j1 in range(stateind.nmany):
        state = stateind.get_state(j1)
        # Iterate over single particle states
        for j0 in tleads:
            (j3, j2), tamp = j0, tleads[j0]
            # Calculate fermion sign for added/removed electron in a given state
            fsign = np.power(-1, sum(state[0:j2]))
            if state[j2] == 0:
                statep = list(state)
                statep[j2] = 1
                ind = stateind.get_ind(statep)
                Xba[j3, ind, j1] += fsign*np.conj(tamp)
            else:
                statep = list(state)
                statep[j2] = 0
                ind = stateind.get_ind(statep)
                Xba[j3, ind, j1] += fsign*tamp
    return Xba

def construct_full_pmtr(vecslst, stateind, mtype=complex):
    """
    From definite charge eigenvectors constructs full eigenvectors
    defined by indexing in stateind.

    Parameters
    ----------
    veclst : list of arrays
        List of size ncharge containing arrays defining eigenvector matrices for given charge.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of pmtr matrix. For example, float, complex, etc.

    Returns
    -------
    pmtr : array
        nmany by nmany numpy array containing many-body eigenvectors.
    """
    pmtr = np.zeros((stateind.nmany, stateind.nmany), dtype=mtype)
    if stateind.indexing == 'sz':
        # Iterate over charges
        for charge in range(stateind.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, stateind.nsingle):
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                # Iterate over many-body states for given charge
                for j1, j2 in itertools.product(range(len(stateind.szlst[charge][szind])), range(len(stateind.szlst[charge][szind]))):
                    pmtr[stateind.szlst[charge][szind][j1], stateind.szlst[charge][szind][j2]] = vecslst[charge][szind][j1, j2]
    elif stateind.indexing == 'ssq':
        # Iterate over charges
        for charge in range(stateind.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, stateind.nsingle):
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                # Iterate over total spin ssq
                for ssq in ssqrange(charge, sz, stateind.nsingle):
                    ssqind = ssq_to_ind(ssq, sz)
                    # Iterate over many-body states for given charge
                    for j1, j2 in itertools.product(range(len(stateind.szlst[charge][szind])), range(len(stateind.ssqlst[charge][szind][ssqind]))):
                        pmtr[stateind.szlst[charge][szind][j1], stateind.ssqlst[charge][szind][ssqind][j2]] = vecslst[charge][szind][ssqind][j1, j2]
    else:
        # Iterate over charges
        for charge in range(stateind.ncharge):
            # Iterate over many-body states for given charge
            for j1, j2 in itertools.product(range(len(stateind.chargelst[charge])), range(len(stateind.chargelst[charge]))):
                pmtr[stateind.chargelst[charge][j1], stateind.chargelst[charge][j2]] = vecslst[charge][j1, j2]
    return pmtr

def rotate_Xba(Xba0, vecslst, stateind, indexing='n', mtype=complex):
    """
    Rotates tunneling amplitude matrix Xba0 in Fock basis to Xba,
    which is in eigenstate basis of the quantum dot.

    Parameters
    ----------
    Xba0 : array
        nleads by nmany by nmany numpy array, giving tunneling amplitudes in Fock basis.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    indexing : string
        Specifies what kind of rotation procedure to use. Default is stateind.indexing.
    mtype : type
        Defines type of Xba matrix. For example, float, complex, etc.

    Returns
    -------
    Xba : array
        nleads by nmany by nmany numpy array containing many-body tunneling amplitudes.
        The returned Xba corresponds to the quantum dot eigenbasis.
    """
    if indexing == 'n':
        indexingp = stateind.indexing
    else:
        indexingp = indexing
    Xba = np.zeros((stateind.nleads, stateind.nmany, stateind.nmany), dtype=mtype)
    if indexingp == 'Lin':
        pmtr = construct_full_pmtr(vecslst, stateind, mtype)
        for l in range(stateind.nleads):
            # Calculate many-body tunneling matrix Xba=P^(-1).Xba0.P
            # in eigenbasis of Hamiltonian from tunneling matrix Xba0 in Fock basis.
            # pmtr.conj().T denotes the conjugate transpose of pmtr.
            Xba[l] = np.dot(pmtr.conj().T, np.dot(Xba0[l], pmtr))
    elif indexingp == 'sz':
        for l, charge in itertools.product(range(stateind.nleads), range(stateind.ncharge-1)):
            szrng = szrange(charge, stateind.nsingle)
            # Lead labels from 0 to nleads//2 correspond to spin up
            # and nleads//2+1 to nleads-1 correspond to spin down
            if charge >= stateind.ncharge//2:
                szrng = szrng[0:-1] if l < stateind.nleads//2 else szrng[1:]
            # s=+1/-1 add/remove spin up/donw
            s = +1 if l < stateind.nleads//2 else -1
            for sz in szrng:
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                szind2 = sz_to_ind(sz+s, charge+1, stateind.nsingle)
                i1 = stateind.szlst[charge][szind][0]
                i2 = stateind.szlst[charge][szind][-1] + 1
                i3 = stateind.szlst[charge+1][szind2][0]
                i4 = stateind.szlst[charge+1][szind2][-1] + 1
                Xba[l, i1:i2][:, i3:i4] = np.dot(vecslst[charge][szind].conj().T, np.dot(Xba0[l, i1:i2][:, i3:i4], vecslst[charge+1][szind2]))
                Xba[l, i3:i4][:, i1:i2] = Xba[l, i1:i2][:, i3:i4].conj().T
    elif indexingp == 'ssq':
        for l, charge in itertools.product(range(stateind.nleads), range(stateind.ncharge-1)):
            szrng = szrange(charge, stateind.nsingle)
            # Lead labels from 0 to nleads//2 correspond to spin up
            # and nleads//2+1 to nleads-1 correspond to spin down
            if charge >= stateind.ncharge//2:
                szrng = szrng[0:-1] if l < stateind.nleads//2 else szrng[1:]
            # s=+1/-1 add/remove spin up/donw
            s = +1 if l < stateind.nleads//2 else -1
            for sz in szrng:
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                szind2 = sz_to_ind(sz+s, charge+1, stateind.nsingle)
                i1 = stateind.szlst[charge][szind][0]
                i2 = stateind.szlst[charge][szind][-1] + 1
                i3 = stateind.szlst[charge+1][szind2][0]
                i4 = stateind.szlst[charge+1][szind2][-1] + 1
                vecslst1 = np.concatenate(vecslst[charge][szind], axis=1)
                vecslst2 = np.concatenate(vecslst[charge+1][szind2], axis=1)
                Xba[l, i1:i2][:, i3:i4] = np.dot(vecslst1.conj().T, np.dot(Xba0[l, i1:i2][:, i3:i4], vecslst2))
                Xba[l, i3:i4][:, i1:i2] = Xba[l, i1:i2][:, i3:i4].conj().T
    elif indexingp == 'charge':
        for l, charge in itertools.product(range(stateind.nleads), range(stateind.ncharge-1)):
            i1 = stateind.chargelst[charge][0]
            i2 = stateind.chargelst[charge][-1] + 1
            i3 = stateind.chargelst[charge+1][0]
            i4 = stateind.chargelst[charge+1][-1] + 1
            Xba[l, i1:i2][:, i3:i4] = np.dot(vecslst[charge].conj().T, np.dot(Xba0[l, i1:i2][:, i3:i4], vecslst[charge+1]))
            Xba[l, i3:i4][:, i1:i2] = Xba[l, i1:i2][:, i3:i4].conj().T
    return Xba
#---------------------------------------------------------------------------------------------------

def make_tleads_mtr(tleads, nleads, nsingle, mtype=complex):
    """
    Makes single particle tunneling matrix from a list or a dictionary.

    Parameters
    ----------
    tleads : list or dictionary
        Contains single particle tunneling amplitudes
    nleads : int
        Number of leads.
    nsingle : int
        Number of single particle states.
    mtype : type
        Defines type of tleads matrix. For example, float, complex, etc.

    Returns
    -------
    tleads_mtr : array
        nleads by nsingle numpy array containing single particle tunneling amplitudes.
    """
    tleads_mtr = np.zeros((nleads, nsingle), dtype=mtype)
    htype = type(tleads).__name__
    for j0 in tleads:
        if htype == 'list':    j1, j2, tamp = j0
        elif htype == 'dict': (j1, j2), tamp = j0, tleads[j0]
        tleads_mtr[j1, j2] += tamp
    return tleads_mtr

def make_tleads_dict(tleads):
    """
    Makes single particle tunneling amplitude dictionary.

    Parameters
    ----------
    tleads : list, dict, or array
        Contains single particle tunneling amplitudes.

    Returns
    -------
    tleads_dict : dictionary
        Dictionary containing tunneling amplitudes.
        tleads[(lead, state)] gives the tunneling amplitude.
    """
    htype = type(tleads).__name__
    if htype == 'list':
        tleads_dict = {}
        for j0 in tleads:
            j1, j2, tamp = j0
            tleads_dict.update({(j1, j2):tamp})
        return tleads_dict
    elif htype == 'ndarray':
        nleads, nsingle = tleads.shape
        tleads_dict = {}
        for j1 in range(nleads):
            for j2 in range(nsingle):
                if tleads[j1, j2] != 0:
                    tleads_dict.update({(j1, j2):tleads[j1, j2]})
        return tleads_dict
    elif htype == 'dict':
        return tleads
#---------------------------------------------------------------------------------------------------

class LeadsTunneling(object):
    """
    Class for defining tunneling amplitude from leads to the quantum dot.

    Attributes
    ----------
    nleads : int
        Number of the leads.
    tleads : list, dict, or array
        list, dictionary or numpy array defining single particle tunneling amplitudes.
        numpy array has to be nleads by nsingle.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mulst : list
        List containing chemical potentials of the leads.
    tlst : list
        List containing temperatures of the leads.
    dlst : list
        List containing bandwidths of the leads.
    mtype : type
        Defines type of Xba0 and Xba matrices. For example, float, complex, etc.
    Xba0 : array
        nleads by nmany by nmany array, which contains many-body tunneling amplitude matrix in Fock basis.
    Xba : list
        nleads by nmany by nmany array, which contains many-body tunneling amplitude matrix,
        which is used in calculations.
    """

    def __init__(self, nleads, tleads, stateind, mulst=0, tlst=0, dlst=0, mtype=complex):
        """Initialization of the LeadsTunneling class."""
        self.tleads = make_tleads_dict(tleads)
        self.stateind = stateind
        self.stateind.nleads = nleads
        self.mulst = np.array(mulst)
        self.tlst = np.array(tlst)
        self.dlst = np.array(dlst)
        self.mtype = mtype
        self.Xba0 = construct_Xba(self.tleads, stateind, mtype)
        self.Xba = self.Xba0

    def add(self, tleads={}, updateq=True):
        """
        Adds a value to single particle tunneling amplitudes and correspondingly redefines
        many-body tunneling matrix Xba.

        Parameters
        ----------
        tleads : dict
            Dictionary describing what values to add.
            For example, tleads[(lead, state)] = value to add.
        updateq : bool
            Specifies if the values of the single particle amplitudes will be updated.
            The many-body tunneling amplitudes Xba will be updates in either case.
        """
        self.Xba0 = construct_Xba(tleads, self.stateind, self.mtype, self.Xba0)
        if updateq:
            for j0 in tleads:
                try:    self.tleads[j0] += tleads[j0]       # if tleads[j0] != 0:
                except: self.tleads.update({j0:tleads[j0]}) # if tleads[j0] != 0:

    def change(self, tleads={}, updateq=True):
        """
        Changes the values of the single particle tunneling amplitudes and correspondingly redefines
        many-body tunneling matrix Xba.

        Parameters
        ----------
        tleads : dict
            Dictionary describing which tunneling amplitudes to change.
            For example, tleads[(lead, state)] = the new value.
        updateq : bool
            Specifies if the values of the single particle amplitudes will be updated.
            The many-body tunneling amplitudes Xba will be updates in either case.
        """
        tleadsp = tleads if type(tleads).__name__ == 'dict' else make_tleads_dict(tleads)
        # Find the differences from the previous tunneling amplitudes
        tleads_add = {}
        for j0 in tleadsp:
            try:
                tleads_diff = tleadsp[j0]-self.tleads[j0]
                if tleads_diff != 0:
                    tleads_add.update({j0:tleads_diff})
                    if updateq: self.tleads[j0] += tleads_diff
            except:
                tleads_diff = tleadsp[j0]
                if tleads_diff != 0:
                    tleads_add.update({j0:tleads_diff})
                    if updateq: self.tleads.update({j0:tleads_diff})
        # Add the differences
        self.add(tleads_add, False)

    def rotate(self, vecslst, indexing='n'):
        """
        Rotates tunneling amplitude matrix Xba0 in Fock basis to Xba,
        which is in eigenstate basis of the quantum dot.

        Parameters
        ----------
        veclst : list of arrays
            List of size ncharge containing arrays defining eigenvector matrices for given charge.
        indexing : string
            Specifies what kind of rotation procedure to use. Default is stateind.indexing.
        """
        self.Xba = rotate_Xba(self.Xba0, vecslst, self.stateind, indexing, self.mtype)

    def use_Xba0(self):
        """
        Sets the Xba matrix for calculation to Xba0 in the Fock basis.
        """
        self.Xba = self.Xba0

    def update_Xba0(self, nleads, tleads, mtype=complex):
        """
        Updates the Xba0 in the Fock basis using new single-particle tunneling amplitudes.

        Parameters
        ----------
        tleads : array
            The new single-particle tunneling amplitudes. See attribute tleads.
        """
        self.stateind.nleads = nleads
        self.tleads = make_tleads_dict(tleads)
        self.mtype = mtype
        self.Xba0 = construct_Xba(tleads, self.stateind, mtype)
