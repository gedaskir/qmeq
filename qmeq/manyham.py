"""Module for constructing many-body quantum dot Hamiltonian and diagonalising it."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from .indexing import empty_szlst
from .indexing import empty_ssqlst
from .indexing import szrange
from .indexing import ssqrange
from .indexing import sz_to_ind
from .indexing import ssq_to_ind

def construct_ham_coulomb(coulomb, statelst, stateind, mtype=float, hamq=False, ham_=None):
    """
    Constructs many-body Hamiltonian for given Coulomb matrix elements.

    Parameters
    ----------
    coulomb : dict
        Dictionary containing coulomb matrix elements. The dictionary is of the format
        coulomb[(m, n, k, l)] = U, where m, n, k, l are the state labels
        and U is the strength of the coulomb interaction between these states.
        Note that only the matrix elements k>l, n>m have to be specified.
    statelst : list
        List of indices of states under consideration.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    hamq : bool
        If true the old values of the Hamiltonian are not used.
    ham_ : None or array
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.
        The values in coulomb are added to ham_coulomb.
    mtype : type
        Defines type of ham_coulomb matrix. For example, float, complex, etc.

    Returns
    -------
    ham_coulomb : array
        Matrix representation of many-body Coulomb Hamiltonian in Fock basis for states statelst.
    """
    nstates = len(statelst)
    if hamq and not (ham_ is None):
        ham_coulomb = ham_
    else:
        ham_coulomb = np.zeros((nstates, nstates), dtype=mtype)
    # Iterate over many-body states
    for j1 in range(nstates):
        state = stateind.get_state(statelst[j1])
        # Iterate over Coulomb matrix elements
        for j2 in coulomb:
            # The suggested convetion is: k>l, n>m
            (m, n, k, l), U = j2, coulomb[j2]
            if state[k] == 1 and state[l] == 1 and k != l:
                # Calculate fermion sign due to two removed electrons in a given state
                # Note that if k is larger than l additional sign appears for flipping k with l
                fsign = np.power(-1, sum(state[0:k])+sum(state[0:l])) * (-1 if k > l else +1)
                statep = list(state)
                statep[k] = 0
                statep[l] = 0
                if statep[m] == 0 and statep[n] == 0 and m != n:
                    # Calculate fermion sign due to two added electrons in a given state
                    # Note that if n is smaller than m additional sign appears for flipping m with n
                    fsign = fsign*np.power(-1, sum(statep[0:n])+sum(statep[0:m])) * (+1 if n > m else -1)
                    statep[m] = 1
                    statep[n] = 1
                    ind = statelst.index(stateind.get_ind(statep))
                    ham_coulomb[ind, j1] += U*fsign
    return ham_coulomb

def construct_ham_hopping(hsingle, statelst, stateind, mtype=float, hamq=False, ham_=None):
    """
    Constructs many-body Hamiltonian for given single-particle hamiltonian.

    Parameters
    ----------
    hsingle : dict
        Dictionary corresponding to single-particle hopping (tunneling) Hamiltonian.
        The dictionary is of the format hsingle[(i, j)] = hij, where i, j are the state labels
        and hij is the matrix element of the single particle Hamiltonian.
    statelst : list
        List of indices of states under consideration.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    hamq : bool
        If true the old values of the Hamiltonian are not used.
    ham_ : None or array
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.
        The values in hsingle are added to ham.
    mtype : type
        Defines type of ham_coulomb matrix. For example, float, complex, etc.

    Returns
    -------
    ham_coulomb : array
        Matrix representation of many-body Hamiltonian in Fock basis for states statelst.
    """
    nstates = len(statelst)
    if hamq and not (ham_ is None):
        ham_hopping = ham_
    else:
        ham_hopping = np.zeros((nstates, nstates), dtype=mtype)
    # Iterate over many-body states
    for j1 in range(nstates):
        state = stateind.get_state(statelst[j1])
        # Iterate over single particle Hamiltonian
        for j0 in hsingle:
            (j2, j3), hop = j0, hsingle[j0]
            if j2 == j3:
                if state[j2] == 1:
                    ham_hopping[j1, j1] += hop
            # Remove particle from j2 single particle state, add particle in j3 single particle state
            elif state[j2] == 1 and state[j3] == 0:
                # Calculate fermion sign for added/removed electrons in a given state
                # Note that if j3 is larger than j2 additional sign appears for flipping j3 with j2
                fsign = np.power(-1, sum(state[0:j2])+sum(state[0:j3])) * (+1 if j2 > j3 else -1)
                statep = list(state)
                statep[j2] = 0
                statep[j3] = 1
                ind = statelst.index(stateind.get_ind(statep))
                ham_hopping[ind, j1] += hop*fsign
                ham_hopping[j1, ind] += hop.conjugate()*fsign
    return ham_hopping

def construct_manybody_eigenstates(hsingle, coulomb, statelst, stateind, mtype=float, hamq=False, ham_=None):
    """
    Calculates eigenstates of many-body Hamiltonian (described by hsingle and coulomb).

    Parameters
    ----------
    hsingle : array
        nsingle by nsingle array corresponding to single-particle hopping (tunneling) Hamiltonian.
    coulomb : list of lists
        List containing coulomb matrix elements.
        coulomb[i] is list of format [m, n, k, l, U], where m, n, k, l are state labels
        and U is strength of coulomb interaction between these states.
        Note that only the matrix elements k>l, n>m have to be specified.
    statelst : list
        List of indices of states under consideration.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of ham_coulomb matrix. For example, float, complex, etc.
    hamq : bool
        If true the old values of the Hamiltonian are not used.
    ham_ : None or array
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.

    Returns
    -------
    ham_vals : array
        len(statelst) by 1 array containing eigenvalues of Hamiltonian.
    ham_vecs : array
        len(statelst) by len(statelst) array containing eigenvector matrix of Hamiltonian.
        Columns of ham_vecs correspond to particular eigenvectors.
    """
    if hamq and not (ham_ is None):
        ham = ham_
    else:
        ham_coulomb = construct_ham_coulomb(coulomb, statelst, stateind, mtype)
        ham_hopping = construct_ham_hopping(hsingle, statelst, stateind, mtype)
        ham = ham_coulomb+ham_hopping
    ham_vals, ham_vecs = np.linalg.eigh(ham)
    return ham_vals, ham_vecs

def construct_Ea_manybody(valslst, stateind):
    """
    Makes a single array corresponding to a given indexing and containing eigenvalues of Hamiltonian.

    Parameters
    ----------
    valslst : list of arrays
        List containing eigenvalues. List entry valslst[charge] is an array of definite charge eigenvalues.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.

    Returns
    -------
    Ea : array
        nmany by 1 array containing eigenvalues of the Hamiltonian.
    """
    Ea = np.zeros(stateind.nmany, dtype=float)
    if stateind.indexing == 'sz':
        # Iterate over charges
        for charge in range(stateind.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, stateind.nsingle):
                # Iterate over many-body states for given charge and sz
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                for ind in range(len(stateind.szlst[charge][szind])):
                    # The mapping of many-body states is according to szlst
                    Ea[stateind.szlst[charge][szind][ind]] = valslst[charge][szind][ind]
    elif stateind.indexing == 'ssq':
        # Iterate over charges
        for charge in range(stateind.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, stateind.nsingle):
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                # Iterate over total spin ssq
                for ssq in ssqrange(charge, sz, stateind.nsingle):
                    ssqind = ssq_to_ind(ssq, sz)
                    # Iterate over many-body states for given charge, sz, and ssq
                    for ind in range(len(stateind.ssqlst[charge][szind][ssqind])):
                        # The mapping of many-body states is according to ssqlst
                        Ea[stateind.ssqlst[charge][szind][ssqind][ind]] = valslst[charge][szind][ssqind][ind]
    else:
        # Iterate over charges
        for charge in range(stateind.ncharge):
            # Iterate over many-body states for given charge
            for ind in range(len(stateind.chargelst[charge])):
                # The mapping of many-body states is according to chargelst
                Ea[stateind.chargelst[charge][ind]] = valslst[charge][ind]
    return Ea

#---------------------------------------------------------------------------------------------------
# manyhamssq.py
def operator_sm(charge, sz, stateind):
    """
    Construct the operator :math:`S^{-}`, which acts on the states with given :math:`S^{z}`.

    Parameters
    ----------
    charge : int
        States with given charge (number of electrons) are considered.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.

    Returns
    -------
    sm : array
        Matrix (numpy array) describing the specific :math:`S^{-}` operator.
        The operator is expressed in the Fock basis.
    """
    # The minimum possible value of sz for given number of
    # single particle states nsingle and number of particle charge
    szmin = -min(charge, stateind.nsingle-charge)
    # Acting with S^{-} on the states with smallest possible sz gives 0
    if sz-2 < szmin: return 0
    states = stateind.get_lst(charge, sz)
    statesm = stateind.get_lst(charge, sz-2)
    sm = np.zeros((len(statesm), len(states)), dtype=int)
    # States up to index dind have spin up
    # and the states from dind to nsingle have spin down
    dind = int(stateind.nsingle/2)
    for j1 in range(len(states)):
        state = stateind.get_state(states[j1])
        for j2 in range(int(stateind.nsingle/2)):
            if state[j2] == 1 and state[dind+j2] == 0:
                fsign = np.power(-1, sum(state[0:j2])+sum(state[0:dind+j2])+1)
                state2 = list(state)
                # Flip the spin in single particle state from up to down
                state2[j2] = 0
                state2[dind+j2] = 1
                ind = statesm.index(stateind.get_ind(state2))
                sm[ind, j1] = 2*fsign
    return sm

def operator_sp(charge, sz, stateind):
    """
    Construct the operator :math:`S^{+}`, which acts on the states with given sz.

    Parameters
    ----------
    charge : int
        States with given charge (number of electrons) are considered.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.

    Returns
    -------
    sp : array
        Matrix (numpy array) describing the specific :math:`S^{+}` operator.
        The operator is expressed in the Fock basis.
    """
    # The maximum possible value of sz for given number of
    # single particle states nsingle and number of particle charge
    szmax = min(charge, stateind.nsingle-charge)
    # Acting with S^{+} on the states with largest possible sz gives 0
    if sz+2 > szmax: return 0
    states = stateind.get_lst(charge, sz)
    statesp = stateind.get_lst(charge, sz+2)
    sp = np.zeros((len(statesp), len(states)), dtype=int)
    # States up to index dind have spin up
    # and the states from dind to nsingle have spin down
    dind = int(stateind.nsingle/2)
    for j1 in range(len(states)):
        state = stateind.get_state(states[j1])
        for j2 in range(int(stateind.nsingle/2)):
            if state[j2] == 0 and state[dind+j2] == 1:
                fsign = np.power(-1, sum(state[0:j2])+sum(state[0:dind+j2]))
                state2 = list(state)
                # Flip the spin in single particle state from down to up
                state2[j2] = 1
                state2[dind+j2] = 0
                ind = statesp.index(stateind.get_ind(state2))
                sp[ind, j1] = 2*fsign
    return sp

def operator_ssquare(charge, sz, stateind):
    """
    Construct the operator :math:`S^{2}`, which acts on the states with given sz.
    :math:`S^{2} = (S^{+}S^{-}+S^{-}S^{+})/2+S^{z}S^{z}`

    Parameters
    ----------
    charge : int
        States with given charge (number of electrons) are considered.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.

    Returns
    -------
    ssquare : array
        Matrix (numpy array) describing the specific :math:`S^{+}` operator.
        The operator is expressed in the Fock basis.
    """
    szmin = -min(charge, stateind.nsingle-charge)
    szmax = min(charge, stateind.nsingle-charge)
    sp_sm = np.dot(operator_sp(charge, sz-2, stateind), operator_sm(charge, sz, stateind)) if sz-2 >= szmin else 0
    sm_sp = np.dot(operator_sm(charge, sz+2, stateind), operator_sp(charge, sz, stateind)) if sz+2 <= szmax else 0
    sz_square = np.eye(len(stateind.get_lst(charge, sz)), dtype=int)*sz**2
    ssquare = (sp_sm+sm_sp)//2+sz_square
    return ssquare

def ssquare_eigenstates(charge, sz, stateind, prnt=False):
    """
    Find the eigenstates of the operator :math:`S^{2}` for given sz.

    Parameters
    ----------
    charge : int
        States with given charge (number of electrons) are considered.
    sz : int
        Value of sz.
        sz odd corresponds to fractional spin. For example, 1->1/2, 3->3/2, etc.
        sz even corresponds to integer spin. For example, 0->0, 2->1, etc.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    prnt : bool
        If true, then print the eigenstates.

    Returns
    -------
    ssq_eigvec : list of arrays
        List of numpy arrays containing the eingenstates of :math:`S^{2}`.
        ssq_eigvec[ssqind] gives the eigenstate matrix for :math:`S^{2}` =ssq.
    """
    if charge%2 != sz%2:
        print("WARNING: charge and sz need to have the same parity. Return 0.")
        return 0
    ssq_eigvec = [0 for j in ssqrange(charge, sz, stateind.nsingle)]
    ssquare = operator_ssquare(charge, sz, stateind)
    eigval, eigvec = np.linalg.eigh(ssquare)
    eigval = np.rint(-1 + np.sqrt(1+eigval))
    for ssq in ssqrange(charge, sz, stateind.nsingle):
        v = np.nonzero(eigval == ssq)
        ssqind = ssq_to_ind(ssq, sz)
        ssq_eigvec[ssqind] = eigvec[:][:, v[0][0]:v[0][-1]+1]
        if prnt:
            print('-'*30)
            print('charge =', charge, ', sz =', sz, ', ssq =', ssq)
            print(ssq_eigvec[ssqind])
    return ssq_eigvec

def ssquare_all_szlow(stateind, prnt=False):
    """
    Find the eigenstates of the operator :math:`S^{2}` for sz=0 or 1.

    Parameters
    ----------
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    prnt : bool
        If true, then print the eigenstates.

    Returns
    -------
    ssq_eigvec : list of arrays
        List of numpy arrays containing the eingenstates of :math:`S^{2}` for lowest abs(sz) value possible.
        ssq_eigvec[charge][ssq] gives the eigenstate matrix for given charge and :math:`S^{2}` =ssq.
    """
    ssq_eigvec_all_szlow = [0 for i in range(stateind.ncharge)]
    for charge in range(stateind.ncharge):
        ssq_eigvec_all_szlow[charge] = ssquare_eigenstates(charge, -(charge%2), stateind)
    if prnt:
        for charge in range(stateind.ncharge):
            for ssq in ssqrange(charge, -(charge%2), stateind.nsingle):
                print('-'*30)
                print('charge =', charge, ', sz =', -(charge%2), ', ssq =', ssq)
                print(ssq_eigvec_all_szlow[charge][ssq//2])
    return ssq_eigvec_all_szlow

def construct_manybody_eigenstates_ssq(charge, sz, ssq, hsingle, coulomb, stateind, mtype=float, hamq=False, ham_=None):
    """
    Find the many-body eigenstates of the Hamiltonian, which are also the eigenstates of sz and :math:`S^{2}`
    for given charge, sz and :math:`S^{2}`.

    Parameters
    ----------
    charge : int
        Charge (number of particles) of the states to consider.
    sz : int
        sz value of the states to consider.
    ssq : int
        :math:`S^{2}` value of the states to consider.
    hsingle : dict
        Dictionary corresponding to single-particle hopping (tunneling) Hamiltonian.
        The dictionary is of the format hsingle[(i, j)] = hij, where i, j are the state labels
        and hij is the matrix element of the single particle Hamiltonian.
    coulomb : dict
        Dictionary containing coulomb matrix elements. The dictionary is of the format
        coulomb[(m, n, k, l)] = U, where m, n, k, l are the state labels
        and U is the strength of the coulomb interaction between these states.
        Note that only the matrix elements k>l, n>m have to be specified.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of ham matrix. For example, float, complex, etc.
    hamq : bool
        If true the old values of the Hamiltonian are not used.
    ham_ : None or array
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.
        The values in hsingle and coulomb are added to ham.

    Returns
    -------
    eigvalp : array
        Eigenvalues of the Hamltonian for given charge, sz, :math:`S^{2}`.
    eigvecssq : array
        Corresponding eigenvectors of the Hamiltonian.
    """
    statelst = stateind.get_lst(charge=charge, sz=sz)
    if hamq and not (ham_ is None):
        ham = ham_
    else:
        ham = (construct_ham_coulomb(coulomb, statelst, stateind, mtype)
              +construct_ham_hopping(hsingle, statelst, stateind, mtype))
    ssq_eigvec = ssquare_eigenstates(charge, sz, stateind)
    ssqind = ssq_to_ind(ssq, sz)
    # Write the Hamiltonian in the S^{2} basis for given value of S^{2}=ssq
    hamp = np.dot(ssq_eigvec[ssqind].T, np.dot(ham,  ssq_eigvec[ssqind]))
    eigvalp, eigvecp = np.linalg.eigh(hamp)
    eigvecssq = np.dot(ssq_eigvec[ssqind], eigvecp)
    return eigvalp, eigvecssq

def construct_manybody_eigenstates_ssq_all(charge, hsingle, coulomb, stateind, mtype=float, hamq=False, ham_=None):
    """
    Find the many-body eigenstates of the Hamiltonian for given charge.
    The eigenstates are also the eigenstates of sz and :math:`S^{2}`.

    Parameters
    ----------
    charge : int
        Charge (number of particles) of the states to consider.
    hsingle : dict
        Dictionary corresponding to single-particle hopping (tunneling) Hamiltonian.
        The dictionary is of the format hsingle[(i, j)] = hij, where i, j are the state labels
        and hij is the matrix element of the single particle Hamiltonian.
    coulomb : dict
        Dictionary containing coulomb matrix elements. The dictionary is of the format
        coulomb[(m, n, k, l)] = U, where m, n, k, l are the state labels
        and U is the strength of the coulomb interaction between these states.
        Note that only the matrix elements k>l, n>m have to be specified.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of ham matrix. For example, float, complex, etc.
    hamq : bool
        If true the old values of the Hamiltonian are not used.
    ham_ : None or array
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.
        The values in hsingle and coulomb are added to ham.

    Returns
    -------
    valslst : list of arrays
        Eigenvalues of the Hamltonian for given charge.
        valslst[szind][ssqind] are the eigenvalues for given sz and :math:`S^{2}`.
    vecslst : list of arrays
        Eigenvectors corresponding to valslst.
    """
    szlow = -(charge%2)
    nsingle = stateind.nsingle
    #valslst = [[None] for k in ssqrange(charge, szlow, nsingle)]
    valslst = [[[None] for k in ssqrange(charge, j, nsingle)] for j in szrange(charge, nsingle)]
    vecslst = [[[None] for k in ssqrange(charge, j, nsingle)] for j in szrange(charge, nsingle)]
    #
    statelst = stateind.get_lst(charge=charge, sz=szlow)
    if hamq and not (ham_ is None):
        ham = ham_
    else:
        ham = (construct_ham_coulomb(coulomb, statelst, stateind, mtype)
              +construct_ham_hopping(hsingle, statelst, stateind, mtype))
    #NOTE: Can be an argument of the function instead
    ssq_eigvec = ssquare_eigenstates(charge, szlow, stateind)
    #
    # Find eigenstates corresponding to sz=1 or 0
    szind = sz_to_ind(szlow, charge, nsingle)
    for ssq in ssqrange(charge, szlow, nsingle):
        ssqind = ssq_to_ind(ssq, szlow)
        hamp = np.dot(ssq_eigvec[ssqind].T, np.dot(ham,  ssq_eigvec[ssqind]))
        eigvalp, eigvecp = np.linalg.eigh(hamp)
        eigvecssq = np.dot(ssq_eigvec[ssqind], eigvecp)
        valslst[szind][ssqind] = eigvalp
        vecslst[szind][ssqind] = eigvecssq
    # Construct the eigenstates with larger (positive) sz by acting with S^{+}
    szmax = min(charge, stateind.nsingle-charge)
    for sz in range(szlow, szmax-1, 2):
        szind = sz_to_ind(sz, charge, nsingle)
        szind2 = sz_to_ind(sz+2, charge, nsingle)
        for ssq in ssqrange(charge, sz+2, nsingle):
            ssqind = ssq_to_ind(ssq, sz)
            ssqind2 = ssq_to_ind(ssq, sz+2)
            sp = operator_sp(charge, sz, stateind)
            fct = np.sqrt(ssq*(ssq+2) - sz*(sz+2))
            vecslst[szind2][ssqind2] = np.dot(sp, vecslst[szind][ssqind])/fct
            valslst[szind2][ssqind2] = valslst[szind][ssqind]
    # Construct the eigenstates with smaller (negative) sz by acting with S^{-}
    for sz in range(szlow, -szmax+1, -2):
        szind = sz_to_ind(sz, charge, nsingle)
        szind2 = sz_to_ind(sz-2, charge, nsingle)
        for ssq in ssqrange(charge, sz-2, nsingle):
            ssqind = ssq_to_ind(ssq, sz)
            ssqind2 = ssq_to_ind(ssq, sz-2)
            sm = operator_sm(charge, sz, stateind)
            fct = np.sqrt(ssq*(ssq+2) - sz*(sz-2))
            vecslst[szind2][ssqind2] = np.dot(sm, vecslst[szind][ssqind])/fct
            valslst[szind2][ssqind2] = valslst[szind][ssqind]
    return valslst, vecslst
#---------------------------------------------------------------------------------------------------

def make_hsingle_mtr(hsingle, nsingle, mtype=float):
    """
    Makes single particle Hamiltonian matrix from a list or a dictionary.

    Parameters
    ----------
    hsingle : list or dictionary
        Contains single particle tunneling amplitudes
    nsingle : int
        Number of single particle states.
    mtype : type
        Defines type of tleads matrix. For example, float, complex, etc.

    Returns
    -------
    hsingle_mtr : array
        hsingle by hsingle numpy array containing single particle Hamiltonian.
    """
    hsingle_mtr = np.zeros((nsingle, nsingle), dtype=mtype)
    htype = type(hsingle).__name__
    for j0 in hsingle:
        if htype == 'list':    j1, j2, hop = j0
        elif htype == 'dict': (j1, j2), hop = j0, hsingle[j0]
        hsingle_mtr[j1, j2] += hop
        if j1 != j2:
            hsingle_mtr[j2, j1] += hop.conjugate()
    return hsingle_mtr

def make_hsingle_dict(hsingle):
    """
    Makes single particle Hamiltonian dictionary.

    Parameters
    ----------
    hsingle : list, dict, or array
        Contains single particle Hamiltonian.

    Returns
    -------
    hsingle_dict : dictionary
        Dictionary containing single particle Hamiltonian.
        hsingle[(state1, state2)] gives the matrix element.
    """
    htype = type(hsingle).__name__
    if htype == 'list':
        hsingle_dict = {}
        for j0 in hsingle:
            j1, j2, hop = j0
            hsingle_dict.update({(j1, j2):hop})
        return hsingle_dict
    elif htype == 'ndarray':
        nsingle = hsingle.shape[0]
        hsingle_dict = {}
        for j1 in range(nsingle):
            for j2 in range(j1, nsingle):
                if hsingle[j1, j2] != 0:
                    hsingle_dict.update({(j1, j2):hsingle[j1, j2]})
        return hsingle_dict
    elif htype == 'dict':
        return hsingle

def make_coulomb_dict(coulomb):
    """
    Makes Coulomb matrix element dictionary.

    Parameters
    ----------
    coulomb : list, dict, or array
        Contains coulomb matrix elements.

    Returns
    -------
    coulomb_dict : dictionary
        Dictionary containing coulomb matrix element.
        coulomb[(state1, state2, state3, state4)] gives the coulomb matrix element U.
    """
    htype = type(coulomb).__name__
    if htype == 'list' or htype == 'ndarray':
        coulomb_dict = {}
        for j0 in coulomb:
            m, n, k, l, U = j0
            coulomb_dict.update({(m, n, k, l):U})
        return coulomb_dict
    elif htype == 'dict':
        return coulomb
#---------------------------------------------------------------------------------------------------

class QuantumDot(object):
    """
    Class for constructing and diagonalising many-body Hamiltonian describing the quantum dot.

    Attributes
    ----------
    hsingle : list, dict, or array
        List, dictionary, or array corresponding to single-particle hopping (tunneling) Hamiltonian.
        On input list or array gets converted to dictionary.
    coulomb : list, dict, or array
        List, dictionary, or containing coulomb matrix elements.
        For dictionary:    coulomb[(m, n, k, l)] = U, where m, n, k, l are the state labels.
        For list or array: coulomb[i] is list of the format [m, n, k, l, U].
        U is the strength of the coulomb interaction between the states (m, n, k, l).
        Note that only the matrix elements k>l, n>m have to be specified.
        On input list or array gets converted to dictionary.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines the type of the Hamiltonian matrices. For example, float, complex, etc.
    hamq : bool
        The procedures add() and change() make changes to many-body Hamiltonian only if hamq is True.
    ham_ssq_q : bool
        If ham_ssq_q is False then for 'ssq' indexing only Hamiltonian with lowest abs(sz) value is stored.
    valslst : list of arrays
        List containing ncharge entries, where valslst[charge] is an array having Hamiltonian eigenvalues for definite charge.
    vecslst : list of arrays
        List containing ncharge entries, where vecslst[charge] is an array corresponding to Hamiltonian eigenvectors for definite charge.
    hamlst : list of arrays
        List containing ncharge entries, where hamlst[charge] is an array corresponding to many-body Hamiltonian for definite charge.
    Ea : array
        nmany by 1 array containing many-body Hamiltonian eigenvalues.
    """

    def __init__(self, hsingle, coulomb, stateind,
                 mtype=float, hamq=True, ham_ssq_q=False):
        """Initialization of the QuantumDot class."""
        self.hsingle = make_hsingle_dict(hsingle)
        self.coulomb = make_coulomb_dict(coulomb)
        self.stateind = stateind
        self.mtype = mtype
        self.hamq, self.ham_ssq_q = hamq, ham_ssq_q
        if stateind.indexing == 'sz':
            self.valslst = empty_szlst(stateind.nsingle, True)
            self.vecslst = empty_szlst(stateind.nsingle, True)
            self.hamlst = empty_szlst(stateind.nsingle, True)
        elif stateind.indexing == 'ssq':
            self.valslst = empty_ssqlst(stateind.nsingle, True)
            self.vecslst = empty_ssqlst(stateind.nsingle, True)
            self.hamlst = empty_szlst(stateind.nsingle, True)
        else:
            self.valslst = [None]*stateind.ncharge
            self.vecslst = [None]*stateind.ncharge
            self.hamlst = [None]*stateind.ncharge
        if hamq: self.add(self.hsingle, self.coulomb, False)
        self.Ea = np.zeros(stateind.nmany, dtype=float)
        self.Ea_ext = None

    def add(self, hsingle={}, coulomb={}, updateq=True):
        """
        Adds a value to single particle Hamiltonian and Coulomb matrix elements
        and correspondingly redefines many-body Hamiltonian.

        Parameters
        ----------
        hsingle : dict
            Dictionary describing what single-particle Hamiltonian values to add.
            For example, hsingle[(state1, state2)] = value to add.
        coulomb : dict
            Dictionary describing what Coulomb matrix element values to add.
            For example, coulomb[(state1, state2, state3, state4)] = value to add.
        updateq : bool
            Specifies if the values of the single particle Hamiltonian
            and the Coulomb matrix elements will be updated.
            The many-body Hamiltonian will be updates in either case.
        """
        if self.hamq:
            if self.stateind.indexing == 'sz':
                for charge in range(self.stateind.ncharge):
                    for sz in szrange(charge, self.stateind.nsingle):
                        szlst = self.stateind.get_lst(charge, sz)
                        szind = sz_to_ind(sz, charge, self.stateind.nsingle)
                        self.hamlst[charge][szind] = construct_ham_coulomb(coulomb, szlst, self.stateind, self.mtype, self.hamq, self.hamlst[charge][szind])
                        self.hamlst[charge][szind] = construct_ham_hopping(hsingle, szlst, self.stateind, self.mtype, self.hamq, self.hamlst[charge][szind])
            elif self.stateind.indexing == 'ssq':
                for charge in range(self.stateind.ncharge):
                    szlow = -(charge%2)
                    szrng = szrange(charge, self.stateind.nsingle) if self.ham_ssq_q else [szlow]
                    for sz in szrng:
                        szlst = self.stateind.get_lst(charge, sz)
                        szind = sz_to_ind(sz, charge, self.stateind.nsingle)
                        self.hamlst[charge][szind] = construct_ham_coulomb(coulomb, szlst, self.stateind, self.mtype, self.hamq, self.hamlst[charge][szind])
                        self.hamlst[charge][szind] = construct_ham_hopping(hsingle, szlst, self.stateind, self.mtype, self.hamq, self.hamlst[charge][szind])
            else:
                for charge in range(self.stateind.ncharge):
                    chargelst = self.stateind.get_lst(charge)
                    self.hamlst[charge] = construct_ham_coulomb(coulomb, chargelst, self.stateind, self.mtype, self.hamq, self.hamlst[charge])
                    self.hamlst[charge] = construct_ham_hopping(hsingle, chargelst, self.stateind, self.mtype, self.hamq, self.hamlst[charge])
            if updateq:
                for j0 in hsingle:
                    try:    self.hsingle[j0] += hsingle[j0]       # if hsingle[j0] != 0:
                    except: self.hsingle.update({j0:hsingle[j0]}) # if hsingle[j0] != 0:
                for j0 in coulomb:
                    try:    self.coulomb[j0] += coulomb[j0]       # if coulomb_diff != 0:
                    except: self.hsingle.update({j0:coulomb[j0]}) # if coulomb_diff != 0:

    def change(self, hsingle={}, coulomb={}, updateq=True):
        """
        Changes the values of the single particle Hamiltonian and Coulomb matrix elements
        and correspondingly redefines many-body Hamiltonian.

        Parameters
        ----------
        hsingle : dict
            Dictionary describing what single-particle Hamiltonian values to add.
            For example, hsingle[(state1, state2)] = value to add.
        coulomb : dict
            Dictionary describing what Coulomb matrix element values to add.
            For example, coulomb[(state1, state2, state3, state4)] = value to add.
        updateq : bool
            Specifies if the values of the single particle Hamiltonian
            and the Coulomb matrix elements will be updated.
            The many-body Hamiltonian will be updates in either case.
        """
        if self.hamq:
            hsinglep = hsingle if type(hsingle).__name__ == 'dict' else make_hsingle_dict(hsingle)
            coulombp = coulomb if type(coulomb).__name__ == 'dict' else make_coulomb_dict(coulomb)
            #
            hsingle_add = {}
            coulomb_add = {}
            # Find the differences from the previous single-particle Hamiltonian
            # and from the previous Coulomb matrix elements
            for j0 in hsinglep:
                try:
                    hsingle_diff = hsinglep[j0]-self.hsingle[j0]
                    if hsingle_diff != 0:
                        hsingle_add.update({j0:hsingle_diff})
                        if updateq: self.hsingle[j0] += hsingle_diff
                except:
                    hsingle_diff = hsinglep[j0]
                    if hsingle_diff != 0:
                        hsingle_add.update({j0:hsingle_diff})
                        if updateq: self.hsingle.update({j0:hsingle_diff})
            for j0 in coulombp:
                try:
                    coulomb_diff = coulombp[j0]-self.coulomb[j0]
                    if coulomb_diff != 0:
                        coulomb_add.update({j0:coulomb_diff})
                        if updateq: self.coulomb[j0] += coulomb_diff
                except:
                    coulomb_diff = coulombp[j0]
                    if coulomb_diff != 0:
                        coulomb_add.update({j0:coulomb_diff})
                        if updateq: self.coulomb.update({j0:coulomb_diff})
            # Then add the differences
            self.add(hsingle_add, coulomb_add, False)

    def diagonalise_charge(self, charge='n', sz='n', ssq='n'):
        """Diagonalises Hamiltonian for given charge.
           Additionally, spin projection 'sz' and toltal spin 'ssq' can be specified
           if those symmetries exist for considered model.

        Parameters
        ----------
        charge : int
            Charge under consideration.
        sz : int
            Spin projection.
        ssq : int
            Total spin.
        """
        if charge == 'n':
            return None
        elif sz == 'n':
            chargelst = self.stateind.get_lst(charge)
            self.valslst[charge], self.vecslst[charge] = construct_manybody_eigenstates(self.hsingle, self.coulomb, chargelst,
                                                                                       self.stateind, self.mtype,
                                                                                       self.hamq, self.hamlst[charge])
            return self.valslst[charge], self.vecslst[charge]
        elif ssq == 'n' and (self.stateind.indexing == 'sz' or self.stateind.indexing == 'ssq'):
            szlst = self.stateind.get_lst(charge, sz)
            szind = sz_to_ind(sz, charge, self.stateind.nsingle)
            self.valslst[charge][szind], self.vecslst[charge][szind] = construct_manybody_eigenstates(self.hsingle, self.coulomb, szlst,
                                                                                                     self.stateind, self.mtype,
                                                                                                     self.hamq, self.hamlst[charge][szind])
            return self.valslst[charge][szind], self.vecslst[charge][szind]
        elif self.stateind.indexing == 'ssq':
            szind = sz_to_ind(sz, charge, self.stateind.nsingle)
            ssqind = ssq_to_ind(ssq, sz)
            self.valslst[charge][szind][ssqind], self.vecslst[charge][szind][ssqind] = construct_manybody_eigenstates_ssq(charge, sz, ssq,
                                                                                                                         self.hsingle, self.coulomb,
                                                                                                                         self.stateind, self.mtype,
                                                                                                                         self.hamq, self.hamlst[charge][szind])
            return self.valslst[charge][szind][ssqind], self.vecslst[charge][szind][ssqind]
        else:
            print("WARNING: No indexing by 'sz' or 'ssq'. Return None.")
            return None

    def diagonalise(self):
        """Diagonalises Hamiltonians for all charge states."""
        if self.stateind.indexing == 'sz':
            for charge in range(self.stateind.ncharge):
                for sz in szrange(charge, self.stateind.nsingle):
                    self.diagonalise_charge(charge, sz)
        elif self.stateind.indexing == 'ssq':
            for charge in range(self.stateind.ncharge):
                szlow = -(charge%2)
                szind = sz_to_ind(szlow, charge, self.stateind.nsingle)
                self.valslst[charge], self.vecslst[charge] = construct_manybody_eigenstates_ssq_all(charge, self.hsingle, self.coulomb,
                                                                                                   self.stateind, self.mtype,
                                                                                                   self.hamq, self.hamlst[charge][szind])
        else:
            for charge in range(self.stateind.ncharge):
                self.diagonalise_charge(charge)
        self.set_Ea()

    def set_Ea(self):
        """Sets the many-body eigenstates using construct_Ea_manybody()."""
        self.Ea = construct_Ea_manybody(self.valslst, self.stateind)
