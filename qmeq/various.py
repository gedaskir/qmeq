from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from .indexing import szrange
from .indexing import ssqrange
from .indexing import sz_to_ind
from .indexing import ssq_to_ind

def get_charge(sys, b):
    '''
    Get charge of the many-body state b

    Parameters
    ----------
    sys : Transport, Transport2vN
        The system given as Transport or Transport2vN object.
    b : int
        Label of the many-body state.

    Returns
    -------
    int
        Charge of the many-body state b.
    '''
    return sum(sys.si.get_state(b))

def multiarray_sort(arr, srt=[0]):
    '''
    Sort rows of a two-dimensional array for a given
    hierarchy of rows.

    Parameters
    ----------
    arr : array
        A two-dimensional numpy array.
    srt : list
        List specifying in which order of rows to sort.

    Returns
    -------
    array
        A sorted array.
    '''
    ind = np.lexsort([arr[i] for i in reversed(srt)])
    return (arr.T[ind]).T

def sort_eigenstates(sys, srt='n'):
    '''
    Sort many-body states of the system by given order of properties.

    Parameters
    ----------
    sys : Transport, Transport2vN
        The system given as Transport or Transport2vN object.
    srt : list
        List specifying in which order of properties to sort.
        For example, in the case of 'ssq' indexing we have  such convention:
        0 - energy
        1 - charge
        2 - spin projection S^z
        3 - total spin S^2
        The default sorting order for 'ssq' indexing is srt=[1, 2, 3, 0]

    Modifies
    --------
    sys.qd.Ea_ext : array
        A two-dimensional numpy array containing in the zeroth row energies,
        first row charge, and etc.
    sys.si.states_order : array
        A numpy row containing a new order of many-body states.
    '''
    if srt != 'n':
        if sys.qd.Ea_ext is None:
            sys.qd.Ea_ext = construct_Ea_extended(sys)
        else:
            sys.qd.Ea_ext[0] = sys.qd.Ea
        srt.append(-1)
        sys.si.states_order = np.array( multiarray_sort(sys.qd.Ea_ext, srt)[-1], dtype=int)
    else:
        sys.si.states_order = range(sys.si.nmany)

def get_phi0(sys, b_, bp_):
    '''
    Get the reduced density matrix element corresponding to
    mnay-body states b and bp.

    Parameters
    ----------
    sys : Transport, Transport2vN
        The system given as Transport or Transport2vN object.
    b_, bp_ : int
        Labels of the many-body states.
        For example, in the case of 'ssq' indexing we have  such convention:
        0 - energy
        1 - charge
        2 - spin projection S^z
        3 - total spin S^2
        The default sorting order for 'ssq' indexing is srt=[1, 2, 3, 0]

    Returns
    --------
    phi0bbp : complex
        A matrix element of the reduced density matrix (complex number).
    '''
    b = sys.si.states_order[b_]
    bp = sys.si.states_order[bp_]
    bcharge = sum(sys.si.get_state(b))
    bpcharge = sum(sys.si.get_state(bp))
    phi0bbp = 0.0
    if bcharge == bpcharge:
        ind = sys.si.get_ind_dm0(b, bp, bcharge, maptype=1)
        conj = sys.si.get_ind_dm0(b, bp, bcharge, maptype=3)
        if ind != -1:
            ndm0, npauli = sys.si.ndm0, sys.si.npauli
            phi0bbp = ( sys.phi0[ind] + 1j*sys.phi0[ndm0-npauli+ind]
                                          * (+1 if conj else -1)
                                          * (0 if ind < npauli else 1) )
    return phi0bbp

def construct_Ea_extended(sys):
    '''
    Constructs an array containing properties of the many-body states,
    like energy, charge, spin-projection S^{z}, etc.


    Parameters
    ----------
    sys : Transport, Transport2vN
        The system given as Transport or Transport2vN object.

    Returns
    --------
    Ea_ext : array
        A two-dimensional numpy array containing in the zeroth row energies,
        first row charge, and etc. We have such convention:
        Ea_ext[property number, state number]

        For example, for 'charge' indexing we have such properties:
        0 - energy
        1 - charge
        2 - state number

        and for 'ssq' indexing we have:
        0 - energy
        1 - charge
        2 - spin projection S^{z}
        3 - total spin S^{2}
        4 - state number
    '''
    (si, Ea) = (sys.si, sys.qd.Ea)
    if si.indexing == 'sz':
        Ea_ext = np.zeros((4, len(Ea)), dtype=float)
        Ea_ext[0] = sys.qd.Ea
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
                    Ea_ext[1,sn] = charge
                    Ea_ext[2,sn] = sz
    elif si.indexing == 'ssq':
        Ea_ext = np.zeros((5, len(Ea)), dtype=float)
        Ea_ext[0] = sys.qd.Ea
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
                        Ea_ext[1,sn] = charge
                        Ea_ext[2,sn] = sz
                        Ea_ext[3,sn] = ssq
    else:
        # Iterate over charges
        Ea_ext = np.zeros((3, len(Ea)), dtype=float)
        Ea_ext[0] = sys.qd.Ea
        Ea_ext[2] = np.arange(si.nmany)
        for charge in range(si.ncharge):
            # Iterate over many-body states for given charge
            for ind in range(len(si.chargelst[charge])):
                # The mapping of many-body states is according to chargelst
                sn = si.chargelst[charge][ind]
                Ea_ext[1,sn] = charge
    return Ea_ext

def remove_states(sys, dE):
    '''
    Remove the states with energy dE larger than the ground state
    for the transport calculations.

    Parameters
    ----------
    sys : Transport, Transport2vN
        The system given as Transport or Transport2vN object.
    dE : float
        Energy above the ground state.

    Modifies
    --------
    sys.si.statesdm : list
        List containing indices of many-body state under consideration.
    '''
    Emax = min(sys.qd.Ea)+dE
    statesdm = [[] for i in range(sys.si.nsingle+1)]
    for charge in range(sys.si.ncharge):
        for b in sys.si.chargelst[charge]:
            if sys.qd.Ea[b] < Emax:
                statesdm[charge].append(b)
    sys.si.set_statesdm(statesdm)

def use_all_states(sys):
    '''
    Use all states for the transport calculations.

    Parameters
    ----------
    sys : Transport, Transport2vN
        The system given as Transport or Transport2vN object.

    Modifies
    --------
    sys.si.statesdm : list
        List containing indices of many-body state under consideration.
    '''
    sys.si.set_statesdm(sys.si.chargelst)

def print_state(sys, b_, eps=0.0, prntq=True, filename=None, separator=''):
    '''
    Print properties of given many-body eigenstate of the quantum dot Hamiltonain

    Parameters
    ----------
    sys : Transport, Transport2vN
        The system given as Transport or Transport2vN object.
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
        String added at the begining of the output.
    '''
    si = sys.si
    b = si.states_order[b_]
    Eb = sys.qd.Ea[b]
    sout = ('Energy: '+ str(Eb)+'\n'
           +'Eigenstate coefficients:')
    #
    if si.indexing == 'sz':
        charge, sz, alpha = si.ind_qn[b]
        sout = 'S^z [in units of 1/2]: '+ str(sz) + '\n' + sout
        szind = sz_to_ind(sz, charge, si.nsingle)
        ind = si.szlst[charge][szind].index(b)
        for j1 in range(len(si.szlst[charge][szind])):
            sn = si.szlst[charge][szind][j1]
            val = sys.qd.vecslst[charge][szind][j1, ind]
            if abs(val) >= eps:
                sout = sout + '\n'+'    |'+si.get_state(sn, strq=True)+'>: ' + str(val)
    elif si.indexing == 'ssq':
        charge, sz, ssq, alpha = si.ind_qn[b]
        sout = ('S^z [in units of 1/2]: ' + str(sz) + '\n'
               +'S^2 [in units of 1/2]: ' + str(ssq) + '\n' + sout)
        szind = sz_to_ind(sz, charge, si.nsingle)
        ssqind = ssq_to_ind(ssq, sz)
        ind = si.ssqlst[charge][szind][ssqind].index(b)
        for j1 in range(len(si.szlst[charge][szind])):
            sn = si.szlst[charge][szind][j1]
            val = sys.qd.vecslst[charge][szind][ssqind][j1, ind]
            if abs(val) >= eps:
                sout = sout + '\n'+'    |'+si.get_state(sn, strq=True)+'>: ' + str(val)
    else:
        charge, alpha = si.ind_qn[b]
        #ind = si.dictdm[b]
        ind = si.chargelst[charge].index(b)
        for j1 in range(len(si.chargelst[charge])):
            sn = si.chargelst[charge][j1]
            val = sys.qd.vecslst[charge][j1, ind]
            if abs(val) >= eps:
                sout = sout + '\n'+'    |'+si.get_state(sn, strq=True)+'>: ' + str(val)
    #
    sout = (separator
           +'Sorted label: ' + str(b_) + '\n'
           +'Original label: ' + str(b) + '\n'
           +'Charge: ' + str(charge) + '\n' + sout)
    #
    if not filename is None:
        with open(filename, 'a') as f:
            f.writelines(sout+'\n')
    if prntq:
        print(sout)
    pass

def print_all_states(sys, filename, eps=0.0, separator='', mode='w'):
    '''
    Prints properties of all many-body eigenstates to a file.

    Parameters
    ----------
    sys : Transport, Transport2vN
        The system given as Transport or Transport2vN object.
    filename : string
        File to which to print eigenstate properties. The output is appended to this file.
    eps : float
        Value at which expansion coefficients of the many-body eigenstate in Fock basis
        are neglected.
    separator : string
        String added at the begining of the output.
    mode : string
        Mode in which the file will be treated. For example, to append use 'a' or
        to write into clea file use 'w'.
    '''
    with open(filename, mode) as f:
        f.close()
    for ind in range(sys.si.nmany):
        print_state(sys, ind, eps, False, filename, separator)
    pass
