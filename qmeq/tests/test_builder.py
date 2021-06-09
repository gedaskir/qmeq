import numpy as np
from numpy.linalg import norm
from qmeq.builder import *
import qmeq
import itertools

EPS = 1e-11
CHECK_PY = False
PRNTQ = False
SERIAL_TRIPLE_DOT = False


class ParametersDoubleDotSpinful(object):

    def __init__(self):
        tL, tR = 2.0, 1.0
        self.tleads = {(0,0): tL, (2,2): tL,
                       (1,1): tR, (3,3): tR,
                       (0,1): 0.3*tL, (2,3): 0.3*tL,
                       (1,0): 0.1*tR, (3,2): 0.1*tR}
        #
        e1, e2, omega = -10, -12, 20
        self.hsingle = {(0,0):e1, (2,2):e1,
                        (1,1):e2, (3,3):e2,
                        (0,1):omega, (2,3):omega}
        uintra, uinter = 80, 30
        self.coulomb = {(0,2,2,0):uintra, (1,3,3,1):uintra,
                        (0,1,1,0):uinter, (0,3,3,0):uinter,
                        (1,2,2,1):uinter, (2,3,3,2):uinter}
        #
        vbias, temp, dband = 5.0, 50.0, 1000.0
        self.mulst = [vbias/2, -vbias/2, vbias/2, -vbias/2]
        self.tlst = [temp, temp, temp, temp]
        self.dlst = [dband, dband, dband, dband]
        self.nsingle, self.nleads = 4, 4


class ParametersDoubleDotSpinless(object):

    def __init__(self):
        tL, tR = 2.0, 1.0
        self.tleads = {(0,0): tL, (1,1): tR}
        #
        e1, e2, omega = -10, -12, 20
        self.hsingle = {(0,0):e1, (1,1):e2, (0,1):omega}
        uinter = 30.
        self.coulomb = {(0,1,1,0):uinter}
        #
        vbias, temp, dband = 5.0, 25.0, 1000.0
        self.mulst = [vbias/2, -vbias/2]
        self.tlst = [temp, temp]
        self.dlst = [dband, dband]
        #
        self.nsingle, self.nleads = 2, 2
        self.kpnt = np.power(2, 9)


class ParametersSingleOrbitalSpinful(object):

    def __init__(self):
        tL, tR = 2.0, 1.0
        self.tleads = {(0, 0):tL, (1, 0):tR,
                       (2, 1):tL, (3, 1):tR}
        #
        vgate = 0.0
        self.hsingle = {(0, 0): vgate,
                        (1, 1): vgate}
        #
        U = 300.0
        self.coulomb = {(0,1,1,0):U}
        #
        vbias, temp, dband = 5.0, 50.0, 1000.0
        self.mulst = [vbias/2, -vbias/2, vbias/2, -vbias/2]
        self.tlst = [temp, temp, temp, temp]
        self.dlst = [dband, dband, dband, dband]
        #
        self.nsingle, self.nleads = 2, 4
        self.kpnt = np.power(2, 9)

class ParametersDoubleOrbitalSpinful(object):

    def __init__(self):
        tL, tR = np.sqrt(1 / (2 * np.pi)), np.sqrt(1 / (2 * np.pi))
        self.tleads = {(0, 0): tL, (1, 0): tR, (2, 1): tL, (3, 1): tR,
                  (0, 2): tL, (1, 2): tR, (2, 3): tL, (3, 3): tR}
        #
        U = 100000
        Up = U / 8
        delta = U / 4
        self.coulomb = {(0, 1, 1, 0): U, (2, 3, 3, 2): U, (0, 2, 2, 0): Up, (0, 3, 3, 0): Up,
                   (1, 2, 2, 1): Up, (1, 3, 3, 1): Up}
        #
        vgate = 6500
        e0, e1, e2, e3 = -delta / 2, delta / 2, 2 * U - delta / 2, 2 * U + delta / 2
        self.hsingle = {(0, 0): e0 - vgate, (1, 1): e1 - vgate, (2, 2): e2 - vgate, (3, 3): e3 - vgate}
        #
        vbias = 10
        temp = 450
        dband = 1e6 * temp
        self.mulst = [-vbias/2, vbias/2, -vbias/2, vbias/2]
        self.tlst = [temp, temp, temp, temp]
        self.dlst = [dband, dband, dband, dband]
        #
        self.nsingle, self.nleads = 4, 4


class Calcs(object):
    def __init__(self):
        pass


def save_Builder_double_dot_spinful(fname='data_builder.py'):
    p = ParametersDoubleDotSpinful()
    # data = {}
    kerns = ['Pauli', 'Redfield', '1vN', 'Lindblad', 'pyPauli', 'pyRedfield', 'py1vN', 'pyLindblad']
    # kerns = ['Pauli']
    itypes = [0, 1, 2]
    data = 'data = {\n'
    for kerntype, itype in itertools.product(kerns, itypes):
        if kerntype in {'Pauli', 'pyPauli', 'Lindblad', 'pyLindblad'} and itype in [0, 1]:
            continue

        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, itype=itype)
        system.solve()
        attr = kerntype+str(itype)
        data = data+' '*4+'\''+attr+'current\': '+str(system.current.tolist())+',\n'
        data = data+' '*4+'\''+attr+'energy_current\': '+str(system.energy_current.tolist())
        data = data + ('\n    }' if kerntype == 'pyLindblad' and itype == 2 else ',\n')
    #
    with open(fname, 'w') as f:
        f.write(data)


def test_Builder_double_dot_spinful():
    from qmeq.tests.data_builder import data
    p = ParametersDoubleDotSpinful()
    calcs = Calcs()

    # Check if the results agree with previously calculated data
    kerns = ['Pauli', 'Redfield', '1vN', 'Lindblad', 'RTD']
    kerns += ['pyPauli', 'pyRedfield', 'py1vN', 'pyLindblad', 'pyRTD'] if CHECK_PY else []
    itypes = [0, 1, 2]
    repetitions = 3
    for kerntype, itype in itertools.product(kerns, itypes):
        if kerntype in {'Pauli', 'pyPauli', 'Lindblad', 'pyLindblad'} and itype in [0, 1]:
            continue
        elif kerntype in {'RTD', 'pyRTD'} and itype in [0, 2]:
            continue

        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, itype=itype)

        for i in range(repetitions):
            system.solve()
            attr = kerntype+str(itype)
            setattr(calcs, attr, system)

            if PRNTQ:
                print('kerntype - ', kerntype, 'itype - ', itype, 'repetition - ', i)
                print('current')
                print(system.current)
                print( data[attr+'current'] )
                print('energy_current')
                print(system.energy_current)
                print( data[attr+'energy_current'] )
                print('differences:')
                print( norm(system.current - data[attr+'current']) )
                print( norm(system.energy_current - data[attr+'energy_current']) )
                print()

            assert norm(system.current - data[attr+'current']) < EPS
            assert norm(system.energy_current - data[attr+'energy_current']) < EPS

    # Check least-squares solution with non-square matrix, i.e., symq=False
    for kerntype in kerns:
        itype = 1 if kerntype in ('pyRTD', 'RTD') else 2
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, itype=itype, symq=False)

        for i in range(repetitions):
            system.solve()
            attr = kerntype+str(itype)
            for param in ['current', 'energy_current']:
                assert norm(getattr(system, param) - data[attr+param]) < EPS

    kerns.remove('RTD')
    if CHECK_PY:
        kerns.remove('pyRTD')
    # Check matrix-free methods
    for kerntype in kerns:
        itype = 2
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, itype=itype, mfreeq=True)

        for i in range(repetitions):
            system.solve()
            attr = kerntype+str(itype)
            for param in ['current', 'energy_current']:
                assert norm(getattr(system, param) - data[attr+param]) < 1e-4

    # Check results with different indexing
    indexings = ['Lin', 'charge', 'sz', 'ssq']
    for kerntype, indexing in itertools.product(kerns, indexings):
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, itype=2, indexing=indexing)

        for i in range(repetitions):
            system.solve()
            attr = kerntype+str(itype)
            for param in ['current', 'energy_current']:
                assert norm(getattr(system, param) - data[attr+param]) < EPS


def test_Builder_double_dot_spinless_2vN():
    data_current = {'2vN': [0.18472226147540757, -0.1847222614754047]}
    data_energy_current = {'2vN': [0.2828749942707809, -0.28333373130210493]}
    #
    kerns = ['2vN']
    kerns += ['py2vN'] if CHECK_PY else []
    indexings = ['Lin', 'charge']
    p = ParametersDoubleDotSpinless()
    for kerntype, indexing in itertools.product(kerns, indexings):
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, indexing=indexing, kpnt=p.kpnt)
        system.solve(niter=5)
        assert norm(system.current - data_current['2vN']) < EPS
        assert norm(system.energy_current - data_energy_current['2vN']) < EPS


def test_Builder_single_orbital_spinful():
    data_current = {'Pauli': [0.08368833245372147, -0.08368833245372037, 0.08368833245372147, -0.08368833245372037],
                    '2vN':   [0.0735967870902393, -0.07359678709023731, 0.07359678709023965, -0.07359678709023706],
                    'RTD':   [0.0201086094980386, -0.0201086094980385,  0.0201086094980385, -0.0201086094980385]}
    data_energy_current = {'Pauli': [0.1254772916094682, -0.12547729160946774, 0.1254772916094682, -0.12547729160946774],
                           '2vN':   [0.8745228418442359, -0.8753178905524672, 0.8745228418440387, -0.8753178905524854],
                           'RTD':  [ 0.1020455310853108, -0.1020455310853086, 0.1020455310853103, -0.102045531085309 ]}
    #
    kerns = ['Pauli', 'Redfield', '1vN', 'Lindblad']
    kerns += ['pyPauli', 'pyRedfield', 'py1vN', 'pyLindblad'] if CHECK_PY else []
    indexings = ['Lin', 'charge', 'sz', 'ssq']
    itypes = [0, 1, 2]
    p = ParametersSingleOrbitalSpinful()
    for kerntype, indexing, itype in itertools.product(kerns, indexings, itypes):
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, indexing=indexing, itype=itype)
        system.solve()
        assert norm(system.current - data_current['Pauli']) < EPS
        assert norm(system.energy_current - data_energy_current['Pauli']) < EPS

    kerns = ['2vN']
    kerns += ['py2vN'] if CHECK_PY else []
    indexings = ['Lin', 'charge']
    for kerntype, indexing in itertools.product(kerns, indexings):
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, indexing=indexing, kpnt=p.kpnt)
        system.solve(niter=5)
        assert norm(system.current - data_current['2vN']) < EPS
        assert norm(system.energy_current - data_energy_current['2vN']) < EPS

    kerns = ['RTD']
    kerns += ['pyRTD'] if CHECK_PY else []
    for kerntype in kerns:
        tL, tR = 1.0, 0.5
        tleads = {(0, 0):tL, (1, 0):tR, (2, 1):tL, (3, 1):tR}
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, itype=1)
        system.solve()
        assert norm(system.current - data_current['RTD']) < EPS
        assert norm(system.energy_current - data_energy_current['RTD']) < EPS

def test_Builder_double_orbital_spinful():
    data_current = {'RTD':   [-3.60896814e-09, 3.60896814e-09, -3.14061580e-10, 3.14061580e-10]}
    data_energy_current = { 'RTD':  [ 1.76461253e-05,  1.70735696e-05, -1.73635214e-05, -1.73561735e-05]}
    data_current_dT = {'RTD': [ 4.44455606601e-08, -4.44455606601e-08, -5.60894493255e-10,  5.60894493255e-10]}
    data_energy_current_dT = {'RTD': [-3.73787963742e-04, 4.08684346509e-04, -4.89209714427e-05, 1.40245886744e-05]}
    #
    kerns = ['RTD']
    kerns += ['pyRTD'] if CHECK_PY else []
    p = ParametersDoubleOrbitalSpinful()
    for kerntype in kerns:
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, indexing='charge', itype=1)
        system.solve()
        assert norm(system.current - data_current['RTD']) < EPS
        assert norm(system.energy_current - data_energy_current['RTD']) < EPS
        #
        temp = system.tlst[0]
        system.change(tlst={0: temp, 1: 2 * temp, 2: temp, 3: 2 * temp})
        system.change(mulst={0: 0, 1: 0, 2: 0, 3: 0})
        system.solve()
        assert norm(system.current - data_current_dT['RTD']) < EPS
        assert norm(system.energy_current - data_energy_current_dT['RTD']) < EPS

def test_Builder_coulomb_symmetry_spin():

    def intas(m, l):
        m = m+1
        l = l+1
        return np.exp(1j*(np.sqrt(l)-np.sqrt(m)))*np.sqrt(1/(np.sqrt(m)+np.sqrt(l)))

    def Vmnkl(m, n, k, l):
        return intas(m, l)*intas(n, k)

    nsingle = 4
    # Spinless, m_less_n=False, herm_c=False
    coulomb0 = {}
    for m, n, k, l in itertools.product(range(nsingle), repeat=4):
        coulomb0.update({(m, n, k, l): Vmnkl(m,n,k,l)/2})

    sys0_spinless = Builder(nsingle=nsingle, coulomb=coulomb0, mtype_qd=complex, m_less_n=False, herm_c=False)
    sys0_spinless.solve(masterq=False)

    # Spinless, m_less_n=False, herm_c=True
    coulomb1 = {}
    for m, n, k, l in itertools.product(range(nsingle), repeat=4):
        if not ((l, k, n, m) in coulomb1.keys()):
            coulomb1.update({(m, n, k, l): Vmnkl(m,n,k,l)/2})

    sys1_spinless = Builder(nsingle=nsingle, coulomb=coulomb1, mtype_qd=complex, m_less_n=False, herm_c=True)
    sys1_spinless.solve(masterq=False)

    # Spinless, m_less_n=True, herm_c=False
    coulomb2 = {}
    for m, n, k, l in itertools.product(range(nsingle), repeat=4):
        if m <= n:
            coulomb2.update({(m, n, k, l): Vmnkl(m,n,k,l)})

    sys2_spinless = Builder(nsingle=nsingle, coulomb=coulomb2, mtype_qd=complex, m_less_n=True, herm_c=False)
    sys2_spinless.solve(masterq=False)

    # Spinless, m_less_n=True, herm_c=True
    coulomb3 = {}
    for m, n, k, l in itertools.product(range(nsingle), repeat=4):
        if m <= n and not ((l, k, n, m) in coulomb3.keys()):
            coulomb3.update({(m, n, k, l): Vmnkl(m,n,k,l)})

    sys3_spinless = Builder(nsingle=nsingle, coulomb=coulomb3, mtype_qd=complex, m_less_n=True, herm_c=True)
    sys3_spinless.solve(masterq=False)

    assert sum(abs(sys1_spinless.Ea-sys0_spinless.Ea)) < EPS
    assert sum(abs(sys2_spinless.Ea-sys0_spinless.Ea)) < EPS
    assert sum(abs(sys3_spinless.Ea-sys0_spinless.Ea)) < EPS

    nsingle = 8
    ns = 4
    # Spinful
    coulomb = {}
    for m, n, k, l in itertools.product(range(ns), repeat=4):
        coulomb.update({(m, n, k, l): Vmnkl(m,n,k,l)/2})
        coulomb.update({(m+ns, n, k, l+ns): Vmnkl(m,n,k,l)/2})
        coulomb.update({(m, n+ns, k+ns, l): Vmnkl(m,n,k,l)/2})
        coulomb.update({(m+ns, n+ns, k+ns, l+ns): Vmnkl(m,n,k,l)/2})

    indexing = 'ssq'
    sys_ref_spinful = Builder(nsingle=nsingle, coulomb=coulomb, mtype_qd=complex, m_less_n=False, herm_c=False, indexing=indexing)
    sys_ref_spinful.solve(masterq=False)
    sys0_spinful = Builder(nsingle=nsingle, coulomb=coulomb0, mtype_qd=complex, m_less_n=False, herm_c=False, symmetry='spin', indexing=indexing)
    sys0_spinful.solve(masterq=False)
    sys1_spinful = Builder(nsingle=nsingle, coulomb=coulomb1, mtype_qd=complex, m_less_n=False, herm_c=True, symmetry='spin', indexing=indexing)
    sys1_spinful.solve(masterq=False)
    sys2_spinful = Builder(nsingle=nsingle, coulomb=coulomb2, mtype_qd=complex, m_less_n=True, herm_c=False, symmetry='spin', indexing=indexing)
    sys2_spinful.solve(masterq=False)
    sys3_spinful = Builder(nsingle=nsingle, coulomb=coulomb3, mtype_qd=complex, m_less_n=True, herm_c=True, symmetry='spin', indexing=indexing)
    sys3_spinful.solve(masterq=False)

    assert sum(abs(sys0_spinful.Ea-sys_ref_spinful.Ea)) < EPS
    assert sum(abs(sys1_spinful.Ea-sys_ref_spinful.Ea)) < EPS
    assert sum(abs(sys2_spinful.Ea-sys_ref_spinful.Ea)) < EPS
    assert sum(abs(sys3_spinful.Ea-sys_ref_spinful.Ea)) < EPS


def serial_triple_dot_coulomb_symmetry_spin():
    # Coulomb matrix elements
    # Intradot terms: u, uex
    # Interdot terms: un, udc, usc
    u, uex, un, udc, usc = 10., 2., 3., -0.5, -0.2

    # ----------- Spinless -----------
    nsingle = 5
    dotindex = [0, 0, 1, 1, 2]
    # m, n, k, l
    coulomb = []
    for m, n, k, l in itertools.product(range(nsingle), repeat=4):
        if m == n == k == l:
            coulomb.append([m,m,m,m,u/2]) # Direct
        if dotindex[m] == dotindex[k]:
            if m == n and k == l and m != k:
                coulomb.append([m,m,k,k,uex/2]) # Exchange
        if m != n and k != l:
            # Intradot
            if dotindex[m] == dotindex[n]:
                if m == l and n == k: coulomb.append([m,n,n,m,u/2])   # Direct
                if m == k and n == l: coulomb.append([m,n,m,n,uex/2]) # Exchange
            # Interdot
            # Note that the pairs (n,k) and (m,l) are located at different dots
            if (dotindex[m] == dotindex[l] and
                dotindex[n] == dotindex[k] and
                abs(dotindex[m]-dotindex[n]) == 1):
                if m == l and n == k:
                    coulomb.append([m,n,n,m,un/2])   # Direct
                if n == k and m != l:
                    sgn = dotindex[m]-dotindex[n]
                    coulomb.append([m,n,n,l,udc/2*sgn])  # Charge-dipole
                if n != k and m == l:
                    sgn = dotindex[n]-dotindex[m]
                    coulomb.append([m,n,k,m,udc/2*sgn])  # Charge-dipole
                if n != k and m != l:
                    coulomb.append([m,n,k,l,usc/2])  # Charge-quadrupole
    coulomb0 = coulomb

    coulomb1 = {(0,0,0,0):u, (1,1,1,1):u, (2,2,2,2):u, (3,3,3,3):u, (4,4,4,4):u,
                (0,1,1,0):u, (2,3,3,2):u,
                #
                (0,0,1,1):uex, (2,2,3,3):uex,
                (0,1,0,1):uex, (2,3,2,3):uex,
                #
                (0,2,2,0):un, (0,3,3,0):un,
                (1,2,2,1):un, (1,3,3,1):un,
                (2,4,4,2):un, (3,4,4,3):un,
                #
                (0,2,2,1):-udc, (0,3,3,1):-udc, (2,4,4,3):-udc,
                (0,2,3,0):+udc, (1,2,3,1):+udc,
                #
                (0,2,3,1):usc, (0,3,2,1):usc,
                # Conjugated terms
                (1,1,0,0):uex, (3,3,2,2):uex,
                #
                (1,2,2,0):-udc, (1,3,3,0):-udc, (3,4,4,2):-udc,
                (0,3,2,0):+udc, (1,3,2,1):+udc,
                #
                (1,3,2,0):usc, (1,2,3,0):usc
                }

    coulomb2 = {(0,0,0,0):u, (1,1,1,1):u, (2,2,2,2):u, (3,3,3,3):u, (4,4,4,4):u,
                (0,1,1,0):u, (2,3,3,2):u,
                #
                (0,0,1,1):uex, (2,2,3,3):uex,
                (0,1,0,1):uex, (2,3,2,3):uex,
                #
                (0,2,2,0):un, (0,3,3,0):un,
                (1,2,2,1):un, (1,3,3,1):un,
                (2,4,4,2):un, (3,4,4,3):un,
                #
                (0,2,2,1):-udc, (0,3,3,1):-udc, (2,4,4,3):-udc,
                (0,2,3,0):+udc, (1,2,3,1):+udc,
                #
                (0,2,3,1):usc, (0,3,2,1):usc
                }

    sys0_spinless = qmeq.Builder(nsingle=5, coulomb=coulomb0, symmetry=None, m_less_n=False, herm_c=False)
    sys0_spinless.solve(masterq=False)
    sys1_spinless = qmeq.Builder(nsingle=5, coulomb=coulomb1, symmetry=None, m_less_n=True, herm_c=False)
    sys1_spinless.solve(masterq=False)
    sys2_spinless = qmeq.Builder(nsingle=5, coulomb=coulomb2, symmetry=None, m_less_n=True, herm_c=True)
    sys2_spinless.solve(masterq=False)

    assert sum(abs(sys1_spinless.Ea-sys0_spinless.Ea)) < EPS
    assert sum(abs(sys2_spinless.Ea-sys0_spinless.Ea)) < EPS

    # ----------- Spinful -----------
    nsingle = 10
    nssl = nsingle//2
    dotindex = [0, 0, 1, 1, 2, 0, 0, 1, 1, 2]
    # m, n, k, l
    coulomb = []
    for m, n, k, l in itertools.product(range(nsingle), repeat=4):
        if m != n and k != l and m//nssl == l//nssl and n//nssl == k//nssl:
            # Intradot
            if dotindex[m] == dotindex[n]:
                if m == l and n == k: coulomb.append([m,n,n,m,u/2]) # Direct
                if m == k and n == l:
                    coulomb.append([m,n,m,n,uex/2]) # Exchange
                    if m+nssl < nsingle:
                        coulomb.append([m,n+nssl,m+nssl,n,uex/2])
                        coulomb.append([m+nssl,n,m,n+nssl,uex/2])
                        coulomb.append([m,m+nssl,n+nssl,n,uex/2])
                        coulomb.append([m+nssl,m,n,n+nssl,uex/2])
            # Interdot
            # Note that the pairs (n,k) and (m,l) are located at different dots
            if (dotindex[m] == dotindex[l] and
                dotindex[n] == dotindex[k] and
                abs(dotindex[m]-dotindex[n]) == 1):
                if m == l and n == k:
                    coulomb.append([m,n,n,m,un/2])   # Direct
                if n == k and m != l:
                    sgn = dotindex[m]-dotindex[n]
                    coulomb.append([m,n,n,l,udc/2*sgn])  # Charge-dipole
                if n != k and m == l:
                    sgn = dotindex[n]-dotindex[m]
                    coulomb.append([m,n,k,m,udc/2*sgn])  # Charge-dipole
                if n != k and m != l:
                    coulomb.append([m,n,k,l,usc/2])  # Charge-quadrupole

    indexing='ssq'
    sys_ref_spinful = qmeq.Builder(nsingle=10, coulomb=coulomb, indexing=indexing)
    sys_ref_spinful.solve(masterq=False)
    sys0_spinful = qmeq.Builder(nsingle=10, coulomb=coulomb0, symmetry='spin', m_less_n=False, indexing=indexing)
    sys0_spinful.solve(masterq=False)
    sys1_spinful = qmeq.Builder(nsingle=10, coulomb=coulomb1, symmetry='spin', m_less_n=True, herm_c=False, indexing=indexing)
    sys1_spinful.solve(masterq=False)
    sys2_spinful = qmeq.Builder(nsingle=10, coulomb=coulomb2, symmetry='spin', m_less_n=True, herm_c=True, indexing=indexing)
    sys2_spinful.solve(masterq=False)

    assert sum(abs(sys0_spinful.Ea-sys_ref_spinful.Ea)) < 10*EPS
    assert sum(abs(sys1_spinful.Ea-sys_ref_spinful.Ea)) < 10*EPS
    assert sum(abs(sys2_spinful.Ea-sys_ref_spinful.Ea)) < 10*EPS


def test_Builder_serial_triple_dot_coulomb_symmetry_spin():
    if SERIAL_TRIPLE_DOT:
        serial_triple_dot_coulomb_symmetry_spin()
