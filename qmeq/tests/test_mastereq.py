import numpy as np
from numpy.linalg import norm
from qmeq.mastereq import *
import qmeq
import itertools

EPS = 1e-11
CHECK_PY = False
PRNTQ = False

class Parameters_double_dot_spinful(object):

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

class Parameters_double_dot_spinless(object):

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

class Parameters_single_orbital_spinful(object):

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

class Calcs(object):
    def __init__(self):
        pass

def save_Builder_double_dot_spinful(fname='data_mastereq.py'):
    p = Parameters_double_dot_spinful()
    #data = {}
    kerns = ['Pauli', 'Redfield', '1vN', 'Lindblad', 'pyPauli', 'py1vN', 'pyLindblad']
    #kerns = ['Pauli']
    itypes = [0, 1, 2]
    data = 'data = {\n'
    dataR = ''
    for kerntype, itype in itertools.product(kerns, itypes):
        if not ( kerntype in {'Pauli', 'pyPauli', 'Lindblad', 'pyLindblad'} and itype in [0, 1] ):
            system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                             kerntype=kerntype, itype=itype)
            system.solve()
            attr = kerntype+str(itype)
            data = data+' '*4+'\''+attr+'current\': '+str(system.current.tolist())+',\n'
            data = data+' '*4+'\''+attr+'energy_current\': '+str(system.energy_current.tolist())
            data = data + ('\n    }' if kerntype is 'pyLindblad' and itype is 2 else ',\n' )
    #
    with open(fname, 'w') as f:
        f.write(data)

def test_Builder_double_dot_spinful():
    from data_mastereq import data
    p = Parameters_double_dot_spinful()
    calcs = Calcs()

    # Check if the results agree with previously calculated data
    kerns = ['Pauli', 'Redfield', '1vN', 'Lindblad']
    kerns += ['pyPauli', 'py1vN', 'pyLindblad'] if CHECK_PY else []
    itypes = [0, 1, 2]
    for kerntype, itype in itertools.product(kerns, itypes):
        if not ( kerntype in {'Pauli', 'pyPauli', 'Lindblad', 'pyLindblad'} and itype in [0, 1] ):
            system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                             kerntype=kerntype, itype=itype)
            system.solve()
            attr = kerntype+str(itype)
            setattr(calcs, attr, system)
            #
            if PRNTQ:
                print(kerntype, itype)
                print(system.tt.current)
                print( data[attr+'current'] )
                print(system.tt.energy_current)
                print( data[attr+'energy_current'] )
                print( norm(system.current - data[attr+'current']) )
                print( norm(system.energy_current - data[attr+'energy_current']) )
            #
            assert norm(system.current - data[attr+'current']) < EPS
            assert norm(system.energy_current - data[attr+'energy_current']) < EPS

    # Check least-squares solution with non-square matrix, i.e., symq=False
    for kerntype in kerns:
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, itype=2, symq=False)
        system.solve()
        attr = kerntype+str(itype)
        for param in ['current', 'energy_current']:
            assert norm(getattr(system, param) - getattr(getattr(calcs, attr), param)) < EPS

    # Check matrix-free methods
    kerns = ['Redfield', '1vN', 'Lindblad']
    kerns += ['py1vN', 'pyLindblad'] if CHECK_PY else []
    for kerntype in kerns:
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, itype=2, mfreeq=True)
        system.solve()
        attr = kerntype+str(itype)
        for param in ['current', 'energy_current']:
            assert norm(getattr(system, param) - getattr(getattr(calcs, attr), param)) < 1e-4

    # Check results with different indexing
    kerns = ['Pauli', 'Redfield', '1vN', 'Lindblad']
    kerns += ['pyPauli', 'py1vN', 'pyLindblad'] if CHECK_PY else []
    indexings = ['Lin', 'charge', 'sz', 'ssq']
    for kerntype, indexing in itertools.product(kerns, indexings):
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, itype=2, indexing=indexing)
        system.solve()
        attr = kerntype+str(itype)
        for param in ['current', 'energy_current']:
            assert norm(getattr(system, param) - getattr(getattr(calcs, attr), param)) < EPS

def test_Builder_double_dot_spinless_2vN():
    data_current = {'2vN': [0.18472226147540757, -0.1847222614754047]}
    data_energy_current = {'2vN': [0.2828749942707809, -0.28333373130210493]}
    #
    kerns = ['2vN']
    kerns += ['py2vN'] if CHECK_PY else []
    indexings = ['Lin', 'charge']
    p = Parameters_double_dot_spinless()
    for kerntype, indexing in itertools.product(kerns, indexings):
        system = Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                         kerntype=kerntype, indexing=indexing, kpnt=p.kpnt)
        system.solve(niter=5)
        assert norm(system.current - data_current['2vN']) < EPS
        assert norm(system.energy_current - data_energy_current['2vN']) < EPS

def test_Builder_single_orbital_spinful():
    data_current = {'Pauli': [0.08368833245372147, -0.08368833245372037, 0.08368833245372147, -0.08368833245372037],
                    '2vN':   [0.0735967870902393, -0.07359678709023731, 0.07359678709023965, -0.07359678709023706]}
    data_energy_current = {'Pauli': [0.1254772916094682, -0.12547729160946774, 0.1254772916094682, -0.12547729160946774],
                           '2vN':   [0.8745228418442359, -0.8753178905524672, 0.8745228418440387, -0.8753178905524854]}
    #
    kerns = ['Pauli', 'Redfield', '1vN', 'Lindblad']
    kerns += ['pyPauli', 'py1vN', 'pyLindblad'] if CHECK_PY else []
    indexings = ['Lin', 'charge', 'sz', 'ssq']
    itypes = [0, 1, 2]
    p = Parameters_single_orbital_spinful()
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

def test_Transport2vN_kpnt():
    # Check generation of energy grid
    si = qmeq.StateIndexingDMc(1)
    qd = qmeq.QuantumDot({}, {}, si)
    leads = qmeq.LeadsTunneling(1, {}, si, {}, {}, {0: 1000})
    funcp = FunctionProperties(kpnt=5, kerntype='2vN')
    tt = Transport2vN(qd, leads, si, funcp)
    assert tt.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    tt.kpnt = 6
    assert tt.Ek_grid.tolist() == [-1000, -600,  -200, 200, 600, 1000]
    #
    system = Builder(1, {}, {}, 1, {}, {}, {}, {0: 1000}, kpnt=5, kerntype='2vN')
    assert system.tt.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    system.kpnt = 6
    assert system.tt.Ek_grid.tolist() == [-1000, -600,  -200, 200, 600, 1000]

def test_Transport2vN_make_Ek_grid():
    si = qmeq.StateIndexingDMc(1)
    qd = qmeq.QuantumDot({}, {}, si)
    leads = qmeq.LeadsTunneling(2, {}, si, {}, {}, {0: [-1000, 1000], 1: [-1000, 1000]})
    funcp = FunctionProperties(kpnt=5, kerntype='2vN')
    tt = Transport2vN(qd, leads, si, funcp)
    tt.make_Ek_grid()
    assert tt.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    tt.leads.change(dlst={0: [-1400, 1000], 1: [-1000, 1000]})
    tt.make_Ek_grid()
    assert tt.Ek_grid.tolist() == [-1400.0, -800.0, -200.0, 400.0, 1000.0]
