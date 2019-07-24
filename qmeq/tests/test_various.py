from numpy.linalg import norm
from qmeq.builder.various import *
import qmeq

EPS = 1e-14


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
        #
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
        vbias, temp, dband = 5.0, 50.0, 1000.0
        self.mulst = [vbias/2, -vbias/2]
        self.tlst = [temp, temp]
        self.dlst = [dband, dband]
        self.dlst2 = [[-dband+vbias/2, dband+vbias/2],
                      [-dband-vbias/2, dband-vbias/2]]
        #
        self.nsingle, self.nleads = 2, 2
        self.kpnt = np.power(2, 9)


def test_get_charge():
    system = qmeq.Builder(4, {}, {}, 0, {}, {}, {}, {})
    assert get_charge(system, 10) == 2


def test_multiarray_sort():
    arr = np.array([[3, 2, 1, 3, 2, 1],
                    [2, 3, 1, 1, 2, 3],
                    [1, 2, 3, 2, 1, 3]])
    assert multiarray_sort(arr, [0,1,2]).tolist() == [[1, 1, 2, 2, 3, 3], [1, 3, 2, 3, 1, 2], [3, 3, 1, 2, 2, 1]]
    assert multiarray_sort(arr, [0,2,1]).tolist() == [[1, 1, 2, 2, 3, 3], [1, 3, 2, 3, 2, 1], [3, 3, 1, 2, 1, 2]]
    assert multiarray_sort(arr, [1,0,2]).tolist() == [[1, 3, 2, 3, 1, 2], [1, 1, 2, 2, 3, 3], [3, 2, 1, 1, 3, 2]]
    assert multiarray_sort(arr, [1,2,0]).tolist() == [[3, 1, 2, 3, 2, 1], [1, 1, 2, 2, 3, 3], [2, 3, 1, 1, 2, 3]]
    assert multiarray_sort(arr, [2,0,1]).tolist() == [[2, 3, 2, 3, 1, 1], [2, 2, 3, 1, 1, 3], [1, 1, 2, 2, 3, 3]]
    assert multiarray_sort(arr, [2,1,0]).tolist() == [[2, 3, 3, 2, 1, 1], [2, 2, 1, 3, 1, 3], [1, 1, 2, 2, 3, 3]]


def test_sort_eigenstates():
    p = ParametersDoubleDotSpinful()
    system = qmeq.Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                          indexing='ssq')
    system.solve(masterq=False)
    # Sort by energy
    system.sort_eigenstates([0,1,2,3])
    inds = system.si.states_order.tolist()
    assert inds == [1, 3, 6, 0, 5, 9, 10, 2, 4, 7, 8, 11, 13, 12, 14, 15]
    assert system.Ea[inds].tolist() == np.sort(system.Ea).tolist()
    # Sort by charge
    sort_eigenstates(system, [1,2,3,0])
    inds = system.si.states_order.tolist()
    assert inds == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert system.Ea[inds].tolist() == system.Ea.tolist()


def test_get_phi0_and_get_phi1():
    p = ParametersDoubleDotSpinless()
    system = qmeq.Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst2,
                          kerntype='1vN', itype=0)
    system.solve()
    assert norm(abs(system.get_phi0(1, 2)) - 0.00284121629354) < EPS
    assert norm(system.get_phi0(1, 2).conjugate() - system.get_phi0(2, 1)) < EPS
    assert norm(abs(get_phi0(system, 1, 2)) - 0.00284121629354) < EPS
    assert norm(get_phi0(system, 1, 2).conjugate() - get_phi0(system, 2, 1)) < EPS
    #
    assert norm( abs(system.get_phi1(0, 1, 0)) - 2.6173227979053406) < EPS
    assert norm(system.get_phi1(0, 1, 0).conjugate() - system.get_phi1(0, 0, 1)) < EPS
    assert norm( abs(get_phi1(system, 0, 1, 0)) - 2.6173227979053406) < EPS
    assert norm(get_phi1(system, 0, 1, 0).conjugate() - get_phi1(system, 0, 0, 1)) < EPS
    #
    system = qmeq.Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                          kerntype='2vN', kpnt=p.kpnt)
    system.solve(niter=5)
    assert norm(abs(system.get_phi0(1, 2)) - 0.0028388555051) < EPS
    assert norm(system.get_phi0(1, 2).conjugate() - system.get_phi0(2, 1)) < EPS
    assert norm(abs(get_phi0(system, 1, 2)) - 0.0028388555051) < EPS
    assert norm(get_phi0(system, 1, 2).conjugate() - get_phi0(system, 2, 1)) < EPS
    #
    assert norm( abs(system.get_phi1(0, 1, 0)) - 2.4806366680863543) < EPS
    assert norm(system.get_phi1(0, 1, 0).conjugate() - system.get_phi1(0, 0, 1)) < EPS
    assert norm( abs(get_phi1(system, 0, 1, 0)) - 2.4806366680863543) < EPS
    assert norm(get_phi1(system, 0, 1, 0).conjugate() - get_phi1(system, 0, 0, 1)) < EPS


def test_construct_Ea_extended():
    data = {'Lin':    [[0.0, -31.024984394500787, -31.024984394500787, -14.182934011942466, 9.024984394500784, 8.0, 8.0, 86.9750156054992, 9.024984394500784, 8.0, 57.87579144142183, 86.9750156054992, 80.30714257052065, 127.02498439450079, 127.02498439450079, 236.0], [0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]],
            'charge': [[0.0, -31.024984394500787, -31.024984394500787, 9.024984394500784, 9.024984394500784, -14.182934011942466, 8.0, 8.0, 8.0, 57.87579144142183, 80.30714257052065, 86.9750156054992, 86.9750156054992, 127.02498439450079, 127.02498439450079, 236.0], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]],
            'sz':     [[0.0, -31.024984394500787, 9.024984394500784, -31.024984394500787, 9.024984394500784, 8.0, -14.182934011942466, 8.0, 57.87579144142183, 80.30714257052065, 8.0, 86.9750156054992, 127.02498439450079, 86.9750156054992, 127.02498439450079, 236.0], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0], [0.0, -1.0, -1.0, 1.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0, -1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]],
            'ssq':    [[0.0, -31.024984394500787, 9.024984394500784, -31.024984394500787, 9.024984394500784, 8.0, -14.182934011942473, 57.875791441421825, 80.30714257052063, 8.0, 8.0, 86.9750156054992, 127.02498439450079, 86.9750156054992, 127.02498439450079, 236.0], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0], [0.0, -1.0, -1.0, 1.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0, -1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]}
    p = ParametersDoubleDotSpinful()
    for indexing in ['Lin', 'charge', 'sz', 'ssq']:
        system = qmeq.Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst,
                              indexing=indexing)
        system.solve(masterq=False)
        assert norm(construct_Ea_extended(system) - data[indexing]) < EPS*10


def test_remove_states():
    p = ParametersDoubleDotSpinful()
    system = qmeq.Builder(p.nsingle, p.hsingle, p.coulomb, p.nleads, p.tleads, p.mulst, p.tlst, p.dlst)
    system.solve()
    #
    system.remove_states(50.0)
    assert system.si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8], [], [], []]
    system.use_all_states()
    assert system.si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15], []]
    #
    remove_states(system, 50.0)
    assert system.si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8], [], [], []]
    use_all_states(system)
    assert system.si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15], []]
