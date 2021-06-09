from numpy.linalg import norm
from qmeq.indexing import StateIndexing
from qmeq.qdot import *

EPS = 1e-13


class ParametersDoubleDotSpinful(object):

    def __init__(self):
        e1, e2, omega = 1, 2, 20
        self.hsingle = {(0,0):e1, (2,2):e1,
                        (1,1):e2, (3,3):e2,
                        (0,1):omega, (2,3):omega}
        uintra, uinter = 80, 30
        self.coulomb = {(0,2,2,0):uintra, (1,3,3,1):uintra,
                        (0,1,1,0):uinter, (0,3,3,0):uinter,
                        (1,2,2,1):uinter, (2,3,3,2):uinter}


def test_construct_ham_coulomb():
    si = StateIndexing(4, indexing='Lin')
    p = ParametersDoubleDotSpinful()
    qd = QuantumDot({}, {}, si)
    statelst = [3, 5, 6, 9, 10, 12]  # si.chargelst[2]
    assert construct_ham_coulomb(qd, p.coulomb, statelst).tolist() == np.diag([30., 80., 30., 30., 80., 30.]).tolist()
    statelst = [15]  # si.chargelst[4]
    assert construct_ham_coulomb(qd, p.coulomb, statelst).tolist() == [[280.]]


def test_construct_ham_hopping():
    si = StateIndexing(4, indexing='Lin')
    p = ParametersDoubleDotSpinful()
    qd = QuantumDot({}, {}, si)
    statelst = [1, 2, 4, 8]  # si.chargelst[1]
    assert construct_ham_hopping(qd, p.hsingle, statelst).tolist() == [[2, 20, 0, 0], [20, 1, 0, 0], [0, 0, 2, 20], [0, 0, 20, 1]]
    statelst = [3, 5, 6, 9, 10, 12]  # si.chargelst[2]
    assert construct_ham_hopping(qd, p.hsingle, statelst).tolist() == [[3, 0, 0, 0, 0, 0], [0, 4, 20, 20, 0, 0], [0, 20, 3, 0, 20, 0], [0, 20, 0, 3, 20, 0], [0, 0, 20, 20, 2, 0], [0, 0, 0, 0, 0, 3]]


def test_construct_manybody_eigenstates():
    si = StateIndexing(4, indexing='Lin')
    p = ParametersDoubleDotSpinful()
    qd = QuantumDot({}, {}, si)
    statelst = [1, 2, 4, 8]  # si.chargelst[1]
    ham_vals, ham_vecs = construct_manybody_eigenstates(qd, p.hsingle, p.coulomb, statelst)
    assert norm(ham_vals - np.array([-18.506249023742555, -18.506249023742555, 21.50624902374256, 21.50624902374256])) < EPS
    statelst = [3, 5, 6, 9, 10, 12]  # si.chargelst[2]
    ham_vals, ham_vecs = construct_manybody_eigenstates(qd, p.hsingle, p.coulomb, statelst)
    assert norm(ham_vals - np.array([10.82683790853667, 33.0, 33.0, 33.0, 82.96879990127198, 105.20436219019136])) < EPS


def test_construct_Ea_manybody():
    valslst = {'Lin':    [[0], [-18, -18, 21, 21], [10, 33, 33, 33, 82, 105], [124, 124, 164, 164], [286]],
               'charge': [[0], [-18, -18, 21, 21], [10, 33, 33, 33, 82, 105], [124, 124, 164, 164], [286]],
               'sz':     [[[0]], [[-18, 21], [-18, 21]], [[33], [10, 33, 82, 105], [33]], [[124, 164], [124, 164]], [[286]]],
               'ssq':    [[[[0]]], [[[-18, 21]], [[-18, 21]]], [[[33]], [[105, 82, 105], [33]], [[33]]], [[[124, 164]], [[124, 164]]], [[[286]]]] }
    data = {'Lin':    [0, -18, -18, 10, 21, 33, 33, 124, 21, 33, 82, 124, 105, 164, 164, 286],
            'charge': [0, -18, -18, 21, 21, 10, 33, 33, 33, 82, 105, 124, 124, 164, 164, 286],
            'sz':     [0, -18, 21, -18, 21, 33, 10, 33, 82, 105, 33, 124, 164, 124, 164, 286],
            'ssq':    [0, -18, 21, -18, 21, 33, 105, 82, 105, 33, 33, 124, 164, 124, 164, 286]}
    for indexing in ['Lin', 'charge', 'sz', 'ssq']: #
        si = StateIndexing(4, indexing=indexing)
        assert construct_Ea_manybody(valslst[indexing], si).tolist() == data[indexing]


def test_operator_sm():
    si = StateIndexing(4, indexing='ssq')
    assert operator_sm(2, 2, si).tolist() == [[0], [-2], [2], [0]]
    assert operator_sm(2, 0, si).tolist() == [[0, -2, 2, 0]]
    assert operator_sm(2, -2, si) == 0


def test_operator_sp():
    si = StateIndexing(4, indexing='ssq')
    assert operator_sp(2, 2, si) == 0
    assert operator_sp(2, 0, si).tolist() == [[0, -2, 2, 0]]
    assert operator_sp(2, -2, si).tolist() == [[0], [-2], [2], [0]]


def test_operator_ssquare():
    si = StateIndexing(4, indexing='ssq')
    assert operator_ssquare(2, 2, si).tolist() == [[8]]
    assert operator_ssquare(2, 0, si).tolist() == [[0, 0, 0, 0], [0, 4, -4, 0], [0, -4, 4, 0], [0, 0, 0, 0]]
    assert operator_ssquare(2, -2, si).tolist() == [[8]]


def test_ssquare_eigenstates():
    si = StateIndexing(4, indexing='ssq')
    assert ssquare_eigenstates(2, -2, si)[0].tolist() == [[1]]
    assert norm(ssquare_eigenstates(2,  0, si)[0] - np.array([[1.0, 0.0, 0.0], [0.0, -0.5*np.sqrt(2), 0.0], [0.0, -0.5*np.sqrt(2), 0.0], [0.0, 0.0, 1.0]])) < EPS
    assert norm(ssquare_eigenstates(2,  0, si)[1] - np.array([[0.0], [-0.5*np.sqrt(2)], [0.5*np.sqrt(2)], [0.0]])) < EPS
    assert ssquare_eigenstates(2, -2, si)[0].tolist() == [[1]]
    pass


def test_ssquare_all_szlow():
    si = StateIndexing(4, indexing='ssq')
    rez = ssquare_all_szlow(si)
    assert rez[0][0].tolist() == [[1.0]]
    assert rez[1][0].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert norm(rez[2][0].tolist() - np.array([[1.0, 0.0, 0.0], [0.0, -0.5*np.sqrt(2), 0.0], [0.0, -0.5*np.sqrt(2), 0.0], [0.0, 0.0, 1.0]])) < EPS
    assert norm(rez[2][1].tolist() - np.array([[0.0], [-0.5*np.sqrt(2)], [0.5*np.sqrt(2)], [0.0]])) < EPS
    assert rez[3][0].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert rez[4][0].tolist() == [[1.0]]


def test_construct_manybody_eigenstates_ssq():
    data = {(0,0): np.array([10.826837908536675, 82.96879990127198, 105.20436219019135]),
            (-2,2): np.array([33.0]),
            ( 0,2): np.array([33.0]),
            (+2,2): np.array([33.0])}
    si = StateIndexing(4, indexing='ssq')
    p = ParametersDoubleDotSpinful()
    qd = QuantumDot({}, {}, si)
    #
    eigvalp, eigvecssq = construct_manybody_eigenstates_ssq(qd, 2, 0, 0, p.hsingle, p.coulomb)
    assert norm(eigvalp - data[(0,0)]) < EPS
    #
    eigvalp, eigvecssq = construct_manybody_eigenstates_ssq(qd, 2, -2, 2, p.hsingle, p.coulomb)
    assert norm(eigvalp - data[(-2,2)]) < EPS
    eigvalp, eigvecssq = construct_manybody_eigenstates_ssq(qd, 2,  0, 2, p.hsingle, p.coulomb)
    assert norm(eigvalp - data[( 0,2)]) < EPS
    eigvalp, eigvecssq = construct_manybody_eigenstates_ssq(qd, 2, +2, 2, p.hsingle, p.coulomb)
    assert norm(eigvalp - data[(+2,2)]) < EPS
    #
    valslst, vecslst = construct_manybody_eigenstates_ssq_all(qd, 2, p.hsingle, p.coulomb)
    assert norm(valslst[0][0] - data[(-2,2)]) < EPS
    assert norm(valslst[1][0] - data[( 0,0)]) < EPS
    assert norm(valslst[1][1] - data[( 0,2)]) < EPS
    assert norm(valslst[2][0] - data[(+2,2)]) < EPS


def test_make_hsingle_mtr_and_dict():
    si = StateIndexing(4)
    qd = QuantumDot({}, {}, si)
    e1, e2, omega = 1, 2, 20
    hsingle_dict = {(0,0):e1, (2,2):e1, (1,1):e2, (3,3):e2, (0,1):omega, (2,3):omega}
    hsingle_list = [[0,0,e1], [2,2,e1], [1,1,e2], [3,3,e2], [0,1,omega], [2,3,omega]]
    hsingle_mtr = [[e1, omega, 0.0, 0.0], [omega, e2, 0.0, 0.0], [0.0, 0.0, e1, omega], [0.0, 0.0, omega, e2]]
    assert make_hsingle_mtr(hsingle_dict, 4, mtype=float).tolist() == hsingle_mtr
    assert make_hsingle_mtr(hsingle_list, 4, mtype=float).tolist() == hsingle_mtr
    assert make_hsingle_dict(qd, hsingle_list) == hsingle_dict
    assert make_hsingle_dict(qd, np.array(hsingle_mtr)) == hsingle_dict
    assert make_hsingle_dict(qd, hsingle_dict) == hsingle_dict


def test_make_coulomb_dict():
    si = StateIndexing(4)
    qd = QuantumDot({}, {}, si)
    uintra, uinter = 80, 30
    coulomb_dict = {(0,2,2,0):uintra, (1,3,3,1):uintra, (0,1,1,0):uinter, (0,3,3,0):uinter, (1,2,2,1):uinter, (2,3,3,2):uinter}
    coulomb_list = [[0,2,2,0,uintra], [1,3,3,1,uintra], [0,1,1,0,uinter], [0,3,3,0,uinter], [1,2,2,1,uinter], [2,3,3,2,uinter]]
    coulomb_arr = np.array(coulomb_list)
    assert make_coulomb_dict(qd, coulomb_list) == coulomb_dict
    assert make_coulomb_dict(qd, coulomb_arr) == coulomb_dict
    assert make_coulomb_dict(qd, coulomb_dict) == coulomb_dict


def test_QuantumDot(symmetry=None):
    e1, e2, omega = 1, 2, 20
    uintra, uinter = 80, 30
    if symmetry == 'spin':
        nsingle = 4
        hsingle = np.array([[e1, omega], [omega, e2]])
        coulomb = [[0,0,0,0,uintra], [1,1,1,1,uintra], [0,1,1,0,uinter]]
    else:
        nsingle = 4
        hsingle = np.array([[e1, omega, 0.0, 0.0], [omega, e2, 0.0, 0.0], [0.0, 0.0, e1, omega], [0.0, 0.0, omega, e2]])
        coulomb = [[0,2,2,0,uintra], [1,3,3,1,uintra], [0,1,1,0,uinter], [0,3,3,0,uinter], [1,2,2,1,uinter], [2,3,3,2,uinter]]
    #
    data0 = {'Lin':    [0.0, -18.506249023742555, -18.506249023742555, 10.82683790853667, 21.50624902374256, 33.0, 33.0, 124.49375097625745, 21.50624902374256, 33.0, 82.96879990127198, 124.49375097625745, 105.20436219019136, 164.50624902374255, 164.50624902374255, 286.0],
             'charge': [0.0, -18.506249023742555, -18.506249023742555, 21.50624902374256, 21.50624902374256, 10.82683790853667, 33.0, 33.0, 33.0, 82.96879990127198, 105.20436219019136, 124.49375097625745, 124.49375097625745, 164.50624902374255, 164.50624902374255, 286.0],
             'sz':     [0.0, -18.506249023742555, 21.50624902374256, -18.506249023742555, 21.50624902374256, 33.0, 10.82683790853667, 33.0, 82.96879990127198, 105.20436219019136, 33.0, 124.49375097625745, 164.50624902374255, 124.49375097625745, 164.50624902374255, 286.0],
             'ssq':    [0.0, -18.506249023742555, 21.50624902374256, -18.506249023742555, 21.50624902374256, 33.0, 10.826837908536675, 82.96879990127198, 105.20436219019135, 33.0, 33.0, 124.49375097625745, 164.50624902374255, 124.49375097625745, 164.50624902374255, 286.0]}
    data1 = {'Lin':    [0.0, 0.7541138286625813, 0.7541138286625813, 2.9707929658106993, 2.475886171337419, 33.23, 33.23, 65.00260975586771, 2.475886171337419, 33.23, 33.24585473007767, 65.00260975586771, 84.02335230411165, 145.23739024413226, 145.23739024413226, 207.01],
             'charge': [0.0, 0.7541138286625813, 0.7541138286625813, 2.475886171337419, 2.475886171337419, 2.9707929658106993, 33.23, 33.23, 33.23, 33.24585473007767, 84.02335230411165, 65.00260975586771, 65.00260975586771, 145.23739024413226, 145.23739024413226, 207.01],
             'sz':     [0.0, 0.7541138286625813, 2.475886171337419, 0.7541138286625813, 2.475886171337419, 33.23, 2.9707929658106993, 33.23, 33.24585473007767, 84.02335230411165, 33.23, 65.00260975586771, 145.23739024413226, 65.00260975586771, 145.23739024413226, 207.01],
             'ssq':    [0.0, 0.7541138286625813, 2.475886171337419, 0.7541138286625813, 2.475886171337419, 33.23, 2.970792965810705, 33.24585473007765, 84.02335230411163, 33.23, 33.23, 65.00260975586771, 145.23739024413226, 65.00260975586771, 145.23739024413226, 207.01]}
    #
    for indexing in ['Lin', 'charge', 'sz', 'ssq']:
        si = StateIndexing(nsingle, indexing=indexing, symmetry=symmetry)
        qd = QuantumDot(hsingle, coulomb, si)
        if symmetry == 'spin':
            assert qd.hsingle == {(0,0):e1, (2,2):e1, (1,1):e2, (3,3):e2, (0,1):omega, (2,3):omega}
            print(qd.coulomb)
            assert qd.coulomb == {(0,2,2,0):uintra, (1,3,3,1):uintra, (0,1,1,0):uinter, (0,3,3,0):uinter, (1,2,2,1):uinter, (2,3,3,2):uinter,
                                  (0,0,0,0):uintra, (1,1,1,1):uintra, (2,2,2,2):uintra, (3,3,3,3):uintra}
        else:
            assert qd.hsingle == {(0,0):e1, (2,2):e1, (1,1):e2, (3,3):e2, (0,1):omega, (2,3):omega}
            assert qd.coulomb == {(0,2,2,0):uintra, (1,3,3,1):uintra, (0,1,1,0):uinter, (0,3,3,0):uinter, (1,2,2,1):uinter, (2,3,3,2):uinter}
        #
        qd.diagonalise()
        assert norm(qd.Ea - data0[indexing]) < EPS
        if indexing == 'Lin':
            assert qd.hamlst[0].tolist() == [[0.0]]
            assert qd.hamlst[1].tolist() == [[2.0, 20.0, 0.0, 0.0], [20.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 20.0], [0.0, 0.0, 20.0, 1.0]]
            assert qd.hamlst[2].tolist() == [[33.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 84.0, 20.0, 20.0, 0.0, 0.0], [0.0, 20.0, 33.0, 0.0, 20.0, 0.0], [0.0, 20.0, 0.0, 33.0, 20.0, 0.0], [0.0, 0.0, 20.0, 20.0, 82.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 33.0]]
            assert qd.hamlst[3].tolist() == [[145.0, 20.0, 0.0, 0.0], [20.0, 144.0, 0.0, 0.0], [0.0, 0.0, 145.0, 20.0], [0.0, 0.0, 20.0, 144.0]]
            assert qd.hamlst[4].tolist() == [[286.0]]
        elif indexing == 'charge':
            assert qd.hamlst[0].tolist() == [[0.0]]
            assert qd.hamlst[1].tolist() == [[2.0, 20.0, 0.0, 0.0], [20.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 20.0], [0.0, 0.0, 20.0, 1.0]]
            assert qd.hamlst[2].tolist() == [[33.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 84.0, 20.0, 20.0, 0.0, 0.0], [0.0, 20.0, 33.0, 0.0, 20.0, 0.0], [0.0, 20.0, 0.0, 33.0, 20.0, 0.0], [0.0, 0.0, 20.0, 20.0, 82.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 33.0]]
            assert qd.hamlst[3].tolist() == [[145.0, 20.0, 0.0, 0.0], [20.0, 144.0, 0.0, 0.0], [0.0, 0.0, 145.0, 20.0], [0.0, 0.0, 20.0, 144.0]]
            assert qd.hamlst[4].tolist() == [[286.0]]
        elif indexing == 'sz':
            assert qd.hamlst[0][0].tolist() == [[0.0]]
            assert qd.hamlst[1][0].tolist() == [[2.0, 20.0], [20.0, 1.0]]
            assert qd.hamlst[1][1].tolist() == [[2.0, 20.0], [20.0, 1.0]]
            assert qd.hamlst[2][0].tolist() == [[33.0]]
            assert qd.hamlst[2][1].tolist() == [[84.0, 20.0, 20.0, 0.0], [20.0, 33.0, 0.0, 20.0], [20.0, 0.0, 33.0, 20.0], [0.0, 20.0, 20.0, 82.0]]
            assert qd.hamlst[2][2].tolist() == [[33.0]]
            assert qd.hamlst[3][0].tolist() == [[145.0, 20.0], [20.0, 144.0]]
            assert qd.hamlst[3][1].tolist() == [[145.0, 20.0], [20.0, 144.0]]
            assert qd.hamlst[4][0].tolist() == [[286.0]]
        elif indexing == 'ssq':
            assert qd.hamlst[0][0].tolist() == [[0.0]]
            assert qd.hamlst[1][0].tolist() == [[2.0, 20.0], [20.0, 1.0]]
            assert qd.hamlst[1][1] is None
            assert qd.hamlst[2][0] is None
            assert qd.hamlst[2][1].tolist() == [[84.0, 20.0, 20.0, 0.0], [20.0, 33.0, 0.0, 20.0], [20.0, 0.0, 33.0, 20.0], [0.0, 20.0, 20.0, 82.0]]
            assert qd.hamlst[2][2] is None
            assert qd.hamlst[3][0].tolist() == [[145.0, 20.0], [20.0, 144.0]]
            assert qd.hamlst[3][1] is None
            assert qd.hamlst[4][0].tolist() == [[286.0]]
        #
        if symmetry == 'spin':
            qd.add(hsingle={(0,0):1.23, (0,1):0.77}, coulomb={(0,0,0,0):0.55})
            assert norm(norm(make_hsingle_mtr(qd.hsingle, 4) - make_hsingle_mtr({(0,0):e1+1.23, (2,2):e1+1.23, (1,1):e2, (3,3):e2, (0,1):omega+0.77, (2,3):omega+0.77}, 4))) < EPS
            assert norm(qd.coulomb[(0,2,2,0)] - (uintra+0.55)) < EPS
            qd.change(hsingle={(0,0):1.23, (0,1):0.77}, coulomb={(0,0,0,0):0.55})  # , coulomb={1: 2.13}
            assert norm(norm(make_hsingle_mtr(qd.hsingle, 4) - make_hsingle_mtr({(0,0):1.23, (2,2):1.23, (1,1):e2, (3,3):e2, (0,1):0.77, (2,3):0.77}, 4))) < EPS
            assert norm(qd.coulomb[(0,2,2,0)] - 0.55) < EPS
        else:
            qd.add(hsingle={(0,0):1.23, (2,2):1.23, (0,1):0.77, (2,3):0.77}, coulomb={(0,2,2,0):0.55})
            assert norm(norm(make_hsingle_mtr(qd.hsingle, 4) - make_hsingle_mtr({(0,0):e1+1.23, (2,2):e1+1.23, (1,1):e2, (3,3):e2, (0,1):omega+0.77, (2,3):omega+0.77}, 4))) < EPS
            assert norm(qd.coulomb[(0,2,2,0)] - (uintra+0.55)) < EPS
            qd.change(hsingle={(0,0):1.23, (2,2):1.23, (0,1):0.77, (2,3):0.77}, coulomb={(0,2,2,0):0.55})  # , coulomb={1: 2.13}
            assert norm(norm(make_hsingle_mtr(qd.hsingle, 4) - make_hsingle_mtr({(0,0):1.23, (2,2):1.23, (1,1):e2, (3,3):e2, (0,1):0.77, (2,3):0.77}, 4))) < EPS
            assert norm(qd.coulomb[(0,2,2,0)] - 0.55) < EPS

        qd.diagonalise()
        assert norm(qd.Ea - data1[indexing]) < EPS
        if indexing == 'Lin':
            assert norm(qd.hamlst[0] - [[0.0]]) < EPS
            assert norm(qd.hamlst[1] - [[2.0, 0.77, 0.0, 0.0], [0.77, 1.23, 0.0, 0.0], [0.0, 0.0, 2.0, 0.77], [0.0, 0.0, 0.77, 1.23]]) < EPS
            assert norm(qd.hamlst[2] - [[33.23, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 84.0, 0.77, 0.77, 0.0, 0.0], [0.0, 0.77, 33.23, 0.0, 0.77, 0.0], [0.0, 0.77, 0.0, 33.23, 0.77, 0.0], [0.0, 0.0, 0.77, 0.77, 3.01, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 33.23]]) < EPS
            assert norm(qd.hamlst[3] - [[145.23, 0.77, 0.0, 0.0], [0.77, 65.01, 0.0, 0.0], [0.0, 0.0, 145.23, 0.77], [0.0, 0.0, 0.77, 65.01]]) < EPS
            assert norm(qd.hamlst[4] - [[207.01]]) < EPS
        elif indexing == 'charge':
            assert norm(qd.hamlst[0] - [[0.0]]) < EPS
            assert norm(qd.hamlst[1] - [[2.0, 0.77, 0.0, 0.0], [0.77, 1.23, 0.0, 0.0], [0.0, 0.0, 2.0, 0.77], [0.0, 0.0, 0.77, 1.23]]) < EPS
            assert norm(qd.hamlst[2] - [[33.23, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 84.0, 0.77, 0.77, 0.0, 0.0], [0.0, 0.77, 33.23, 0.0, 0.77, 0.0], [0.0, 0.77, 0.0, 33.23, 0.77, 0.0], [0.0, 0.0, 0.77, 0.77, 3.01, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 33.23]]) < EPS
            assert norm(qd.hamlst[3] - [[145.23, 0.77, 0.0, 0.0], [0.77, 65.01, 0.0, 0.0], [0.0, 0.0, 145.23, 0.77], [0.0, 0.0, 0.77, 65.01]]) < EPS
            assert norm(qd.hamlst[4] - [[207.01]]) < EPS
        elif indexing == 'sz':
            assert norm(qd.hamlst[0][0] - [[0.0]]) < EPS
            assert norm(qd.hamlst[1][0] - [[2.0, 0.77], [0.77, 1.23]]) < EPS
            assert norm(qd.hamlst[1][1] - [[2.0, 0.77], [0.77, 1.23]]) < EPS
            assert norm(qd.hamlst[2][0] - [[33.23]]) < EPS
            assert norm(qd.hamlst[2][1] - [[84.0, 0.77, 0.77, 0.0], [0.77, 33.23, 0.0, 0.77], [0.77, 0.0, 33.23, 0.77], [0.0, 0.77, 0.77, 3.01]]) < EPS
            assert norm(qd.hamlst[2][2] - [[33.23]]) < EPS
            assert norm(qd.hamlst[3][0] - [[145.23, 0.77], [0.77, 65.01]]) < EPS
            assert norm(qd.hamlst[3][1] - [[145.23, 0.77], [0.77, 65.01]]) < EPS
            assert norm(qd.hamlst[4][0] - [[207.01]]) < EPS
        elif indexing == 'ssq':
            assert norm(qd.hamlst[0][0] - [[0.0]]) < EPS
            assert norm(qd.hamlst[1][0] - [[2.0, 0.77], [0.77, 1.23]]) < EPS
            assert qd.hamlst[1][1] is None
            assert qd.hamlst[2][0] is None
            assert norm(qd.hamlst[2][1] - [[84.0, 0.77, 0.77, 0.0], [0.77, 33.23, 0.0, 0.77], [0.77, 0.0, 33.23, 0.77], [0.0, 0.77, 0.77, 3.01]]) < EPS
            assert qd.hamlst[2][2] is None
            assert norm(qd.hamlst[3][0] - [[145.23, 0.77], [0.77, 65.01]]) < EPS
            assert qd.hamlst[3][1] is None
            assert norm(qd.hamlst[4][0] - [[207.01]]) < EPS


def test_QuantumDot_spin():
    test_QuantumDot(symmetry='spin')
