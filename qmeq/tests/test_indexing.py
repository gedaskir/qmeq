from qmeq.indexing import *


def test_binarylist_to_integer():
    assert binarylist_to_integer([1, 0, 1, 1, 0, 0]) == 44


def test_integer_to_binarylist():
    assert integer_to_binarylist(33) == [1, 0, 0, 0, 0, 1]
    assert integer_to_binarylist(33, strq=True) == '100001'
    assert integer_to_binarylist(33, binlen=8) == [0, 0, 1, 0, 0, 0, 0, 1]
    assert integer_to_binarylist(33, binlen=8, strq=True) == '00100001'


def test_construct_chargelst():
    assert construct_chargelst(4) == [[0],
                                      [1, 2, 4, 8],
                                      [3, 5, 6, 9, 10, 12],
                                      [7, 11, 13, 14],
                                      [15]]


def test_sz_to_ind():
    assert sz_to_ind(-2, 4, 6) == 0
    assert sz_to_ind( 0, 4, 6) == 1
    assert sz_to_ind(+2, 4, 6) == 2


def test_szrange():
    assert szrange(2, 6) == [-2, 0, 2]
    assert szrange(3, 6) == [-3, -1, 1, 3]
    assert szrange(4, 6) == [-2, 0, 2]


def test_empty_szlst():
    assert empty_szlst(4) == [[[]],
                              [[], []],
                              [[], [], []],
                              [[], []],
                              [[]]]
    assert empty_szlst(4, noneq=True) == [[None],
                                          [None, None],
                                          [None, None, None],
                                          [None, None],
                                          [None]]


def test_construct_szlst():
    assert construct_szlst(4) == [[[0]],
                                  [[1, 2], [4, 8]],
                                  [[3], [5, 6, 9, 10], [12]],
                                  [[7, 11], [13, 14]],
                                  [[15]]]


def test_ssq_to_ind():
    assert ssq_to_ind(2, -2) == 0
    assert ssq_to_ind(2,  0) == 1
    assert ssq_to_ind(2, +2) == 0


def test_ssqrange():
    assert ssqrange(3, 1, 6) == [1, 3]
    assert ssqrange(4, 0, 6) == [0, 2]


def test_empty_ssqlst():
    assert empty_ssqlst(4) == [[[[]]],
                               [[[]], [[]]],
                               [[[]], [[], []], [[]]],
                               [[[]], [[]]],
                               [[[]]]]
    assert empty_ssqlst(4, noneq=True) == [[[None]],
                                           [[None], [None]],
                                           [[None], [None, None], [None]],
                                           [[None], [None]],
                                           [[None]]]


def tezt_construct_ssqlst():
    szlst = construct_szlst(4)
    assert construct_ssqlst(szlst, 4) == [[[[0]]],
                                          [[[1, 2]], [[3, 4]]],
                                          [[[5]], [[6, 7, 8], [9]], [[10]]],
                                          [[[11, 12]], [[13, 14]]],
                                          [[[15]]]]


def test_flatten():
    szlst = construct_szlst(4)
    ssqlst = construct_ssqlst(szlst, 4)
    f1 = flatten(ssqlst)
    f2 = flatten(f1)
    f3 = flatten(f2)
    assert f1 == [[[0]], [[1, 2]], [[3, 4]], [[5]], [[6, 7, 8], [9]], [[10]], [[11, 12]], [[13, 14]], [[15]]]
    assert f2 == [[0], [1, 2], [3, 4], [5], [6, 7, 8], [9], [10], [11, 12], [13, 14], [15]]
    assert f3 == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def test_enum_chargelst():
    chargelst_lin = construct_chargelst(4)
    assert enum_chargelst(chargelst_lin) == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15]]


def test_enum_szlst():
    szlst_lin = construct_szlst(4)
    assert enum_szlst(szlst_lin) == [[[0]], [[1, 2], [3, 4]], [[5], [6, 7, 8, 9], [10]], [[11, 12], [13, 14]], [[15]]]


def test_make_inverse_map():
    chargelst_lin = construct_chargelst(4)
    i = flatten(chargelst_lin)
    assert make_inverse_map(i) == [0, 1, 2, 5, 3, 6, 7, 11, 4, 8, 9, 12, 10, 13, 14, 15]


def test_make_quantum_numbers():
    si = StateIndexing(4, indexing='Lin')
    qn_ind, ind_qn = make_quantum_numbers(si)
    assert qn_ind == {(1, 2): 4, (2, 5): 12, (0, 0): 0, (3, 3): 14, (3, 0): 7, (3, 1): 11, (3, 2): 13, (2, 1): 5, (2, 4): 10, (2, 0): 3, (1, 3): 8, (2, 3): 9, (2, 2): 6, (1, 0): 1, (1, 1): 2, (4, 0): 15}
    assert ind_qn == {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (2, 0), 4: (1, 2), 5: (2, 1), 6: (2, 2), 7: (3, 0), 8: (1, 3), 9: (2, 3), 10: (2, 4), 11: (3, 1), 12: (2, 5), 13: (3, 2), 14: (3, 3), 15: (4, 0)}
    #
    si = StateIndexing(4, indexing='charge')
    qn_ind, ind_qn = make_quantum_numbers(si)
    assert qn_ind == {(1, 2): 3, (2, 5): 10, (0, 0): 0, (3, 3): 14, (3, 0): 11, (3, 1): 12, (3, 2): 13, (2, 1): 6, (2, 4): 9, (2, 0): 5, (1, 3): 4, (2, 3): 8, (2, 2): 7, (1, 0): 1, (1, 1): 2, (4, 0): 15}
    assert ind_qn == {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (1, 2), 4: (1, 3), 5: (2, 0), 6: (2, 1), 7: (2, 2), 8: (2, 3), 9: (2, 4), 10: (2, 5), 11: (3, 0), 12: (3, 1), 13: (3, 2), 14: (3, 3), 15: (4, 0)}
    #
    si = StateIndexing(4, indexing='sz')
    qn_ind, ind_qn = make_quantum_numbers(si)
    assert qn_ind == {(3, -1, 1): 12, (1, 1, 0): 3, (2, -2, 0): 5, (2, 0, 3): 9, (4, 0, 0): 15, (2, 0, 2): 8, (1, -1, 0): 1, (2, 2, 0): 10, (3, 1, 0): 13, (0, 0, 0): 0, (1, -1, 1): 2, (2, 0, 1): 7, (3, 1, 1): 14, (3, -1, 0): 11, (1, 1, 1): 4, (2, 0, 0): 6}
    assert ind_qn == {0: (0, 0, 0), 1: (1, -1, 0), 2: (1, -1, 1), 3: (1, 1, 0), 4: (1, 1, 1), 5: (2, -2, 0), 6: (2, 0, 0), 7: (2, 0, 1), 8: (2, 0, 2), 9: (2, 0, 3), 10: (2, 2, 0), 11: (3, -1, 0), 12: (3, -1, 1), 13: (3, 1, 0), 14: (3, 1, 1), 15: (4, 0, 0)}
    #
    si = StateIndexing(4, indexing='ssq')
    qn_ind, ind_qn = make_quantum_numbers(si)
    assert qn_ind == {(0, 0, 0, 0): 0,
                      (1, -1, 1, 0): 1,
                      (1, -1, 1, 1): 2,
                      (1, 1, 1, 0): 3,
                      (1, 1, 1, 1): 4,
                      (2, -2, 2, 0): 5,
                      (2, 0, 0, 0): 6,
                      (2, 0, 0, 1): 7,
                      (2, 0, 0, 2): 8,
                      (2, 0, 2, 0): 9,
                      (2, 2, 2, 0): 10,
                      (3, -1, 1, 0): 11,
                      (3, -1, 1, 1): 12,
                      (3, 1, 1, 0): 13,
                      (3, 1, 1, 1): 14,
                      (4, 0, 0, 0): 15}
    assert ind_qn == {0: (0, 0, 0, 0),
                      1: (1, -1, 1, 0),
                      2: (1, -1, 1, 1),
                      3: (1, 1, 1, 0),
                      4: (1, 1, 1, 1),
                      5: (2, -2, 2, 0),
                      6: (2, 0, 0, 0),
                      7: (2, 0, 0, 1),
                      8: (2, 0, 0, 2),
                      9: (2, 0, 2, 0),
                      10: (2, 2, 2, 0),
                      11: (3, -1, 1, 0),
                      12: (3, -1, 1, 1),
                      13: (3, 1, 1, 0),
                      14: (3, 1, 1, 1),
                      15: (4, 0, 0, 0)}


def test_StateIndexing():
    si = StateIndexing(4)
    assert si.nsingle == 4
    assert si.indexing == 'Lin'
    assert si.ncharge == 5
    assert si.nmany == 16
    assert si.nleads == 0
    #
    for indexing in ['Lin', None]:
        si = StateIndexing(4, indexing=indexing)
        assert si.chargelst == [[0],[1, 2, 4, 8],[3, 5, 6, 9, 10, 12],[7, 11, 13, 14],[15]]
        assert si.i == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        assert si.j == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    #
    si = StateIndexing(4, indexing='charge')
    assert si.chargelst_lin == [[0], [1, 2, 4, 8], [3, 5, 6, 9, 10, 12], [7, 11, 13, 14], [15]]
    assert si.chargelst == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15]]
    assert si.i == [0, 1, 2, 4, 8, 3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15]
    assert si.j == [0, 1, 2, 5, 3, 6, 7, 11, 4, 8, 9, 12, 10, 13, 14, 15]
    #
    for indexing in ['sz', 'ssq']:
        si = StateIndexing(4, indexing=indexing)
        assert si.chargelst_lin == [[0], [1, 2, 4, 8], [3, 5, 6, 9, 10, 12], [7, 11, 13, 14], [15]]
        assert si.chargelst == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15]]
        assert si.szlst_lin == [[[0]], [[1, 2], [4, 8]], [[3], [5, 6, 9, 10], [12]], [[7, 11], [13, 14]], [[15]]]
        assert si.szlst == [[[0]], [[1, 2], [3, 4]], [[5], [6, 7, 8, 9], [10]], [[11, 12], [13, 14]], [[15]]]
        assert si.i == [0, 1, 2, 4, 8, 3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15]
        assert si.j == [0, 1, 2, 5, 3, 6, 7, 11, 4, 8, 9, 12, 10, 13, 14, 15]
    #
    si = StateIndexing(4, indexing='ssq')
    assert si.ssqlst == [[[[0]]], [[[1, 2]], [[3, 4]]], [[[5]], [[6, 7, 8], [9]], [[10]]], [[[11, 12]], [[13, 14]]], [[[15]]]]
    assert si.get_state(6) == [0, 1, 0, 1]
    assert si.get_state(6, linq=True) == [0, 1, 1, 0]
    assert si.get_state(6, strq=True) == '0101'
    assert si.get_state(6, linq=True, strq=True) == '0110'
    assert si.get_ind([0, 1, 0, 1]) == 6
    assert si.get_ind([0, 1, 1, 0], linq=True) == 6
    assert si.get_lst(charge=2) == [5, 6, 7, 8, 9, 10]
    assert si.get_lst(charge=2, sz=0) == [6, 7, 8, 9]
    assert si.get_lst(charge=2, sz=0, ssq=0) == [6, 7, 8]


def test_StateIndexingPauli_charge():
    si = StateIndexingPauli(4, indexing='charge')
    assert si.npauli_ == 16
    assert si.npauli == 16
    assert list(si.shiftlst0) == [0, 1, 5, 11, 15, 16]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15], []]
    assert list(si.mapdm0) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert list(si.booldm0) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert si.get_ind_dm0(8, 8, 2, maptype=0) == 8
    assert si.get_ind_dm0(8, 8, 2, maptype=1) == 8
    assert si.get_ind_dm0(8, 8, 2, maptype=2) == 1
    #
    si.set_statesdm([[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], []])
    assert si.npauli_ == 11
    assert si.npauli == 11
    assert list(si.shiftlst0) == [0, 1, 1, 7, 11, 11]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [], []]
    assert list(si.mapdm0) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert list(si.booldm0) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert si.get_ind_dm0(8, 8, 2, maptype=0) == 4
    assert si.get_ind_dm0(8, 8, 2, maptype=1) == 4
    assert si.get_ind_dm0(8, 8, 2, maptype=2) == 1


def test_StateIndexingPauli_ssq():
    si = StateIndexingPauli(4, indexing='ssq')
    assert si.npauli_ == 16
    assert si.npauli == 10
    assert list(si.shiftlst0) == [0, 1, 5, 11, 15, 16]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15], []]
    assert list(si.mapdm0) == [0, 1, 2, 1, 2, 3, 4, 5, 6, 3, 3, 7, 8, 7, 8, 9]
    assert list(si.booldm0) == [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    assert si.get_ind_dm0(8, 8, 2, maptype=0) == 8
    assert si.get_ind_dm0(8, 8, 2, maptype=1) == 6
    assert si.get_ind_dm0(8, 8, 2, maptype=2) == 1
    #
    si.set_statesdm([[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], []])
    assert si.npauli_ == 11
    assert si.npauli == 7
    assert list(si.shiftlst0) == [0, 1, 1, 7, 11, 11]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [], []]
    assert list(si.mapdm0) == [0, 1, 2, 3, 4, 1, 1, 5, 6, 5, 6]
    assert list(si.booldm0) == [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
    assert si.get_ind_dm0(8, 8, 2, maptype=0) == 4
    assert si.get_ind_dm0(8, 8, 2, maptype=1) == 4
    assert si.get_ind_dm0(8, 8, 2, maptype=2) == 1


def test_StateIndexingDM_charge():
    si = StateIndexingDM(4, indexing='charge')
    assert si.ndm0_tot == 70
    assert si.ndm0_  == 70
    assert si.ndm0 == 43
    assert si.ndm0r == 70
    assert si.npauli_ == 16
    assert si.npauli == 16
    assert si.ndm1_tot == 56
    assert si.ndm1_ == 56
    assert si.ndm1 == 56
    assert list(si.shiftlst0) == [0, 1, 17, 53, 69, 70]
    assert list(si.shiftlst1) == [0, 4, 28, 52, 56]
    assert list(si.lenlst) == [1, 4, 6, 4, 1]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15], []]
    assert list(si.mapdm0) == [0, 1, 16, 17, 18, 16, 2, 19, 20, 17, 19, 3, 21, 18, 20, 21, 4, 5, 22, 23, 24, 25, 26, 22, 6, 27, 28, 29, 30, 23, 27, 7, 31, 32, 33, 24, 28, 31, 8, 34, 35, 25, 29, 32, 34, 9, 36, 26, 30, 33, 35, 36, 10, 11, 37, 38, 39, 37, 12, 40, 41, 38, 40, 13, 42, 39, 41, 42, 14, 15]
    assert si.inddm0 == {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (3, 3), 4: (4, 4), 5: (5, 5), 6: (6, 6), 7: (7, 7), 8: (8, 8), 9: (9, 9), 10: (10, 10), 11: (11, 11), 12: (12, 12), 13: (13, 13), 14: (14, 14), 15: (15, 15), 16: (1, 2), 17: (1, 3), 18: (1, 4), 19: (2, 3), 20: (2, 4), 21: (3, 4), 22: (5, 6), 23: (5, 7), 24: (5, 8), 25: (5, 9), 26: (5, 10), 27: (6, 7), 28: (6, 8), 29: (6, 9), 30: (6, 10), 31: (7, 8), 32: (7, 9), 33: (7, 10), 34: (8, 9), 35: (8, 10), 36: (9, 10), 37: (11, 12), 38: (11, 13), 39: (11, 14), 40: (12, 13), 41: (12, 14), 42: (13, 14)}
    assert list(si.booldm0) == [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    assert list(si.conjdm0) == [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    assert si.get_ind_dm0(7, 8, 2, maptype=0) == 32
    assert si.get_ind_dm0(7, 8, 2, maptype=1) == 31
    assert si.get_ind_dm0(7, 8, 2, maptype=2) == 1
    assert si.get_ind_dm0(7, 8, 2, maptype=3) == 1
    assert si.get_ind_dm0(8, 7, 2, maptype=2) == 0
    assert si.get_ind_dm0(8, 7, 2, maptype=3) == 0
    assert si.get_ind_dm0(5, 8, 2, maptype=1) == 24
    assert si.get_ind_dm1(5, 4, 1) == 7
    #
    si.set_statesdm([[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], []])
    assert si.ndm0_  == 53
    assert si.ndm0 == 32
    assert si.ndm0r == 53
    assert si.npauli_ == 11
    assert si.npauli == 11
    assert si.ndm1_ == 24
    assert si.ndm1 == 24
    assert list(si.shiftlst0) == [0, 1, 1, 37, 53, 53]
    assert list(si.shiftlst1) == [0, 0, 0, 24, 24]
    assert list(si.lenlst) == [1, 0, 6, 4, 0]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [], []]
    assert list(si.mapdm0) == [0, 1, 11, 12, 13, 14, 15, 11, 2, 16, 17, 18, 19, 12, 16, 3, 20, 21, 22, 13, 17, 20, 4, 23, 24, 14, 18, 21, 23, 5, 25, 15, 19, 22, 24, 25, 6, 7, 26, 27, 28, 26, 8, 29, 30, 27, 29, 9, 31, 28, 30, 31, 10]
    assert si.inddm0 == {0: (0, 0), 1: (5, 5), 2: (6, 6), 3: (7, 7), 4: (8, 8), 5: (9, 9), 6: (10, 10), 7: (11, 11), 8: (12, 12), 9: (13, 13), 10: (14, 14), 11: (5, 6), 12: (5, 7), 13: (5, 8), 14: (5, 9), 15: (5, 10), 16: (6, 7), 17: (6, 8), 18: (6, 9), 19: (6, 10), 20: (7, 8), 21: (7, 9), 22: (7, 10), 23: (8, 9), 24: (8, 10), 25: (9, 10), 26: (11, 12), 27: (11, 13), 28: (11, 14), 29: (12, 13), 30: (12, 14), 31: (13, 14)}
    assert list(si.booldm0) == [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1]
    assert list(si.conjdm0) == [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1]
    assert si.get_ind_dm0(7, 8, 2, maptype=0) == 16
    assert si.get_ind_dm0(7, 8, 2, maptype=1) == 20
    assert si.get_ind_dm0(7, 8, 2, maptype=2) == 1
    assert si.get_ind_dm0(7, 8, 2, maptype=3) == 1
    assert si.get_ind_dm0(8, 7, 2, maptype=2) == 0
    assert si.get_ind_dm0(8, 7, 2, maptype=3) == 0
    assert si.get_ind_dm0(5, 8, 2, maptype=1) == 13
    assert si.get_ind_dm1(5, 4, 1) == 3


def test_StateIndexingDM_ssq():
    si = StateIndexingDM(4, indexing='ssq')
    assert si.ndm0_tot == 70
    assert si.ndm0_  == 70
    assert si.ndm0 == 15
    assert si.ndm0r == 20
    assert si.npauli_ == 16
    assert si.npauli == 10
    assert si.ndm1_tot == 56
    assert si.ndm1_ == 56
    assert si.ndm1 == 56
    assert list(si.shiftlst0) == [0, 1, 17, 53, 69, 70]
    assert list(si.shiftlst1) == [0, 4, 28, 52, 56]
    assert list(si.lenlst) == [1, 4, 6, 4, 1]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15], []]
    assert list(si.mapdm0) == [0, 1, 10, -1, -1, 10, 2, -1, -1, -1, -1, 1, 10, -1, -1, 10, 2, 3, -1, -1, -1, -1, -1, -1, 4, 11, 12, -1, -1, -1, 11, 5, 13, -1, -1, -1, 12, 13, 6, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, 3, 7, 14, -1, -1, 14, 8, -1, -1, -1, -1, 7, 14, -1, -1, 14, 8, 9]
    assert si.inddm0 == {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (5, 5), 4: (6, 6), 5: (7, 7), 6: (8, 8), 7: (11, 11), 8: (12, 12), 9: (15, 15), 10: (1, 2), 11: (6, 7), 12: (6, 8), 13: (7, 8), 14: (11, 12)}
    assert list(si.booldm0) == [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert list(si.conjdm0) == [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    assert si.get_ind_dm0(7, 8, 2, maptype=0) == 32
    assert si.get_ind_dm0(7, 8, 2, maptype=1) == 13
    assert si.get_ind_dm0(7, 8, 2, maptype=2) == 1
    assert si.get_ind_dm0(7, 8, 2, maptype=3) == 1
    assert si.get_ind_dm0(8, 7, 2, maptype=2) == 0
    assert si.get_ind_dm0(8, 7, 2, maptype=3) == 0
    assert si.get_ind_dm0(5, 8, 2, maptype=1) == -1
    assert si.get_ind_dm1(5, 4, 1) == 7
    #
    si.set_statesdm([[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], []])
    assert si.ndm0_  == 53
    assert si.ndm0 == 11
    assert si.ndm0r == 15
    assert si.npauli_ == 11
    assert si.npauli == 7
    assert si.ndm1_ == 24
    assert si.ndm1 == 24
    assert list(si.shiftlst0) == [0, 1, 1, 37, 53, 53]
    assert list(si.shiftlst1) == [0, 0, 0, 24, 24]
    assert list(si.lenlst) == [1, 0, 6, 4, 0]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [], []]
    assert list(si.mapdm0) == [0, 1, -1, -1, -1, -1, -1, -1, 2, 7, 8, -1, -1, -1, 7, 3, 9, -1, -1, -1, 8, 9, 4, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 5, 10, -1, -1, 10, 6, -1, -1, -1, -1, 5, 10, -1, -1, 10, 6]
    assert si.inddm0 == {0: (0, 0), 1: (5, 5), 2: (6, 6), 3: (7, 7), 4: (8, 8), 5: (11, 11), 6: (12, 12), 7: (6, 7), 8: (6, 8), 9: (7, 8), 10: (11, 12)}
    assert list(si.booldm0) == [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert list(si.conjdm0) == [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]
    assert si.get_ind_dm0(7, 8, 2, maptype=0) == 16
    assert si.get_ind_dm0(7, 8, 2, maptype=1) == 9
    assert si.get_ind_dm0(7, 8, 2, maptype=2) == 1
    assert si.get_ind_dm0(7, 8, 2, maptype=3) == 1
    assert si.get_ind_dm0(8, 7, 2, maptype=2) == 0
    assert si.get_ind_dm0(8, 7, 2, maptype=3) == 0
    assert si.get_ind_dm0(5, 8, 2, maptype=1) == -1
    assert si.get_ind_dm1(5, 4, 1) == 3


def test_StateIndexingDMc_charge():
    si = StateIndexingDMc(4, indexing='charge')
    assert si.ndm0_tot == 70
    assert si.ndm0_  == 70
    assert si.ndm0 == 70
    assert si.ndm0r == 124
    assert si.npauli_ == 16
    assert si.npauli == 16
    assert si.ndm1_tot == 56
    assert si.ndm1_ == 56
    assert si.ndm1 == 56
    assert list(si.shiftlst0) == [0, 1, 17, 53, 69, 70]
    assert list(si.shiftlst1) == [0, 4, 28, 52, 56]
    assert list(si.lenlst) == [1, 4, 6, 4, 1]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15], []]
    assert list(si.mapdm0) == [0, 1, 16, 17, 18, 19, 2, 20, 21, 22, 23, 3, 24, 25, 26, 27, 4, 5, 28, 29, 30, 31, 32, 33, 6, 34, 35, 36, 37, 38, 39, 7, 40, 41, 42, 43, 44, 45, 8, 46, 47, 48, 49, 50, 51, 9, 52, 53, 54, 55, 56, 57, 10, 11, 58, 59, 60, 61, 12, 62, 63, 64, 65, 13, 66, 67, 68, 69, 14, 15]
    assert si.inddm0 == {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (3, 3), 4: (4, 4), 5: (5, 5), 6: (6, 6), 7: (7, 7), 8: (8, 8), 9: (9, 9), 10: (10, 10), 11: (11, 11), 12: (12, 12), 13: (13, 13), 14: (14, 14), 15: (15, 15), 16: (1, 2), 17: (1, 3), 18: (1, 4), 19: (2, 1), 20: (2, 3), 21: (2, 4), 22: (3, 1), 23: (3, 2), 24: (3, 4), 25: (4, 1), 26: (4, 2), 27: (4, 3), 28: (5, 6), 29: (5, 7), 30: (5, 8), 31: (5, 9), 32: (5, 10), 33: (6, 5), 34: (6, 7), 35: (6, 8), 36: (6, 9), 37: (6, 10), 38: (7, 5), 39: (7, 6), 40: (7, 8), 41: (7, 9), 42: (7, 10), 43: (8, 5), 44: (8, 6), 45: (8, 7), 46: (8, 9), 47: (8, 10), 48: (9, 5), 49: (9, 6), 50: (9, 7), 51: (9, 8), 52: (9, 10), 53: (10, 5), 54: (10, 6), 55: (10, 7), 56: (10, 8), 57: (10, 9), 58: (11, 12), 59: (11, 13), 60: (11, 14), 61: (12, 11), 62: (12, 13), 63: (12, 14), 64: (13, 11), 65: (13, 12), 66: (13, 14), 67: (14, 11), 68: (14, 12), 69: (14, 13)}
    assert list(si.booldm0) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert si.get_ind_dm0(7, 8, 2, maptype=0) == 32
    assert si.get_ind_dm0(7, 8, 2, maptype=1) == 40
    assert si.get_ind_dm0(7, 8, 2, maptype=2) == True
    assert si.get_ind_dm0(8, 7, 2, maptype=2) == True
    assert si.get_ind_dm0(5, 8, 2, maptype=1) == 30
    assert si.get_ind_dm1(5, 4, 1) == 7
    #
    si.set_statesdm([[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], []])
    assert si.ndm0_  == 53
    assert si.ndm0 == 53
    assert si.ndm0r == 95
    assert si.npauli_ == 11
    assert si.npauli == 11
    assert si.ndm1_ == 24
    assert si.ndm1 == 24
    assert list(si.shiftlst0) == [0, 1, 1, 37, 53, 53]
    assert list(si.shiftlst1) == [0, 0, 0, 24, 24]
    assert list(si.lenlst) == [1, 0, 6, 4, 0]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [], []]
    assert list(si.mapdm0) == [0, 1, 11, 12, 13, 14, 15, 16, 2, 17, 18, 19, 20, 21, 22, 3, 23, 24, 25, 26, 27, 28, 4, 29, 30, 31, 32, 33, 34, 5, 35, 36, 37, 38, 39, 40, 6, 7, 41, 42, 43, 44, 8, 45, 46, 47, 48, 9, 49, 50, 51, 52, 10]
    assert si.inddm0 == {0: (0, 0), 1: (5, 5), 2: (6, 6), 3: (7, 7), 4: (8, 8), 5: (9, 9), 6: (10, 10), 7: (11, 11), 8: (12, 12), 9: (13, 13), 10: (14, 14), 11: (5, 6), 12: (5, 7), 13: (5, 8), 14: (5, 9), 15: (5, 10), 16: (6, 5), 17: (6, 7), 18: (6, 8), 19: (6, 9), 20: (6, 10), 21: (7, 5), 22: (7, 6), 23: (7, 8), 24: (7, 9), 25: (7, 10), 26: (8, 5), 27: (8, 6), 28: (8, 7), 29: (8, 9), 30: (8, 10), 31: (9, 5), 32: (9, 6), 33: (9, 7), 34: (9, 8), 35: (9, 10), 36: (10, 5), 37: (10, 6), 38: (10, 7), 39: (10, 8), 40: (10, 9), 41: (11, 12), 42: (11, 13), 43: (11, 14), 44: (12, 11), 45: (12, 13), 46: (12, 14), 47: (13, 11), 48: (13, 12), 49: (13, 14), 50: (14, 11), 51: (14, 12), 52: (14, 13)}
    assert list(si.booldm0) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert si.get_ind_dm0(7, 8, 2, maptype=0) == 16
    assert si.get_ind_dm0(7, 8, 2, maptype=1) == 23
    assert si.get_ind_dm0(7, 8, 2, maptype=2) == 1
    assert si.get_ind_dm0(8, 7, 2, maptype=2) == 1
    assert si.get_ind_dm0(5, 8, 2, maptype=1) == 13
    assert si.get_ind_dm1(5, 4, 1) == 3


def test_StateIndexingDMc_ssq():
    si = StateIndexingDMc(4, indexing='ssq')
    assert si.ndm0_tot == 70
    assert si.ndm0_  == 70
    assert si.ndm0 == 20
    assert si.ndm0r == 30
    assert si.npauli_ == 16
    assert si.npauli == 10
    assert si.ndm1_tot == 56
    assert si.ndm1_ == 56
    assert si.ndm1 == 56
    assert list(si.shiftlst0) == [0, 1, 17, 53, 69, 70]
    assert list(si.shiftlst1) == [0, 4, 28, 52, 56]
    assert list(si.lenlst) == [1, 4, 6, 4, 1]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [15], []]
    assert list(si.mapdm0) == [0, 1, 10, -1, -1, 11, 2, -1, -1, -1, -1, 1, 10, -1, -1, 11, 2, 3, -1, -1, -1, -1, -1, -1, 4, 12, 13, -1, -1, -1, 14, 5, 15, -1, -1, -1, 16, 17, 6, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, 3, 7, 18, -1, -1, 19, 8, -1, -1, -1, -1, 7, 18, -1, -1, 19, 8, 9]
    assert si.inddm0 == {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (5, 5), 4: (6, 6), 5: (7, 7), 6: (8, 8), 7: (11, 11), 8: (12, 12), 9: (15, 15), 10: (1, 2), 11: (2, 1), 12: (6, 7), 13: (6, 8), 14: (7, 6), 15: (7, 8), 16: (8, 6), 17: (8, 7), 18: (11, 12), 19: (12, 11)}
    assert list(si.booldm0) == [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert si.get_ind_dm0(7, 8, 2, maptype=0) == 32
    assert si.get_ind_dm0(7, 8, 2, maptype=1) == 15
    assert si.get_ind_dm0(7, 8, 2, maptype=2) == 1
    assert si.get_ind_dm0(8, 7, 2, maptype=2) == 1
    assert si.get_ind_dm0(5, 8, 2, maptype=1) == -1
    assert si.get_ind_dm1(5, 4, 1) == 7
    #
    si.set_statesdm([[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], []])
    assert si.ndm0_  == 53
    assert si.ndm0 == 15
    assert si.ndm0r == 23
    assert si.npauli_ == 11
    assert si.npauli == 7
    assert si.ndm1_ == 24
    assert si.ndm1 == 24
    assert list(si.shiftlst0) == [0, 1, 1, 37, 53, 53]
    assert list(si.shiftlst1) == [0, 0, 0, 24, 24]
    assert list(si.lenlst) == [1, 0, 6, 4, 0]
    assert list(si.dictdm) == [0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0]
    assert si.statesdm == [[0], [], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14], [], []]
    assert list(si.mapdm0) == [0, 1, -1, -1, -1, -1, -1, -1, 2, 7, 8, -1, -1, -1, 9, 3, 10, -1, -1, -1, 11, 12, 4, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 5, 13, -1, -1, 14, 6, -1, -1, -1, -1, 5, 13, -1, -1, 14, 6]
    assert si.inddm0 == {0: (0, 0), 1: (5, 5), 2: (6, 6), 3: (7, 7), 4: (8, 8), 5: (11, 11), 6: (12, 12), 7: (6, 7), 8: (6, 8), 9: (7, 6), 10: (7, 8), 11: (8, 6), 12: (8, 7), 13: (11, 12), 14: (12, 11)}
    assert list(si.booldm0) == [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert si.get_ind_dm0(7, 8, 2, maptype=0) == 16
    assert si.get_ind_dm0(7, 8, 2, maptype=1) == 10
    assert si.get_ind_dm0(7, 8, 2, maptype=2) == 1
    assert si.get_ind_dm0(8, 7, 2, maptype=2) == 1
    assert si.get_ind_dm0(5, 8, 2, maptype=1) == -1
    assert si.get_ind_dm1(5, 4, 1) == 3
