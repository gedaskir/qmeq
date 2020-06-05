from qmeq.approach.aprclass import *
from qmeq.approach.base.neumann2 import *
import qmeq


def test_Approach2vN_kpnt():
    system = qmeq.Builder(nleads=1, dband={0: 1000}, kpnt=5, kerntype='2vN')
    appr = Approach2vN(system)
    appr.make_Ek_grid()
    assert appr.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    appr.funcp.kpnt = 6
    appr.make_Ek_grid()
    assert appr.Ek_grid.tolist() == [-1000, -600,  -200, 200, 600, 1000]
    #
    system = qmeq.Builder(1, {}, {}, 1, {}, {}, {}, {0: 1000}, kpnt=5, kerntype='2vN')
    system.appr.make_Ek_grid()
    assert system.appr.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    system.kpnt = 6
    system.appr.make_Ek_grid()
    assert system.appr.Ek_grid.tolist() == [-1000, -600,  -200, 200, 600, 1000]


def test_Approach2vN_make_Ek_grid():
    system = qmeq.Builder(nleads=2, dband={0: [-1000, 1000], 1: [-1000, 1000]}, kpnt=5, kerntype='2vN')
    appr = Approach2vN(system)
    appr.make_Ek_grid()
    assert appr.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    appr.leads.change(dlst={0: [-1400, 1000], 1: [-1000, 1000]})
    appr.make_Ek_grid()
    assert appr.Ek_grid.tolist() == [-1400.0, -800.0, -200.0, 400.0, 1000.0]
