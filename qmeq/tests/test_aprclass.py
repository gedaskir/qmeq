from qmeq.aprclass import *
import qmeq

def test_Approach2vN_kpnt():
    # Check generation of energy grid
    si = qmeq.StateIndexingDMc(1)
    qd = qmeq.QuantumDot({}, {}, si)
    leads = qmeq.LeadsTunneling(1, {}, si, {}, {}, {0: 1000})
    funcp = qmeq.FunctionProperties(kpnt=5, kerntype='2vN')
    tt = Approach2vN(qd, leads, si, funcp)
    assert tt.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    tt.kpnt = 6
    assert tt.Ek_grid.tolist() == [-1000, -600,  -200, 200, 600, 1000]
    #
    system = qmeq.Builder(1, {}, {}, 1, {}, {}, {}, {0: 1000}, kpnt=5, kerntype='2vN')
    assert system.tt.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    system.kpnt = 6
    assert system.tt.Ek_grid.tolist() == [-1000, -600,  -200, 200, 600, 1000]

def test_Approach2vN_make_Ek_grid():
    si = qmeq.StateIndexingDMc(1)
    qd = qmeq.QuantumDot({}, {}, si)
    leads = qmeq.LeadsTunneling(2, {}, si, {}, {}, {0: [-1000, 1000], 1: [-1000, 1000]})
    funcp = qmeq.FunctionProperties(kpnt=5, kerntype='2vN')
    tt = Approach2vN(qd, leads, si, funcp)
    tt.make_Ek_grid()
    assert tt.Ek_grid.tolist() == [-1000, -500, 0, 500, 1000]
    tt.leads.change(dlst={0: [-1400, 1000], 1: [-1000, 1000]})
    tt.make_Ek_grid()
    assert tt.Ek_grid.tolist() == [-1400.0, -800.0, -200.0, 400.0, 1000.0]
