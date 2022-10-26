import numpy as np

from verification_example import solvePoisson, computeErrorNorms


def test_demo():
    assert 0.1 + 0.1 == 0.2, "Whoops, we've failed adding numbers"


def test_poisson():

    from sympy import symbols, pi, cos, sin, log, diff

    L = 1.0
    # Manufactured solution adapted from JCP (2015)
    x, y, T, S = symbols('x y T S')

    T0   = 350
    Tx   = -10.0
    Ty   = 25.0
    aTx  = 1.5
    aTy  = 1.0
    Ti   = 350.0
    aTx2 = 0.75

    T = T0 + Tx*cos(aTx*pi*x/L) + Ty*cos(aTx2*pi*x/L)*sin(aTy*pi*y/L)
    S = diff(T, x, 2) + diff(T, y, 2)

    def source(x1, y1):
        return S.subs({x: x1, y: y1}).evalf()

    xRange = {'min': 0.0*L, 'max': 1.0*L}
    yRange = {'min': 1.0*L, 'max': 1.5*L}
    bcs = {
        'north': lambda x1: T.subs({x: x1, y: yRange['max']}).evalf(),
        'south': lambda x1: T.subs({x: x1, y: yRange['min']}).evalf(),
        'east' : lambda y1: T.subs({x: xRange['max'], y: y1}).evalf(),
        'west' : lambda y1: T.subs({x: xRange['min'], y: y1}).evalf()
    }

    def fExact(x1, y1):
        return T.subs({x: x1, y: y1}).evalf()

    discretisations = [5, 10, 20, 40]
    r = 2.0
    logR = log(r)

    L1 = []
    Linf = []

    nDisc = len(discretisations)

    prev_pL1 = None
    prev_pLinf = None

    tol = 0.1

    for i, d in enumerate(discretisations):
        m = n = d
        fNum, xs, ys = solvePoisson(xRange, yRange, bcs, source, m, n)
        L1_k, Linf_k = computeErrorNorms(fNum, fExact, xs, ys)
        L1.append(L1_k)
        Linf.append(Linf_k)
        if i == 0:
            continue

        p_L1 = log(L1[i-1]/L1[i])/logR
        p_Linf = log(Linf[i-1]/Linf[i])/logR

        assert np.abs(p_L1 - 2.0) < tol
        assert np.abs(p_Linf - 2.0) < tol
