# Author: Rowan Gollan
#
# Demonstrate verification of solution method
# by use of manufactured solution

import numpy
from math import fabs
# from sympy import *


def solvePoisson(xRange, yRange, bcs, source, m, n):
    """Solve Poisson equation with finite-difference method.

    Inputs:
    xRange  : dict with min and max entries for x-range
    yRange  : dict with min and max entries for y-range
    bcs     : dict with north, east, south and west BCs as functions
    source  : source term as function of (x,y)
    m       : number of nodes in i-direction
    n       : number of nodes in j-direction

    Output:
    f       : matrix of solution values (m-rows x n-columns)
    x       : grid of x ordinates
    y       : grid of y ordinates
    """
    dx = (xRange['max']-xRange['min'])/(m-1)
    dy = (yRange['max']-yRange['min'])/(n-1)

    # derived constants that appear as coefficients
    C0 = 1.0/(dx*dx)
    C1 = -2.0/(dx*dx) - 2.0/(dy*dy)
    C2 = 1.0/(dy*dy)

    def ij2p(i, j):
        return i + j*m

    # construct grid points
    x = numpy.linspace(xRange['min'], xRange['max'], m)
    y = numpy.linspace(yRange['min'], yRange['max'], n)

    # 1. Assemble matrix and vector of knowns
    nNodes = m*n
    A = numpy.zeros((nNodes, nNodes))
    b = numpy.zeros((nNodes, 1))
    # 1a. set coefficients for internal node equations
    for i in range(1, m-1):
        for j in range(1, n-1):
            p = ij2p(i, j)
            b[p] = source(x[i], y[j])
            # (i-1,j)
            q = ij2p(i-1, j)
            A[p, q] = C0
            # (i,j)
            q = ij2p(i, j)
            A[p, q] = C1
            # (i+1,j)
            q = ij2p(i+1, j)
            A[p, q] = C0
            # (i,j-1)
            q = ij2p(i, j-1)
            A[p, q] = C2
            # (i,j+1)
            q = ij2p(i, j+1)
            A[p, q] = C2
    # 1b. Set entries on boundaries
    # north
    j = n-1
    for i in range(m):
        p = ij2p(i, j)
        b[p] = bcs['north'](x[i])
        A[p, p] = 1.0
    # east
    i = m-1
    for j in range(1, n-1):
        p = ij2p(i, j)
        b[p] = bcs['east'](y[j])
        A[p, p] = 1.0
    # south
    j = 0
    for i in range(m):
        p = ij2p(i, j)
        b[p] = bcs['south'](x[i])
        A[p, p] = 1.0
    # west
    i = 0
    for j in range(1, n-1):
        p = ij2p(i, j)
        b[p] = bcs['west'](y[j])
        A[p, p] = 1.0

    # 2. Solve the system
    f = numpy.linalg.solve(A, b)
    return numpy.ndarray.reshape(f, (m, n)), x, y


def computeErrorNorms(fNum, fExact, x, y):
    """Compute L1 and Linf norms.

    Inputs:
    fNum    : numerical values a discrete points (x, y)
    fExact  : analytical function of x and y
    x       : grid of x ordinates
    y       : grid of y ordinates

    Outputs:
    L1      : L1 error norm
    Linf    : Linf error norm
    """
    L1 = 0.0
    Linf = 0.0
    nNodes = len(x) * len(y)
    for j in range(len(y)):
        for i in range(len(x)):
            error = fabs(fNum[j, i] - fExact(x[i], y[j]))
            L1 += error
            if (error > Linf):
                Linf = error
    L1 /= nNodes
    return L1, Linf


def main():

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

    print("==============================================================")
    print("   k       m,n      L1           p(L1)    Linf        p(Linf)")
    print("==============================================================")

    L1 = []
    Linf = []

    nDisc = len(discretisations)

    for i, d in enumerate(discretisations):
        m = n = d
        fNum, xs, ys = solvePoisson(xRange, yRange, bcs, source, m, n)
        L1_k, Linf_k = computeErrorNorms(fNum, fExact, xs, ys)
        L1.append(L1_k)
        Linf.append(Linf_k)
        if i == 0:
            print(
                f"   {nDisc-i:d}\t   {m:d}\t    {L1_k:.4e}   --       {Linf_k:.4e}   --")
            continue
        p_L1 = log(L1[i-1]/L1[i])/logR
        p_Linf = log(Linf[i-1]/Linf[i])/logR
        print(
            f"   {nDisc-i:d}\t   {m:d}\t    {L1_k:.4e}   {p_L1:.4f}   {Linf_k:.4e}   {p_Linf:.4f}")

    print("==============================================================")


if __name__ == "__main__":
    main()
