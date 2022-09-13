# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 12:26:32 2022

@author: Declan
"""
import numpy as np
from numpy import inner as ni
import matplotlib.pyplot as plt
import math


def Question_1 ():
    #Set initial guess of y = [1,1,1,1]
    Y_n = np.ones((4, 1))
    #Arbitary Value of functions to initiate condition of while loop
    f1, f2, f3, f4 = 1,1,1,1
    
    #This statement iterates over the absolute value of each of the f values.
    #It then compares the maximum value to 10E-6, the specified max absolute error
    #If the largest error is above max allowable error, the loop runs, otherwise it will not
    while max(abs(f) for f in [f1,f2,f3,f4]) > 10**-6:
        #Set y variable to current initial variables
        y1,y2,y3,y4 = float(Y_n[0]), float(Y_n[1]), float(Y_n[2]), float(Y_n[3])
        
        #Function Equations
        f1 = 0.04*y1**2 - 2*y1/0.2 + y2/0.4
        f2 = 0.16*y2**2 - 2*y2/0.4 + (y3-y1)/0.4
        f3 = 0.36*y3**2 - 2*y3/0.6 + (y4-y2)/0.4
        f4 = 0.64*y4**2 - 2*y4/0.8 + (1-y3)/0.4

        #Function Equation
        F_y = np.vstack((f1, f2, f3, f4))
        
        #Jacobian
        J = np.array([[0.08*y1 - 10, 2.5, 0, 0],
                     [-2.5, 0.32*y2 - 5, 2.5, 0],
                     [0, -2.5, 0.72*y3 - 2/0.6, 2.5],
                     [0, 0, -2.5, 1.28*y4 -2.5]])
        
        #Inverse of Jacobain
        J_inv = np.linalg.inv(J)
        
        #Newtons method to calculate Yn+1
        Y_n1 = Y_n - np.dot(J_inv, F_y)
        
        #Setting Yn+1 to be Yn for the next iteration
        Y_n = Y_n1
    
    # A list of the X values used as points in the systems mesh     
    X = [0.2, 0.4, 0.6, 0.8]
    Y_an = [] #empty list of analytical y solutions
    #The following for loop determines the analytical y solution for each x value
    for x in X:
        y_an = x**2/(0.2*x**5 + 0.8)
        Y_an.append(y_an)
    
    #Plotting the Newton Method results against the Analytical Method results
    fig, ax = plt.subplots()
    ax.plot (X, Y_n, "g", label = "Newtons Method")
    ax.plot (X, Y_an, "r", label = "Analytical Solution")
    ax.legend()
    plt.xlabel("Position x mm")
    plt.ylabel("Y(x)")
    
    plt.show()
    
    return (Y_n)


def Question_2():
    #Initial data from question
    y = [12.02,18.60,23.72,28.78,34.35,38.88,39.29,44.65,43.16,48.39,48.89,47.156,48.35,50.62,51.09]
    t = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    
    #Creating lists of y and t to the power of N for use in least square
    y0 = [number ** 0 for number in y]
    y1 = y
    y2 = [number ** 2 for number in y]
    y3 = [number ** 3 for number in y]
   
    t0 = [number ** 0 for number in t]
    t1 = t
    t2 = [number ** 2 for number in t]
    t3 = [number ** 3 for number in t]
  
  
    #creating array for N=2, using inner products
    N2 = np.array([[ni(t0,t0), ni(t0,t1)],
                   [ni(t1,t0), ni(t1,t1)]])                                  
                  
    #Creating Fucntion vector of N=2, using inner products
    F2 = np.vstack((ni(t0,y), ni(t1,y)))
    
    #Determining the coefficients of N=2 polynomial
    A2 = np.linalg.inv(N2).dot(F2)
    
    #N=2 data fit polynomial fucntion
    Y2 = [A2[1]*t + A2[0] for t in range (1,16)]
    
    
    #creating array for N=4, using inner products
    N4 = np.array([[ni(t0,t0), ni(t0,t1), ni(t0,t2), ni(t0,t3)],
                   [ni(t1,t0), ni(t1,t1), ni(t1,t2), ni(t1,t3)],                                  
                   [ni(t2,t0), ni(t2,t1), ni(t2,t2), ni(t2,t3)],
                   [ni(t3,t0), ni(t3,t1), ni(t3,t2), ni(t3,t3)]])
    
    #Creating Fucntion vector of N=4, using inner products
    F4 = np.vstack((ni(t0,y), ni(t1,y), ni(t2,y), ni(t3,y)))
    
    #Determining the coefficients of N=4 polynomial
    A4 = np.linalg.inv(N4).dot(F4)
    
    #N=4 data fit polynomial fucntion
    Y4 = [A4[3]*t**3 + A4[2]*t**2 + A4[1]*t + A4[0]  for t in range (1,16)]
    
    
    #Generating list of emperical results
    v = [(9.8*68.1)/12*(t/(3.75+t)) for t in range (1,16)]
    
    #creating array for N=2, using inner products
    V2 = np.array([[ni(y0,y0), ni(y0,y1)],
                   [ni(y1,y0), ni(y1,y1)]])
    
    #Creating function vector of emperical results, N = 2
    FV = np.vstack((ni(y0,v), ni(y1,v)))
    
    #Determining coefficients of emperical results, N = 2
    AV = np.linalg.inv(V2).dot(FV)
    
    #N=4 data fit polynomial fucntion for emperical results
    YV = [AV[1]*t + AV[0] for t in range (1,16)]
    
    #Printing the equation of the N=2 polynomial least squares data fit for
    #purpose of discussion
    print(f"{AV[1]}*t + {AV[0]}")
    
    #Plotting Data for part A
    fig, ax = plt.subplots()
    ax.plot(t, Y2, "r", label = "N = 2 Fit")
    ax.plot(t, Y4, "g", label = "N = 4 Fit")
    ax.scatter(t,y, label = "Measured Data")
    ax.legend()
    plt.title("Question 2 Part A")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.show()
    
    #Plotting Data for part B
    fig, ax = plt.subplots()
    ax.scatter(y,v,)
    ax.plot(t, YV, "c", label = "Empirical N = 2 Fit")
    ax.legend()
    plt.title("Question 2 Part B")
    plt.xlabel("Measured")
    plt.ylabel("Emperical")
    plt.show()
    
    
def Question_3():
    modeldata = open("Modeldata.txt", "r")
    M = modeldata.read().split(",")
    M = [float(entries) for entries in M]
    modeldata.close()
    
    turbinedata = open("Turbinedata.txt", "r")
    T = turbinedata.read().split(",")
    T = [float(entries) for entries in T]
    turbinedata.close()
    
    dftmodel = np.fft.fft(M)
    dftturbine = np.fft.fft(T)
    
    x = np.arange(0.0, 1.5, 0.001)
    
    M_real = [i.real for i in dftmodel]
    M_imag = [i.imag for i in dftmodel]
    
    T_real = [i.real for i in dftturbine]
    T_imag = [i.imag for i in dftturbine]
        
    En_M = np.abs(dftmodel)
    En_T = np.abs(dftturbine)
    
    
    sig_M = []
    c_M = []
    counter_M = 0
    
    for i in En_M:
        if i >= 200:
            sig_M.append(i)
            c_M.append(counter_M)
        counter_M += 1
        
    sig_T = []
    c_T = []
    counter_T = 0
    
    for i in En_T:
        if i >= 200:
            sig_T.append(i)
            c_T.append(counter_T)
        counter_T += 1
            
    Freq_M = []
    for i in c_M:
        Freq_M.append(M[i]/0.001)
        
    Freq_T = []
    for i in c_T:
        Freq_T.append(T[i]/0.001)

    print(f"Significant Harmonics Feild Turbine")
    print(f"Signal \t Frequency")
    [print(f"{i:.2f} \t {c:.2f}") for i,c in zip(sig_M, Freq_M)]
    
    print(f"\nSignificant Harmonics Model Turbine")
    print(f"Signal \t Frequency")
    [print(f"{i:.2f} \t {c:.2f}") for i,c in zip(sig_T, Freq_T)]
    
    fig, ax = plt.subplots()
    ax.plot(x, M_real)
    plt.title("Model Real")
    
    fig, ax = plt.subplots()
    ax.plot(x, T_real)
    plt.title("Turbine Real")
    
    fig, ax = plt.subplots()
    ax.plot(x, M_imag)
    plt.title("Model Imaginary")
    
    fig, ax = plt.subplots()
    ax.plot(x, T_imag)
    plt.title("Turbine Imaginary")
    
    fig, ax = plt.subplots()
    ax.plot(x, En_T)
    plt.title("Turbine Energy")
    
    fig, ax = plt.subplots()
    ax.plot(x, En_M)
    plt.title("Model Energy")
    
    plt.show()
     