#  -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 12:26:32 2022

@author: Declan
"""
import numpy as np
from numpy import inner as ni
from numpy.fft import fftshift as shift
import matplotlib.pyplot as plt
import math


def Question_1 ():
    # Set initial guess of y = [1,1,1,1]
    Y_n = np.ones((4, 1))
    # Arbitary Value of functions to initiate condition of while loop
    f1, f2, f3, f4 = 1,1,1,1
    
    # This statement iterates over the absolute value of each of the f values.
    # It then compares the maximum value to 10E-6, the specified max absolute error
    # If the largest error is above max allowable error, the loop runs, otherwise it will not
    while max(abs(f) for f in [f1,f2,f3,f4]) > 10**-6:
        # Set y variable to current initial variables
        y1,y2,y3,y4 = float(Y_n[0]), float(Y_n[1]), float(Y_n[2]), float(Y_n[3])
        
        # Function Equations
        f1 = 0.04*y1**2 - 2*y1/0.2 + y2/0.4
        f2 = 0.16*y2**2 - 2*y2/0.4 + (y3-y1)/0.4
        f3 = 0.36*y3**2 - 2*y3/0.6 + (y4-y2)/0.4
        f4 = 0.64*y4**2 - 2*y4/0.8 + (1-y3)/0.4

        # Function Equation
        F_y = np.vstack((f1, f2, f3, f4))
        
        # Jacobian
        J = np.array([[0.08*y1 - 10, 2.5, 0, 0],
                     [-2.5, 0.32*y2 - 5, 2.5, 0],
                     [0, -2.5, 0.72*y3 - 2/0.6, 2.5],
                     [0, 0, -2.5, 1.28*y4 -2.5]])
        
        # Inverse of Jacobain
        J_inv = np.linalg.inv(J)
        
        # Newtons method to calculate Yn+1
        Y_n1 = Y_n - np.dot(J_inv, F_y)
        
        # Setting Yn+1 to be Yn for the next iteration
        Y_n = Y_n1
    
    #  A list of the X values used as points in the systems mesh     
    X = [0, 0.2, 0.4, 0.6, 0.8, 1]
    Y_an = [] # empty list of analytical y solutions
    # The following for loop determines the analytical y solution for each x value
    for x in X:
        y_an = x**2/(0.2*x**5 + 0.8)
        Y_an.append(y_an)
    
    Y_n = np.insert(Y_n, 0, [0])
    Y_n = np.insert(Y_n, 5, [1])
        
    # Plotting the Newton Method results against the Analytical Method results
    fig, ax = plt.subplots()
    ax.plot (X, Y_n, "g", label = "Newtons Method")
    ax.plot (X, Y_an, "r", label = "Analytical Solution")
    ax.legend()
    plt.title("Newton's vs Analytical Method")
    plt.xlabel("Position x mm")
    plt.ylabel("Y(x)")
    
    plt.show()
    
    return (Y_n)


def Question_2():
    # Initial data from question
    y = [12.02,18.60,23.72,28.78,34.35,38.88,39.29,44.65,43.16,48.39,48.89,47.156,48.35,50.62,51.09]
    t = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    
    # Creating lists of y and t to the power of N for use in least square
    y0 = [number ** 0 for number in y]
    y1 = y
    y2 = [number ** 2 for number in y]
    y3 = [number ** 3 for number in y]
   
    t0 = [number ** 0 for number in t]
    t1 = t
    t2 = [number ** 2 for number in t]
    t3 = [number ** 3 for number in t]
  
  
    # creating array for N=2, using inner products
    N2 = np.array([[ni(t0,t0), ni(t0,t1)],
                   [ni(t1,t0), ni(t1,t1)]])                                  
                  
    # Creating Fucntion vector of N=2, using inner products
    F2 = np.vstack((ni(t0,y), ni(t1,y)))
    
    # Determining the coefficients of N=2 polynomial
    A2 = np.linalg.inv(N2).dot(F2)
    
    # N=2 data fit polynomial fucntion
    Y2 = [A2[1]*t + A2[0] for t in range (1,16)]
    
    
    # creating array for N=4, using inner products
    N4 = np.array([[ni(t0,t0), ni(t0,t1), ni(t0,t2), ni(t0,t3)],
                   [ni(t1,t0), ni(t1,t1), ni(t1,t2), ni(t1,t3)],                                  
                   [ni(t2,t0), ni(t2,t1), ni(t2,t2), ni(t2,t3)],
                   [ni(t3,t0), ni(t3,t1), ni(t3,t2), ni(t3,t3)]])
    
    # Creating Fucntion vector of N=4, using inner products
    F4 = np.vstack((ni(t0,y), ni(t1,y), ni(t2,y), ni(t3,y)))
    
    # Determining the coefficients of N=4 polynomial
    A4 = np.linalg.inv(N4).dot(F4)
    
    # N=4 data fit polynomial fucntion
    Y4 = [A4[3]*t**3 + A4[2]*t**2 + A4[1]*t + A4[0]  for t in range (1,16)]
    
    
    # Generating list of emperical results
    v = [(9.8*68.1)/12*(t/(3.75+t)) for t in range (1,16)]
    
    # creating array for N=2, using inner products
    V2 = np.array([[ni(y0,y0), ni(y0,y1)],
                   [ni(y1,y0), ni(y1,y1)]])
    
    # Creating function vector of emperical results, N = 2
    FV = np.vstack((ni(y0,v), ni(y1,v)))
    
    # Determining coefficients of emperical results, N = 2
    AV = np.linalg.inv(V2).dot(FV)
    
    # N=4 data fit polynomial fucntion for emperical results
    YV = [AV[1]*t + AV[0] for t in range (1,51)]
    
    # Printing the equation of the N=2 polynomial least squares data fit for
    # purpose of discussion
    print(f"{AV[1]}*t + {AV[0]}")
    
    # Plotting Data for part A
    fig, ax = plt.subplots()
    ax.plot(t, Y2, "r", label = "N = 2 Fit")
    ax.plot(t, Y4, "g", label = "N = 4 Fit")
    ax.scatter(t,y, label = "Measured Data")
    ax.legend()
    plt.title("Question 2 Part A")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.show()
    
    # Plotting Data for part B
    fig, ax = plt.subplots()
    ax.scatter(y,v,)
    ax.plot(np.linspace(0,50,50), YV, "c", label = "Empirical N = 2 Fit")
    ax.legend()
    plt.title("Question 2 Part B")
    plt.xlabel("Measured")
    plt.ylabel("Emperical")
    plt.show()
    
    # Plotting Data for part B discussion
    fig, ax = plt.subplots()
    ax.scatter(t, y, label = "Measaured Data")
    ax.scatter(t, v, label = "Expected Data")
    ax.legend()
    plt.title("Measured vs Expected Velocities")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.show()    
    
def Question_3():
    modeldata = open("Modeldata.txt", "r") # Opening the Model data set
    M = modeldata.read().split(",") # Splitting the data stream at the commas
    M = [float(entries) for entries in M] # Saving the data points as floats into a list
    modeldata.close() # Closing the text file
    
    turbinedata = open("Turbinedata.txt", "r") # Opening the Turbine data set
    T = turbinedata.read().split(",") # Splitting the data stream at the commas
    T = [float(entries) for entries in T] # Saving the data points as floats into a list
    turbinedata.close() # Closing the text file
    
    dftmodel = np.fft.fft(M) # Conducting a discrete fourier transform on the Model Data
    dftturbine = np.fft.fft(T) # Conducting a discrete fourier transform on the Turbine Data
    
    # The fftshift functions are used to reorder the elements to account for the positive and negative frequencies
    x_M = shift(np.fft.fftfreq(dftmodel.size, d = 0.001)) # Creating freqencies to serve as x-axis for graphs
    x_T = shift(np.fft.fftfreq(dftturbine.size, d = 0.001)) # Creating freqencies to serve as x-axis for graphs
    
    M_real = [i.real for i in dftmodel] # Isolating the real components of the fourier transform of the model data
    M_imag = [i.imag for i in dftmodel] # Isolating the imaginary components of the fourier transform of the model data
    
    T_real = [i.real for i in dftturbine] # Isolating the real components of the fourier transform of the turbine data
    T_imag = [i.imag for i in dftturbine] # Isolating the imaginary components of the fourier transform of the turbine data
        
    En_M = np.abs(dftmodel) # determining the magnitude of each model data point
    En_T = np.abs(dftturbine) # determining the magnitude of each turbine data point
    
    sig_M = [] # empty list creating for storing values of significant harmonics in the model
    c_M = [] # empty list for saving the position of each significant harmonic
    counter_M = 0
    
    # For loop to iterate over the model energies, any data point with a magnitude
    # Greater than 200 is idenitified as a significant harmonic and appended to the
    # significant harmonics list, additionally the counter position is appened as well
    for i in En_M:
        if i >= 200:
            sig_M.append(i)
            c_M.append(counter_M)
        counter_M += 1
        
    sig_T = [] # empty list creating for storing values of significant harmonics in the turbine
    c_T = [] # empty list for saving the position of each significant harmonic
    counter_T = 0
    
    # For loop to iterate over the turbine energies, any data point with a magnitude
    # Greater than 200 is idenitified as a significant harmonic and appended to the
    # significant harmonics list, additionally the counter position is appened as well
    for i in En_T:
        if i >= 200:
            sig_T.append(i)
            c_T.append(counter_T)
        counter_T += 1
    
    # Detetermining the frequencies of each significatn harmonic using the counter list
    Freq_M = []
    for i in c_M:
        Freq_M.append(x_M[i])
    
    # Detetermining the frequencies of each significatn harmonic using the counter list
    Freq_T = []
    for i in c_T:
        Freq_T.append(x_T[i])
    
    # Prints the magnitude of the energy of each signficant model turbine harmonic and its frequency
    print(f"Significant Harmonics Model Turbine")
    print(f"Signal \t Frequency")
    [print(f"{i:.2f} \t {c:.2f}") for i,c in zip(sig_M, Freq_M)]
    
    # Prints the magnitude of the energy of each signficant feild turbine harmonic and its frequency
    print(f"\nSignificant Harmonics Feild Turbine")
    print(f"Signal \t Frequency")
    [print(f"{i:.2f} \t {c:.2f}") for i,c in zip(sig_T, Freq_T)]
    
    # Plots models real components of the DFT
    fig, ax = plt.subplots()
    ax.plot(x_M, shift(M_real))
    plt.title("Model Real")
    
    # Plots turbines real components of the DFT
    fig, ax = plt.subplots()
    ax.plot(x_T, shift(T_real))
    plt.title("Turbine Real")
    
    # Plots models imaginary components of the DFT
    fig, ax = plt.subplots()
    ax.plot(x_M, shift(M_imag))
    plt.title("Model Imaginary")
    
    # Plots turbines imaginary components of the DFT
    fig, ax = plt.subplots()
    ax.plot(x_T, shift(T_imag))
    plt.title("Turbine Imaginary")
    
    # Plots turbines energies
    fig, ax = plt.subplots()
    ax.plot(x_T, shift(En_T))
    plt.title("Turbine Energy")
    
    # Plots models energies
    fig, ax = plt.subplots()
    ax.plot(x_M, shift(En_M))
    plt.title("Model Energy")
    
    plt.show()
    
def Question_4():
    x_range = np.linspace(0,1,1000) #Interval of [0,1] with 1000 data points for smoothness of graph
    solutions_a = [] # Empty list to append solutions to
    # This loop iterates over every x value and sums the fourier series results for each index
    for x in x_range:
        f_x = sum((-2*(-1)**n)/(n*np.pi)*np.sin(n*np.pi*x) for n in range(1,11))
        solutions_a.append(f_x)
    
    solutions_b = [] # Empty list to append solutions to
    # This loop iterates over every x value and sums the fourier series results for each index
    for i in range(1,11):
        Bn = sum((-2*(-1)**n)/(n*np.pi) for n in range (1,i+1))
        solutions_b.append(Bn)
    
    # Plot of fourier series terms
    fig, ax = plt.subplots()
    ax.plot(x_range, solutions_a)
    plt.title("Fourier Series")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    
    # Plot of magnitude of coefficients by index
    fig, ax = plt.subplots()
    ax.plot(range(1,11), solutions_b)
    plt.title("Magnitude of Coefficients by index")
    plt.xlabel("n")
    plt.ylabel("Magnitude of Fourier Coefficients")
    
    # Plot of Log Log plot
    fig, ax = plt.subplots()
    ax.loglog(range(1,11), solutions_b)
    plt.title("Fourier Series")
    plt.xlabel("n")
    plt.ylabel("Bn")
     