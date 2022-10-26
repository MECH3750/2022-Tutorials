# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:57:49 2022

@author: dhutc
"""

import numpy as np
import math as math

#Coefficients and Constants
L = 0.001 #lenght
r = 2.5E-6 #radius
h = 1000 #Heat Transfer Coefficient
P = (2*math.pi*r) #Perimeter
k = 200 #Thermal Conductivity
A = math.pi*r**2 #Cross-sectional area


T_a = 293.15
T_0 = 353.15
T_n = 343.15
O0 = T_0 - T_a
On = T_n - T_a

def approx_T(n):
    delta_x = L/(n-1)
    B = ((h*P)/(k*A))**0.5
    sigma = -2-B**2*(delta_x)**2
    upper_vector = np.hstack((0,np.ones(n-2)))
    middle_vector = np.hstack((1,sigma*np.ones(n-2),1))
    lower_vector = np.hstack((np.ones(n-2),0))
    UM = np.diag(upper_vector,1)
    MM = np.diag(middle_vector)
    LM = np.diag(lower_vector,-1)
    M = UM + MM + LM
    
    B = np.vstack(np.hstack((O0,np.zeros(n-2),On)))
    
    O = np.linalg.solve(M,B)
    print(O)
    
def determine_T(n):
    B = ((h*P)/(k*A))**0.5
    delta_x = L/(n-1)
    x = 0
    theta_array = []
    while x < L:
        theta = (On*np.sinh(B*x) + O0*np.sinh(B*(L-x)))/(np.sinh(B*L))
        theta_array.append(theta)
        x += delta_x
    np.vstack((np.array(theta_array)))
    print(theta_array)
    
        
        
