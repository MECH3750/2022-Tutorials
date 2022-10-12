# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:21:02 2022

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt

w, h = 5, 4

# Initial Conditions

T = np.full(w*h, 25+273.15)

bottom_boundary = np.array([110, 100, 90, 80, 70]) + 273.15
i = 0
j = np.arange(0, w)
s = i*w + j
T[s] = bottom_boundary

top_boundary = np.array([0, 10, 20, 30, 40]) + 273.15
i = h-1
j = np.arange(0, w)
s = i*w + j
T[s] = top_boundary

left_boundary = np.array([110, 65, 25, 0]) + 273.15
i = np.arange(0, h)
j = 0
s = i*w + j
T[s] = left_boundary

right_boundary = np.array([70, 60, 50, 40]) + 273.15
i = np.arange(0, h)
j = w-1
s = i*w + j
T[s] = right_boundary

print(T)

# Set up system of equations

k = 54
rho = 7860
Cp = 490
delta_x = 0.025
delta_t = 10

sigma = k/(rho*Cp) * delta_t / (delta_x**2)
print(sigma)


# M @ T^{m+1} = T^{m}
M = np.zeros((w*h, w*h))
for i in range(h):  # Row
    for j in range(w):  # Column
        s = i*w + j
        
        # Bottom 
        if i == 0:
            M[s,s] = 1
            continue
        # Top
        elif i == h-1:
            M[s,s] = 1
            continue
        # Left 
        elif j == 0:
            M[s,s] = 1
            continue
        # Right
        elif j == w-1:
            M[s,s] = 1
            continue
        
        # Horizontal derivative (d^2T/dx^2)
        # T_{i,j-1} -2*T_{i,j} + T_{i,j+1}
        # T_{s-1} - 2*T_s + T_{s+1}
        M[s,s-1] += -1*sigma
        M[s,s]   += +2*sigma
        M[s,s+1] += -1*sigma
        
        # Vertical derivative (d^2T/dy^2)
        # T_{i-1,j} -2*T_{i,j} + T_{i+1,j}
        # T_{s-w} - 2*T_s + T_{s+w}
        M[s,s-w] += -1*sigma
        M[s,s]   += +2*sigma
        M[s,s+w] += -1*sigma
        
        # One side of time derivative
        M[s,s] += 1

fig, ax = plt.subplots()
ax.matshow(M)
plt.show()

# Iterating system of linear equations to get solution

for _ in range(5):
    T_new = np.linalg.solve(M, T)
    T = T_new
    
fig, ax = plt.subplots()
ax.matshow(T.reshape((h, w)), origin='lower')
plt.show()