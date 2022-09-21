# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:09:28 2022

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt

# int = ... -1, 0, 1, 2, 3, 4, ...
# float = 0.1, 0.11, 0.111, ...

data = [1, 2, 3, 4, 5]  # <-- All ints

one_array = np.ones_like(data, dtype=float)
# Copy the shape AND the dtype

term = 1
one_array += term

print(one_array)

# Initial conditions

U_0 = 1
L   = 10
T   = 5

fig, ax = plt.subplots()

x = np.linspace(0, L, num=51)
f = np.where(
    x < L/2,         # Conditional
    2*U_0/L * x,     # if conditional == True:
    2*U_0/L * (L-x)  # else: 
)
    
ax.plot(x, f)
plt.show()

# 3D plots

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(...)
# ax.plot_wireframe(...)

x, t = np.meshgrid(
    np.linspace(0, L, num=51),  # Space dimension
    np.linspace(0, T, num=21),  # Time dimension
    indexing='ij'
)

f = np.where(
    x < L/2,         # Conditional
    2*U_0/L * x,     # if conditional == True:
    2*U_0/L * (L-x)  # else: 
)
    
ax.plot_surface(x, t, f)
