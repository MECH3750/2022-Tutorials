# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:38:54 2022

@author: dhutc
"""

import numpy as np
import matplotlib as plt

x = np.linspace(0, 5)
f_x = x(np.pi-x)

p1 = np.sin(x)
p2 = np.sin(2*x)
p3 = np.sin(3*x)

a1 = 8/np.pi
a2 = 0
a3 = 8/(27*np.pi)


P = np.column_stack((p1, p2, p3))

fitted_trend = a1 * p1 + a2 * p2 + a3 * p3

fig, ax = plt.subplots()
ax.plot(x,f_x)
ax.plot(x, fitted_trend, "r")

plt.show()

      