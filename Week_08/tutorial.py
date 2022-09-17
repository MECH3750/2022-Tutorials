# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:55:03 2022

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, fftfreq

f = np.array([1, 2-1j, -1j, -1+2j])

print(f"Data points: {f}")

# FFT coefficients
a = fft(f)

print(f"Coefficients: {a}")

check = ifft(a)

error = np.abs(check - f)
print(f"The maximum reconstruction error: {np.max(error)}") 


# Part 2a)

N = 8
n = np.arange(0, N)  # n = 0, 1, ..., N-1
f = np.sin(2*np.pi*n/N)  # equation a)

_, ax = plt.subplots()
ax.plot(n, f)
plt.show()

a = fft(f)  # Coefficients
_, ax = plt.subplots()
# ax.plot(n, a)  # <-- Complex value!
# ax.plot(n, a.real, label="Real")
# ax.plot(n, a.imag, label="Imaginary")

w = fftfreq(f.size, d=1)  # Sample spacing (inverse sample frequency)
ax.plot(fftshift(w), fftshift(a.real), label="Real")
ax.plot(fftshift(w), fftshift(a.imag), label="Imaginary")

ax.legend()
plt.show()
