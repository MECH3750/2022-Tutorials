# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:38:46 2022

@author: alexa
"""
import numpy as np

def forward_difference(f, x, h):
    approx = (f(x+h) - f(x)) / h
    return approx


def backwards_difference(f, x, h):
    approx = (f(x)-f(x-h)) / h
    return approx

def central_difference(f, x, h):
    approx = (f(x+h)-f(x-h)) / (2*h)
    return approx

def func_a(x):
    return x**3

def func_b(x):
    return 3*x**2 - 2*x

def func_c(x):
    return np.sin(x)

def investiagte_diff(f, x, h):
    fwd = forward_difference(f, x, h)
    b = backwards_difference(f, x, h)
    c = central_difference(f, x, h)
    print(f"{fwd}\n{b}\n{c}")


