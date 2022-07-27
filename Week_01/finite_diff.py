# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:38:46 2022

@author: alexa
"""

def forward_difference(f, x, h):
    approx = (f(x+h) - f(x)) / h
    return approx


def func_a(x):
    return x**3


