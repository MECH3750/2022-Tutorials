# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:02:54 2022

@author: dhutc
"""
import numpy as np
import matplotlib.pyplot as plt

def fact(n):
    output = 1
    for term in range(1,n+1):
        output *= term
    return(output)
    
def exptaylor(n, x):
    output = 0
    for term in range (0,n+1):  
        output += 1/fact(term) * x**term
    return output

def better_exp(n,x):
    term = 1
    output = term    
    for i in range(1,n+1):
        term *= x/ i 
        output += term
    return output

def plot_exp(l,h, ax):
    x = np.linspace(l, h, 101)
    y = np.exp(x)
    ax.plot(x,y)
    return

def plot_taylor(l,h, ax):
    x = np.linspace(l, h, 101)
    y = []
    for term in x:
        y = better_exp(1, x)
    ax.plot(x,y)
    
def compare(l,h):
    fig, ax = plt.subplots()
    plot_exp(l,h, ax)
    plot_taylor(l, h, ax)
    plt.show()    
    
    
    