# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:07:31 2022

@author: alexa
"""


def fact(n):
    # Return n! 
    # n! = n * (n-1) * (n-2) * ... * 2 * 1
    output = 1
    for term in range(1, n+1):
        # Loop through values term=[1,n]=[1,n+1)
        output *= term  # output = output * term
        
    return output


def exptaylor(n, x):
    # Return Taylor series of exp(x) using n terms
    # exp(x) = 
    #          1         = 1/fact(0) * x**0
    #        + x         = 1/fact(1) * x**1
    #        + x**2 / 2  = 1/fact(2) * x**2
    #        + ...       = ...
    #                    = 1/fact(term) * x**term
    output = 0
    for term in range(0, n+1):
        # Loop through values term=[0,n]=[0,n+1)
        output += 1/fact(term) * x**term
    
    return output


def better_exp(n, x):
    # Return Taylor series of exp(x) using n terms
    # exp(x) = 
    #          1         = 1/1 * x**0
    #        + x         = 1/(1) * x
    #        + x**2 / 2  = 1/2*x * (1/(1) * x)
    #        + x**3 / 6  = 1/3*x * (1/2*x * (1/(1) * x))
    #        + ...       = ...
    #                    = 1/fact(term) * x**term
    term = 1  # 0th term
    output = term
    for i in range(1, n+1):
        term *= x / i
        output += term
        
    return output
        
















    