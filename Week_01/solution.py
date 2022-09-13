__author__ = "Alex Muirhead"

import matplotlib.pyplot as plt
import numpy as np

# ====================== Construct difference functions ======================


def forward_difference(f, x, h):
    # Accuracy: O(h)
    approx = (f(x+h) - f(x)) / h
    return approx


def backward_difference(f, x, h):
    # Accuracy: O(h)
    approx = (f(x) - f(x-h)) / h
    return approx


def central_difference(f, x, h):
    # Accuracy: O(h**2)
    approx = (f(x+h) - f(x-h)) / (2*h)
    return approx


# ===================== Construct mathematical functions =====================


def f_a(x):
    # Analytical derivative: 3*x**2
    return x**3


def df_a(x):
    return 3*x**2


def f_b(x):
    # Analytical derivative: 6*x - 2
    return 3*x**2 - 2*x


def df_b(x):
    return 6*x - 2


def f_c(x):
    # Analytical derivative: cos(x)
    return np.sin(x)


def df_c(x):
    return np.cos(x)


# ========================== Plotting Approximations =========================

domain = np.linspace(-5, 5, num=101)  # Generate 101 numbers in domain

# Test different functions

fig, ax = plt.subplots()

for f, df in zip((f_a, f_b, f_c), (df_a, df_b, df_c)):
    approx = forward_difference(f, domain, h=1E-01)
    ax.plot(domain, approx)
    ax.plot(domain, df(domain), color='k', linestyle='--')

plt.show()

# Test different approximations (using Function C)

fig, ax = plt.subplots()
ax.plot(domain, df_c(domain), label="Exact", color="black")

for diff in (forward_difference, backward_difference, central_difference):
    approx = diff(f_c, domain, h=1E-01)
    ax.plot(domain, approx, label=diff.__name__)

ax.legend()
plt.show()

# Test different step sizes / values of h (using Function C & central diff)

fig, ax = plt.subplots()
ax.plot(domain, df_c(domain), label="Exact", color="black")

for h in np.geomspace(1E-01, 1E-05, num=5):
    approx = central_difference(f_c, domain, h=h)
    ax.plot(domain, approx, label=f"h={h:.1e}")

ax.legend()
plt.show()

# Show convergence of function approximations (using Function C)

fig, ax = plt.subplots()

step_sizes = np.geomspace(1E-01, 1E-05)
for diff in (forward_difference, backward_difference, central_difference):
    approx = diff(f_c, x=1, h=step_sizes)  # Do at x=1 for now
    error = np.abs(approx - df_c(x=1))
    ax.loglog(step_sizes, error, label=diff.__name__)

ax.loglog(step_sizes, step_sizes, "k--", label="O(h)")
ax.loglog(step_sizes, step_sizes**2, "k-.", label="O(h^2)")

ax.legend()
plt.show()
