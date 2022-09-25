import numpy as np
import matplotlib.pyplot as plt
# Import "colour maps" (cm) for nice colours
from matplotlib import cm

U_0 = 1     # Height of peak
L   = 10    # Range of spatial domain
T   = 5     # Range of temporal domain
D   = 0.7   # Decay constant


def initial_conditions(x):

    f = np.where(
        x < L/2,         # Condition (True or False at each index)
        2*U_0/L * x,     # if condition:
        2*U_0/L * (L-x)  # else:
    )

    return f


def analytical_solution(x, t, *, terms):

    u = U_0/2 * np.ones_like(x, dtype=float)  # NOTE: Ensure float!

    for n in range(terms):
        k = (2*n+1)           # Wavenumber
        w = k * 2*np.pi / L   # Frequency

        # Demonstrate the separation of variables in each term
        #                               Spatial => F(x) -----------
        u -= U_0 * (2/k/np.pi)**2 * np.exp(-D*t*w**2) * np.cos(x*w)
        #                           ----------------- Temporal decay => G(t)

    return u


# 1D plot

x = np.linspace(0, L, num=51)
f = initial_conditions(x)

_, ax = plt.subplots()
ax.plot(x, f)
plt.show()


# 2D plot

# NOTE: Creates a 2D array with shape (51, 11)
x, t = np.meshgrid(
    np.linspace(0, L, num=51),  # Spatial
    np.linspace(0, T, num=11),  # Temporal
    indexing='ij'
    # NOTE: Using 'ij' indexing ensures that first index is 'x'
    #       and second index is 't'
)

f = initial_conditions(x)
u = analytical_solution(x, t, terms=10)

_, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_wireframe(x, t, f)  # type: ignore
ax.plot_surface(x, t, u, cmap=cm.get_cmap("coolwarm"))  # type: ignore
plt.show()
