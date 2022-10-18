import numpy as np
import matplotlib.pyplot as plt

TOL = np.finfo(float).resolution


def C_to_K(degrees_celsius):
    return degrees_celsius + 273.15


class IndexTool:
    def __init__(self, shape):
        self._shape   = shape
        self._indices = np.indices(shape)

    def __getitem__(self, index):
        index  = (slice(None, None), *index)
        coords = self._indices[index]
        return np.ravel_multi_index(coords, self._shape)


# Set up our constants
Lx = 10.0 * 0.01  # [m]
Ly =  7.5 * 0.01  # [m]

rho = 7860  # [kg/m^3]
c_p = 490   # [J/kg.K]
k   = 54    # [W/m.K]

alpha = k / (c_p * rho)  # [m^2/s]

dx = 2.5 * 0.01  # [m]
dy = 2.5 * 0.01  # [m]
dt = 100         # [s]

# Calculate the CFL numbers
sigma_x = alpha * dt / dx**2
sigma_y = alpha * dt / dy**2

# Create array sizes
nx = int((Lx+TOL) // dx) + 1
ny = int((Ly+TOL) // dy) + 1  # NOTE: Tolerance for floating point error

# Find total number of points in array.
ns = nx * ny

# Dividing small floating point numbers sometimes doesn't give the expected
# answer. For example, 0.075 / 0.025 = 2.9999999999999996
# Without the tolerance, this gives ny = 3 rather than the expected 4.

print(f'The grid is {nx} wide, and {ny} tall')
# Configure initial conditions
current_temperature = C_to_K(50) * np.ones(ns)

# Configure boundary conditions
bottom_edge = C_to_K(np.linspace(110, 70, num=nx))
top_edge    = C_to_K(np.linspace(  0, 40, num=nx))

left_edge  = C_to_K(np.linspace(110,  0, num=ny))
right_edge = C_to_K(np.linspace( 70, 40, num=ny))

ind = IndexTool(shape=(ny, nx))

# Constants to represent the indices of boundaries
N = ind[-1, :]
S = ind[+0, :]
E = ind[:, -1]
W = ind[:, +0]

# Boundary condition
current_temperature[S] = bottom_edge
current_temperature[N] = top_edge
current_temperature[W] = left_edge
current_temperature[E] = right_edge

D2_x = sigma_x * (np.eye(ns, k=-nx) - 2*np.eye(ns, k=0) + np.eye(ns, k=+nx))
D2_y = sigma_y * (np.eye(ns, k=-1)  - 2*np.eye(ns, k=0) + np.eye(ns, k=+1))

implicit_matrix = np.eye(ns) - D2_x - D2_y

# Add boundary conditions
dirichlet = np.eye(ns)  # T^{m+1} = T^{m}
for edge in [N, S, E, W]:
    implicit_matrix[edge] = dirichlet[edge]

plt.imshow(implicit_matrix)
plt.show()

for _ in range(100):  # 100 timesteps
    # M @ T{t+1} = T{t}
    next_temperature = np.linalg.solve(implicit_matrix, current_temperature)
    current_temperature = next_temperature.copy()


plt.imshow(current_temperature.reshape(ny, nx) - 273.15, origin='lower')
plt.show()
