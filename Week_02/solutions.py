from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D axes
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tabulate import tabulate

from tutorial_session import better_exp


def Q2to4():
    """Code for ICT questions 2--4."""

    domain = np.linspace(-2, 2)

    fig, ax = plt.subplots()
    ax.plot(domain, np.exp(domain), 'k--', label='Exact')
    for n in range(1, 4):
        ax.plot(domain, better_exp(n, domain), label=f'{n} Terms')

    ax.set_title('Comparison of exp and Taylor Series approximation')
    ax.set_xlim((domain.min(), domain.max()))
    ax.legend()


def Q5():
    """Code for ICT question 5."""

    domain = np.linspace(-1, 1, num=9)
    # Construct a grid, such that a pair of elements (one from each array)
    # gives the corresponding (x, y) coordinates.
    X, Y = np.meshgrid(domain, domain)
    Z = X**2 + Y**2  # Construct function values

    # Use and modify base code from
    # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

    fig, ax = plt.subplots(subplot_kw={"projection": '3d'})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


def lntaylor(n, x):
    """
    Efficient Taylor series expansion of ln(x+1) at x=1

    term 0: -1 {not included}
    term 1:   (x-1)**1 / (2**1 * 1)
    term 2: - (x-1)**2 / (2**2 * 2)
    term 3:   (x-1)**3 / (2**3 * 3)
    term 4: - (x-1)**4 / (2**4 * 4)
    """
    term = -1
    output = np.log(2.)  # <- 0th term
    for i in range(1, n):
        term *= (1-x) / 2
        output += term / i
    return output


def Q6():
    """Code for ICT question 6."""

    x, h   = 1, np.logspace(-1, -6, 6)
    exact  = np.log(1+x+h)
    approx = lntaylor(3, x+h)

    error = np.abs(exact - approx)
    print(tabulate({
        'ln(1+x)'    : exact,
        'Taylor'     : approx,
        'Error'      : error,
        'Error / h^2': error / h**2,
        'Error / h^3': error / h**3,
        'Error / h^4': error / h**4
    }, headers='keys'))


def fvec(xvec):
    """Vector function for Q7."""
    x, y = xvec
    return np.array([
        x**4 + y**4 - 1,
        x**2 - y**2 + 1
    ])


def jacobian(xvec):
    """Jacobian function for Q7."""
    x, y = xvec
    return np.array([
        [4*x**3, 4*y**3],
        [2*x,    -2*y  ]
    ])


def newton_solver(initial, func, grad=None, step=0.02):
    state = initial.copy()
    value = func(state)

    goal = np.zeros_like(value)

    if grad is None:
        # Use derivative approximation when no analytical is provided
        delta = np.eye(state.size) * step

        def grad(x):
            return (func(x+delta) - func(x)) / step

    while not np.allclose(value, goal, atol=1e-4, rtol=1e-4):
        state -= np.linalg.solve(grad(state), func(state))
        value = func(state)

    return state


def Q7to9():
    """Code for ICT questions 7 to 9."""

    initial = np.array([1, 1], dtype=float)

    x_sol, y_sol = newton_solver(initial, fvec, grad=jacobian)
    print(f'The analytical solution is at x={x_sol:.3f}, y={y_sol:.3f}')

    x_sol, y_sol = newton_solver(initial, fvec)
    print(f'The numerical solution is at  x={x_sol:.3f}, y={y_sol:.3f}')


if __name__ == '__main__':
    questions = (Q2to4, Q5, Q6, Q7to9)
    for question in questions:
        input(f'Press `Enter` to run {question.__name__} ')

        plt.close('all')        # <- Close all existing figures
        question()
        plt.show(block=False)   # <- Allow code execution to continue

    input('Press `Enter` to quit the program.')
