import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def smooth_product(u, v, domain=(0, 1)):
    approx, _ = quad(lambda x: u(x)*v(x), *domain)  # Discard error
    return approx


def least_squares(func, basis, inner, **kwargs):
    # Take the inner product of each pair of basis
    lhs = np.array([
        [inner(p, q, **kwargs) for p in basis]
        for q in basis
    ])

    # The the inner product of the function with each basis
    rhs = np.array([
        inner(p, func, **kwargs) for p in basis
    ])

    # Create the approximation
    return np.linalg.solve(lhs, rhs)


def Q1():
    legendre = ('1', '2*x - 1', '6*x**2 - 6*x + 1')
    basis    = [  # Generate the functions based off strings
        np.vectorize(eval('lambda x:' + poly))
        for poly in legendre
    ]

    coeffs = least_squares(
        np.exp, basis, smooth_product, domain=(0, 1))

    domain = np.linspace(0, 1)
    approx = coeffs @ [p(domain) for p in basis]

    fig, ax = plt.subplots()
    ax.plot(domain, np.exp(domain), label='Exact')
    ax.plot(domain, approx, label='Approx')


def Q2():
    basis_funcs = [  # Generate the functions based off strings
        np.vectorize(eval('lambda x:' + poly))
        for poly in ('1', 'x', 'x**2')
    ]

    x_data, y_data = np.loadtxt('data_points.txt', unpack=True)
    basis_data = [p(x_data) for p in basis_funcs]

    coeffs = least_squares(
        y_data, basis_data, np.inner)

    domain = np.linspace(x_data.min(), x_data.max())
    approx = coeffs @ [p(domain) for p in basis_funcs]

    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, label='Exact')
    ax.plot(domain, approx, 'k-', label='Approx')


def Q3():
    basis = [  # Generate the functions based off strings
        eval('lambda x:' + func)
        for func in ('np.sin(x)', 'np.sin(2*x)', 'np.sin(3*x)')
    ]

    def function(x):
        return x*(np.pi-x)

    coeffs = least_squares(
        function, basis,
        smooth_product, domain=(0, np.pi)
    )

    for coeff, exact in zip(coeffs, (8/np.pi, 0, 8/(27*np.pi))):
        print(f'Coefficient error: {abs(coeff-exact):.3e}')

    domain = np.linspace(0, np.pi)
    approx = coeffs @ [p(domain) for p in basis]

    fig, ax = plt.subplots()
    ax.plot(domain, function(domain), label='Exact')
    ax.plot(domain, approx, label='Approx')


if __name__ == '__main__':
    questions = (Q1, Q2, Q3)
    for question in questions:
        input(f'Press `Enter` to run {question.__name__} ')

        plt.close('all')        # <- Close all existing figures
        question()
        plt.show(block=False)   # <- Allow code execution to continue

    input('Press `Enter` to quit the program.')
