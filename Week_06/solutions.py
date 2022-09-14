# Standard NumPy and Matplotlib import
import matplotlib.pyplot as plt
import numpy as np


def Q1():
    domain = np.linspace(-2*np.pi, +2*np.pi, num=501)

    _, ax = plt.subplots()
    # Plot y=1 over the domain [0, pi]
    ax.plot((0, np.pi), (1, 1), label='Exact')

    N = [5, 21]
    approx = np.zeros_like(domain)  # Make it the same size
    for n in range(1, max(N)+1):
        #         ------------- Make sure we cover terms we want
        a_n = 4/(n*np.pi) if (n % 2 == 1) else 0
        #                    ------------ Check if n is even or odd
        approx += a_n * np.sin(domain*n)

        if n in N:
            ax.plot(domain, approx, label=f'N={n}')

    ax.legend()
    plt.show()


def Q2():
    domain = np.linspace(-2*np.pi, +2*np.pi, num=501)

    _, ax = plt.subplots()
    # Plot y=x^2 over the domain [0, pi]
    small_domain = np.linspace(-np.pi, +np.pi, num=501)
    ax.plot(
        small_domain, small_domain**2,
        label='Exact'
    )

    N = [5, 21]
    approx = (np.pi**2 / 3) * np.ones_like(domain)  # Make it the same size
    for n in range(1, max(N)+1):
        #         ------------- Make sure we cover terms we want
        b_n = 4*(-1)**n / (n**2)
        approx += b_n * np.cos(domain*n)

        if n in N:
            ax.plot(domain, approx, label=f'N={n}')

    ax.legend()
    plt.show()


def Q3():
    domain = np.linspace(-2*np.pi, +2*np.pi, num=501)

    _, ax = plt.subplots()
    # Plot y=x over the domain [0, pi]
    small_domain = np.linspace(-np.pi, +np.pi, num=501)
    ax.plot(
        small_domain, small_domain,
        label='Exact'
    )

    N = [5, 21]
    approx = np.zeros_like(domain)  # Make it the same size
    for n in range(1, max(N)+1):
        #         ------------- Make sure we cover terms we want
        a_n = 2*(-1)**(n+1) / n
        approx += a_n * np.sin(domain*n)

        if n in N:
            ax.plot(domain, approx, label=f'N={n}')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    questions = (Q1, Q2, Q3)
    for question in questions:
        input(f'Press `Enter` to run {question.__name__} ')

        plt.close('all')        # <- Close all existing figures
        question()
        plt.show(block=False)   # <- Allow code execution to continue

    input('Press `Enter` to quit the program.')
