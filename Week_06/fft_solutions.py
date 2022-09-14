# Standard NumPy and Matplotlib import
import matplotlib.pyplot as plt
import numpy as np

# Import FFT library
import scipy.fft as fft


def Q1():
    sample = np.linspace(0, np.pi, num=501)
    exact  = np.ones_like(sample)

    _, ax = plt.subplots()
    ax.plot(sample, exact)

    N = [5, 21]
    amplitudes = np.asarray(fft.dst(exact, norm="forward", type=1))
    #            ^^^^^^^^^^ This isn't strictly necessary, but it just
    #                       makes the type checker happy

    domain = np.linspace(-2*np.pi, +2*np.pi, num=501)
    approx = np.zeros_like(domain)  # Make it the same size
    for n in range(1, max(N)+1):
        a_n = 2*amplitudes[n-1]
        approx += a_n * np.sin(domain*n)

        if n in N:
            ax.plot(domain, approx, label=f'N={n}')

    plt.show()


def Q2():
    sample = np.linspace(0, np.pi, num=501)
    exact  = sample**2

    _, ax = plt.subplots()
    ax.plot(sample, exact)

    N = [5, 21]
    amplitudes = np.asarray(fft.dct(exact, norm="forward"))
    #            ^^^^^^^^^^ This isn't strictly necessary, but it just
    #                       makes the type checker happy

    domain = np.linspace(-2*np.pi, +2*np.pi, num=501)
    approx = amplitudes[0] * np.ones_like(domain)  # Make it the same size
    for n in range(1, max(N)+1):
        a_n = 2*amplitudes[n]
        approx += a_n * np.cos(domain*n)

        if n in N:
            ax.plot(domain, approx, label=f'N={n}')

    plt.show()


def Q3():
    sample = np.linspace(0, np.pi, num=501)
    exact  = sample

    _, ax = plt.subplots()
    ax.plot(sample, exact)

    N = [5, 21]
    amplitudes = np.asarray(fft.dst(exact, norm="forward", type=1))
    #            ^^^^^^^^^^ This isn't strictly necessary, but it just
    #                       makes the type checker happy

    domain = np.linspace(-2*np.pi, +2*np.pi, num=501)
    approx = np.zeros_like(domain)  # Make it the same size
    for n in range(1, max(N)+1):
        a_n = 2*amplitudes[n-1]
        approx += a_n * np.sin(domain*n)

        if n in N:
            ax.plot(domain, approx, label=f'N={n}')

    plt.show()


if __name__ == '__main__':
    questions = (Q1, Q2, Q3)
    for question in questions:
        input(f'Press `Enter` to run {question.__name__} ')

        plt.close('all')        # <- Close all existing figures
        question()
        plt.show(block=False)   # <- Allow code execution to continue

    input('Press `Enter` to quit the program.')
