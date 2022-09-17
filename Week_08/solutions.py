# Standard NumPy and Matplotlib import
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from numpy.fft import fft, ifft, fftfreq, fftshift


def Q1():
    # Use `j` for imaginary number. Must have a numeral in front of it!
    f = np.array([1, 2-1j, -1j, -1+2j])
    a = fft(f)

    print(f"Fourier coefficients: {a}")

    check = ifft(a)
    print(f"Data from inverse: {check}")


def plot_coeffs(axes: Axes, timeseries: np.ndarray):
    """Helper function to plot FFT coefficients for a given timeseries

    Parameters
    ----------
    axes : Axes
        The axes on which to plot the coefficients
    timeseries : np.ndarray
        The timeseries data to transform and plot
    """
    frequencies  = fftfreq(timeseries.size, d=1)  # Assume sample rate of 1
    coefficients = fft(timeseries)

    axes.plot(
        fftshift(frequencies),
        fftshift(coefficients.real),
    )
    axes.plot(
        fftshift(frequencies),
        fftshift(coefficients.imag)
    )


def Q2():
    N = 8
    n = np.arange(0, N)

    _, axes = plt.subplots(2, 2, sharey=True, sharex=True)
    axes = np.asarray(axes)  # Make the type checker happy

    # Part a)
    f = np.sin(2*np.pi*n/N)
    plot_coeffs(axes[0, 0], f)

    # Part b)
    f = np.cos(2*np.pi*n/N)
    plot_coeffs(axes[0, 1], f)

    # Part c)
    f = np.cos(4*np.pi*n/N)
    plot_coeffs(axes[1, 0], f)

    # Part d)
    f = 2*np.sin(2*np.pi*n/N) + 0.5*np.cos(4*np.pi*n/N)
    plot_coeffs(axes[1, 1], f)

    plt.show()


def Q3():
    for N in [8, 16]:  # Let's not repeat code
        n = np.arange(0, N)
        q_1 = np.exp(2j*np.pi*n*1/N)
        q_2 = np.exp(2j*np.pi*n*2/N)

        print(f"Checking vectors for {N=}")

        print(f"<q1, q2> = {np.vdot(q_1, q_2):.3f}")
        print(f"<q1, q1> = {np.vdot(q_1, q_1):.3f}")
        print(f"<q2, q2> = {np.vdot(q_2, q_2):.3f}")

        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1.plot(n, q_1.real, q_1.imag)
        ax2.plot(n, q_2.real, q_2.imag)

        ax1.legend(["Real", "Imaginary"])
        ax2.legend(["Real", "Imaginary"])
        ax1.set(title="$q^{(1)}$", xlabel="n")
        ax2.set(title="$q^{(2)}$", xlabel="n")

    plt.show()


if __name__ == '__main__':
    questions = [Q1, Q2, Q3]
    for question in questions:
        input(f'Press `Enter` to run {question.__name__} ')

        plt.close('all')        # <- Close all existing figures
        question()
        plt.show(block=False)   # <- Allow code execution to continue

    input('Press `Enter` to quit the program.')
