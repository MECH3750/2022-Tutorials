__author__ = "Alex Muirhead"


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


if __name__ == "__main__":
    # Test the timings of the code!

    from timeit import Timer

    import numpy as np
    from matplotlib import pyplot as plt

    import_stmt = "from __main__ import exptaylor, better_exp"

    print("This might take a minute...")

    term_range   = np.arange(5, 100, 5, dtype=int)  # Ensure ints
    naive_times  = []
    better_times = []

    for n in range(5, 100, 5):
        iters, secs = Timer(f"exptaylor({n}, 5)", setup=import_stmt).autorange()
        time = secs / iters

        naive_times.append(time * 1E06)

        iters, secs = Timer(f"better_exp({n}, 5)", setup=import_stmt).autorange()
        time = secs / iters

        better_times.append(time * 1E06)

    fig, ax = plt.subplots()
    ax.plot(term_range, naive_times, label="Naive method")
    ax.plot(term_range, better_times, label="Better method")
    ax.set_xlabel("Terms of approximation")
    ax.set_ylabel("Average time (us)")
    ax.legend()
    plt.show(block=True)
