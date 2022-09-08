# Solutions

While the ICT session questions required analytical calculations for the
coefficients of the sine and cosine approximations, we can use the
[`scipy.fft`](https://docs.scipy.org/doc/scipy/reference/fft.html) library to
numerically evaluate these coefficients. These solutions are given in
[fft_solutions.py](./fft_solutions.py), and are similar to the analytical
solutions for the visualisation.

## Q1 and Q3

Both Q1 and Q3 involve purely sine representations. If the coefficients $a_n$
are known (from the `fft` module), then we can reconstruct the sample with

$$
y(x) = 2\sum_{n=1}^{N} a_{n-1} \sin\left(nx\right).
$$

Note that we index the coefficients with $n-1$, since the `fft` module does not
return a "zeroth" coefficent.

In order to calculate the coefficients, we can call `scipy.fft.dst`. The main
restriction with this function, is that it assumes the data we give it is on
the domain $[0, \pi]$. Sometimes, this means scaling the $x$ in our above
reconstruction.

```python
from scipy import fft

#                --- Discrete Sine Transform
amplitudes = fft.dst(data, norm="forward")  # <- Ensures correct scaling
#                    ---- Assumes data is over [0, pi]

approx = 0
for n in range(1, N+1):
    approx += amplitudes[n-1] * np.sin(n*x)
```

## Q2

A similar approach can be taken to Q2, except that we use `scipy.fft.dct` for
the discrete cosine transform. The recreation is slightly different too, as we
have to account for the "zeroth" coefficient being at a zero frequency, and is
technically counted twice.

$$
y(x) = a_{0} + 2\sum_{n=1}^{N} a_{n} \cos\left(nx\right).
$$

In Python, this can be done as
```python
from scipy import fft

#                --- Discrete Sine Transform
amplitudes = fft.dct(data, norm="forward")  # <- Ensures correct scaling
#                    ---- Assumes data is over [0, pi]

approx = amplitudes[0]
for n in range(1, N+1):
    approx += amplitudes[n] * np.cos(n*x)
```