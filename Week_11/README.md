# Advanced Solutions

Here, we describe a more general and maintainable approach to constructing
finite difference equations in NumPy. This approach can be applied to systems
of any classification (i.e. parabolic, hyperbolic, elliptic).

We shall use the heat diffusion equation as our example here. Instead of fully
expanding all the finite difference approximations, we keep each approximation
grouped together, as follows. Note, we use the linear indexing $s = i\times w +
j$.

$$
T_{s}^{m+1} - T_{s}^m
\approx
\sigma_x \left(
T_{s-w}^{m+1} -
2 T_{s}^{m+1} +
T_{s+w}^{m+1}
\right) +
\sigma_y \left(
T_{s-1}^{m+1} -
2 T_{s}^{m+1} +
T_{s+1}^{m+1}
\right)
$$

where $\sigma_x$ and $\sigma_y$ are now the CFL numbers (Fourier coefficients)
for the spacing in $x$ and $y$ respectively.

$$
\sigma_x = \frac{k}{\rho C_p} \frac{\Delta t}{\Delta x^2}
,\qquad
\sigma_y = \frac{k}{\rho C_p} \frac{\Delta t}{\Delta y^2}
$$

Let's focus on one of the finite approximations, without the CFL number. Here,
we use the Kronecker Delta function $\delta_{r,s}$ to expand the indices of the
equation. The function is 1 when $r=s$ and 0 otherwise.

$$
\begin{align*}
T_{s-1}^{m+1} -
2 T_{s}^{m+1} +
T_{s+1}^{m+1}
&=
T_{r}^{m+1}\delta_{r,s-1} -
2 T_{r}^{m+1}\delta_{r,s} +
T_{r}^{m+1}\delta_{r,s+1}
\\
&=
\left(
\delta_{r,s-1} -
2 \delta_{r,s} +
\delta_{r,s+1}
\right)T_{r}^{m+1}
\end{align*}
$$

While it may look foreign, the matrix corresponding to $M_{i,j} = \delta_{i,j}$
is just the identity matrix! Ones where the row is equal to the column (the
main diagonal) and zeros elsewhere. Conversely, the shifted functions
$\delta_{r,s-1}$ and $\delta_{r,s+1}$ are *shifted* identity matrices, where
the main diagonal is shifted by 1.

$$
\delta_{r,s} =
\begin{bmatrix}
1 & 0 & 0 & \dots \\
0 & 1 & 0 &       \\
0 & 0 & 1         \\
\vdots & & & \ddots
\end{bmatrix},
\quad
\delta_{r,s-1} =
\begin{bmatrix}
0 & 1 & 0 & \dots \\
0 & 0 & 1 &       \\
0 & 0 & 0         \\
\vdots & & & \ddots
\end{bmatrix},
\quad
\delta_{r,s+1} =
\begin{bmatrix}
0 & 0 & 0 & \dots \\
1 & 0 & 0 &       \\
0 & 1 & 0         \\
\vdots & & & \ddots
\end{bmatrix}
$$

Returning to the finite difference approximation, we can group these matrices
together with their CFL number under the term $D_y^{(2)}$, to denote the finite
approximation of the second derivative in $y$.

$$
\left[D_{y}^{(2)}\right]_{r,s} = \sigma_y\left(
\delta_{r,s-1} -
2 \delta_{r,s} +
\delta_{r,s+1}
\right)
$$

Our entire PDE then becomes

$$
\begin{gather*}
T_{s}^{m+1} - T_{s}^m
\approx
\left[D_{x}^{(2)}\right]_{r,s} T_{r}^{m+1} +
\left[D_{y}^{(2)}\right]_{r,s} T_{r}^{m+1}
\\
T_{s}^{m+1} -
\left[D_{x}^{(2)}\right]_{r,s} T_{r}^{m+1} -
\left[D_{y}^{(2)}\right]_{r,s} T_{r}^{m+1}
\approx T_{s}^m
\\
\left[I - D_{x}^{(2)} - D_{y}^{(2)}\right]_{r,s} T_{r}^{m+1}
\approx T_{s}^m
\end{gather*}
$$

We can make these matrices trivially in NumPy, with the `np.eye` and `np.diag`
functions. These are very similar functions, with the main difference being
that `np.eye` will be a shifted identify, whereas `np.diag` constructs the
diagonal from an array of values.

```pycon
>>> import numpy as np
>>> np.eye(3)  # delta_{r,s}
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

>>> np.eye(3, k=+1)  # delta_{r,s-1}
array([[0., 1., 0.],
       [0., 0., 1.],
       [0., 0., 0.]])

>>> a = np.array([1, 2, 3])
>>> np.diag(a, k=-1)  # delta_{r,s+1} a_{s}
array([[0, 0, 0, 0],
       [1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0]])
```

An example of creating the $D_{x}^{(2)}$ matrix and the full system is shown in
[the solution file](./solutions.py).
