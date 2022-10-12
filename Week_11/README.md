# Solution

## Grid Indexing

While the grid indexing can be arbitary, it makes sense to choose an indexing
where neighbouring elements are stored close by, and where the index can be
quickly calculated from grid coordinates.

Consider the (relatively standard) grid indexing below:

```
 Row starts at a multiple of 5 (the width)
 vv
(15) == (16) == (17) == (18) == (19)  \
 ||      ||      ||      ||      ||   :
 ||      ||      ||      ||      ||   :
(10) == (11) == (12) == (13) == (14)  :
 ||      ||      ||      ||      ||   : height = 4
 ||      ||      ||      ||      ||   :
(05) == (06) == (07) == (08) == (09)  :
 ||      ||      ||      ||      ||   :
 ||      ||      ||      ||      ||   :
(00) == (01) == (02) == (03) == (04)  /

 \................................/
             width = 5
```

Here, the grid index $s$ can be calculated from $s = i\times w + j$ where $w$
is the width of the grid. We can see the horizontally neighbouring elements are
separated by 1, and vertically neighbouring elements by $w$.

$$
\Delta_x s = \frac{\partial s}{\partial j}\Delta j \Rightarrow 1,
\qquad
\Delta_y s = \frac{\partial s}{\partial i}\Delta i \Rightarrow w
$$

Following this through to our finite difference approximations, we can see that

$$
\begin{gather}
\frac{\partial u}{\partial x}
\approx \frac{u_{i,j+1} - u_{i,j-1}}{\Delta x}
= \frac{u_{s+1} - u_{s-1}}{\Delta x}
\\
\frac{\partial u}{\partial y}
\approx \frac{u_{i+1,j} - u_{i-1,j}}{\Delta y}
= \frac{u_{s+w} - u_{s-w}}{\Delta y}
\end{gather}
$$

## Finite Difference

The heat equation in 2D follows the PDE

$$
\frac{\partial T}{\partial t}
= \frac{k}{\rho C_p} \nabla^2 T
= \frac{k}{\rho C_p} \left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right)
$$

Using the finite difference of the second derivative in $x$ as

$$
\frac{\partial^2 T}{\partial x^2} = \frac{T_{i-1,j} - 2T_{i,j} +
T_{i+1,j}}{\Delta x^2} + \mathcal{O}(\Delta x^2)
$$

We can re-arrange the heat equation PDE to the following implicit scheme

$$
(1+4\sigma)T_{i,j}^{m+1} -
\sigma \left(
T_{i-1,j}^{m+1} +
T_{i+1,j}^{m+1} +
T_{i,j-1}^{m+1} +
T_{i,j+1}^{m+1}
\right) \approx T_{i,j}^m
$$
