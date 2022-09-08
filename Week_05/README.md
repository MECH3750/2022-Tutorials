# Solutions

The advanced solution this week demonstrates how much code you can save when
you create useful functions that can be reused in multiple places.

At the top of [solutions.py](solutions.py), we've created the `least_squares`
function, which takes some function to approximate, the basis to use, and the
inner product to apply. It also allows you to pass any extra arguments to the
inner product with `**kwargs` (key-word arguments, i.e. `domain=(0, 1)`).

```python
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
```

You can see this is a very simple function, and yet also very general.
This allows it to be used for all three questions, vastly reducing the work!
Check out [solutions.py](solutions.py) to see how it's done.

## Note:

Remember, if you use this function in your assignments, you will need to
reference it.
