import jax
import jax.numpy as jnp

def next_multiple(i: int, N: int) -> int:
    r"""
    Return smallest integer multiple of N greater than or equal to integer i.
    
    Parameters
    ----------
    i : int
        The input number.
    N : int
        The multiple to align to.

    Returns
    -------
    j : int
        The next multiple of N.

    Notes
    -----
    Example with N = 3:

    >>> for i in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
    ...     print(f"{i} -> {next_multiple(i, 4)}")
    -4 -> -3
    -3 -> -3
    -2 ->  0
    -1 ->  0
     0 ->  0
     1 ->  3
     2 ->  3
     3 ->  3
     4 ->  6
    """
    j = ((i + N - 1) // N) * N
    return j


def C_nm(n: int, m: int) -> complex:
    r"""
    Compute the complex-valued modulation coefficient :math:`C_{nm}`.

    This coefficient alternates between 1 and :math:`i` to apply modulation
    in the WDM transform.

    Parameters
    ----------
    n : int
        Time index.
    m : int
        Frequency index.

    Returns
    -------
    complex
        Coefficient :math:`C_{nm}`, equal to 1 or :math:`i` depending on 
        parity of :math:`n+m`.
    """
    return 1.0 if (n + m) % 2 == 0 else 1.0j


def circular_slice(x, start, end):
    r"""
    Return a circular (wrap-around) slice from a 1D JAX array.

    This function slices the array `x`, from index `start` to `end`. Negative
    indices, or indices greater than the length of the array, are allowed and
    will wrap around the array circularly. 

    Parameters
    ----------
    x : jnp.ndarray of shape (N,)
        A 1D JAX array from which to take the circular slice.
    start : int
        The starting index of the slice. 
    end : int
        The end index of the slice. Usually be greater than `start`, otherwise
        an empty array will be returned.

    Returns
    -------
    y : jnp.ndarray
        A 1D JAX array containing the circular slice.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> arr = jnp.arange(10)
    >>> circular_slice(arr, 3, 7)
    Array([3, 4, 5, 6], dtype=int64)
    >>> circular_slice(arr, 3, 7)
    Array([8, 9, 0, 1], dtype=int64)
    >>> circular_slice(arr, 3, 7)
    Array([8, 9, 0, 1], dtype=int64)
    """
    n = x.shape[0]
    indices = (jnp.arange(start, end) % n)
    y = x[indices]
    return y

