from functools import partial
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


@partial(jax.jit, static_argnums=(1, 2, 3))
def overlapping_windows(x: jnp.ndarray, K: int, Nt: int, Nf: int) -> jnp.ndarray:
    """
    Extract overlapping, wrapped windows from input array `x`.

    Parameters
    ----------
    x : jnp.ndarray, shape (N,)
        Input array to extract windows from.
    K : int
        Window length (must be even).
    Nt : int
        Number of windows (time steps).
    Nf : int
        Hop size between window centers.

    Returns
    -------
    windows : jnp.ndarray, shape (Nt, K)
        Array of overlapping windows with wraparound indexing.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> Nt = 4
    >>> Nf = 4
    >>> K = 8
    >>> x = jnp.arange(Nt*Nf)
    >>> overlapping_windows(x, K, Nt, Nf)
    array([ [12, 13, 14, 15,  0,  1,  2,  3],
            [ 0,  1,  2,  3,  4,  5,  6,  7],
            [ 4,  5,  6,  7,  8,  9, 10, 11],
            [ 8,  9, 10, 11, 12, 13, 14, 15] ])
    """
    N = x.shape[0]
    
    # Centered window indices relative to each window center
    k_offsets = jnp.arange(-K//2, K//2)
    
    # Window center indices
    centers = jnp.arange(Nt) * Nf
    
    # Create full (Nt, K) index matrix with wraparound
    idx = (centers[:,jnp.newaxis] + k_offsets[jnp.newaxis,:]) % N
    
    return x[idx]
