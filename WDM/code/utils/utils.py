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


def pad_signal(x : jnp.ndarray, N : int, where: str = 'end') -> jnp.ndarray:
    r"""
    The transform method requires the input time series signal to have a 
    specific length :math:`N`. This method can be used to zero-pad any 
    signal to the desired length.

    This function also returns a Boolean mask that can be used later to 
    recover arrays of the original length.

    Parameters
    ----------
    x : jnp.ndarray
        Input signal to be padded.
    N : int
        Desired length of the output signal.
    where : str
        Where to add the padding. Options are 'end', 'start', or 'equal' 
        which puts the zero padding at the end of the signal, the start of 
        the signal, or equally at both ends respectively. Optional.

    Returns
    -------
    x_padded : jnp.ndarray
        Padded signal to length N, with zeros added at the end.
    mask : jnp.ndarray
        Boolean mask indicating the valid part of the padded signal.

    Notes
    -----
    The Boolean mask can be used to get back to the original signal; i.e.
    `x_padded[mask]` will recover the original signal, `x`.
    """
    x = jnp.asarray(x)

    n = len(x)
    padding_length = N - n

    assert padding_length >= 0, \
        f"Input signal length {n} exceeds desired length {N}."

    mask = jnp.full(N, True, dtype=bool)

    if where == 'end':
        x_padded = jnp.pad(x, (0, padding_length), 
                            mode='constant', constant_values=0)
        mask = mask.at[n:].set(False)
    elif where == 'start':
        x_padded = jnp.pad(x, (padding_length, 0), 
                            mode='constant', constant_values=0)
        mask = mask.at[:padding_length].set(False)
    elif where == 'equal':
        a = padding_length // 2
        b = padding_length - a
        x_padded = jnp.pad(x, (a, b),
                            mode='constant', constant_values=0)
        mask = mask.at[:a].set(False)
        mask = mask.at[n + a:].set(False)
    else:
        raise ValueError(f"Invalid padding location {where=}.")

    return x_padded, mask
