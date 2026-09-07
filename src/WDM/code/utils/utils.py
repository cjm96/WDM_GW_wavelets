import os
from functools import partial
import jax
import jax.numpy as jnp


#: Fraction of the machine's memory that one working set may occupy before
#: the reference transforms refuse to run. Unlike
#: `filters.FILTER_TABLE_BLOCK_BYTES`, which is a block size chosen so that
#: many blocks fit at once, this is a refusal threshold: the arrays it
#: guards are built whole and cannot be blocked. Raise it if you have
#: headroom.
MAX_WORKING_SET_FRACTION = 0.25

#: Working-set limit used when the machine's memory cannot be detected.
MAX_WORKING_SET_BYTES_FALLBACK = 1 << 30  # 1 GiB


def detect_memory_bytes() -> int:
    r"""
    Total memory available to JAX, in bytes.

    Prefers the memory limit of the default JAX device, so that a run on an
    accelerator is bounded by device memory rather than host memory. The CPU
    backend reports no such limit, in which case the host's physical memory
    is used. If neither can be determined - `os.sysconf` is absent on
    Windows - `MAX_WORKING_SET_BYTES_FALLBACK` is returned.

    Returns
    -------
    nbytes : int
        Total memory, in bytes.
    """
    try:
        stats = jax.devices()[0].memory_stats()
        if stats is not None and stats.get('bytes_limit'):
            return int(stats['bytes_limit'])
    except Exception:
        pass

    try:
        return int(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
    except (ValueError, AttributeError, OSError):
        return MAX_WORKING_SET_BYTES_FALLBACK


def check_working_set(nbytes: int, what: str, instead: str) -> None:
    r"""
    Refuse to build an array that would not fit in memory.

    The reference and truncated transforms build dense arrays whose size
    grows as :math:`N^2` (the wavelet bases) or :math:`qNN_f` (the truncated
    window transform). At production grid sizes these reach tens or hundreds
    of GiB - enough to take a machine down rather than merely run slowly -
    so it is better to refuse with an explanation than to start allocating.

    The guarded methods are jitted with `self` static, so :math:`N`,
    :math:`N_t`, :math:`N_f` and :math:`K` are compile-time constants at the
    call site: this check costs arithmetic at trace time and nothing at all
    at run time.

    Parameters
    ----------
    nbytes : int
        Estimated peak working set, in bytes.
    what : str
        Description of the array being built, naming its shape. Used in the
        error message.
    instead : str
        Name of the production method to suggest instead.

    Returns
    -------
    None

    Raises
    ------
    MemoryError
        If `nbytes` exceeds `MAX_WORKING_SET_FRACTION` of detected memory.
    """
    limit = MAX_WORKING_SET_FRACTION * detect_memory_bytes()

    if nbytes > limit:
        raise MemoryError(
            f"{what} needs about {format_bytes(nbytes)}, which exceeds the "
            f"working-set limit of {format_bytes(limit)} "
            f"({100*MAX_WORKING_SET_FRACTION:.0f}% of detected memory). "
            f"This method is intended for testing and debugging; use "
            f"`{instead}` instead, which does not build this array. To "
            f"override, raise "
            f"`WDM.code.utils.utils.MAX_WORKING_SET_FRACTION`.")


def format_bytes(nbytes: float) -> str:
    r"""
    Render a number of bytes in the largest unit that keeps it above one.

    Parameters
    ----------
    nbytes : float
        A number of bytes.

    Returns
    -------
    text : str
        Human-readable size.

    Notes
    -----
    Example:

    >>> for n in [512, 1 << 18, 1 << 28, 1 << 34, 1 << 44]:
    ...     print(f"{n} -> {format_bytes(n)}")
    512 -> 512.00 B
    262144 -> 256.00 KiB
    268435456 -> 256.00 MiB
    17179869184 -> 16.00 GiB
    17592186044416 -> 16.00 TiB
    """
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if nbytes < 1024.:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.
    return f"{nbytes:.2f} TiB"


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
    ...     print(f"{i:2d} -> {next_multiple(i, 3):2d}")
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
    >>> print(overlapping_windows(x, K, Nt, Nf))
    [[12 13 14 15  0  1  2  3]
     [ 0  1  2  3  4  5  6  7]
     [ 4  5  6  7  8  9 10 11]
     [ 8  9 10 11 12 13 14 15]]
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
