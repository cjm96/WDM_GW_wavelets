import jax
import jax.numpy as jnp
import numpy as np

from WDM.code.discrete_wavelet_transform.WDM import WDM_transform


def overlapping_windows(x : np.array, 
                        num_perseg : int, 
                        num_overlap : int):
    """
    Slice input time series `x` into overlapping windows. Parts of the time 
    series at the end that do not fit into a complete window are discarded.

    Parameters
    ----------
    x : ndarray
        Input array. Array shape=(..., N). Windows are taken along last axis.
    num_perseg : int
        Window length.
    num_overlap : int
        Number of overlapping samples between windows.

    Returns
    -------
    windows : ndarray
        Input array. Array shape=(..., num_windows, num_perseg).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.arange(15)
    >>> windows = overlapping_segments(x, 20, 10)
    >>> windows
    array([[ 0,  1,  2,  3,  4,  5],
        [ 4,  5,  6,  7,  8,  9],
        [ 8,  9, 10, 11, 12, 13]]) # final 14 from original array is discarded
    """
    x = np.asarray(x)

    N = x.shape[-1]
    leading = x.shape[:-1]

    assert 0<num_perseg<= N, \
        f"Must have 0<num_perseg<={N}, got {num_perseg=}."
    assert 0<=num_overlap<num_perseg, \
        f"Must have 0<=num_overlap<{num_perseg}, got {num_overlap=}."
    
    step = num_perseg - num_overlap

    num_windows = (N-num_overlap) // step

    windows = np.empty(leading+(num_windows, num_perseg), dtype=x.dtype)

    for i in range(num_windows):
        start = i*step
        end = start + num_perseg
        windows[...,i,:] = x[...,start:end]

    return windows
