from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


#: Working-set budget, in bytes, for one frequency block of
#: `build_filter_tables`. The whole point of the blocking is that the peak
#: allocation stops depending on :math:`N`, so this only has to be small
#: enough to fit and large enough that the matrix products stay efficient.
FILTER_TABLE_BLOCK_BYTES = 1 << 28  # 256 MiB


def time_delay_filter_Tl_reference(ell : int,
                         delta : float,
                         freqs : jnp.array,
                         window_FD : jnp.array,
                         dT : float,
                         df : float) -> float:
    r"""
    The time-delay filter for the case :math:`m'=m` is defined as

    .. math::
        T_{\ell}(\delta)=\int\mathrm{d}f\exp(2\pi i f(\ell\Delta T-\delta)) 
                            |\tilde{\Phi}(f)|^2 .

    This function is SLOW and is NOT used in production; `build_filter_tables`
    builds the same quantity for a whole grid of lags and delays at once.
    It is kept as the reference implementation of the defining integral,
    against which the tabulated version is tested.

    Parameters
    ----------
    ell : int
        The time index difference :math:`\ell=n-n'`.
    delta : float
        The time delay :math:`\delta`, in the time units of `wdm`.
    freqs : jnp.ndarray
        The sample frequencies of the wdm object time series.
    window_FD : jnp.ndarray
        The frequency-domain Meyer window function, :math:`\tilde{\Phi}(f)` of 
        the wdm object.
    dT : float
        Time resolution of the the wdm object.
    df : float
        The frequency resolution of the wdm object time series.

    Returns
    -------
    T_l : float
        The time-delay filter :math:`T_{\ell}(\delta)`.
    """
    integrand = jnp.exp(2*jnp.pi*(1j)*freqs*(ell*dT-delta)) * window_FD**2

    T_l = jnp.sum(integrand) * df

    return float(T_l.real)


def time_delay_filter_Tprimel_reference(ell : int,
                         delta : float,
                         freqs : jnp.array,
                         window_FD : jnp.array,
                         dT : float,
                         dF : float,
                         N : int,
                         df : float) -> float:
    r"""
    The time-delay filter for the case :math:`m'=m\pm 1` is defined as

    .. math::
        T'_{\ell}(\delta)=\int\mathrm{d}f\exp(2\pi i f(\ell\Delta T-\delta)) 
                            \tilde{\Phi}\left(f-\frac{1}{2}\Delta F\right)
                            \tilde{\Phi}\left(f+\frac{1}{2}\Delta F\right) .

    This function is SLOW and is NOT used in production; `build_filter_tables`
    builds the same quantity for a whole grid of lags and delays at once.
    It is kept as the reference implementation of the defining integral,
    against which the tabulated version is tested.

    Parameters
    ----------
    ell : int
        The time index difference :math:`\ell=n-n'`.
    delta : float
        The time delay :math:`\delta`, in the time units of `wdm`.
    freqs : jnp.ndarray
        The sample frequencies of the wdm object time series.
    window_FD : jnp.ndarray
        The frequency-domain Meyer window function, :math:`\tilde{\Phi}(f)` of 
        the wdm object.
    dT : float
        Time resolution of the the wdm object.
    dF : float 
        Frequency resolution of the wdm object.
    N : int 
        Length of the input time series. 
    df : float
        The frequency resolution of the wdm object time series.

    Returns
    -------
    Tprime_l : float
        The time-delay filter :math:`T'_{\ell}(\delta)`.
    """
    indices = jnp.arange(N)

    shift = int(0.5*dF/df)

    integrand = jnp.exp(2*jnp.pi*(1j)*freqs*(ell*dT-delta)) * \
                    window_FD[(indices-shift)%N] * \
                    window_FD[(indices+shift)%N]

    Tprime_l = jnp.sum(integrand) * df

    return float(Tprime_l.real)

def window_support(window_FD : jnp.array,
                   shift : int) -> tuple:
    r"""
    Bound the frequency samples that can contribute to the filter integrals.

    The Meyer window is compactly supported: `window_FD` is exactly zero
    outside a band of roughly :math:`(1-A_{\rm frac})N_t` samples centred on
    :math:`f=0`, which on production grids is a factor :math:`\sim N_f`
    shorter than the array itself.

    Both window products used by `build_filter_tables` vanish wherever
    `window_FD` does. That is immediate for :math:`\tilde{\Phi}^2`. For the
    rolled product each factor is the same support translated by
    :math:`\mp` `shift`, so the product is supported on the overlap of the
    two translates, which is contained in the untranslated support - provided
    neither translate wraps around the array edge.

    Parameters
    ----------
    window_FD : jnp.ndarray
        The frequency-domain Meyer window, :math:`\tilde{\Phi}(f)`,
        shape=(N,).
    shift : int
        The half-bin index offset :math:`\Delta F/2\delta f` applied to the
        window in the :math:`T'_{\ell}` integrand.

    Returns
    -------
    start, stop : int
        Half-open index range. `(0, N)` is returned if the window is nowhere
        non-zero or if the translates would wrap, so the caller always
        integrates over a superset of the true support.
    """
    N = int(jnp.shape(window_FD)[0])

    nonzero = np.flatnonzero(np.asarray(window_FD) != 0.0)
    if nonzero.size == 0:
        return 0, N

    start, stop = int(nonzero[0]), int(nonzero[-1]) + 1

    if start - shift < 0 or stop + shift > N:
        return 0, N

    return start, stop


def _choose_freq_block(num_lags : int,
                       num_deltas : int,
                       num_freqs : int,
                       max_bytes : int) -> int:
    r"""
    Largest number of frequency samples whose temporaries fit in `max_bytes`.

    One block of `_filter_tables_block` holds the lag phase (`num_lags`
    complex values per frequency), its two windowed copies (`2*num_lags`) and
    the delay phase (`num_deltas`); the accumulator itself does not scale with
    the block.
    """
    bytes_per_freq = 16 * (3*num_lags + num_deltas)

    block = int(max_bytes // bytes_per_freq) if bytes_per_freq > 0 else num_freqs

    return max(1, min(int(num_freqs), block))


@partial(jax.jit, static_argnums=(4, 5))
def _filter_tables_block(lags_dT : jnp.array,
                         deltas : jnp.array,
                         freqs : jnp.array,
                         window_FD : jnp.array,
                         indices : tuple,
                         shift : int) -> jnp.array:
    r"""
    The contribution of one contiguous block of frequency samples.

    `indices` is `(start, stop)`, a half-open range into the frequency axis.
    It is static so that the block shape is a compile-time constant and every
    full-sized block reuses the same compiled kernel.

    The windows are gathered at the block's own indices rather than rolled
    over the whole array, which is what keeps the working set independent of
    :math:`N`. `jnp.roll(w, s)[j]` is `w[(j-s)%N]`, hence the index signs.
    """
    start, stop = indices
    N = window_FD.shape[0]

    index = jnp.arange(start, stop)

    windows = jnp.stack([window_FD[(index-shift) % N]
                            * window_FD[(index+shift) % N],
                         window_FD[index]**2])

    block_freqs = freqs[index]

    lag_phase = jnp.exp(2*jnp.pi*(1j)*jnp.outer(lags_dT, block_freqs))
    delta_phase = jnp.exp(-2*jnp.pi*(1j)*jnp.outer(block_freqs, deltas))

    return jnp.real((lag_phase*windows[:, jnp.newaxis, :]) @ delta_phase)


def build_filter_tables(lags : jnp.array,
                        deltas : jnp.array,
                        freqs : jnp.array,
                        window_FD : jnp.array,
                        dT : float,
                        dF : float,
                        df : float,
                        max_bytes : int = FILTER_TABLE_BLOCK_BYTES
                        ) -> jnp.array:
    r"""
    Evaluate both time-delay filters on a grid of lags and delays.

    :math:`T_{\ell}` and :math:`T'_{\ell}` are the same integral over a
    different window product, so they are built together. The lag phase and
    the delay phase separate, which turns the whole grid into two matrix
    products rather than a loop over :math:`(\ell,\delta)` pairs.

    The frequency axis is the contraction index, so nothing of size :math:`N`
    appears in the result - only in the intermediates. Those intermediates are
    what used to make this function unusable at year-plus durations: written
    as a single pair of matrix products they need
    :math:`16(3A+B)N` bytes, which is tens of gigabytes for
    :math:`N\sim 10^7`, and it is the choice of `num_interp_points` (:math:`B`)
    that tips a working configuration over the edge. Two things bound it here:

    - the integral is restricted to the window's compact support (see
      `window_support`), which is exact rather than an approximation and on
      production grids removes a factor of order :math:`N_f`;
    - what remains is accumulated over frequency blocks sized by `max_bytes`,
      so the peak allocation is capped whatever the grid.

    Parameters
    ----------
    lags : jnp.ndarray
        The time index differences :math:`\ell`, dtype=int, shape=(A,).
    deltas : jnp.ndarray
        The time delays :math:`\delta`, shape=(B,).
    freqs : jnp.ndarray
        The sample frequencies of the wdm object time series.
    window_FD : jnp.ndarray
        The frequency-domain Meyer window, :math:`\tilde{\Phi}(f)`.
    dT : float
        Time resolution of the wdm object.
    dF : float
        Frequency resolution of the wdm object.
    df : float
        The frequency resolution of the wdm object time series.
    max_bytes : int
        Working-set budget for one frequency block. Optional.

    Returns
    -------
    tables : jnp.ndarray
        shape=(2, A, B). Index 0 is :math:`T'_{\ell}`, index 1 is
        :math:`T_{\ell}`.
    """
    lags = jnp.asarray(lags)
    deltas = jnp.asarray(deltas)

    num_lags = int(lags.shape[0])
    num_deltas = int(deltas.shape[0])

    shift = int(0.5*dF/df)

    start, stop = window_support(window_FD, shift)

    block = _choose_freq_block(num_lags, num_deltas, stop-start, max_bytes)

    lags_dT = lags*dT

    tables = jnp.zeros((2, num_lags, num_deltas))

    for lo in range(start, stop, block):
        tables += _filter_tables_block(lags_dT,
                                       deltas,
                                       freqs,
                                       window_FD,
                                       (lo, min(lo+block, stop)),
                                       shift)

    return tables * df
