import jax
import jax.numpy as jnp


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

def build_filter_tables(lags : jnp.array,
                        deltas : jnp.array,
                        freqs : jnp.array,
                        window_FD : jnp.array,
                        dT : float,
                        dF : float,
                        df : float) -> jnp.array:
    r"""
    Evaluate both time-delay filters on a grid of lags and delays.

    :math:`T_{\ell}` and :math:`T'_{\ell}` are the same integral over a
    different window product, so they are built together. The lag phase and
    the delay phase separate, which turns the whole grid into two matrix
    products rather than a loop over :math:`(\ell,\delta)` pairs.

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

    Returns
    -------
    tables : jnp.ndarray
        shape=(2, A, B). Index 0 is :math:`T'_{\ell}`, index 1 is
        :math:`T_{\ell}`.
    """
    lag_phase = jnp.exp(2*jnp.pi*(1j)*jnp.outer(lags*dT, freqs))
    delta_phase = jnp.exp(-2*jnp.pi*(1j)*jnp.outer(freqs, deltas))

    shift = int(0.5*dF/df)

    windows = jnp.stack([jnp.roll(window_FD, shift)
                            * jnp.roll(window_FD, -shift),
                         window_FD**2])

    return jnp.real((lag_phase*windows[:, jnp.newaxis, :]) @ delta_phase) * df
