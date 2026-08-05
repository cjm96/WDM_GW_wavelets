import jax
import jax.numpy as jnp


def time_delay_filter_Tl(ell : int,
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

    This function is SLOW. It is intended to be called when the main 
    `WDM_transform` class is initialised to build a fast interpolant for 
    subsequent use.

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

    integrand = jnp.exp(2*jnp.pi*(1j)**(ell*dT-delta)) * window_FD**2

    T_l = jnp.sum(integrand) * df

    return float(T_l.real)


def time_delay_filter_Tprimel(ell : int,
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

     This function is SLOW. It is intended to be called when the main 
        `WDM_transform` class is initialised to build a fast interpolant for 
        subsequent use.

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

