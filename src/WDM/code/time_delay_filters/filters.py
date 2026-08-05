import jax
import jax.numpy as jnp

from WDM.code.discrete_wavelet_transform import WDM
from WDM.code.utils.utils import C_nm


def time_delay_filter_Tl(ell : int,
                         delta_t : float,
                         freqs : jnp.array,
                         window_FD : jnp.array,
                         dT : float,
                         df : float) -> float:
    r"""
    The time-delay filter for the case :math:`m'=m` is defined as

    .. math::
        T_{\ell}(\delta t)=\int\mathrm{d}f\exp(2\pi i f(\ell\Delta T-\delta t)) 
                            |\tilde{\Phi}(f)|^2 .

    This function is SLOW. It is intended to be called when the main 
    `WDM_transform` class is initialised to build a fast interpolant for 
    subsequent use.

    Parameters
    ----------
    ell : int
        The time index difference :math:`\ell=n-n'`.
    delta_t : float
        The time delay :math:`\delta t`, in the time units of `wdm`.
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
        The time-delay filter :math:`T_{\ell}(\delta t)`.
    """

    integrand = jnp.exp(2*jnp.pi*(1j)**(ell*dT-delta_t)) * window_FD**2

    T_l = jnp.sum(integrand) * df

    return float(T_l.real)


def time_delay_filter_Tprimel(ell : int,
                         delta_t : float,
                         freqs : jnp.array,
                         window_FD : jnp.array,
                         dT : float,
                         dF : float,
                         N : int,
                         df : float) -> float:
    r"""
    The time-delay filter for the case :math:`m'=m\pm 1` is defined as

    .. math::
        T'_{\ell}(\delta t)=\int\mathrm{d}f\exp(2\pi i f(\ell\Delta T-\delta t)) 
                            \tilde{\Phi}\left(f-\frac{1}{2}\Delta F\right)
                            \tilde{\Phi}\left(f+\frac{1}{2}\Delta F\right) .

     This function is SLOW. It is intended to be called when the main 
        `WDM_transform` class is initialised to build a fast interpolant for 
        subsequent use.

    Parameters
        ----------
        ell : int
            The time index difference :math:`\ell=n-n'`.
        delta_t : float
            The time delay :math:`\delta t`, in the time units of `wdm`.
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
        The time-delay filter :math:`T'_{\ell}(\delta t)`.
    """
    indices = jnp.arange(N)

    shift = int(0.5*dF/df)

    integrand = jnp.exp(2*jnp.pi*(1j)*freqs*(ell*dT-delta_t)) * \
                    window_FD[(indices-shift)%N] * \
                    window_FD[(indices+shift)%N]

    Tprime_l = jnp.sum(integrand) * df

    return float(Tprime_l.real)


def time_delay_X(wdm : WDM.WDM_transform, 
                 n : int, 
                 nprime : int, 
                 m : int, 
                 mprime : int, 
                 delta_t : float) -> float:
    r"""
    Compute the time-delay matrix element :math:`X_{nn';mm'}(\delta t)`,

    .. math::
        X_{nn';mm'}(\delta t) = \int\mathrm{d}t g_{nm}(t+\delta t)g^*_{n'm'}(t).

    This will return zero unless :math:`m'=m`, or :math:`m'=m\pm 1`.

    Parameters
    ----------
    wdm : WDM.WDM_transform
        An instance of the WDM_transform class. This defines the wavelet basis.
    n : int
        The time index :math:`n`.
    nprime : int
        The time index :math:`n'`.
    m : int
        The frequency index :math:`m`.
    mprime : int
        The frequency index :math:`m'`.
    delta_t : float
        The time delay :math:`\delta t`, in the time units of `wdm`.

    Returns
    -------
    X : float
        The time-delay matrix element :math:`X_{nn';mm'}(\delta t)`.
    """
    ell = n - nprime

    if m == mprime:
        Tl = time_delay_filter_Tl(wdm, ell, delta_t)
        X = (-1)**(ell*m) * \
                jnp.conj(C_nm(n,m)) * C_nm(nprime,m) * Tl * \
                jnp.exp(2*jnp.pi*(1j)*m*wdm.dF*delta_t)

    elif mprime == m+1:
        Tprimel = time_delay_filter_Tprimel(wdm, ell, delta_t)
        X = (-1)**(ell*m) * (-1j)**(ell) * \
                jnp.conj(C_nm(n,m)) * C_nm(nprime,mprime) * \
                Tprimel * jnp.exp(2*jnp.pi*(1j)*(m+0.5)*wdm.dF*delta_t)
        
    elif mprime == m-1:
        Tprimel = time_delay_filter_Tprimel(wdm, ell, delta_t)
        X = (-1)**(ell*m) * (+1j)**(ell) * \
                jnp.conj(C_nm(n,m)) * C_nm(nprime,mprime) * \
                Tprimel * jnp.exp(2*jnp.pi*(1j)*(m-0.5)*wdm.dF*delta_t)

    else:
        X = 0.0

    return X.real

