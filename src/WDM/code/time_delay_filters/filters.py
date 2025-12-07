import jax
import jax.numpy as jnp

from WDM.code.discrete_wavelet_transform import WDM
from WDM.code.utils.utils import C_nm


def time_delay_filter_Tl(wdm : WDM.WDM_transform, 
                         ell : int,
                         delta_t : float) -> float:
    r"""
    The time-delay filter for the case :math:`m'=m` is defined as

    .. math::
        T_{\ell}(\delta t)=\int\mathrm{d}f\exp(2\pi i f(\ell\Delta T-\delta t)) 
                            |\tilde{\Phi}(f)|^2 .

    Parameters
    ----------
    wdm : WDM.WDM_transform
        An instance of the WDM_transform class. This defines the wavelet basis.
    ell : int
        The time index difference :math:`\ell=n-n'`.
    delta_t : float
        The time delay :math:`\delta t`, in the time units of `wdm`.

    Returns
    -------
    T_l : float
        The time-delay filter :math:`T_{\ell}(\delta t)`.
    """
    integrand = jnp.exp(2*jnp.pi*(1j)*wdm.freqs*(ell*wdm.dT-delta_t)) * \
                  wdm.window_FD**2

    T_l = jnp.sum(integrand) * wdm.df

    return float(T_l.real)


def time_delay_filter_Tprimel(wdm : WDM.WDM_transform, 
                              ell : int,
                              delta_t : float) -> complex:
    r"""
    The time-delay filter for the case :math:`m'=m\pm 1` is defined as

    .. math::
        T'_{\ell}(\delta t)=\int\mathrm{d}f\exp(2\pi i f(\ell\Delta T-\delta t)) 
                            \tilde{\Phi}\left(f-\frac{1}{2}\Delta F\right)
                            \tilde{\Phi}\left(f+\frac{1}{2}\Delta F\right) .

    Parameters
    ----------
    wdm : WDM.WDM_transform
        An instance of the WDM_transform class. This defines the wavelet basis.
    ell : int
        The time index difference :math:`\ell=n-n'`.
    delta_t : float
        The time delay :math:`\delta t`, in the time units of `wdm`.

    Returns
    -------
    Tprime_l : float
        The time-delay filter :math:`T'_{\ell}(\delta t)`.
    """
    indices = jnp.arange(wdm.N)

    shift = int(0.5*wdm.dF/wdm.df)

    integrand = jnp.exp(2*jnp.pi*(1j)*wdm.freqs*(ell*wdm.dT-delta_t)) * \
                    wdm.window_FD[(indices-shift)%wdm.N] * \
                    wdm.window_FD[(indices+shift)%wdm.N]

    Tprime_l = jnp.sum(integrand) * wdm.df

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

