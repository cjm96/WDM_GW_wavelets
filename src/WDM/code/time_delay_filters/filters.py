import numpy as np
import jax.numpy as jnp

from WDM.code.utils.Meyer import Meyer


#: Gauss-Legendre nodes on each smooth piece of the window, before the extra
#: nodes `_quadrature` adds for the oscillation of the integrand.
NUM_QUADRATURE_NODES = 64


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

    This evaluates the integral as a sum over the FFT frequencies of a record
    of :math:`N` samples, so it is periodic in :math:`\ell\Delta T-\delta`
    with the record length. `build_filter_tables` evaluates the integral
    itself, which differs from this sum only by those periodic copies.

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

    This evaluates the integral as a sum over the FFT frequencies of a record
    of :math:`N` samples, so it is periodic in :math:`\ell\Delta T-\delta`
    with the record length. `build_filter_tables` evaluates the integral
    itself, which differs from this sum only by those periodic copies.

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

    # sampled at freqs + offset so both windows land on the grid
    shift_lo, shift_hi, offset = half_bin_shifts(dF, df)

    integrand = jnp.exp(2*jnp.pi*(1j)*(freqs+offset)*(ell*dT-delta)) * \
                    window_FD[(indices-shift_lo)%N] * \
                    window_FD[(indices+shift_hi)%N]

    Tprime_l = jnp.sum(integrand) * df

    return float(Tprime_l.real)


def half_bin_shifts(dF : float, df : float) -> tuple:
    r"""
    Where to sample the :math:`T'_{\ell}` integrand so that both windows
    :math:`\tilde{\Phi}(f\mp\Delta F/2)` fall on samples of `window_FD`.

    The two windows are :math:`\Delta F/\delta f = N_t/2` samples apart, a
    whole number, but each sits :math:`N_t/4` samples from :math:`f`, which is
    a half-integer :math:`k+\tfrac12` unless :math:`N_t` is a multiple of four.
    In that case the integrand is sampled at :math:`f_j+\delta f/2`, which puts
    the windows at :math:`f_{j-k}` and :math:`f_{j+k+1}`, both on the grid.

    Parameters
    ----------
    dF : float
        Frequency resolution of the wdm object.
    df : float
        The frequency resolution of the wdm object time series.

    Returns
    -------
    shift_lo, shift_hi : int
        Index offsets of the lower and upper window: :math:`k` and :math:`k`,
        or :math:`k` and :math:`k+1`.
    offset : float
        Frequency offset of the integrand samples, :math:`0` or
        :math:`\delta f/2`.
    """
    # Nt/2, a whole number since Nt is even; round only absorbs float error
    full = int(round(dF/df))

    shift_lo = full // 2
    shift_hi = full - shift_lo

    # the split is uneven, and the samples need moving, only when full is odd
    offset = 0.5*df if full % 2 else 0.0

    return shift_lo, shift_hi, offset


def window_in_bins(u : jnp.array, d : int, A_frac : float) -> jnp.array:
    r"""
    The Meyer window as a function of frequency in units of :math:`\Delta F`,
    :math:`\Phi(u)=\sqrt{\Delta F}\,\tilde{\Phi}(u\Delta F)`, normalised so
    that :math:`\int\mathrm{d}u\,\Phi(u)^2=1`.

    Parameters
    ----------
    u : jnp.ndarray
        Frequency in units of :math:`\Delta F`.
    d : int
        Steepness parameter of the window.
    A_frac : float
        Bandwidth fraction of the flat top.

    Returns
    -------
    Phi : jnp.ndarray
        The window, shaped like `u`.
    """
    two_pi = 2.*jnp.pi
    return jnp.sqrt(two_pi) * Meyer(two_pi*jnp.asarray(u), d,
                                    A_frac*two_pi, (1.-2.*A_frac)*two_pi)


def _quadrature(kinks : list, x_max : float) -> tuple:
    r"""
    Gauss-Legendre nodes and weights on each interval between consecutive
    `kinks`, with enough nodes per interval to resolve
    :math:`\cos(\pi u x)` up to :math:`|x|=` `x_max`.
    """
    nodes, weights = [], []
    for a, b in zip(kinks[:-1], kinks[1:]):
        num = NUM_QUADRATURE_NODES + int(np.ceil(0.5*np.pi*x_max*(b - a)))
        g, w = np.polynomial.legendre.leggauss(num)
        nodes.append(0.5*(b - a)*g + 0.5*(b + a))
        weights.append(0.5*(b - a)*w)
    return np.concatenate(nodes), np.concatenate(weights)


def filter_functions(x : jnp.array, d : int, A_frac : float) -> jnp.array:
    r"""
    The time-delay filters as functions of :math:`x=\ell-\delta/\Delta T`,

    .. math::
        T(x)=\int\mathrm{d}u\,\Phi(u)^2\cos(\pi u x) , \qquad
        T'(x)=\int\mathrm{d}u\,\Phi\left(u-\tfrac12\right)
                              \Phi\left(u+\tfrac12\right)\cos(\pi u x) ,

    with :math:`\Phi` the window in units of :math:`\Delta F`
    (`window_in_bins`). These are :math:`T_{\ell}(\delta)` and
    :math:`T'_{\ell}(\delta)` for every tiling: they depend on the window
    shape alone. Both integrands are even and smooth between the window's
    kinks, so each is a Gauss-Legendre sum over :math:`u\geq0`, split there.

    Parameters
    ----------
    x : jnp.ndarray
        :math:`\ell-\delta/\Delta T`. Any shape.
    d : int
        Steepness parameter of the window.
    A_frac : float
        Bandwidth fraction of the flat top.

    Returns
    -------
    filters : jnp.ndarray
        shape=(2,)+x.shape. Index 0 is :math:`T'`, index 1 is :math:`T`.
    """
    x = jnp.asarray(x)
    x_max = float(jnp.max(jnp.abs(x))) if x.size else 0.0

    edge = 1. - A_frac                      # where the window reaches zero

    # Phi(u)^2: kinks where |u| meets A_frac or the edge
    u, w = _quadrature(sorted({0., A_frac, edge}), x_max)
    weights_T = w * window_in_bins(u, d, A_frac)**2

    # Phi(u-1/2)Phi(u+1/2) is zero beyond u = edge - 1/2; kinks where u -+ 1/2
    # meets +-A_frac or +-edge
    top = edge - 0.5
    kinks = {0., top} | {k for c in (A_frac, edge) for k in (0.5 - c, c - 0.5)
                         if 0. < k < top}
    up, wp = _quadrature(sorted(kinks), x_max)
    weights_Tp = wp * window_in_bins(up - 0.5, d, A_frac) \
                    * window_in_bins(up + 0.5, d, A_frac)

    flat = x.reshape(-1)
    T = 2.*jnp.cos(jnp.pi*jnp.outer(flat, u)) @ weights_T
    Tp = 2.*jnp.cos(jnp.pi*jnp.outer(flat, up)) @ weights_Tp

    return jnp.stack([Tp, T]).reshape((2,) + x.shape)


def build_filter_tables(lags : jnp.array,
                        delta_over_dT : jnp.array,
                        d : int,
                        A_frac : float) -> jnp.array:
    r"""
    Evaluate both time-delay filters on a grid of lags and delays,
    :math:`T_{\ell}(\delta)=T(\ell-\delta/\Delta T)` and likewise for
    :math:`T'`; see `filter_functions`.

    Parameters
    ----------
    lags : jnp.ndarray
        The time index differences :math:`\ell`, dtype=int, shape=(A,).
    delta_over_dT : jnp.ndarray
        The time delays in units of :math:`\Delta T`, shape=(B,).
    d : int
        Steepness parameter of the window.
    A_frac : float
        Bandwidth fraction of the flat top.

    Returns
    -------
    tables : jnp.ndarray
        shape=(2, A, B). Index 0 is :math:`T'_{\ell}`, index 1 is
        :math:`T_{\ell}`.
    """
    x = jnp.asarray(lags)[:, jnp.newaxis] \
            - jnp.asarray(delta_over_dT)[jnp.newaxis, :]
    return filter_functions(x, d, A_frac)
