import numpy as np
import jax.numpy as jnp

from WDM.code.utils.Meyer import Meyer


#: Gauss-Legendre nodes on each smooth piece of the window, before the extra
#: nodes `_quadrature` adds for the oscillation of the integrand.
NUM_QUADRATURE_NODES = 64


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
