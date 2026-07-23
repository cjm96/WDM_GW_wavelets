"""Direct reference definitions of individual WDM time-delay filters.

These routines are intentionally simple and are used for derivation-level and
small-grid tests.  Production response calculations use the prepared JAX
operators in ``plans.py`` and ``_time_shift_jax.py``.
"""

from __future__ import annotations

import jax.numpy as jnp

from WDM.code.discrete_wavelet_transform import WDM
from WDM.code.utils.utils import C_nm


def time_delay_filter_Tl(
    wdm: WDM.WDM_transform,
    ell: int,
    delta_t: float,
) -> float:
    r"""Return the diagonal-frequency delay filter :math:`T_\ell(\delta t)`."""

    integrand = (
        jnp.exp(
            2j
            * jnp.pi
            * wdm.freqs
            * (ell * wdm.dT - delta_t)
        )
        * wdm.window_FD**2
    )
    return float((jnp.sum(integrand) * wdm.df).real)


def time_delay_filter_Tprimel(
    wdm: WDM.WDM_transform,
    ell: int,
    delta_t: float,
) -> float:
    r"""Return the adjacent-frequency filter :math:`T'_\ell(\delta t)`."""

    indices = jnp.arange(wdm.N)
    shift = int(0.5 * wdm.dF / wdm.df)
    integrand = (
        jnp.exp(
            2j
            * jnp.pi
            * wdm.freqs
            * (ell * wdm.dT - delta_t)
        )
        * wdm.window_FD[(indices - shift) % wdm.N]
        * wdm.window_FD[(indices + shift) % wdm.N]
    )
    return float((jnp.sum(integrand) * wdm.df).real)


def time_delay_X(
    wdm: WDM.WDM_transform,
    n: int,
    nprime: int,
    m: int,
    mprime: int,
    delta_t: float,
) -> float:
    r"""Return one real WDM delay-matrix element.

    The retained basis approximation couples only ``m'=m`` and the two
    adjacent frequency bins.
    """

    ell = n - nprime

    if m == mprime:
        Tl = time_delay_filter_Tl(wdm, ell, delta_t)
        value = (
            (-1) ** (ell * m)
            * jnp.conj(C_nm(n, m))
            * C_nm(nprime, m)
            * Tl
            * jnp.exp(2j * jnp.pi * m * wdm.dF * delta_t)
        )
    elif mprime == m + 1:
        Tprime = time_delay_filter_Tprimel(wdm, ell, delta_t)
        value = (
            (-1) ** (ell * m)
            * (-1j) ** ell
            * jnp.conj(C_nm(n, m))
            * C_nm(nprime, mprime)
            * Tprime
            * jnp.exp(2j * jnp.pi * (m + 0.5) * wdm.dF * delta_t)
        )
    elif mprime == m - 1:
        Tprime = time_delay_filter_Tprimel(wdm, ell, delta_t)
        value = (
            (-1) ** (ell * m)
            * (+1j) ** ell
            * jnp.conj(C_nm(n, m))
            * C_nm(nprime, mprime)
            * Tprime
            * jnp.exp(2j * jnp.pi * (m - 0.5) * wdm.dF * delta_t)
        )
    else:
        value = 0.0

    return float(jnp.real(value))
