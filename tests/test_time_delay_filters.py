import jax
import jax.numpy as jnp

import WDM
from WDM.code.discrete_wavelet_transform import WDM
from WDM.code.time_delay_filters.filters import time_delay_filter_Tl
from WDM.code.time_delay_filters.filters import time_delay_filter_Tprimel
from WDM.code.time_delay_filters.filters import time_delay_X

from scipy.interpolate import interp1d


def test_filter_functions():
    r"""
    Test the time delay filter functions - just check that these evaluate 
    and return variables of the correct type.
    """
    wdm = WDM.WDM_transform(dt=0.5, 
                            Nf=8, 
                            N=64,
                            q=4,
                            calc_m0=True)
    
    ell = 0
    delta_t = 1.0

    Tl = time_delay_filter_Tl(wdm, ell, delta_t)
    assert isinstance(Tl, float), "oh dear"

    Tprimel = time_delay_filter_Tprimel(wdm, ell, delta_t)
    assert isinstance(Tprimel, float), "oh dear"


def test_filter_X_orthogonality():
    r"""
    Test the orthogonality of the time-delay matrix elements. We should have 
    the following property hold with zero time delay:

    .. math::
        X_{nn';mm'}(\delta t = 0) = \delta_{nn'} \delta_{mm'}
    """
    wdm = WDM.WDM_transform(dt=0.5, 
                            Nf=8, 
                            N=64,
                            q=4,
                            calc_m0=True)
    
    delta_t = 0.0
    
    for n in range(wdm.Nt):
        for n_ in range(wdm.Nt):

            for m in range(wdm.Nf):
                for m_ in range(wdm.Nf):

                    X = time_delay_X(wdm, n, n_, m, m_, delta_t)

                    expected = 1.0 if (n == n_ and m == m_) else 0.0

                    assert jnp.isclose(X, expected), \
                        "the X coefficients are not orthogonal!"
                    

def test_filter_X_expressions():
    r"""
    Test the expressions of the time-delay matrix elements implemented in the 
    function `time_delay_X` by comparing against direct numerical integration
    of the defining expression,

    .. math::
        X_{nn';mm'}(\delta t)=\int\mathrm{d}t g_{nm}(t+\delta t)g^*_{n'm'}(t).

    This test only checks for :math:`n` times indices away from the edges of the 
    allowed range (`boundary=12`) to avoid edge effects.
    """
    Nf = 8
    Nt = 32

    wdm = WDM.WDM_transform(dt=0.5, 
                            Nf=Nf, 
                            N=Nf*Nt,
                            q=4,
                            calc_m0=True)
    
    delta_t = 0.5 * wdm.dT

    boundary = 12 # avoid periodic edge effects not captured by interpolation

    for n in range(boundary, wdm.Nt - boundary):
        for m in range(wdm.Nf):

            for n_ in range(boundary, wdm.Nt - boundary):
                for m_ in range(wdm.Nf):

                    if m==0 or m_==0:
                        pass

                    else:
                        X_expression = time_delay_X(wdm, n, n_, m, m_, delta_t)

                        g_nprime_mprime = wdm.gnm(n_, m_)
                        g_nm_shifted = interp1d(wdm.times, wdm.gnm(n,m), 
                                                bounds_error=False, 
                                                fill_value=0.0)(wdm.times+delta_t)

                        X_direct_integral = wdm.dt*jnp.sum(g_nprime_mprime*g_nm_shifted)

                        assert jnp.isclose(X_expression, X_direct_integral, atol=1e-3, rtol=1e-3), \
                            "the X coefficients do not match the direct integral!" + \
                            f" n={n}, m={m}, n'={n_}, m'={m_}, delta_t={delta_t}: " + \
                            f"X_expression={X_expression}, X_direct_integral={X_direct_integral}"

