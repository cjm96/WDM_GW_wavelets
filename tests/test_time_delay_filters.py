import jax
import jax.numpy as jnp

import WDM
from WDM.code.discrete_wavelet_transform import WDM
from WDM.code.time_delay_filters.filters import time_delay_filter_Tl
from WDM.code.time_delay_filters.filters import time_delay_filter_Tprimel



def test_filter_functions():
    r"""
    Test the time delay filter functions - just check that these evaluate 
    and return variables of the correct type.
    """
    wdm = WDM.WDM_transform(dt=0.5, 
                            Nf=8, 
                            N=256,
                            q=4,
                            calc_m0=True)

    L = 25  # the maximum lag index
    N = 10  # the number of interpolation points
    wdm.build_time_delay_filter_interpolants(L, N)

    ell = jnp.array(jnp.arange(-L, L+1), dtype=int)
    delta = jnp.linspace(-wdm.dT/2, wdm.dT/2, 100)

    Tl = wdm.time_delay_filter_Tl(ell, delta)
    assert Tl.shape==(len(ell), len(delta)), "Tl array wrong shape"

    Tprimel = wdm.time_delay_filter_Tprimel(ell, delta)
    assert Tprimel.shape==(len(ell), len(delta)), "Tprimel array wrong shape"


def test_time_delay():
    r"""
    Test the time delay method - check that shifting a signal using the 
    time-delay filters gives the same result as simply interpolating the signal
    in the time domain.
    """
    fs = 0.2    # sampling frequency [Hz]
    T  = 1.0e5  # duration [s]
    f0 = 3.0e-3 # central frequency [Hz]
    w  = T/15.  # duration [s]

    sine_gauss = lambda t: jnp.exp(-0.5*((t-T/2.)/w)**2)*jnp.sin(2*jnp.pi*f0*t)

    times = jnp.arange(0, T, 1/fs)
    signal = sine_gauss(times)

    N, Nf= times.shape[0], 200
    wdm = WDM.WDM_transform(dt=1/fs, Nf=Nf, N=N, q=32, calc_m0=True)

    w = wdm(signal)

    L = 25  # the maximum lag index
    N = 10  # the number of interpolation points
    wdm.build_time_delay_filter_interpolants(L, N)

    tn = jnp.arange(wdm.Nt)*wdm.dT

    delta_constant = lambda t: (T/10.)*jnp.ones_like(t)

    # shift the signal in the time domain
    shifted_signal_constant_t = jnp.interp(times + delta_constant(times), 
                                           times, 
                                           signal)

    # shift the signal in the time-frequency domain
    w_shifted_constant = wdm.apply_variable_time_shift(w, delta_constant(tn))
    shifted_signal_constant_tf = wdm.idwt(w_shifted_constant)

    # check the methods agree
    residuals = shifted_signal_constant_tf - shifted_signal_constant_t
    eps = jnp.max(jnp.abs(residuals))
    tol = 1.0e-4
    assert eps<tol, "residuals are too large"
    

