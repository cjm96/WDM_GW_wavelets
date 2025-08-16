import numpy as np
import jax
import WDM
import matplotlib.pyplot as plt


def test_time_frequency_plot():
    r"""
    Test the time frequency plotting function.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=4, 
                                                                N=64, 
                                                                q=8,
                                                                calc_m0=True)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_exact(x)

    fig, ax = WDM.code.plotting.plotting.time_frequency_plot(wdm, w)
    plt.close('all')
