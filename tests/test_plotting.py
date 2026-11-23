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
                                                                q=8)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_exact(x)

    fig, ax = WDM.code.plotting.plotting.time_frequency_plot(wdm, w)
    plt.close('all')


def test_time_frequency_plot_colour_options():
    r"""
    The colour options reach the image, and the defaults are unchanged.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5,
                                                                Nf=4,
                                                                N=64,
                                                                q=8)
    x = jax.random.normal(jax.random.key(0), shape=(wdm.N,))
    w = wdm.forward_transform_exact(x)
    plot = WDM.code.plotting.plotting.time_frequency_plot

    _, ax = plot(wdm, w)
    assert ax.images[0].get_cmap().name == 'jet'
    assert ax.images[0].colorbar.ax.get_ylabel() == 'Magnitude'

    _, ax = plot(wdm, w, part='real', cmap='RdBu_r', vmin=-1., vmax=1.,
                 label='X')
    assert ax.images[0].get_cmap().name == 'RdBu_r'
    assert ax.images[0].get_clim() == (-1., 1.)
    assert ax.images[0].colorbar.ax.get_ylabel() == 'X'
    plt.close('all')
