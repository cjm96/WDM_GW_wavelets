import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from WDM.code.discrete_wavelet_transform.WDM import WDM_transform
from typing import Tuple


def time_domain_plot(wdm : WDM_transform, 
                     x: jnp.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    r"""
    Plot the time-domain signal.

    Parameters
    ----------
    wdm : WDM_transform
        The WDM transform object.
    x : jnp.ndarray
        Array shape (N,). Input time-domain signal to be plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the wavelets were plotted.

    Notes
    -----
    This function does not call `plt.show()`. The user is responsible
    for displaying or saving the plot.
    """
    assert x.shape == (wdm.N,), \
                f"Input signal must have shape ({wdm.N},), got {x.shape=}"
    
    fig, ax = plt.subplots()
    ax.plot(wdm.times, x)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Signal $x(t)$')
    return fig, ax


def frequency_domain_plot(wdm : WDM_transform, 
                          x: jnp.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    r"""
    Plot the frequency-domain signal.

    Parameters
    ----------
    wdm : WDM_transform
        The WDM transform object.
    x : jnp.ndarray
        Array shape (N,). Input time-domain signal to be plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the wavelets were plotted.

    Notes
    -----
    This function does not call `plt.show()`. The user is responsible
    for displaying or saving the plot.
    """
    assert x.shape == (wdm.N,), \
                f"Input signal must have shape ({wdm.N},), got {x.shape=}"
    
    data = jnp.abs(jnp.fft.fft(x))
    mask = wdm.freqs >= 0.

    fig, ax = plt.subplots()
    ax.loglog(wdm.freqs[mask], data[mask])
    ax.set_xlabel(r'Frequency $f$')
    ax.set_ylabel(r'Signal $|\tilde{X}(f)|$')
    return fig, ax


def time_frequency_plot(wdm : WDM_transform, 
                        w: jnp.ndarray, 
                        part='abs',
                        scale='linear') -> Tuple[plt.Figure, plt.Axes]:
    r"""
    Plot the time-frequency coefficients of the WDM transform.

    Parameters
    ----------
    wdm : WDM_transform
        The WDM transform object.
    w : jnp.ndarray of shape (Nt, Nf)
        WDM time-frequency coefficients to be plotted.
    part : str
        Part of the coefficients to plot. Options are 'abs' for magnitude, 
        'real', or 'imag'. Default is 'abs'. Optional.
    scale : str
        Scale of the colour axis of the plot. Passed to matplotlib. 
        Options are 'linear' or 'log'. Default is 'linear'. Logarithmic 
        scale should only be used with part='abs' otherwise problems with 
        negative values will occur. Optional.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the wavelets were plotted.

    Notes
    -----
    This function does not call `plt.show()`. The user is responsible
    for displaying or saving the plot.
    """
    assert w.shape == (wdm.Nt, wdm.Nf), \
                f"Input signal must have shape ({wdm.Nt}, {wdm.Nf}), " \
                f"got {w.shape=}"

    if part == 'abs':
        data = jnp.abs(w)
    elif part == 'real':
        data = jnp.real(w)
    elif part == 'imag':
        data = jnp.imag(w)
    else:
        raise ValueError(f"Invalid {part=}. " + 
                            "Choose 'abs', 'real', or 'imag'.")

    fig, ax = plt.subplots()
    if scale == 'linear':
        im = ax.imshow(data.T, aspect='auto', origin='lower', 
                    extent=[0., wdm.T, 0., wdm.f_Ny], cmap='jet')
        fig.colorbar(im, label='Magnitude', ax=ax)
    elif scale == 'log':
        im = ax.imshow(jnp.log10(data).T, aspect='auto', origin='lower', 
                    extent=[0., wdm.T, 0., wdm.f_Ny], cmap='jet')
        fig.colorbar(im, label='log10 Magnitude', ax=ax)
    else:
        raise ValueError(f"Invalid {scale=}. Choose 'linear' or 'log'.")
    
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Frequency $f$')
    return fig, ax