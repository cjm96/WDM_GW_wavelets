import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import WDM


def test_padding():
    r"""
    Test the padding function in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=1.0, 
                                                                Nf=2**5, 
                                                                N=2**6)

    x = np.array([9.,9.,9.])

    for where in ['end', 'start', 'equal']:

        x_padded, mask = wdm.pad_signal(x, where=where)

        assert len(x_padded)%wdm.Nf == 0, \
            "Length of padded signal must be a multiple of Nf"
        
        assert jnp.all(jnp.array_equal(x, x_padded[mask])), \
            "Padded signal should match original signal when mask is applied"
        

def test_Gnm():
    r"""
    Test the frequency-domain Gnm function in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=1.0, 
                                                                Nf=16, 
                                                                N=512)
    
    for m in range(3):
        for n in range(3):
            G = wdm.Gnm(n, m)

    G = wdm.Gnm(n=0, m=0, freq=jnp.array([0.0, 0.5]))
    assert G.shape == (2,), \
        "Gnm should return an array with the same shape as the input freqs"


def test_gnm():
    r"""
    Test the time-domain gnm function in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=1.0, 
                                                                Nf=16, 
                                                                N=512)
    
    for m in range(3):
        for n in range(3):
            g = wdm.gnm(n, m)

    g = wdm.gnm(n=0, m=0, time=jnp.array([0.0, 1.0]))
    assert g.shape == (2,), \
        "gnm should return an array with the same shape as the input freqs"


def test_orthonormality():
    r"""
    Test the orthonormality of the WDM wavelets.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=1.0, 
                                                                Nf=4, 
                                                                N=16)

    for m in range(wdm.Nf):
        for n in range(wdm.Nt):
            for m_ in range(wdm.Nf):
                for n_ in  range(wdm.Nt):
                    if n == n_ and m == m_:
                        expected = 1.0
                    else:
                        expected = 0.0
                    actual = np.sum(
                                    wdm.gnm(n=n, m=m) * wdm.gnm(n=n_, m=m_)
                                ) * wdm.dt * 2*np.pi
                    check = np.isclose(actual, expected, 
                                       rtol=1.0e-5, atol=1.0e-5)
                    assert check, \
                        f"Failed for (n,m)=({n},{m}), (n',m')=({n_},{m_}): " \
                        f"expected {expected}, got {actual}"


def test_exact_transforms():
    r"""
    Test the orthonormality of the WDM wavelets.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=1., 
                                                                Nf=16, 
                                                                N=512, 
                                                                q=5)

    x = np.random.randn(wdm.N) # white noise

    w = wdm.forward_transform_exact(x)

    x_ = wdm.inverse_transform_exact(w)

    assert np.allclose(x, x_, rtol=1.0e-3, atol=1.0e-3), \
        "Inverse transform did not recover original signal"
    

def test_truncated_transforms():
    r"""
    Test the orthonormality of the WDM wavelets.
    """
    pass


def test_fast_transforms():
    r"""
    Test the orthonormality of the WDM wavelets.
    """
    pass
