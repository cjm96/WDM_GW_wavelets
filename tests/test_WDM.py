import numpy as np
import jax
import jax.numpy as jnp
import WDM


def test_x64():
    r"""
    Test that the WDM module is using float64 precision.
    """
    assert jax.config.read("jax_enable_x64"), \
        "WDM module should be using float64 precision, check the __init__ file."


def test_padding():
    r"""
    Test the padding function in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=1.0, 
                                                                Nf=4, 
                                                                N=64,
                                                                q=6)

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
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=16, 
                                                                N=512)

    basis = wdm.gnm_basis() # shape (N, Nt, Nf)

    basis = basis.reshape(basis.shape[0], -1) # shape (N, Nt*Nf)

    I = basis @ basis.T * 2. * jnp.pi * wdm.dt # compute pairwise inner products

    assert jnp.allclose(I, jnp.eye(wdm.N), atol=1e-3, rtol=1e-3), \
        "Orthonormality condition failed"


def test_exact_transforms():
    r"""
    Test the exact wavelet transform.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=16, 
                                                                N=512, 
                                                                q=5)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_exact(x)

    x_ = wdm.inverse_transform_exact(w)

    assert np.allclose(x, x_, rtol=1.0e-2, atol=1.0e-2), \
        "Inverse transform did not recover original signal"
    

def test_truncated_transforms():
    r"""
    Test the truncated wavelet transform.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=4, 
                                                                N=32, 
                                                                q=4)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_exact(x)

    W = wdm.forward_transform_truncated(x)

    assert np.allclose(W, w, rtol=1.0e-2, atol=1.0e-2), \
        "Truncated transform did not match exact transform"


def test_truncated_window_transform():
    r"""
    Test the truncated window expression for the wavelet transform.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=8, 
                                                                N=256, 
                                                                q=8)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_truncated(x)

    W = wdm.forward_transform_truncated_window(x)

    assert np.allclose(W, w, rtol=1.0e-2, atol=1.0e-2), \
        "Truncated window transform did not match truncated transform"
    

def test_truncated_windowed_fft_transform():
    r"""
    Test the truncated windowed fft expression for the wavelet transform.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=8, 
                                                                N=256, 
                                                                q=8)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_truncated(x)

    W = wdm.forward_transform_truncated_windowed_fft(x)

    assert np.allclose(W[:,1:], w[:,1:], rtol=1.0e-2, atol=1.0e-2), \
        "Truncated windowed fft transform did not match truncated transform"
    

def test_time_domain_basis_functions():
    r"""
    Test that the two expressions for the time-domain basis wavelets 
    :math:`g_{nm}[k]` give the same results
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=4, 
                                                                N=64,
                                                                q=6)
    
    basis_one = wdm.gnm_basis() # shape (N, Nt, Nf)

    basis_two = jnp.array([[wdm.gnm(n,m) for m in range(wdm.Nf)] 
                       for n in range(wdm.Nt)])
    basis_two = jnp.transpose(basis_two, (2, 0, 1)) 

    assert jnp.allclose(basis_one, basis_two, rtol=1.0e-3, atol=1.0e-3), \
        "Time-domain basis functions do not match between two expressions"
    

def test_inverse_transform_fast():
    r"""
    Test the inverse transform fast and exact methods agree.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=16, 
                                                                N=512, 
                                                                q=5)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_exact(x)

    x_fast = wdm.inverse_transform_fast(w)

    x_exact = wdm.inverse_transform_exact(w)

    assert np.allclose(x_fast, x_exact, rtol=1.0e-2, atol=1.0e-2), \
        "Inverse transform fast did not match exact inverse transform"


    

