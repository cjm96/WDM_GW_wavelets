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
    Test the frequency-domain Gnm functions in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=8, 
                                                                N=64,
                                                                q=4,
                                                                calc_m0=True)
    
    Gnm_basis = wdm.Gnm_basis()

    assert Gnm_basis.shape == (wdm.N, wdm.Nt, wdm.Nf), \
        "Gnm_basis should return an array with shape (N, Nt, Nf)."
    
    Gnm_basis_slow = jnp.transpose(
                        jnp.array([[wdm.Gnm(n,m) 
                                 for m in range(wdm.Nf)] 
                                  for n in range(wdm.Nt)]), 
                        (2, 0, 1))
    
    assert jnp.allclose(Gnm_basis, Gnm_basis_slow, rtol=1.0e-3, atol=1.0e-3), \
        "The two methods for computing Gnm_basis should match."
    
    # reshape to (N, Nt*Nf) for orthonormality check
    Gnm_basis = Gnm_basis.reshape(Gnm_basis.shape[0], -1) 

    I = jnp.conj(Gnm_basis) @ Gnm_basis.T * wdm.df

    assert jnp.allclose(I, jnp.eye(wdm.N), atol=1e-3, rtol=1e-3), \
        "The Gnm_basis should be orthonormal."
    

def test_gnm():
    r"""
    Test the time-domain gnm functions in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=8, 
                                                                N=64,
                                                                q=4,
                                                                calc_m0=True)
    
    gnm_basis = wdm.gnm_basis()

    assert gnm_basis.shape == (wdm.N, wdm.Nt, wdm.Nf), \
        "Gnm_basis should return an array with shape (N, Nt, Nf)."
    
    gnm_basis_slow = jnp.transpose(
                        jnp.array([[wdm.gnm(n,m) 
                                 for m in range(wdm.Nf)] 
                                  for n in range(wdm.Nt)]), 
                        (2, 0, 1))
    
    assert jnp.allclose(gnm_basis, gnm_basis_slow, rtol=1.0e-3, atol=1.0e-3), \
        "The two methods for computing gnm_basis should match."
    
    # reshape to (N, Nt*Nf) for orthonormality check
    gnm_basis = gnm_basis.reshape(gnm_basis.shape[0], -1) 

    I = jnp.conj(gnm_basis) @ gnm_basis.T * wdm.dt

    assert jnp.allclose(I, jnp.eye(wdm.N), atol=1e-3, rtol=1e-3), \
        "The gnm_basis should be orthonormal."
    

def test_exact_transform():
    r"""
    Test the exact wavelet transform.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=16, 
                                                                N=512, 
                                                                q=5,
                                                                calc_m0=True)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_exact(x)

    x_ = wdm.inverse_transform(w)

    assert np.allclose(x, x_, rtol=1.0e-2, atol=1.0e-2), \
        "Inverse transform did not recover original signal."
    

def test_truncated_transform():
    r"""
    Test the exact wavelet transform.
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

    w_ = wdm.forward_transform_truncated(x)

    assert np.allclose(w, w_, rtol=1.0e-2, atol=1.0e-2), \
        "Truncated transform did not agree with the exact transform."
    

def test_short_fft_transform():
    r"""
    Test the short FFT transform.
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

    w_ = wdm.forward_transform_short_fft(x)

    assert np.allclose(w, w_, rtol=1.0e-2, atol=1.0e-2), \
        "Truncated transform did not agree with the exact transform."
    
    
