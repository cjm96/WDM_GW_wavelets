import numpy as np
import jax
import jax.numpy as jnp
import WDM


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
    
    for n in range(wdm.Nt):
        for m in range(wdm.Nf):
            assert jnp.allclose(Gnm_basis[:,n,m], Gnm_basis_slow[:,n,m], 
                                rtol=1.0e-3, atol=1.0e-3), \
                f"gnm_basis at n={n}, m={m} does not match the slow method."
    
    assert jnp.allclose(Gnm_basis, Gnm_basis_slow, rtol=1.0e-3, atol=1.0e-3), \
        f"The two methods for computing Gnm_basis should match."
    
    # reshape to (N, Nt*Nf) for orthonormality check
    Gnm_basis = Gnm_basis.reshape(Gnm_basis.shape[0], -1)

    I = jnp.conj(Gnm_basis) @ Gnm_basis.T * wdm.df

    assert jnp.allclose(I.real, jnp.eye(wdm.N), atol=1e-3, rtol=1e-3), \
        f"The Gnm_basis should be orthonormal."
    

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
    
    for n in range(wdm.Nt):
        for m in range(wdm.Nf):
            assert jnp.allclose(gnm_basis[:,n,m], gnm_basis_slow[:,n,m], 
                                rtol=1.0e-3, atol=1.0e-3), \
                f"gnm_basis at n={n}, m={m} does not match the slow method."
    
    assert jnp.allclose(gnm_basis, gnm_basis_slow, rtol=1.0e-3, atol=1.0e-3), \
        f"The two methods for computing gnm_basis should match."

    # reshape to (N, Nt*Nf) for orthonormality check
    gnm_basis = gnm_basis.reshape(gnm_basis.shape[0], -1) 

    I = gnm_basis @ gnm_basis.T * wdm.dt

    assert jnp.allclose(I, jnp.eye(wdm.N), atol=1e-3, rtol=1e-3), \
        f"The gnm_basis should be orthonormal."
    

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

    x_ = wdm.inverse_transform_exact(w)

    assert np.allclose(x, x_, rtol=1.0e-3, atol=1.0e-3), \
        "Inverse transform did not recover original signal."
    

def test_inverse_transforms():
    r"""
    Test that the two methods for performing the inverse transform agree.
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

    x = wdm.inverse_transform_exact(w)
    x_ = wdm.inverse_transform(w)

    assert np.allclose(x, x_, rtol=1.0e-3, atol=1.0e-3), \
        "Inverse transforms don't agree."


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
    print(w)

    w_ = wdm.forward_transform_truncated(x)

    assert np.allclose(w, w_, rtol=1.0e-3, atol=1.0e-3), \
        "Truncated transform did not agree with the exact transform."
    

def test_truncated_window_transform():
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

    w_ = wdm.forward_transform_truncated_window(x)

    assert np.allclose(w, w_, rtol=1.0e-3, atol=1.0e-3), \
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

    assert np.allclose(w, w_, rtol=1.0e-3, atol=1.0e-3), \
        "Short FFT transform did not agree with the exact transform."
    

def test_short_fft():
    r"""
    Check the conventions in our short FFT method.

    .. math::

        X_n[j] = \sum_{k=-K/2}^{K/2-1} \exp(2\pi i kj/K) x[nN_f+k] \phi[k]
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=4, 
                                                                N=64, 
                                                                q=4)
    
    x = jnp.arange(wdm.N, dtype=wdm.jax_dtype)

    X = wdm.short_fft(x)

    k_vals = jnp.arange(-wdm.K//2, wdm.K//2)
    j_vals = jnp.arange(wdm.K)
    n_vals = jnp.arange(wdm.Nt)
    kj = k_vals[:,jnp.newaxis,jnp.newaxis] * j_vals[jnp.newaxis,jnp.newaxis,:]  
    nNf_plus_k = n_vals[jnp.newaxis,:,jnp.newaxis]*wdm.Nf + \
                                    k_vals[:,jnp.newaxis,jnp.newaxis]
    my_short_ffft = jnp.sum(jnp.exp(2*jnp.pi*(1j)*kj/wdm.K) * \
                            x[nNf_plus_k] * \
                            wdm.window_TD[k_vals,jnp.newaxis,jnp.newaxis], 
                        axis=0)
    
    assert jnp.allclose(my_short_ffft, X, rtol=1.0e-3, atol=1.0e-3), \
        "Short FFT conventions are wrong."





def test_fft_transform():
    r"""
    Test the FFT transform.
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

    w_ = wdm.forward_transform_fft(x)

    assert np.allclose(w, w_, rtol=1.0e-3, atol=1.0e-3), \
        "FFT transform did not agree with the exact transform."
    
    
