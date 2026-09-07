import numpy as np
import pytest
import jax
import jax.numpy as jnp
import WDM
import WDM.code.utils.utils


def test_reference_transforms_refuse_to_exhaust_memory():
    r"""
    Test that the reference transforms refuse a working set that will not
    fit, rather than trying to allocate it.

    These methods build dense arrays that grow as :math:`N^2` (the bases) or
    :math:`qNN_f` (the truncated window transform), which reach tens of GiB
    at production grid sizes. The limit is pinned to something tiny here so
    that a small grid trips it: the point is to prove the guard is wired
    into each method, and nothing large is ever allocated.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5,
                                                                Nf=4,
                                                                N=64,
                                                                q=8)

    x = jnp.zeros(wdm.N)

    guarded = [("Gnm_basis", lambda: wdm.Gnm_basis()),
               ("gnm_basis", lambda: wdm.gnm_basis()),
               ("forward_transform_truncated_window",
                    lambda: wdm.forward_transform_truncated_window(x))]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(WDM.code.utils.utils,
                      "MAX_WORKING_SET_FRACTION", 1.0e-12)

        for name, call in guarded:
            with pytest.raises(MemoryError):
                call()

    # with the limit restored, the same calls go through
    for name, call in guarded:
        assert call() is not None, \
            f"{name} should succeed once the working set fits."


def test_Gnm():
    r"""
    Test the frequency-domain Gnm functions in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=8, 
                                                                N=64,
                                                                q=4)
    
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
                                                                q=4)
    
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
                                                                q=5)

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
                                                                N=512)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_exact(x)

    x = wdm.inverse_transform_exact(w)
    x_ = wdm.inverse_transform(w)

    assert np.allclose(x, x_, rtol=1.0e-3, atol=1.0e-3), \
        "Inverse transforms don't agree."

    w_lots = np.repeat(w[np.newaxis,:,:], 3, axis=0)

    x_lots = wdm.inverse_transform(w_lots)
    
    assert (x_lots.shape==(3, wdm.N) and 
            np.allclose(x, x_lots[0], rtol=1.0e-3, atol=1.0e-3)), \
        "Inverse transform vecorisation is not behaving correctly."



def test_truncated_transform():
    r"""
    Test the exact wavelet transform.
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
                                                                q=8)

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
                                                                q=8)

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
    
    x = jnp.arange(wdm.N, dtype=wdm.jax_dtype) # test signal

    X = wdm.short_fft(x)

    k_vals = jnp.arange(-wdm.K//2, wdm.K//2)
    j_vals = jnp.arange(wdm.K)
    n_vals = jnp.arange(wdm.Nt)
    kj = k_vals[:,jnp.newaxis,jnp.newaxis] * j_vals[jnp.newaxis,jnp.newaxis,:]  
    nNf_plus_k = n_vals[jnp.newaxis,:,jnp.newaxis]*wdm.Nf + \
                                    k_vals[:,jnp.newaxis,jnp.newaxis]
    my_short_ffft = jnp.sum(jnp.exp(2*jnp.pi*(1j)*kj/wdm.K) * \
                            x[nNf_plus_k%wdm.N] * \
                            wdm.window_TD[k_vals%wdm.N,jnp.newaxis,jnp.newaxis],
                        axis=0)
    
    assert jnp.allclose(my_short_ffft, X, rtol=1.0e-3, atol=1.0e-3), \
        f"Short FFT conventions are wrong."


def test_fft_transform_thisone():
    r"""
    Test the FFT transform.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.333, 
                                                                Nf=4, 
                                                                N=64, 
                                                                q=8)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.forward_transform_exact(x)

    w_ = wdm.forward_transform_fft(x)

    assert np.allclose(w, w_, rtol=1.0e-3, atol=1.0e-3), \
        "FFT transform did not agree with the exact transform."
    
    x_lots = np.repeat(x[np.newaxis,:], 3, axis=0)

    w_lots = wdm.forward_transform_fft(x_lots)
    
    assert (w_lots.shape==(3, wdm.Nt, wdm.Nf) and 
            np.allclose(w_, w_lots, rtol=1.0e-3, atol=1.0e-3)), \
        "Inverse transform vecorisation is not behaving correctly."

def test_fft_transform_m0():
    r"""
    Test the frequency-domain m=0 calculation against the exact transform.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(
        dt=0.333,
        Nf=4,
        N=64,
        q=8,
    )

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,))

    w_exact = wdm.forward_transform_exact(x)
    w_fft = wdm.forward_transform_fft(x)

    M = wdm.Nt // 2

    # DC part of packed m=0 column
    assert np.allclose(
        w_exact[:M, 0],
        w_fft[:M, 0],
        rtol=1.0e-6,
        atol=1.0e-6,
    ), "FFT zero-frequency m=0 terms do not agree with exact transform."

    # Nyquist part of packed m=0 column
    assert np.allclose(
        w_exact[M:, 0],
        w_fft[M:, 0],
        rtol=1.0e-6,
        atol=1.0e-6,
    ), "FFT Nyquist-frequency m=0 terms do not agree with exact transform."


def test_fft_transform_m0_independent_of_q():
    seed = 1234
    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)

    N = 64
    Nf = 4
    dt = 0.333

    x = jax.random.normal(subkey, shape=(N,))

    wdm_q2 = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(
        dt=dt,
        Nf=Nf,
        N=N,
        q=2,
    )

    wdm_q8 = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(
        dt=dt,
        Nf=Nf,
        N=N,
        q=8,
    )

    w_q2 = wdm_q2.forward_transform_fft(x)
    w_q8 = wdm_q8.forward_transform_fft(x)

    assert np.allclose(
        w_q2[:, 0],
        w_q8[:, 0],
        rtol=1.0e-10,
        atol=1.0e-10,
    ), "FFT m=0 coefficients should be independent of q."