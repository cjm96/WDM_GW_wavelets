import numpy as np
import pytest
import jax
import jax.numpy as jnp
import WDM
import WDM.code.utils.utils


#: Tolerance for the FFT transforms. These are exact rather than truncated,
#: and agree with the exact transform to around 1e-13, so this leaves a few
#: orders of headroom for platform variation while still being tight enough
#: to catch a reintroduced dependence on the truncation parameter q.
TOL = 1.0e-10

#: Grid geometries for the m=0 tests, as (Nf, N, q). The m=0 algorithm folds
#: Nt frequency samples onto Nt/2, so these span Nt from 32 down to the
#: degenerate Nt=2, where that fold has a single sample. Note q <= Nt/2.
M0_GEOMETRIES = [(2, 64, 8),
                 (4, 64, 8),
                 (8, 64, 4),
                 (32, 64, 1),
                 (16, 512, 5)]


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


def test_fft_transform():
    r"""
    Test the FFT transform.

    The FFT transform is exact rather than truncated, so this is held to a
    much tighter tolerance than the truncated transforms above.
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

    assert np.allclose(w, w_, rtol=TOL, atol=TOL), \
        "FFT transform did not agree with the exact transform."

    x_lots = np.repeat(x[np.newaxis,:], 3, axis=0)

    w_lots = wdm.forward_transform_fft(x_lots)

    assert (w_lots.shape==(3, wdm.Nt, wdm.Nf) and
            np.allclose(w_, w_lots, rtol=TOL, atol=TOL)), \
        "Forward transform vecorisation is not behaving correctly."


@pytest.mark.parametrize("Nf, N, q", M0_GEOMETRIES)
def test_fft_transform_m0(Nf, N, q):
    r"""
    Test the frequency-domain m=0 calculation against the exact transform,
    in both directions.

    The m=0 column packs two families of edge wavelets, the zero-frequency
    ones in its first half and the Nyquist ones in its second, and the FFT
    algorithm folds Nt frequency samples onto Nt/2 to compute them. The
    geometries here vary Nt from 32 down to 2, so that the degenerate fold
    at Nt=2 is covered as well as the comfortable cases.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(
        dt=0.333,
        Nf=Nf,
        N=N,
        q=q,
    )

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,))

    w_exact = wdm.forward_transform_exact(x)
    w_fft = wdm.forward_transform_fft(x)

    M = wdm.Nt // 2

    # DC part of packed m=0 column
    assert np.allclose(w_exact[:M, 0], w_fft[:M, 0], rtol=TOL, atol=TOL), \
        "FFT zero-frequency m=0 terms do not agree with exact transform."

    # Nyquist part of packed m=0 column
    assert np.allclose(w_exact[M:, 0], w_fft[M:, 0], rtol=TOL, atol=TOL), \
        "FFT Nyquist-frequency m=0 terms do not agree with exact transform."

    # the inverse direction: isolate the m=0 column, so that only the m=0
    # branch of the inverse contributes to the reconstruction
    w_m0 = jnp.zeros_like(w_exact).at[:, 0].set(w_exact[:, 0])

    assert np.allclose(wdm.inverse_transform_exact(w_m0),
                       wdm.inverse_transform_fft(w_m0),
                       rtol=TOL, atol=TOL), \
        "FFT inverse of the m=0 column does not agree with exact transform."


def test_fft_transform_independent_of_q():
    r"""
    Test that the FFT transform does not depend on the truncation parameter.

    The FFT transform uses the frequency-domain window only, so q should not
    enter anywhere - not just in the m=0 column, which is why the whole grid
    is compared here.
    """
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

    assert np.allclose(wdm_q2.forward_transform_fft(x),
                       wdm_q8.forward_transform_fft(x),
                       rtol=TOL, atol=TOL), \
        "FFT coefficients should be independent of q."


def test_inverse_transform_fft():
    r"""
    Test the FFT inverse transform against the exact inverse transform.
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

    assert np.allclose(wdm.inverse_transform_exact(w),
                       wdm.inverse_transform_fft(w),
                       rtol=TOL, atol=TOL), \
        "FFT inverse transform did not agree with the exact inverse."

    x_ = wdm.inverse_transform_fft(w)

    # one leading axis
    w_lots = np.repeat(w[np.newaxis,:,:], 3, axis=0)

    x_lots = wdm.inverse_transform_fft(w_lots)

    assert (x_lots.shape==(3, wdm.N) and
            np.allclose(x_, x_lots[0], rtol=TOL, atol=TOL)), \
        "Inverse transform vecorisation is not behaving correctly."

    # two leading axes
    w_more = np.repeat(w_lots[np.newaxis,...], 2, axis=0)

    x_more = wdm.inverse_transform_fft(w_more)

    assert (x_more.shape==(2, 3, wdm.N) and
            np.allclose(x_, x_more[0,0], rtol=TOL, atol=TOL)), \
        "Inverse transform should accept more than one leading axis."


def test_inverse_transform_fft_independent_of_q():
    r"""
    Test that the FFT inverse transform does not depend on the truncation
    parameter, for the same reason as the forward transform.
    """
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

    w = wdm_q2.forward_transform_fft(x)

    assert np.allclose(wdm_q2.inverse_transform_fft(w),
                       wdm_q8.inverse_transform_fft(w),
                       rtol=TOL, atol=TOL), \
        "The inverse FFT transform should be independent of q."


def test_dwt_idwt_roundtrip():
    r"""
    Test that the production transform pair is a bijection, in both
    directions.

    Recovering the signal from its coefficients is the obvious direction.
    Recovering an arbitrary coefficient grid is the stronger statement: it
    holds only because the wavelets form a complete orthonormal basis, which
    in turn requires the m=0 column to be packed and unpacked consistently
    by both transforms.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5,
                                                                Nf=8,
                                                                N=512,
                                                                q=8)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    assert np.allclose(wdm.idwt(wdm.dwt(x)), x, rtol=TOL, atol=TOL), \
        "idwt(dwt(x)) did not recover the original signal."

    key, subkey = jax.random.split(key)
    w = jax.random.normal(subkey, shape=(wdm.Nt, wdm.Nf))

    assert np.allclose(wdm.dwt(wdm.idwt(w)), w, rtol=TOL, atol=TOL), \
        "dwt(idwt(w)) did not recover the original coefficients."


def test_fft_transform_parseval():
    r"""
    Test that the transform conserves energy.

    A round trip cannot detect a normalisation error that is inverse between
    the forward and inverse transforms, because it cancels; comparing the
    energy of the coefficients against the energy of the signal can.
    """
    seed = 1234
    key = jax.random.key(seed)

    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5,
                                                                Nf=8,
                                                                N=512,
                                                                q=8)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, shape=(wdm.N,)) # white noise

    w = wdm.dwt(x)

    assert np.allclose(jnp.sum(w**2), jnp.sum(x**2)*wdm.dt,
                       rtol=TOL, atol=TOL), \
        "Sum of squared wavelet coefficients should equal dt times the " \
        "sum of squared samples."


@pytest.mark.parametrize("Nf, N, q", M0_GEOMETRIES)
def test_fft_transform_m0_conventions(Nf, N, q):
    r"""
    Test where the m=0 coefficients of a pure DC or Nyquist signal land.

    The m=0 column holds the zero-frequency wavelets in its first half and
    the Nyquist wavelets in its second. Unlike the tests above, this pins
    that convention against the signals themselves rather than against
    another implementation: swapping the two halves in both the forward and
    inverse transforms would leave every round trip, and Parseval, intact.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5,
                                                                Nf=Nf,
                                                                N=N,
                                                                q=q)

    M = wdm.Nt // 2

    k_vals = jnp.arange(wdm.N)

    # a pure DC signal belongs in the first half of the m=0 column, and a
    # pure Nyquist signal in the second; neither belongs anywhere else
    for name, x, occupied, empty in [
            ("zero-frequency", jnp.ones(wdm.N), slice(None, M), slice(M, None)),
            ("Nyquist", (-1.)**k_vals, slice(M, None), slice(None, M))]:

        w = wdm.dwt(x)

        assert np.abs(w[occupied, 0]).max() > 0.1, \
            f"A pure {name} signal should occupy its half of the m=0 column."

        assert np.allclose(w[empty, 0], 0.0, rtol=TOL, atol=TOL), \
            f"A pure {name} signal should not appear in the other half of " \
            f"the m=0 column."

        assert np.allclose(w[:, 1:], 0.0, rtol=TOL, atol=TOL), \
            f"A pure {name} signal should not leak into the m>0 columns."

        assert np.allclose(jnp.sum(w**2), jnp.sum(x**2)*wdm.dt,
                           rtol=TOL, atol=TOL), \
            f"A pure {name} signal should conserve energy."


@pytest.mark.parametrize("method_name", ["forward_transform_truncated",
                                         "forward_transform_truncated_window",
                                         "forward_transform_short_fft"])
def test_truncated_methods_m0_matches_grid_accuracy(method_name):
    r"""
    Test that the truncated transforms treat the m=0 column no worse than
    the rest of the grid.

    These methods compute the m=0 coefficients from truncated time-domain
    wavelets, so unlike the FFT transform their m=0 column does depend on q.
    That is intended: what matters is that it is not a weak point, so its
    error should track the error everywhere else, and vanish with it once
    the wavelets are no longer truncated at q = Nt/2.
    """
    seed = 1234
    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)

    Nf, N = 8, 512

    x = jax.random.normal(subkey, shape=(N,))

    # a truncated case, where both errors should be comparable
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5,
                                                                Nf=Nf,
                                                                N=N,
                                                                q=8)

    w_exact = wdm.forward_transform_exact(x)
    w = getattr(wdm, method_name)(x)

    error_m0 = np.abs(w[:, 0] - w_exact[:, 0]).max()
    error_rest = np.abs(w[:, 1:] - w_exact[:, 1:]).max()

    assert error_m0 <= 10 * error_rest, \
        f"{method_name} m=0 error of {error_m0:.2e} is disproportionate to " \
        f"its error of {error_rest:.2e} elsewhere on the grid."

    # and an untruncated case, where both should vanish
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5,
                                                                Nf=Nf,
                                                                N=N,
                                                                q=(N//Nf)//2)

    w_exact = wdm.forward_transform_exact(x)
    w = getattr(wdm, method_name)(x)

    assert np.allclose(w, w_exact, rtol=TOL, atol=TOL), \
        f"{method_name} should agree with the exact transform when the " \
        f"wavelets are not truncated."