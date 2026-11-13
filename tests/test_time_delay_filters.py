import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import betainc

import WDM
from WDM.code.discrete_wavelet_transform import WDM
from WDM.code.time_delay_filters.filters import filter_functions



def test_filter_functions():
    r"""
    Test the time delay filter functions - just check that these evaluate 
    and return variables of the correct type.
    """
    wdm = WDM.WDM_transform(dt=0.5, 
                            Nf=8, 
                            N=256,
                            q=4)

    L = 25  # the maximum lag index
    N = 10  # the number of interpolation points
    tables = wdm.build_time_delay_filter_interpolants(L, N)

    ell = jnp.array(jnp.arange(-L, L+1), dtype=int)
    delta = jnp.linspace(-wdm.dT/2, wdm.dT/2, 100)

    Tl = wdm.time_delay_filter_Tl(tables, ell, delta)
    assert Tl.shape==(len(ell), len(delta)), "Tl array wrong shape"

    Tprimel = wdm.time_delay_filter_Tprimel(tables, ell, delta)
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
    wdm = WDM.WDM_transform(dt=1/fs, Nf=Nf, N=N, q=32)

    w = wdm(signal)

    L = 25  # the maximum lag index
    N = 10  # the number of interpolation points
    tables = wdm.build_time_delay_filter_interpolants(L, N)

    tn = jnp.arange(wdm.Nt)*wdm.dT

    delta_constant = lambda t: (T/10.)*jnp.ones_like(t)

    # shift the signal in the time domain
    shifted_signal_constant_t = jnp.interp(times + delta_constant(times), 
                                           times, 
                                           signal)

    # shift the signal in the time-frequency domain
    w_shifted_constant = wdm.apply_variable_time_shift(tables, w, delta_constant(tn))
    shifted_signal_constant_tf = wdm.idwt(w_shifted_constant)

    # check the methods agree
    residuals = shifted_signal_constant_tf - shifted_signal_constant_t
    eps = jnp.max(jnp.abs(residuals))
    tol = 1.0e-4
    assert eps<tol, "residuals are too large"


def _spectral_shift(wdm, g, s):
    r"""
    Evaluate :math:`g(t+s)` by multiplying by a linear phase in the frequency
    domain. The wavelets are periodic and band-limited on the FFT grid, so this
    is exact for any :math:`s` - unlike linear interpolation, which is only
    exact when :math:`s` is a whole number of samples.

    Parameters
    ----------
    wdm : WDM_transform
        The wdm object supplying the time grid.
    g : jnp.ndarray
        Time series to shift, shape=(N,).
    s : float
        The shift, in the time units of `wdm`.

    Returns
    -------
    g_shifted : jnp.ndarray
        The shifted time series, shape=(N,).
    """
    freqs = jnp.fft.fftfreq(wdm.N, d=wdm.dt)
    return jnp.real(jnp.fft.ifft(jnp.fft.fft(g)*jnp.exp(2j*jnp.pi*freqs*s)))


def test_constant_shift_off_bin_for_every_Nt_parity():
    r"""
    An off-bin signal shifted by a constant delay must come out right whether
    or not :math:`N_t` is a multiple of four.

    :math:`T'_{\ell}` enters only the :math:`\sigma=\pm1` terms, which matter
    only where neighbouring windows overlap, so the tone sits at
    :math:`r=0.49` of a band. The reference is the spectral shift, exact on
    the periodic grid the shift wraps on.
    """
    for Nt in (124, 126):
        wdm = WDM.WDM_transform(dt=4.0, Nf=50, N=50*Nt)
        tables = wdm.build_time_delay_filter_interpolants(25, 128)

        f = 10.49*wdm.dF
        signal = jnp.cos(2*jnp.pi*f*wdm.times)
        delta = 70.0

        shifted = wdm.apply_variable_time_shift(tables,
                                                wdm.dwt(signal),
                                                jnp.full(wdm.Nt, delta))
        exact = wdm.dwt(_spectral_shift(wdm, signal, delta))

        a, b = shifted[:, 1:], exact[:, 1:]
        mismatch = 1.0 - jnp.sum(a*b)/jnp.sqrt(jnp.sum(a*a)*jnp.sum(b*b))

        assert mismatch < 1.0e-6, \
            f"off-bin constant shift mismatch {float(mismatch):.2e} at " \
            f"{Nt=} ({Nt % 4=})"


def test_time_delay_matrix_X_orthogonality():
    r"""
    With zero time delay the time-delay matrix must reduce to the identity,

    .. math::
        X_{nn';mm'}(0) = \delta_{nn'}\delta_{mm'} ,

    i.e. the :math:`l=0,\sigma=0` element is one and every other element is
    zero.
    """
    L = 4
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32, q=4)
    tables = wdm.build_time_delay_filter_interpolants(L, 65)

    n = jnp.arange(wdm.Nt)
    m = jnp.arange(wdm.Nf)
    delta = jnp.zeros(wdm.Nt)

    for l in range(0, L+1):
        for sigma in (-1, 0, 1):
            X = wdm.time_delay_matrix_X(tables, n, m, l, sigma, delta)

            expected = 1.0 if (l == 0 and sigma == 0) else 0.0

            # stay away from the time edges and the m=0 / Nyquist bins
            eps = jnp.max(jnp.abs(X[8:24, 1:7] - expected))

            assert eps < 1.0e-12, \
                f"X is not orthogonal at zero delay: {l=} {sigma=} {eps=}"


def test_time_delay_matrix_X_definitional_symmetry():
    r"""
    Swapping the two index pairs and reversing the delay leaves the defining
    integral unchanged,

    .. math::
        X_{nn';mm'}(\delta t) = X_{n'n;m'm}(-\delta t) ,

    which for the array-valued method means

    .. math::
        X(l,\sigma;\delta)_{nm} = X(-l,-\sigma;-\delta)_{(n-l)(m+\sigma)} .

    This is a purely internal check - it needs no reference integral - and it
    catches sign errors and index offsets in the individual
    :math:`\sigma` branches.
    """
    L = 4
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32, q=4)
    tables = wdm.build_time_delay_filter_interpolants(L, 65)

    n = jnp.arange(wdm.Nt)
    m = jnp.arange(wdm.Nf)
    delta = 0.31*wdm.dT

    for l in range(0, L+1):
        for sigma in (-1, 0, 1):
            lhs = wdm.time_delay_matrix_X(tables, n, m, l, sigma,
                                          delta*jnp.ones(wdm.Nt))
            rhs = wdm.time_delay_matrix_X(tables, n, m, -l, -sigma,
                                          -delta*jnp.ones(wdm.Nt))

            eps = jnp.max(jnp.abs(lhs[12:21, 2:6]
                                  - rhs[12-l:21-l, 2+sigma:6+sigma]))

            assert eps < 1.0e-12, \
                f"X violates X_nn';mm'(dt)=X_n'n;m'm(-dt): {l=} {sigma=} {eps=}"


def test_time_delay_matrix_X_matches_direct_integral():
    r"""
    Compare the closed-form expressions in `time_delay_matrix_X` against direct
    numerical integration of the defining expression,

    .. math::
        X_{nn';mm'}(\delta t)=\int\mathrm{d}t\,
                              g_{nm}(t+\delta t)\,g^*_{n'm'}(t) ,

    with :math:`n'=n-l`, :math:`m'=m+\sigma` and :math:`\delta t=-\delta`.

    Both :math:`m` and :math:`m'` are kept inside :math:`1\leq m\leq N_f-2`.
    The :math:`m=0` bin is excluded from the time-shift operation, so elements
    coupling to it are not expected to agree; likewise the Nyquist bin. Time
    indices are kept away from the edges to avoid wrap-around.

    The atoms here are periodic in the 32-row record, so the direct integral
    also picks up the filters one record length away, at most
    :math:`6\times10^{-6}` for these lags; the tolerance covers that.
    """
    L = 4
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32, q=4)
    tables = wdm.build_time_delay_filter_interpolants(L, 129)

    n_idx = jnp.arange(wdm.Nt)
    m_idx = jnp.arange(wdm.Nf)
    delta_t = 0.31*wdm.dT

    for l in (0, 1, 2):
        for sigma in (-1, 0, 1):
            X = wdm.time_delay_matrix_X(tables, n_idx, m_idx, l, sigma,
                                        delta_t*jnp.ones(wdm.Nt))

            for n in range(12, 21, 2):
                for m in range(1, wdm.Nf-1):

                    if not 1 <= m + sigma <= wdm.Nf-2:
                        continue

                    g_shifted = _spectral_shift(wdm, wdm.gnm(n, m), -delta_t)
                    g_nprime_mprime = wdm.gnm(n-l, m+sigma)

                    X_direct = wdm.dt*jnp.sum(g_nprime_mprime*g_shifted)

                    assert jnp.isclose(X[n, m], X_direct, atol=1.0e-4,
                                       rtol=1.0e-4), \
                        "X does not match the direct integral: " + \
                        f"{n=} {m=} {l=} {sigma=}, " + \
                        f"expression={X[n, m]}, integral={X_direct}"


def test_apply_variable_time_shift_matches_X_reference():
    r"""
    The variable time shift is defined as the sum

    .. math::
        \tilde{\omega}_{nm}=\sum_{l,\sigma}\omega_{(n-l)(m+\sigma)}\,
                            X_{n(n-l);m(m+\sigma)}(-\delta_n) .

    Check that `apply_variable_time_shift` reproduces that sum when it is
    assembled explicitly from `time_delay_matrix_X`. This pins the optimised
    kernel to the readable per-:math:`(l,\sigma)` expressions, so the two can
    never drift apart.
    """
    L = 4
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32, q=4)
    tables = wdm.build_time_delay_filter_interpolants(L, 65)

    key = jax.random.PRNGKey(0)
    wdm_coeff = jax.random.normal(key, (wdm.Nt, wdm.Nf))
    delta = 0.3*wdm.dT*jnp.sin(jnp.arange(wdm.Nt)/5.0)

    n_idx = jnp.arange(wdm.Nt)
    m_idx = jnp.arange(wdm.Nf)

    reference = jnp.zeros((wdm.Nt, wdm.Nf))
    for l in range(-L, L+1):
        for sigma in (-1, 0, 1):
            X = wdm.time_delay_matrix_X(tables, n_idx, m_idx, l, sigma, delta)

            # omega_{(n-l)(m+sigma)}, wrapped periodically
            coeff = jnp.roll(jnp.roll(wdm_coeff, shift=l, axis=0),
                             shift=-sigma, axis=1)

            reference = reference + coeff*X

    shifted = wdm.apply_variable_time_shift(tables, wdm_coeff, delta)

    eps = jnp.max(jnp.abs(shifted - reference))/jnp.max(jnp.abs(reference))

    assert eps < 1.0e-12, \
        f"the kernel does not match the explicit sum over X: {eps=}"


def test_variable_time_shift_linear_ramp():
    r"""
    As `test_time_delay`, but for a genuinely *variable* time shift
    :math:`\delta(t)=a+bt` rather than a constant one, since the constant case
    does not exercise the dependence of the delay on the time index.

    Accuracy is quantified by the mismatch against a time-domain shift of the
    same signal.
    """
    fs = 0.2    # sampling frequency [Hz]
    T  = 1.0e5  # duration [s]
    f0 = 3.0e-3 # central frequency [Hz]
    width = T/15.  # duration [s]

    sine_gauss = lambda t: jnp.exp(-0.5*((t-T/2.)/width)**2) \
                            * jnp.sin(2*jnp.pi*f0*t)

    times = jnp.arange(0, T, 1/fs)
    signal = sine_gauss(times)

    wdm = WDM.WDM_transform(dt=1/fs, Nf=200, N=times.shape[0], q=32)
    w = wdm(signal)

    L = 25  # the maximum lag index
    tables = wdm.build_time_delay_filter_interpolants(L, 10)

    tn = jnp.arange(wdm.Nt)*wdm.dT

    starting_shift = T/10.
    gradient = 1/1000.

    delta_variable = lambda t: starting_shift + gradient*t

    # shift the signal in the time domain
    shifted_t = jnp.interp(times + delta_variable(times), times, signal)

    # shift the signal in the time-frequency domain
    shifted_tf = wdm.idwt(wdm.apply_variable_time_shift(tables, w,
                                                        delta_variable(tn)))

    overlap = jnp.abs(jnp.dot(shifted_t, shifted_tf)) \
                / jnp.sqrt(jnp.dot(shifted_t, shifted_t)
                           * jnp.dot(shifted_tf, shifted_tf))
    mismatch = 1.0 - overlap

    assert mismatch < 1.0e-5, \
        f"variable time shift is not accurate enough: {mismatch=}"



def test_rebuilding_interpolants_takes_effect():
    r"""
    Rebuilding the interpolants with a finer grid must change the answer.

    The filter tables are passed into the jitted methods as an argument rather
    than read from `self`. Were they read from `self` instead, they would be
    frozen into the compiled code on the first call and every later rebuild
    would be silently ignored - this test fails if that regresses.
    """
    wdm = WDM.WDM_transform(dt=0.5, Nf=16, N=16*32, q=4)

    lags = jnp.arange(-3, 4)
    delta = jnp.array([0.1*wdm.dT, -0.2*wdm.dT])

    coarse_tables = wdm.build_time_delay_filter_interpolants(3, 17)
    coarse = wdm.time_delay_filter_Tl(coarse_tables, lags, delta)

    fine_tables = wdm.build_time_delay_filter_interpolants(3, 513)
    fine = wdm.time_delay_filter_Tl(fine_tables, lags, delta)

    assert not jnp.all(coarse == fine), \
        "rebuilding the interpolants had no effect - stale jit cache?"


def test_filter_tables_finite_when_sample_sits_on_band_edge():
    r"""
    A frequency sample landing exactly on the window band edge must not
    produce nan.

    `absw <= A+B` and `(absw-A)/B <= 1` are not the same test in floating
    point, so the roll-off argument can exceed one by an ulp on a sample the
    band-edge branch has already claimed. `nu_d` is nan there. This grid puts
    a sample exactly on the edge - before the fix every entry of the table
    came back nan.
    """
    wdm = WDM.WDM_transform(dt=15.0, Nf=1024, N=1024*64, q=16)

    assert jnp.all(jnp.isfinite(wdm.window_FD)), \
        "the frequency-domain window is not finite on the band edge"

    tables = wdm.build_time_delay_filter_interpolants(4, 17)

    assert jnp.all(jnp.isfinite(tables)), \
        "the time-delay filter tables are not finite on the band edge"


def _window_in_bins(u, d=4, A_frac=0.25):
    r"""
    The Meyer window in units of :math:`\Delta F`, written out here from its
    definition so that the comparison below does not share code with the
    package.
    """
    a = abs(u)
    if a < A_frac:
        return 1.0
    if a > 1.0 - A_frac:
        return 0.0
    return np.cos(0.5*np.pi*betainc(d, d, (a - A_frac)/(1.0 - 2.0*A_frac)))


def test_filter_tables_match_the_defining_integrals():
    r"""
    The filters must equal their defining integrals,

    .. math::
        T(x)=\int\mathrm{d}u\,\Phi(u)^2\cos(\pi u x) , \qquad
        T'(x)=\int\mathrm{d}u\,\Phi\left(u-\tfrac12\right)
                              \Phi\left(u+\tfrac12\right)\cos(\pi u x) ,

    evaluated by adaptive quadrature, including at the largest lag a table
    holds, where the integrand oscillates fastest. :math:`T(0)=1` is the
    window's normalisation.
    """
    xs = np.array([0.0, 0.31, -1.7, 4.5, 12.25, -25.5])
    kinks = [-0.75, -0.25, 0.0, 0.25, 0.75]

    computed = np.asarray(filter_functions(jnp.asarray(xs), 4, 0.25))

    for i, x in enumerate(xs):
        T = quad(lambda u: _window_in_bins(u)**2*np.cos(np.pi*u*x),
                 -1.0, 1.0, points=kinks, limit=400, epsabs=1e-13)[0]
        Tp = quad(lambda u: _window_in_bins(u - 0.5)*_window_in_bins(u + 0.5)
                            * np.cos(np.pi*u*x),
                  -1.0, 1.0, points=kinks, limit=400, epsabs=1e-13)[0]

        assert abs(computed[1, i] - T) < 1.0e-12, \
            f"T({x}) = {computed[1, i]}, integral {T}"
        assert abs(computed[0, i] - Tp) < 1.0e-12, \
            f"T'({x}) = {computed[0, i]}, integral {Tp}"

    assert abs(computed[1, 0] - 1.0) < 1.0e-13, \
        f"T(0) = {computed[1, 0]}, not 1"


def test_filter_tables_are_the_same_for_every_tiling():
    r"""
    The tables depend on the window shape alone, not on :math:`N_f`,
    :math:`N_t` or the cadence.
    """
    tilings = ((0.5, 8, 8*32), (4.0, 50, 50*126), (10.0, 1024, 1024*64))

    tables = [WDM.WDM_transform(dt=dt, Nf=Nf, N=N)
                 .build_time_delay_filter_interpolants(5, 33)
              for dt, Nf, N in tilings]

    for other, (dt, Nf, N) in zip(tables[1:], tilings[1:]):
        eps = float(jnp.max(jnp.abs(other - tables[0])))
        assert eps < 1.0e-14, \
            f"tables differ by {eps} between tilings ({dt=}, {Nf=}, {N=})"


def test_constant_shift_with_fewer_rows_than_lags():
    r"""
    A constant shift must stay accurate when the record has fewer rows than
    the lag sum has terms, :math:`N_t<2L+1`. Lags :math:`\ell` and
    :math:`\ell-N_t` then read the same row, and each must carry its own
    filter value.

    The delay sits on a table node, so the residual is the lag truncation.
    The reference is the spectral shift, exact on the periodic grid.
    """
    L, P = 25, 129
    wdm = WDM.WDM_transform(dt=4.0, Nf=2048, N=2048*32)
    tables = wdm.build_time_delay_filter_interpolants(L, P)

    f = jnp.round(500.40625*wdm.dF*wdm.T)/wdm.T
    signal = jnp.cos(2*jnp.pi*f*wdm.times + 0.3)
    delta = wdm.dT*(-0.5 + 40/(P - 1))

    shifted = wdm.apply_variable_time_shift(tables,
                                            wdm.dwt(signal),
                                            jnp.full(wdm.Nt, delta))
    exact = wdm.dwt(_spectral_shift(wdm, signal, delta))

    eps = float(jnp.linalg.norm(shifted - exact)/jnp.linalg.norm(exact))

    assert eps < 1.0e-4, \
        f"constant shift with Nt={wdm.Nt} < 2L+1={2*L+1}: relative L2 {eps:.2e}"
