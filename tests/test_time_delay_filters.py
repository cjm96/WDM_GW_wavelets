import inspect

import jax
import jax.numpy as jnp

import WDM
from WDM.code.discrete_wavelet_transform import WDM
from WDM.code.time_delay_filters.filters import (
                                        FILTER_TABLE_BLOCK_BYTES,
                                        _choose_freq_block,
                                        build_filter_tables,
                                        time_delay_filter_Tl_reference,
                                        time_delay_filter_Tprimel_reference,
                                        window_support)



def test_filter_functions():
    r"""
    Test the time delay filter functions - just check that these evaluate 
    and return variables of the correct type.
    """
    wdm = WDM.WDM_transform(dt=0.5, 
                            Nf=8, 
                            N=256,
                            q=4,
                            calc_m0=True)

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
    wdm = WDM.WDM_transform(dt=1/fs, Nf=Nf, N=N, q=32, calc_m0=True)

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


def test_time_delay_matrix_X_orthogonality():
    r"""
    With zero time delay the time-delay matrix must reduce to the identity,

    .. math::
        X_{nn';mm'}(0) = \delta_{nn'}\delta_{mm'} ,

    i.e. the :math:`l=0,\sigma=0` element is one and every other element is
    zero.
    """
    L = 4
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32, q=4, calc_m0=True)
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
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32, q=4, calc_m0=True)
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
    """
    L = 4
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32, q=4, calc_m0=True)
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
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32, q=4, calc_m0=True)
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

    wdm = WDM.WDM_transform(dt=1/fs, Nf=200, N=times.shape[0], q=32,
                            calc_m0=True)
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
    wdm = WDM.WDM_transform(dt=0.5, Nf=16, N=16*32, q=4, calc_m0=True)

    lags = jnp.arange(-3, 4)
    delta = jnp.array([0.1*wdm.dT, -0.2*wdm.dT])

    coarse_tables = wdm.build_time_delay_filter_interpolants(3, 17)
    coarse = wdm.time_delay_filter_Tl(coarse_tables, lags, delta)

    fine_tables = wdm.build_time_delay_filter_interpolants(3, 513)
    fine = wdm.time_delay_filter_Tl(fine_tables, lags, delta)

    assert not jnp.all(coarse == fine), \
        "rebuilding the interpolants had no effect - stale jit cache?"


def test_window_support_bounds_both_window_products():
    r"""
    Restricting the filter integrals to the window support must be exact.

    `window_support` measures the range from the two window products, but it
    is `build_filter_tables` that has to be able to use the answer, so check
    it against the products as that function forms them: `window_FD**2` and
    the half-bin-shifted product. Everything outside the returned range must
    be identically zero in both, otherwise the trim would silently discard
    signal. The synthetic windows elsewhere in this file cover the awkward
    cases; this one covers the grids the code is actually run on, over a
    spread of `A_frac` since that sets how much of the array the window fills.
    """
    for Nf, N, A_frac in ((8, 8*64, 0.25), (2, 2*2048, 0.25),
                          (64, 64*128, 0.4), (128, 128*32, 0.1)):
        wdm = WDM.WDM_transform(dt=0.5, Nf=Nf, N=N, q=4, A_frac=A_frac)

        shift = int(0.5*wdm.dF/wdm.df)
        start, stop = window_support(wdm.window_FD, shift)

        products = jnp.stack([jnp.roll(wdm.window_FD, shift)
                                  * jnp.roll(wdm.window_FD, -shift),
                              wdm.window_FD**2])

        outside = jnp.concatenate([products[:, :start], products[:, stop:]],
                                  axis=1)

        assert jnp.all(outside == 0.0), \
            f"window support [{start}, {stop}) drops non-zero samples " \
            f"for {Nf=} {N=} {A_frac=}"


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


def test_filter_tables_match_the_defining_integrals():
    r"""
    The tabulation must still evaluate the integrals it claims to evaluate.

    Restricting to the window support and accumulating over frequency blocks
    are both changes to *how* the integrals are summed, so neither may move
    the answer. Comparing the tabulation against itself at two block sizes
    cannot see that: both paths are trimmed identically, so a trim that
    dropped signal would agree with itself and pass. The anchor has to be
    outside the blocked code path entirely, which is what the two
    `..._reference` functions are for - they sum the defining integrand over
    the full frequency axis, one :math:`(\ell,\delta)` pair at a time.

    Three block regimes are checked, which is what makes this the only test
    the blocked path needs: one block covering the whole support, one block
    per frequency sample, and a size that divides the support unevenly. The
    last is the interesting one - the short final block is a different static
    shape, so it is separately compiled, and it is the only regime in which
    an off-by-one in the `min(lo+block, stop)` bound can show up.
    """
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*64, q=4)

    lags = jnp.arange(-3, 4)
    deltas = jnp.linspace(-0.5*wdm.dT, 0.5*wdm.dT, 5)

    shift = int(0.5*wdm.dF/wdm.df)
    start, stop = window_support(wdm.window_FD, shift)

    assert stop-start < wdm.window_FD.shape[0], \
        "this grid does not exercise the support trim - the comparison " \
        "against the reference integrals would be vacuous"

    args = (lags, deltas, wdm.freqs, wdm.window_FD, wdm.dT, wdm.dF, wdm.df)

    # one block holds three lag-phase copies and the delay phase per frequency
    uneven = 20*16*(3*len(lags) + len(deltas))

    assert (stop-start) % 20, \
        f"a support of {stop-start} samples divides evenly into blocks of " \
        f"20 - this budget no longer leaves a short final block"

    # shape (2, 2L+1, num_interp_points); index 0 is T', index 1 is T
    for kwargs, name in (({}, "one block"),
                         ({"max_bytes": 1}, "one sample per block"),
                         ({"max_bytes": uneven}, "short final block")):
        tables = build_filter_tables(*args, **kwargs)

        for i, ell in enumerate(lags):
            for j, delta in enumerate(deltas):
                T_l = time_delay_filter_Tl_reference(int(ell),
                                                     float(delta),
                                                     wdm.freqs,
                                                     wdm.window_FD,
                                                     wdm.dT,
                                                     wdm.df)

                Tprime_l = time_delay_filter_Tprimel_reference(int(ell),
                                                               float(delta),
                                                               wdm.freqs,
                                                               wdm.window_FD,
                                                               wdm.dT,
                                                               wdm.dF,
                                                               wdm.N,
                                                               wdm.df)

                assert jnp.allclose(tables[1, i, j], T_l,
                                    atol=1.0e-12, rtol=0.0), \
                    f"tabulated T_l disagrees with the defining integral " \
                    f"by {abs(float(tables[1, i, j])-T_l)} at {ell=} " \
                    f"{delta=} ({name} block)"

                assert jnp.allclose(tables[0, i, j], Tprime_l,
                                    atol=1.0e-12, rtol=0.0), \
                    f"tabulated T'_l disagrees with the defining integral " \
                    f"by {abs(float(tables[0, i, j])-Tprime_l)} at {ell=} " \
                    f"{delta=} ({name} block)"


def test_frequency_block_size_stops_growing_with_the_grid():
    r"""
    The claim the blocking exists to support is that the peak allocation no
    longer depends on :math:`N`, and that is a property of the block size
    alone - nothing downstream of `_choose_freq_block` can restore the bound
    if it hands back a block that grows with the grid.

    The budget is restated here rather than imported: one block holds the lag
    phase (`num_lags` complex values per frequency), its two windowed copies
    and the delay phase (`num_deltas`), so `16*(3*num_lags+num_deltas)` bytes
    per frequency sample. A test that reused the implementation's own
    arithmetic could not detect that arithmetic being wrong.

    The degenerate budget matters too: a block of zero would make
    `build_filter_tables` loop forever rather than merely run out of memory.
    """
    num_lags, num_deltas = 51, 100        # L_trunc=25, a typical delay grid

    bytes_per_freq = 16*(3*num_lags + num_deltas)
    budget = 256*1024**2

    hundred_k = _choose_freq_block(num_lags, num_deltas, 10**5, budget)
    ten_m = _choose_freq_block(num_lags, num_deltas, 10**7, budget)

    assert hundred_k == ten_m, \
        f"block size grew with the frequency axis, {hundred_k} -> {ten_m}; " \
        f"the peak allocation still depends on N"

    assert ten_m*bytes_per_freq <= budget, \
        f"a block of {ten_m} frequency samples needs " \
        f"{ten_m*bytes_per_freq} bytes, over the {budget} budget"

    small = _choose_freq_block(num_lags, num_deltas, 32, budget)

    assert small == 32, \
        f"a grid smaller than the budget was split into blocks of {small}"

    for max_bytes in (0, 1):
        block = _choose_freq_block(num_lags, num_deltas, 1000, max_bytes)

        assert block == 1, \
            f"a budget of {max_bytes} bytes gave a block of {block}; " \
            f"anything below one frequency sample does not terminate"


def test_block_budget_has_a_single_definition():
    r"""
    Every `max_bytes` default must be the one module constant.

    `WDM_transform.build_time_delay_filter_interpolants` always forwards its
    own default, so `build_filter_tables`'s default never applies to calls
    made through the class. If the two were written out separately, lowering
    the budget in one place would leave every call through the class on the
    old value, and the change would look like it had done nothing.
    """
    signatures = (inspect.signature(build_filter_tables),
                  inspect.signature(
                      WDM.WDM_transform.build_time_delay_filter_interpolants))

    for signature in signatures:
        default = signature.parameters["max_bytes"].default

        assert default is FILTER_TABLE_BLOCK_BYTES, \
            f"max_bytes defaults to {default}, not the module constant " \
            f"{FILTER_TABLE_BLOCK_BYTES}; the budget has more than one " \
            f"definition and the two can drift"


def test_window_support_covers_a_wrapped_shifted_product():
    r"""
    The trim must hold when the shifted window wraps around the array edge.

    This is the case that cannot be reasoned about from the window's own
    support. `jnp.roll` is circular, so with the support near an edge and a
    large enough `shift` the two translated copies meet at the far end of the
    array - the product is non-zero in a region the window itself never
    touches. A `window_support` that read the window alone would trim to the
    window's support, integrate where the product is identically zero, and
    return zero for :math:`T'_{\ell}` with no error raised.

    A Meyer window on a sane grid comes nowhere near this - the support is
    narrow and centred, thousands of samples clear of both edges - so the
    window here is built by hand. The point is that the answer no longer
    depends on that comfortable margin holding.
    """
    N, shift = 64, 30

    window_FD = jnp.zeros(N).at[0:11].set(1.0)

    indices = jnp.arange(N)
    product = window_FD[(indices-shift) % N] * window_FD[(indices+shift) % N]

    start, stop = window_support(window_FD, shift)

    outside = jnp.concatenate([product[:start], product[stop:]])

    assert jnp.all(outside == 0.0), \
        f"support [{start}, {stop}) drops non-zero samples of the wrapped " \
        f"product, which is non-zero on " \
        f"{jnp.nonzero(product)[0].min()}..{jnp.nonzero(product)[0].max()}"

    assert jnp.sum(product[start:stop]) == jnp.sum(product), \
        "the trimmed integral of the wrapped product lost weight"


def test_window_support_handles_a_window_that_is_never_non_zero():
    r"""
    A window with no support at all has no range to measure, and the
    conservative answer - integrate over everything - is the only safe one.
    """
    N = 64

    assert window_support(jnp.zeros(N), 3) == (0, N), \
        "a window that is nowhere non-zero must not be trimmed"


def test_window_support_still_trims_interior_support():
    r"""
    Measuring the products rather than the window must not cost the trim.

    Every other test here would pass a `window_support` that had given up and
    returned the whole array, so one case has to pin that it still returns a
    tight range when the products really are confined.
    """
    N, shift = 64, 3

    interior = jnp.zeros(N).at[20].set(1.0).at[30].set(1.0)

    assert window_support(interior, shift) == (20, 31), \
        "interior support clear of both edges must still be trimmed"
