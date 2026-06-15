import jax
import jax.numpy as jnp
import numpy as np
import pytest

import WDM
from WDM.code.discrete_wavelet_transform import WDM
from WDM.code.time_delay_filters.filters import time_delay_filter_Tl
from WDM.code.time_delay_filters.filters import time_delay_filter_Tprimel
from WDM.code.time_delay_filters.filters import time_delay_X
from WDM.code.time_delay_filters import time_shift_fast as tsf

from scipy.interpolate import interp1d


def test_filter_functions():
    r"""
    Test the time delay filter functions - just check that these evaluate 
    and return variables of the correct type.
    """
    wdm = WDM.WDM_transform(dt=0.5, 
                            Nf=8, 
                            N=64,
                            q=4,
                            calc_m0=True)
    
    ell = 0
    delta_t = 1.0

    Tl = time_delay_filter_Tl(wdm, ell, delta_t)
    assert isinstance(Tl, float), "oh dear"

    Tprimel = time_delay_filter_Tprimel(wdm, ell, delta_t)
    assert isinstance(Tprimel, float), "oh dear"


def test_kernel_wdm_inherits_A_frac():
    tsf._KERNEL_WDM_CACHE.clear()
    tsf._KERNEL_PRECOMP_CACHE.clear()

    wdm_data_020 = WDM.WDM_transform(
        dt=10.0,
        Nf=100,
        N=20000,
        q=1,
        calc_m0=False,
        d=4,
        A_frac=0.20,
    )

    wdm_ker_020 = tsf._build_kernel_wdm_like(
        wdm_data_020,
        Nker=5200,
        Nf=100,
        d=4,
        calc_m0=True,
    )

    assert np.isclose(float(wdm_ker_020.A_frac), 0.20)

    wdm_data_030 = WDM.WDM_transform(
        dt=10.0,
        Nf=100,
        N=20000,
        q=1,
        calc_m0=False,
        d=4,
        A_frac=0.30,
    )

    wdm_ker_030 = tsf._build_kernel_wdm_like(
        wdm_data_030,
        Nker=5200,
        Nf=100,
        d=4,
        calc_m0=True,
    )

    assert np.isclose(float(wdm_ker_030.A_frac), 0.30)
    assert wdm_ker_020 is not wdm_ker_030


def test_filter_X_orthogonality():
    r"""
    Test the orthogonality of the time-delay matrix elements. We should have 
    the following property hold with zero time delay:

    .. math::
        X_{nn';mm'}(\delta t = 0) = \delta_{nn'} \delta_{mm'}
    """
    wdm = WDM.WDM_transform(dt=0.5, 
                            Nf=8, 
                            N=64,
                            q=4,
                            calc_m0=True)
    
    delta_t = 0.0
    
    for n in range(wdm.Nt):
        for n_ in range(wdm.Nt):

            for m in range(wdm.Nf):
                for m_ in range(wdm.Nf):

                    X = time_delay_X(wdm, n, n_, m, m_, delta_t)

                    expected = 1.0 if (n == n_ and m == m_) else 0.0

                    assert jnp.isclose(X, expected), \
                        "the X coefficients are not orthogonal!"
                    

def test_filter_X_expressions():
    r"""
    Test the expressions of the time-delay matrix elements implemented in the 
    function `time_delay_X` by comparing against direct numerical integration
    of the defining expression,

    .. math::
        X_{nn';mm'}(\delta t)=\int\mathrm{d}t g_{nm}(t+\delta t)g^*_{n'm'}(t).

    This test only checks for :math:`n` times indices away from the edges of the 
    allowed range (`boundary=12`) to avoid edge effects.
    """
    Nf = 8
    Nt = 32

    wdm = WDM.WDM_transform(dt=0.5, 
                            Nf=Nf, 
                            N=Nf*Nt,
                            q=4,
                            calc_m0=True)
    
    delta_t = 0.5 * wdm.dT

    boundary = 12 # avoid periodic edge effects not captured by interpolation

    for n in range(boundary, wdm.Nt - boundary):
        for m in range(wdm.Nf):

            for n_ in range(boundary, wdm.Nt - boundary):
                for m_ in range(wdm.Nf):

                    if m==0 or m_==0:
                        pass

                    else:
                        X_expression = time_delay_X(wdm, n, n_, m, m_, delta_t)

                        g_nprime_mprime = wdm.gnm(n_, m_)
                        g_nm_shifted = interp1d(wdm.times, wdm.gnm(n,m), 
                                                bounds_error=False, 
                                                fill_value=0.0)(wdm.times+delta_t)

                        X_direct_integral = wdm.dt*jnp.sum(g_nprime_mprime*g_nm_shifted)

                        assert jnp.isclose(X_expression, X_direct_integral, atol=1e-3, rtol=1e-3), \
                            "the X coefficients do not match the direct integral!" + \
                            f" n={n}, m={m}, n'={n_}, m'={m_}, delta_t={delta_t}: " + \
                            f"X_expression={X_expression}, X_direct_integral={X_direct_integral}"


def test_variable_shift_batch_matches_single_calls():
    r"""Batch target-mode assembly should match repeated single-job calls."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=8,
        N=64,
        q=4,
        calc_m0=True,
    )

    rng = np.random.default_rng(1234)
    jobs = []
    for phase in (0.0, 0.3, -0.2):
        w_xi = rng.normal(size=(wdm.Nt, wdm.Nf)) + 1j * rng.normal(size=(wdm.Nt, wdm.Nf))
        t_shift = 0.15 * np.sin(np.linspace(0.0, 2.0 * np.pi, wdm.Nt) + phase)
        jobs.append((w_xi.astype(np.complex128), t_shift.astype(np.float64)))

    batch_out = tsf.wdm_time_shift_variable_batch(
        wdm,
        jobs,
        Nf=wdm.Nf,
        L_trunc=3,
        batch_chunk=2,
        tl_tp_mode="interp",
        tl_tp_interp_points=16,
        tl_tp_interp_pad=0.05,
    )
    single_out = [
        tsf.wdm_time_shift_variable(
            wdm,
            w_xi,
            t_shift,
            Nf=wdm.Nf,
            L_trunc=3,
            tl_tp_mode="interp",
            tl_tp_interp_points=16,
            tl_tp_interp_pad=0.05,
        )
        for w_xi, t_shift in jobs
    ]

    assert len(batch_out) == len(single_out)
    for batch_arr, single_arr in zip(batch_out, single_out):
        assert np.allclose(batch_arr, single_arr, atol=1e-5, rtol=1e-8)


def test_variable_shift_chunked_matches_legacy():
    """Chunked target-mode backend should match legacy assembly in exact mode."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=8,
        N=64,
        q=4,
        calc_m0=True,
    )

    rng = np.random.default_rng(2024)
    w_xi = rng.normal(size=(wdm.Nt, wdm.Nf)) + 1j * rng.normal(size=(wdm.Nt, wdm.Nf))
    t_shift = 0.1 * np.sin(np.linspace(0.0, 2.0 * np.pi, wdm.Nt))

    out_chunked = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked",
        assembly_precision="complex128",
        row_chunk_size=8,
    )
    out_legacy = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="legacy",
    )

    assert np.allclose(out_chunked, out_legacy, atol=1e-12, rtol=1e-12)


def test_variable_shift_chunked_fast_precision():
    """Complex64 chunked backend should be close to complex128 chunked output."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=8,
        N=64,
        q=4,
        calc_m0=True,
    )

    rng = np.random.default_rng(2025)
    w_xi = rng.normal(size=(wdm.Nt, wdm.Nf)) + 1j * rng.normal(size=(wdm.Nt, wdm.Nf))
    t_shift = 0.12 * np.sin(np.linspace(0.0, 2.0 * np.pi, wdm.Nt))

    out_c128 = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked",
        assembly_precision="complex128",
        row_chunk_size=8,
    )
    out_c64 = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked",
        assembly_precision="complex64",
        row_chunk_size=8,
    )

    assert np.allclose(out_c64, out_c128, atol=1e-6, rtol=1e-5)


def test_variable_shift_chunked_batch_matches_loop():
    """Chunked batch route should match looping over single chunked calls."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=8,
        N=64,
        q=4,
        calc_m0=True,
    )

    rng = np.random.default_rng(2026)
    jobs = []
    for phase in (0.0, 0.4):
        w_xi = rng.normal(size=(wdm.Nt, wdm.Nf)) + 1j * rng.normal(size=(wdm.Nt, wdm.Nf))
        t_shift = 0.1 * np.sin(np.linspace(0.0, 2.0 * np.pi, wdm.Nt) + phase)
        jobs.append((w_xi.astype(np.complex128), t_shift.astype(np.float64)))

    batch_out = tsf.wdm_time_shift_variable_batch(
        wdm,
        jobs,
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked",
        assembly_precision="complex64",
        row_chunk_size=8,
        batch_chunk=2,
    )
    single_out = [
        tsf.wdm_time_shift_variable(
            wdm,
            w_xi,
            t_shift,
            Nf=wdm.Nf,
            L_trunc=3,
            tl_tp_mode="exact",
            assembly_backend="lagfirst_chunked",
            assembly_precision="complex64",
            row_chunk_size=8,
        )
        for w_xi, t_shift in jobs
    ]

    assert len(batch_out) == len(single_out)
    for batch_arr, single_arr in zip(batch_out, single_out):
        assert np.allclose(batch_arr, single_arr, atol=1e-6, rtol=1e-5)


# Tests for the new lagblock backend
def test_lagblock_backend_single_job_c128():
    """Lagblock backend should match lagfirst_chunked baseline in complex128 mode."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=16,
        N=256,
        q=4,
        calc_m0=True,
    )

    rng = np.random.default_rng(5000)
    w_xi = rng.normal(size=(wdm.Nt, wdm.Nf)) + 1j * rng.normal(size=(wdm.Nt, wdm.Nf))
    t_shift = 0.12 * np.sin(np.linspace(0.0, 2.0 * np.pi, wdm.Nt))

    # Reference: current lagfirst_chunked
    out_reference = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked",
        assembly_precision="complex128",
        row_chunk_size=32,
    )

    # Test: lagblock with lag_block_size=1 (should match reference)
    out_lagblock_1 = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked_lagblock",
        assembly_precision="complex128",
        row_chunk_size=32,
        lag_block_size=1,
    )

    # Should be near machine precision
    rel_l2 = np.linalg.norm(out_lagblock_1 - out_reference) / np.linalg.norm(out_reference)
    max_abs = np.max(np.abs(out_lagblock_1 - out_reference))
    assert rel_l2 < 1e-12, f"lag_block_size=1: rel_l2={rel_l2:.2e} exceeds 1e-12"
    assert max_abs < 1e-11, f"lag_block_size=1: max_abs={max_abs:.2e} exceeds 1e-11"

    # Test with various lag block sizes
    for lag_block_size in [2, 4, 8]:
        out_lagblock = tsf.wdm_time_shift_variable(
            wdm,
            w_xi.astype(np.complex128),
            t_shift.astype(np.float64),
            Nf=wdm.Nf,
            L_trunc=3,
            tl_tp_mode="exact",
            assembly_backend="lagfirst_chunked_lagblock",
            assembly_precision="complex128",
            row_chunk_size=32,
            lag_block_size=lag_block_size,
        )
        rel_l2 = np.linalg.norm(out_lagblock - out_reference) / np.linalg.norm(out_reference)
        max_abs = np.max(np.abs(out_lagblock - out_reference))
        assert rel_l2 < 1e-12, f"lag_block_size={lag_block_size}: rel_l2={rel_l2:.2e} exceeds 1e-12"
        assert max_abs < 1e-11, f"lag_block_size={lag_block_size}: max_abs={max_abs:.2e} exceeds 1e-11"


def test_lagblock_backend_single_job_c64():
    """Lagblock backend should match lagfirst_chunked baseline in complex64 mode."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=16,
        N=256,
        q=4,
        calc_m0=True,
    )

    rng = np.random.default_rng(5001)
    w_xi = rng.normal(size=(wdm.Nt, wdm.Nf)) + 1j * rng.normal(size=(wdm.Nt, wdm.Nf))
    t_shift = 0.12 * np.sin(np.linspace(0.0, 2.0 * np.pi, wdm.Nt))

    # Reference: current lagfirst_chunked in complex64
    out_reference = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked",
        assembly_precision="complex64",
        row_chunk_size=32,
    )

    # Test: lagblock with lag_block_size=1 in complex64
    out_lagblock = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked_lagblock",
        assembly_precision="complex64",
        row_chunk_size=32,
        lag_block_size=1,
    )

    # Should be close in float32 precision
    rel_l2 = np.linalg.norm(out_lagblock - out_reference) / np.linalg.norm(out_reference)
    max_abs = np.max(np.abs(out_lagblock - out_reference))
    assert rel_l2 < 1e-5, f"lag_block_size=1 c64: rel_l2={rel_l2:.2e} exceeds 1e-5"
    assert max_abs < 1e-4, f"lag_block_size=1 c64: max_abs={max_abs:.2e} exceeds 1e-4"

    # Test with larger lag block size
    out_lagblock_large = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked_lagblock",
        assembly_precision="complex64",
        row_chunk_size=32,
        lag_block_size=4,
    )
    rel_l2 = np.linalg.norm(out_lagblock_large - out_reference) / np.linalg.norm(out_reference)
    assert rel_l2 < 1e-5, f"lag_block_size=4 c64: rel_l2={rel_l2:.2e} exceeds 1e-5"


def test_lagblock_backend_odd_lag_block_boundary():
    """Lagblock backend should handle partial (non-divisible) lag blocks correctly."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=16,
        N=256,
        q=4,
        calc_m0=True,
    )

    rng = np.random.default_rng(5002)
    w_xi = rng.normal(size=(wdm.Nt, wdm.Nf)) + 1j * rng.normal(size=(wdm.Nt, wdm.Nf))
    t_shift = 0.12 * np.sin(np.linspace(0.0, 2.0 * np.pi, wdm.Nt))

    # L_trunc=5 gives n_lag=11, not divisible by 4 or 8
    out_reference = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=5,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked",
        assembly_precision="complex128",
        row_chunk_size=32,
    )

    # Test with lag_block_size that doesn't divide n_lag evenly
    for lag_block_size in [4, 8]:
        out_lagblock = tsf.wdm_time_shift_variable(
            wdm,
            w_xi.astype(np.complex128),
            t_shift.astype(np.float64),
            Nf=wdm.Nf,
            L_trunc=5,
            tl_tp_mode="exact",
            assembly_backend="lagfirst_chunked_lagblock",
            assembly_precision="complex128",
            row_chunk_size=32,
            lag_block_size=lag_block_size,
        )
        rel_l2 = np.linalg.norm(out_lagblock - out_reference) / np.linalg.norm(out_reference)
        max_abs = np.max(np.abs(out_lagblock - out_reference))
        assert rel_l2 < 1e-12, f"boundary test lag_block_size={lag_block_size}: rel_l2={rel_l2:.2e}"
        assert max_abs < 1e-11, f"boundary test lag_block_size={lag_block_size}: max_abs={max_abs:.2e}"


def test_lagblock_backend_alias_names():
    """Lagblock backend aliases should all work."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=8,
        N=128,
        q=4,
        calc_m0=True,
    )

    rng = np.random.default_rng(5003)
    w_xi = rng.normal(size=(wdm.Nt, wdm.Nf)) + 1j * rng.normal(size=(wdm.Nt, wdm.Nf))
    t_shift = 0.1 * np.sin(np.linspace(0.0, 2.0 * np.pi, wdm.Nt))

    # Test that all aliases produce the same result
    out_full_name = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=2,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked_lagblock",
        assembly_precision="complex128",
        row_chunk_size=16,
        lag_block_size=2,
    )

    out_alias_lagblock = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=2,
        tl_tp_mode="exact",
        assembly_backend="lagblock",
        assembly_precision="complex128",
        row_chunk_size=16,
        lag_block_size=2,
    )

    out_alias_lagfirst = tsf.wdm_time_shift_variable(
        wdm,
        w_xi.astype(np.complex128),
        t_shift.astype(np.float64),
        Nf=wdm.Nf,
        L_trunc=2,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_lagblock",
        assembly_precision="complex128",
        row_chunk_size=16,
        lag_block_size=2,
    )

    assert np.allclose(out_full_name, out_alias_lagblock, atol=1e-12)
    assert np.allclose(out_full_name, out_alias_lagfirst, atol=1e-12)


def test_lagblock_backend_validation():
    """Lagblock backend should validate lag_block_size correctly."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=8,
        N=128,
        q=4,
        calc_m0=True,
    )

    w_xi = np.ones((wdm.Nt, wdm.Nf), dtype=np.complex128)
    t_shift = np.zeros(wdm.Nt, dtype=np.float64)

    # Test lag_block_size=0 should raise ValueError
    with pytest.raises(ValueError, match="lag_block_size must be >= 1"):
        tsf.wdm_time_shift_variable(
            wdm,
            w_xi,
            t_shift,
            Nf=wdm.Nf,
            L_trunc=2,
            assembly_backend="lagfirst_chunked_lagblock",
            lag_block_size=0,
        )

    # Test lag_block_size<0 should raise ValueError
    with pytest.raises(ValueError, match="lag_block_size must be >= 1"):
        tsf.wdm_time_shift_variable(
            wdm,
            w_xi,
            t_shift,
            Nf=wdm.Nf,
            L_trunc=2,
            assembly_backend="lagfirst_chunked_lagblock",
            lag_block_size=-1,
        )


def test_lagblock_backend_batch_job():
    """Lagblock batch shifting should match single-job calls."""
    if not getattr(tsf, "_JAX_AVAILABLE", False):
        pytest.skip("JAX assembly backend is not available.")

    wdm = WDM.WDM_transform(
        dt=0.5,
        Nf=16,
        N=256,
        q=4,
        calc_m0=True,
    )

    rng = np.random.default_rng(5004)
    jobs = []
    for phase in (0.0, 0.3):
        w_xi = rng.normal(size=(wdm.Nt, wdm.Nf)) + 1j * rng.normal(size=(wdm.Nt, wdm.Nf))
        t_shift = 0.12 * np.sin(np.linspace(0.0, 2.0 * np.pi, wdm.Nt) + phase)
        jobs.append((w_xi.astype(np.complex128), t_shift.astype(np.float64)))

    # Batch call with lagblock
    batch_out = tsf.wdm_time_shift_variable_batch(
        wdm,
        jobs,
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="lagfirst_chunked_lagblock",
        assembly_precision="complex128",
        row_chunk_size=32,
        lag_block_size=2,
        batch_chunk=2,
    )

    # Single job calls with lagblock
    single_out = [
        tsf.wdm_time_shift_variable(
            wdm,
            w_xi,
            t_shift,
            Nf=wdm.Nf,
            L_trunc=3,
            tl_tp_mode="exact",
            assembly_backend="lagfirst_chunked_lagblock",
            assembly_precision="complex128",
            row_chunk_size=32,
            lag_block_size=2,
        )
        for w_xi, t_shift in jobs
    ]

    assert len(batch_out) == len(single_out)
    for batch_arr, single_arr in zip(batch_out, single_out):
        rel_l2 = np.linalg.norm(batch_arr - single_arr) / np.linalg.norm(single_arr)
        assert rel_l2 < 1e-12, f"batch/single mismatch: rel_l2={rel_l2:.2e}"

