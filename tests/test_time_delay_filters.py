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

