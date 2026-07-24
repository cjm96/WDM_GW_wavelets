"""Core regression tests for the maintained WDM delay implementations."""

from __future__ import annotations

import numpy as np
import pytest

import WDM
from WDM.code.discrete_wavelet_transform import WDM as WDM_module
from WDM.code.time_delay_filters.filters import (
    time_delay_X,
    time_delay_filter_Tl,
    time_delay_filter_Tprimel,
)
from WDM.code.time_delay_filters import time_shift_fast as tsf


def _small_shift_case(*, real_coefficients: bool, num_jobs: int = 2):
    rng = np.random.default_rng(20260723)
    Nf = 16
    Nt = 16
    wdm = WDM_module.WDM_transform(
        dt=0.5,
        Nf=Nf,
        N=Nf * Nt,
        q=4,
        calc_m0=True,
        d=4,
        A_frac=0.25,
    )

    coefficients = rng.normal(size=(num_jobs, Nt, Nf))
    if not real_coefficients:
        coefficients = coefficients + 1j * rng.normal(
            size=(num_jobs, Nt, Nf)
        )

    time = np.linspace(0.0, 2.0 * np.pi, Nt)
    delays = np.stack(
        [0.12 * np.sin(time + 0.3 * index) for index in range(num_jobs)]
    )
    return wdm, coefficients, delays


def test_filter_functions_return_real_scalars():
    wdm = WDM_module.WDM_transform(
        dt=0.5,
        Nf=8,
        N=64,
        q=4,
        calc_m0=True,
    )
    assert isinstance(time_delay_filter_Tl(wdm, 0, 1.0), float)
    assert isinstance(time_delay_filter_Tprimel(wdm, 0, 1.0), float)


def test_kernel_wdm_inherits_A_frac():
    tsf._KERNEL_WDM_CACHE.clear()
    tsf._KERNEL_PRECOMP_CACHE.clear()

    first = WDM_module.WDM_transform(
        dt=10.0,
        Nf=100,
        N=20000,
        q=1,
        calc_m0=False,
        d=4,
        A_frac=0.20,
    )
    second = WDM_module.WDM_transform(
        dt=10.0,
        Nf=100,
        N=20000,
        q=1,
        calc_m0=False,
        d=4,
        A_frac=0.30,
    )

    first_kernel = tsf._build_kernel_wdm_like(
        first,
        Nker=5200,
        Nf=100,
        d=4,
        calc_m0=True,
    )
    second_kernel = tsf._build_kernel_wdm_like(
        second,
        Nker=5200,
        Nf=100,
        d=4,
        calc_m0=True,
    )

    assert np.isclose(float(first_kernel.A_frac), 0.20)
    assert np.isclose(float(second_kernel.A_frac), 0.30)
    assert first_kernel is not second_kernel


def test_filter_X_is_identity_at_zero_delay():
    wdm = WDM_module.WDM_transform(
        dt=0.5,
        Nf=8,
        N=64,
        q=4,
        calc_m0=True,
    )
    for n in range(wdm.Nt):
        for nprime in range(wdm.Nt):
            for m in range(wdm.Nf):
                for mprime in range(wdm.Nf):
                    expected = float(n == nprime and m == mprime)
                    assert np.isclose(
                        time_delay_X(wdm, n, nprime, m, mprime, 0.0),
                        expected,
                    )


@pytest.mark.parametrize("real_coefficients", [True, False])
def test_production_matches_reference(real_coefficients):
    wdm, coefficients, delays = _small_shift_case(
        real_coefficients=real_coefficients,
        num_jobs=1,
    )
    common = dict(
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
    )

    reference = tsf.wdm_time_shift_variable(
        wdm,
        coefficients[0],
        delays[0],
        assembly_backend="reference",
        assembly_precision="complex128",
        **common,
    )
    production = tsf.wdm_time_shift_variable(
        wdm,
        coefficients[0],
        delays[0],
        assembly_backend="production",
        assembly_precision="complex128",
        row_chunk_size=8,
        lag_block_size=3,
        **common,
    )

    np.testing.assert_allclose(production, reference, rtol=1e-12, atol=1e-12)
    if real_coefficients:
        assert np.issubdtype(production.dtype, np.floating)
    else:
        assert np.issubdtype(production.dtype, np.complexfloating)


def test_production_complex64_is_close_to_reference():
    wdm, coefficients, delays = _small_shift_case(
        real_coefficients=True,
        num_jobs=1,
    )
    reference = tsf.wdm_time_shift_variable(
        wdm,
        coefficients[0],
        delays[0],
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="reference",
        assembly_precision="complex128",
    )
    production = tsf.wdm_time_shift_variable(
        wdm,
        coefficients[0],
        delays[0],
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="production",
        assembly_precision="complex64",
        row_chunk_size=8,
        lag_block_size=3,
    )
    np.testing.assert_allclose(production, reference.real, rtol=2e-5, atol=2e-6)
    assert production.dtype == np.float32


@pytest.mark.parametrize("real_coefficients", [True, False])
def test_reordered_production_matches_baseline(real_coefficients):
    wdm, coefficients, delays = _small_shift_case(
        real_coefficients=real_coefficients,
        num_jobs=1,
    )
    kwargs = dict(
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="exact",
        assembly_backend="production",
        assembly_precision="complex64",
        row_chunk_size=8,
        lag_block_size=3,
    )

    baseline = tsf.wdm_time_shift_variable(
        wdm,
        coefficients[0],
        delays[0],
        assembly_variant="baseline",
        **kwargs,
    )
    reordered = tsf.wdm_time_shift_variable(
        wdm,
        coefficients[0],
        delays[0],
        assembly_variant="reordered",
        **kwargs,
    )

    np.testing.assert_allclose(
        reordered,
        baseline,
        rtol=2e-5,
        atol=2e-6,
    )


def test_batch_matches_repeated_single_calls():
    wdm, coefficients, delays = _small_shift_case(
        real_coefficients=True,
        num_jobs=3,
    )
    jobs = list(zip(coefficients, delays))
    kwargs = dict(
        Nf=wdm.Nf,
        L_trunc=3,
        tl_tp_mode="interp",
        tl_tp_interp_points=16,
        assembly_backend="production",
        assembly_precision="complex64",
        row_chunk_size=8,
        lag_block_size=3,
    )

    batch = tsf.wdm_time_shift_variable_batch(
        wdm,
        jobs,
        batch_chunk=2,
        **kwargs,
    )
    singles = [
        tsf.wdm_time_shift_variable(wdm, values, delay, **kwargs)
        for values, delay in jobs
    ]

    for batch_values, single_values in zip(batch, singles):
        np.testing.assert_allclose(
            batch_values,
            single_values,
            rtol=2e-5,
            atol=2e-6,
        )


def test_historical_production_alias_maps_to_production():
    assert tsf._normalize_assembly_backend(
        "lagfirst_chunked_lagblock"
    ) == "production"
