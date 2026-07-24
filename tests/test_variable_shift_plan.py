"""Regression tests for ``VariableShiftBatchPlan``."""

from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest
import WDM

from WDM.code.time_delay_filters.config import VariableShiftPlanConfig
from WDM.code.time_delay_filters.plans import VariableShiftBatchPlan
from WDM.code.time_delay_filters.time_shift_fast import (
    wdm_time_shift_variable_batch,
)


@pytest.fixture
def variable_shift_case():
    rng = np.random.default_rng(123456)
    Nf = 16
    Nt = 8
    wdm = WDM.WDM.WDM_transform(
        dt=1.0,
        Nf=Nf,
        N=Nf * Nt,
        q=Nt // 2,
        calc_m0=True,
        d=4,
        A_frac=0.25,
    )

    num_jobs = 3
    coefficients = rng.normal(size=(num_jobs, Nt, Nf))
    centre = np.linspace(7.0, 9.0, Nt)
    delays = np.stack(
        [
            centre,
            centre + 0.15 * np.sin(np.linspace(0.0, 2.0 * np.pi, Nt)),
            centre - 0.10 * np.cos(np.linspace(0.0, 2.0 * np.pi, Nt)),
        ]
    )
    return wdm, coefficients, delays


@pytest.mark.parametrize(
    "config, rtol, atol",
    [
        (
            VariableShiftPlanConfig.reference(lag_truncation=3),
            1e-11,
            1e-12,
        ),
        (
            VariableShiftPlanConfig.production(
                lag_truncation=3,
                interpolation_points=16,
                row_chunk_size=8,
                lag_block_size=3,
                batch_chunk=2,
            ),
            2e-5,
            2e-6,
        ),
    ],
)
def test_plan_matches_direct_batch(variable_shift_case, config, rtol, atol):
    wdm, coefficients, delays = variable_shift_case
    direct = wdm_time_shift_variable_batch(
        wdm,
        list(zip(coefficients, delays)),
        L_trunc=config.lag_truncation,
        batch_chunk=config.batch_chunk,
        tl_tp_mode=config.tl_tp_mode,
        tl_tp_interp_points=config.tl_tp_interp_points,
        tl_tp_interp_pad=config.tl_tp_interp_pad,
        tl_tp_interp_kind=config.tl_tp_interp_kind,
        assembly_backend=config.assembly_backend,
        assembly_precision=config.assembly_precision,
        row_chunk_size=config.row_chunk_size,
        lag_block_size=config.lag_block_size,
    )

    plan = VariableShiftBatchPlan.build(wdm, delays, config=config)
    planned = plan.apply(coefficients)
    np.testing.assert_allclose(
        planned,
        np.stack(direct),
        rtol=rtol,
        atol=atol,
    )


def test_plan_reuse_does_not_rebuild_tl_tp(variable_shift_case, monkeypatch):
    wdm, coefficients, delays = variable_shift_case
    config = VariableShiftPlanConfig.production(
        lag_truncation=3,
        interpolation_points=16,
        row_chunk_size=8,
        lag_block_size=3,
    )
    plan = VariableShiftBatchPlan.build(wdm, delays, config=config)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Tl/Tp construction was called during apply().")

    monkeypatch.setattr(
        "WDM.code.time_delay_filters.plans._build_TlTp_from_shift_matrix",
        fail_if_called,
    )
    monkeypatch.setattr(
        "WDM.code.time_delay_filters.plans._build_TlTp_from_shift_matrix_interp",
        fail_if_called,
    )

    first = plan.apply(coefficients)
    second = plan.apply(0.5 * coefficients)
    np.testing.assert_allclose(second, 0.5 * first, rtol=2e-5, atol=2e-6)


def test_production_plan_omits_checkerboard_and_uses_production_precision(
    variable_shift_case,
):
    wdm, _, delays = variable_shift_case
    config = VariableShiftPlanConfig.production(
        lag_truncation=3,
        interpolation_points=16,
        row_chunk_size=8,
        lag_block_size=3,
    )
    plan = VariableShiftBatchPlan.build(wdm, delays, config=config)

    assert plan.Cnm is None
    assert plan.delays.dtype == np.float32
    assert plan.ell_all.dtype == np.int32
    assert plan.Tl_all.dtype == np.complex64
    assert plan.Tp_all.dtype == np.complex64


def test_reference_plan_retains_checkerboard(variable_shift_case):
    wdm, _, delays = variable_shift_case
    plan = VariableShiftBatchPlan.build(
        wdm,
        delays,
        config=VariableShiftPlanConfig.reference(lag_truncation=3),
    )
    assert plan.Cnm is not None
    assert plan.Cnm.dtype == np.complex128


def test_real_device_application_returns_real_array(variable_shift_case):
    wdm, coefficients, delays = variable_shift_case
    config = VariableShiftPlanConfig.production(
        lag_truncation=3,
        interpolation_points=16,
        row_chunk_size=8,
        lag_block_size=3,
    )
    plan = VariableShiftBatchPlan.build(wdm, delays, config=config)
    output = plan.apply_device(coefficients)
    output.block_until_ready()
    assert str(output.dtype) == "float32"


def test_cpu_real_device_application_matches_explicit_complex_input(
    variable_shift_case,
):
    if jax.default_backend() != "cpu":
        pytest.skip("The real-via-complex production path is CPU-specific.")

    wdm, coefficients, delays = variable_shift_case
    config = VariableShiftPlanConfig.production(
        lag_truncation=3,
        interpolation_points=16,
        row_chunk_size=8,
        lag_block_size=3,
    )
    plan = VariableShiftBatchPlan.build(wdm, delays, config=config)

    real_output = plan.apply_device(coefficients)
    complex_output = plan.apply_device(coefficients.astype(np.complex64))
    real_output.block_until_ready()
    complex_output.block_until_ready()

    assert str(real_output.dtype) == "float32"
    assert str(complex_output.dtype) == "complex64"
    np.testing.assert_array_equal(
        np.asarray(real_output),
        np.asarray(complex_output.real),
    )
    np.testing.assert_array_equal(
        np.asarray(complex_output.imag),
        np.zeros_like(np.asarray(complex_output.imag)),
    )


def test_plan_rejects_wrong_coefficient_shape(variable_shift_case):
    wdm, coefficients, delays = variable_shift_case
    plan = VariableShiftBatchPlan.build(
        wdm,
        delays,
        config=VariableShiftPlanConfig.production(
            lag_truncation=3,
            row_chunk_size=8,
            lag_block_size=3,
        ),
    )
    with pytest.raises(ValueError, match="Expected coefficients with shape"):
        plan.apply(coefficients[:, :-1, :])


def test_config_can_be_replaced_for_benchmark_chunks():
    base = VariableShiftPlanConfig.production(lag_truncation=3)
    updated = replace(base, row_chunk_size=16, lag_block_size=4)
    assert updated.row_chunk_size == 16
    assert updated.lag_block_size == 4


def test_real_split_plan_matches_production(variable_shift_case):
    wdm, coefficients, delays = variable_shift_case
    base = VariableShiftPlanConfig.production(
        lag_truncation=3, interpolation_points=16,
        row_chunk_size=8, lag_block_size=3, batch_chunk=2,
    )
    production_plan = VariableShiftBatchPlan.build(wdm, delays, config=base)
    real_split_plan = VariableShiftBatchPlan.build(
        wdm, delays,
        config=replace(base, assembly_backend="production_real_split"),
    )
    production = production_plan.apply_device(coefficients)
    candidate = real_split_plan.apply_device(coefficients)
    production.block_until_ready()
    candidate.block_until_ready()
    assert str(candidate.dtype) == "float32"
    np.testing.assert_allclose(
        np.asarray(candidate), np.asarray(production), rtol=2e-5, atol=2e-6
    )
