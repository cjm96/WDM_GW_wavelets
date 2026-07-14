"""Regression tests for ``VariableShiftBatchPlan``.

Place this in the WDM_GW_wavelets test directory and adapt the fixture import
names to the existing test suite.  The numerical fixture only needs to return:

    wdm           WDM transform object
    coefficients  ndarray, shape (B, Nt, Nf)
    delays        ndarray, shape (B, Nt)
"""

from __future__ import annotations

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
    """Small deterministic batch suitable for fast regression tests."""

    rng = np.random.default_rng(123456)

    Nf = 16
    Nt = 8
    N = Nf * Nt
    wdm = WDM.WDM.WDM_transform(
        dt=1.0,
        Nf=Nf,
        N=N,
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
        ],
        axis=0,
    )
    return wdm, coefficients, delays


@pytest.mark.parametrize(
    "config, rtol, atol",
    [
        (
            VariableShiftPlanConfig.reference(lag_truncation=3),
            1.0e-11,
            1.0e-12,
        ),
        (
            VariableShiftPlanConfig.production(
                lag_truncation=3,
                interpolation_points=16,
            ),
            2.0e-5,
            2.0e-6,
        ),
    ],
)
def test_plan_matches_legacy_batch(variable_shift_case, config, rtol, atol):
    wdm, coefficients, delays = variable_shift_case

    jobs = list(zip(coefficients, delays))
    legacy = wdm_time_shift_variable_batch(
        wdm,
        jobs,
        L_trunc=config.lag_truncation,
        batch_chunk=config.batch_chunk,
        use_jax=config.use_jax,
        tl_tp_mode=config.tl_tp_mode,
        tl_tp_interp_points=config.tl_tp_interp_points,
        tl_tp_interp_pad=config.tl_tp_interp_pad,
        tl_tp_interp_kind=config.tl_tp_interp_kind,
        tl_tp_interp_backend=config.tl_tp_interp_backend,
        assembly_backend=config.assembly_backend,
        assembly_precision=config.assembly_precision,
        row_chunk_size=config.row_chunk_size,
        lag_block_size=config.lag_block_size,
        job_block_size=config.job_block_size,
        assembly_vmap=config.assembly_vmap,
        jax_pad_last_chunk=config.jax_pad_last_chunk,
    )

    plan = VariableShiftBatchPlan.build(
        wdm,
        delays,
        config=config,
    )
    planned = plan.apply(coefficients)

    np.testing.assert_allclose(
        planned,
        np.stack(legacy, axis=0),
        rtol=rtol,
        atol=atol,
    )


def test_plan_can_be_reused_without_rebuilding(variable_shift_case, monkeypatch):
    wdm, coefficients, delays = variable_shift_case
    config = VariableShiftPlanConfig.production(
        lag_truncation=3,
        interpolation_points=16,
    )

    plan = VariableShiftBatchPlan.build(wdm, delays, config=config)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Tl/Tp construction was called during plan.apply().")

    # Patch the helper names in the plans module, where apply() would look them
    # up if it rebuilt them.  A correct apply() never touches these functions.
    monkeypatch.setattr(
        "WDM.code.time_delay_filters.plans._build_TlTp_from_shift_matrix",
        fail_if_called,
    )
    monkeypatch.setattr(
        "WDM.code.time_delay_filters.plans._build_TlTp_from_shift_matrix_interp",
        fail_if_called,
    )
    monkeypatch.setattr(
        "WDM.code.time_delay_filters.plans._build_TlTp_from_shift_matrix_interp_jax",
        fail_if_called,
    )

    first = plan.apply(coefficients)
    second = plan.apply(0.5 * coefficients)

    np.testing.assert_allclose(second, 0.5 * first, rtol=2.0e-5, atol=2.0e-6)


def test_plan_rejects_wrong_coefficient_shape(variable_shift_case):
    wdm, coefficients, delays = variable_shift_case
    plan = VariableShiftBatchPlan.build(
        wdm,
        delays,
        config=VariableShiftPlanConfig.production(lag_truncation=3),
    )

    with pytest.raises(ValueError, match="Expected coefficients with shape"):
        plan.apply(coefficients[:, :-1, :])