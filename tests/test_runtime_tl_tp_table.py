
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("WDM")

import WDM
from WDM.code.time_delay_filters.config import VariableShiftPlanConfig
from WDM.code.time_delay_filters.plans import VariableShiftKernelContext
from WDM.code.time_delay_filters.time_shift_fast import (
    _build_TlTp_from_shift_matrix_interp,
    _build_kernel_wdm_like,
    _build_signed_lag_idx,
    _get_kernel_precomputes,
    _resolve_ell_range,
    choose_Nker,
)


def _small_wdm():
    Nf = 16
    Nt = 8
    return WDM.WDM.WDM_transform(
        dt=1.0,
        Nf=Nf,
        N=Nf * Nt,
        q=Nt // 2,
        calc_m0=True,
        d=4,
        A_frac=0.25,
    )


def test_runtime_table_matches_historical_local_interpolator():
    wdm = _small_wdm()
    delays = np.stack(
        [
            np.linspace(-3.0, 2.0, wdm.Nt),
            np.linspace(-2.5, 1.5, wdm.Nt),
        ]
    )
    config = VariableShiftPlanConfig.production(
        lag_truncation=3,
        interpolation_points=64,
        row_chunk_size=8,
        lag_block_size=3,
        batch_chunk=2,
    )
    context = VariableShiftKernelContext.build(
        wdm,
        delay_min=float(delays.min()),
        delay_max=float(delays.max()),
        config=config,
        interpolation_points=64,
        interpolation_pad=0.0,
    )
    table_Tl, table_Tp = context.interpolation_table.evaluate(delays)

    ell_all, offset = _resolve_ell_range(wdm.Nt, 3)
    Nker = choose_Nker(offset, wdm.Nf)
    wdm_kernel, freqs_u, W0_u, W1_u, scale = _get_kernel_precomputes(
        wdm, Nker, wdm.Nf, None
    )
    idx = _build_signed_lag_idx(ell_all, wdm.Nf, int(wdm_kernel.N))
    old_Tl, old_Tp = _build_TlTp_from_shift_matrix_interp(
        delays,
        freqs_u,
        W0_u,
        W1_u,
        scale,
        idx,
        interp_points=64,
        interp_pad=0.0,
        interp_kind="linear",
    )
    np.testing.assert_allclose(table_Tl, old_Tl, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(table_Tp, old_Tp, rtol=0.0, atol=0.0)


def test_runtime_context_reuses_table_across_delay_fields():
    rng = np.random.default_rng(1234)
    wdm = _small_wdm()
    config = VariableShiftPlanConfig.production(
        lag_truncation=3,
        interpolation_points=32,
        row_chunk_size=8,
        lag_block_size=3,
        batch_chunk=2,
    )
    context = VariableShiftKernelContext.build(
        wdm,
        delay_min=-10.0,
        delay_max=10.0,
        config=config,
        interpolation_points=256,
    )
    coefficients = rng.normal(size=(2, wdm.Nt, wdm.Nf))
    first_delays = np.stack(
        [np.linspace(-2.0, 2.0, wdm.Nt), np.linspace(-1.0, 3.0, wdm.Nt)]
    )
    second_delays = first_delays + 0.25

    table_id = id(context.interpolation_table)
    first = context.apply(coefficients, first_delays)
    second = context.apply(coefficients, second_delays)

    assert id(context.interpolation_table) == table_id
    assert first.shape == second.shape == coefficients.shape
    assert not np.allclose(first, second)
