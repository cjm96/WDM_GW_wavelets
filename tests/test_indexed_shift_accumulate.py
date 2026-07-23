"""Regression tests for indexed grouped shift accumulation."""

from dataclasses import replace

import numpy as np
import pytest

import WDM
from WDM.code.time_delay_filters.config import VariableShiftPlanConfig
from WDM.code.time_delay_filters.plans import VariableShiftBatchPlan


def _fixture():
    rng = np.random.default_rng(73142)
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

    num_sources = 3
    num_jobs = 7
    num_outputs = 3
    sources = rng.normal(size=(num_sources, Nt, Nf)) + 1j * rng.normal(
        size=(num_sources, Nt, Nf)
    )
    source_indices = np.asarray([0, 1, 2, 0, 2, 1, 0], dtype=np.int32)
    output_indices = np.asarray([0, 0, 1, 2, 1, 2, 0], dtype=np.int32)
    weights = np.asarray([1, -1, 0.5, 1, -0.25, -1, 0.75])
    base = rng.normal(size=(num_outputs, Nt, Nf)) + 1j * rng.normal(
        size=(num_outputs, Nt, Nf)
    )

    time = np.linspace(0.0, 1.0, Nt)
    delays = np.stack(
        [
            7.0 + 0.2 * job + 0.1 * np.sin((job + 1) * time)
            for job in range(num_jobs)
        ]
    )
    return (
        wdm,
        delays,
        sources,
        source_indices,
        output_indices,
        weights,
        base,
        num_outputs,
    )


@pytest.mark.parametrize(
    "config, rtol, atol",
    [
        (
            replace(
                VariableShiftPlanConfig.reference(lag_truncation=3),
                batch_chunk=4,
            ),
            1e-11,
            1e-12,
        ),
        (
            VariableShiftPlanConfig.production(
                lag_truncation=3,
                interpolation_points=16,
                row_chunk_size=8,
                lag_block_size=3,
                batch_chunk=4,
            ),
            3e-5,
            3e-6,
        ),
    ],
)
def test_indexed_accumulate_matches_materialized(config, rtol, atol):
    (
        wdm,
        delays,
        sources,
        source_indices,
        output_indices,
        weights,
        base,
        num_outputs,
    ) = _fixture()

    plan = VariableShiftBatchPlan.build(wdm, delays, config=config)
    shifted = plan.apply(sources[source_indices])
    expected = np.asarray(
        base,
        dtype=np.result_type(base.dtype, shifted.dtype),
    ).copy()
    np.add.at(
        expected,
        output_indices,
        weights[:, None, None] * shifted,
    )

    actual, profile = plan.apply_indexed_and_accumulate(
        sources,
        source_indices,
        output_indices,
        weights,
        num_outputs=num_outputs,
        base=base,
        return_profile=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    assert profile["n_sources"] == sources.shape[0]
    assert profile["n_jobs"] == delays.shape[0]
    assert profile["n_outputs"] == num_outputs
    assert profile["full_materialized_output_bytes"] > profile["returned_output_bytes"]
    assert profile["host_transfer_bytes_saved"] > 0
    assert plan._device_cache

    plan.clear_device_cache()
    assert not plan._device_cache


def test_indexed_accumulate_validates_indices():
    (
        wdm,
        delays,
        sources,
        source_indices,
        output_indices,
        weights,
        base,
        num_outputs,
    ) = _fixture()

    plan = VariableShiftBatchPlan.build(
        wdm,
        delays,
        config=VariableShiftPlanConfig.reference(lag_truncation=2),
    )
    bad_sources = source_indices.copy()
    bad_sources[0] = sources.shape[0]

    with pytest.raises(ValueError, match="source_indices"):
        plan.apply_indexed_and_accumulate(
            sources,
            bad_sources,
            output_indices,
            weights,
            num_outputs=num_outputs,
            base=base,
        )
