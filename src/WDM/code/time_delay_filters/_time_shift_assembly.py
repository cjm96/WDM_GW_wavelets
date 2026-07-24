"""Dispatch helpers for the supported WDM time-shift implementations.

The production API intentionally has one optimized target-mode kernel and one
reference target-mode kernel.  Experimental job-block, prephased, weighted-
source and alternative row-order implementations were removed from dispatch;
their history remains available in version control.
"""

from __future__ import annotations

from ._time_shift_jax import (
    assemble_shift_fixed_jax,
    assemble_shift_target_batch_production_jax,
    assemble_shift_target_batch_factored_phase_jax,
    assemble_shift_target_batch_reference_jax,
    assemble_shift_target_production_jax,
    assemble_shift_target_factored_phase_jax,
    assemble_shift_target_reference_jax,
    assemble_shift_variable_mode_jax,
)


def _assemble_shift_target_dispatch(
    wdm,
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    *,
    Cnm=None,
    assembly_backend="production",
    assembly_precision="complex64",
    row_chunk_size=128,
    lag_block_size=1,
    return_device=False,
):
    """Apply one target-mode shift with the selected supported backend."""

    backend = str(assembly_backend).lower()
    if backend == "production_factored_phase":
        return assemble_shift_target_factored_phase_jax(
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            float(wdm.dF),
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            precision=assembly_precision,
            return_device=return_device,
        )

    if backend == "production":
        return assemble_shift_target_production_jax(
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            float(wdm.dF),
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            precision=assembly_precision,
            return_device=return_device,
        )

    if backend == "reference":
        if Cnm is None:
            raise ValueError("The reference backend requires the Cnm array.")
        return assemble_shift_target_reference_jax(
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            Cnm,
            float(wdm.dF),
            return_device=return_device,
        )

    raise ValueError("assembly_backend must be 'production', 'production_factored_phase', or 'reference'.")


def _assemble_shift_target_batch_dispatch(
    wdm,
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    *,
    Cnm=None,
    assembly_backend="production",
    assembly_precision="complex64",
    row_chunk_size=128,
    lag_block_size=1,
    return_device=False,
):
    """Apply target-mode shifts to a batch of independent jobs."""

    backend = str(assembly_backend).lower()
    if backend == "production_factored_phase":
        return assemble_shift_target_batch_factored_phase_jax(
            w_xi_batch,
            t_shift_batch,
            ell_all,
            offset,
            Tl_batch,
            Tp_batch,
            float(wdm.dF),
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            precision=assembly_precision,
            return_device=return_device,
        )

    if backend == "production":
        return assemble_shift_target_batch_production_jax(
            w_xi_batch,
            t_shift_batch,
            ell_all,
            offset,
            Tl_batch,
            Tp_batch,
            float(wdm.dF),
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            precision=assembly_precision,
            return_device=return_device,
        )

    if backend == "reference":
        if Cnm is None:
            raise ValueError("The reference backend requires the Cnm array.")
        return assemble_shift_target_batch_reference_jax(
            w_xi_batch,
            t_shift_batch,
            ell_all,
            offset,
            Tl_batch,
            Tp_batch,
            Cnm,
            float(wdm.dF),
            return_device=return_device,
        )

    raise ValueError("assembly_backend must be 'production', 'production_factored_phase', or 'reference'.")


def _assemble_shift_fixed_dispatch(
    wdm,
    w_xi,
    delta,
    ell_all,
    offset,
    Tl_vec,
    Tp_vec,
    Cnm,
):
    """Apply the retained high-precision fixed-delay reference operator."""

    return assemble_shift_fixed_jax(
        w_xi,
        delta,
        ell_all,
        offset,
        Tl_vec,
        Tp_vec,
        Cnm,
        float(wdm.dF),
    )


def _assemble_shift_variable_mode_dispatch(
    wdm,
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    *,
    delta_mode,
):
    """Apply the retained source/midpoint reference operator."""

    return assemble_shift_variable_mode_jax(
        w_xi,
        t_shift,
        ell_all,
        offset,
        Tl_all,
        Tp_all,
        Cnm,
        float(wdm.dF),
        delta_mode,
    )
