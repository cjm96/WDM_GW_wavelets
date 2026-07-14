"""Assembly dispatch helpers for WDM time-shift operators.

This module keeps the high-level assembly entrypoints separate from
``time_shift_fast.py`` so the main shift code only handles preprocessing and
dispatch.
"""

from ._time_shift_jax import (
    assemble_shift_fixed_jax,
    assemble_shift_target_jax,
    assemble_shift_target_chunked_jax,
    assemble_shift_target_batch_jax,
    assemble_shift_target_batch_chunked_jax,
    assemble_shift_target_chunked_lagblock_jax,
    assemble_shift_target_batch_chunked_lagblock_jax,
    assemble_shift_target_batch_chunked_lagblock_jobblock_jax,
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
    Cnm,
    use_jax,
    assembly_backend="lagfirst_chunked",
    assembly_precision="complex64",
    row_chunk_size=128,
    lag_block_size=1,
    assembly_vmap=False,
):
    """Dispatch target-mode assembly to the JAX backend.

    The ``use_jax`` flag is retained only for API compatibility with older
    call sites. The active implementation is always the JAX kernel.
    """
    _ = wdm, use_jax
    backend = str(assembly_backend).lower() if assembly_backend is not None else "lagfirst_chunked"
    if backend in ("lagfirst_chunked", "chunked", "auto"):
        return assemble_shift_target_chunked_jax(
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            Cnm,
            float(wdm.dF),
            row_chunk_size=row_chunk_size,
            precision=assembly_precision,
        )

    if backend in ("lagfirst_chunked_lagblock", "lagblock", "lagfirst_lagblock"):
        return assemble_shift_target_chunked_lagblock_jax(
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            Cnm,
            float(wdm.dF),
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            precision=assembly_precision,
        )

    if backend in ("legacy", "row", "lagfirst_row"):
        return assemble_shift_target_jax(
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            Cnm,
            float(wdm.dF),
            assembly_vmap=False,
        )

    if backend == "vmap":
        return assemble_shift_target_jax(
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            Cnm,
            float(wdm.dF),
            assembly_vmap=True,
        )

    raise ValueError("assembly_backend must be lagfirst_chunked, lagfirst_chunked_lagblock, legacy, row, lagfirst_row, vmap, or auto.")


def _assemble_shift_target_batch_dispatch(
    wdm,
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    use_jax,
    assembly_backend="lagfirst_chunked",
    assembly_precision="complex64",
    row_chunk_size=128,
    lag_block_size=1,
    job_block_size=1,
    assembly_vmap=False,
    return_device=False,
):
    """Dispatch batched target-mode assembly to the JAX backend."""
    _ = wdm, use_jax
    backend = str(assembly_backend).lower() if assembly_backend is not None else "lagfirst_chunked"
    if backend in ("lagfirst_chunked", "chunked", "auto"):
        return assemble_shift_target_batch_chunked_jax(
            w_xi_batch,
            t_shift_batch,
            ell_all,
            offset,
            Tl_batch,
            Tp_batch,
            Cnm,
            float(wdm.dF),
            row_chunk_size=row_chunk_size,
            precision=assembly_precision,
            return_device=return_device,
        )

    if backend in ("lagfirst_chunked_lagblock", "lagblock", "lagfirst_lagblock"):
        return assemble_shift_target_batch_chunked_lagblock_jax(
            w_xi_batch,
            t_shift_batch,
            ell_all,
            offset,
            Tl_batch,
            Tp_batch,
            Cnm,
            float(wdm.dF),
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            precision=assembly_precision,
            return_device=return_device,
        )

    if backend in ("lagfirst_chunked_lagblock_jobblock", "lagblock_jobblock", "jobblock"):
        return assemble_shift_target_batch_chunked_lagblock_jobblock_jax(
            w_xi_batch,
            t_shift_batch,
            ell_all,
            offset,
            Tl_batch,
            Tp_batch,
            Cnm,
            float(wdm.dF),
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            job_block_size=job_block_size,
            precision=assembly_precision,
            return_device=return_device,
        )

    return assemble_shift_target_batch_jax(
        w_xi_batch,
        t_shift_batch,
        ell_all,
        offset,
        Tl_batch,
        Tp_batch,
        Cnm,
        float(wdm.dF),
        assembly_vmap=assembly_vmap,
        return_device=return_device,
    )


def _assemble_shift_fixed_dispatch(wdm, w_xi, delta, ell_all, offset, Tl_vec, Tp_vec, Cnm, use_jax, assembly_vmap=False):
    """Dispatch fixed-delay assembly to the JAX backend."""
    _ = wdm, use_jax
    return assemble_shift_fixed_jax(
        w_xi,
        delta,
        ell_all,
        offset,
        Tl_vec,
        Tp_vec,
        Cnm,
        float(wdm.dF),
        assembly_vmap=assembly_vmap,
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
    use_jax,
    delta_mode,
    assembly_vmap=False,
    **_,
):
    """Dispatch non-target variable-delay assembly to the JAX backend."""
    _ = wdm, use_jax
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
        assembly_vmap=assembly_vmap,
    )