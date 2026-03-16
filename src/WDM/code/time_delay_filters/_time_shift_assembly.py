"""Assembly dispatch helpers for WDM time-shift operators.

This module keeps the high-level assembly entrypoints separate from
``time_shift_fast.py`` so the main shift code only handles preprocessing and
dispatch.
"""

from ._time_shift_jax import (
    assemble_shift_fixed_jax,
    assemble_shift_target_jax,
    assemble_shift_variable_mode_jax,
)


def _assemble_shift_target_dispatch(wdm, w_xi, t_shift, ell_all, offset, Tl_all, Tp_all, Cnm, use_jax):
    """Dispatch target-mode assembly to the JAX backend.

    The ``use_jax`` flag is retained only for API compatibility with older
    call sites. The active implementation is always the JAX kernel.
    """
    _ = wdm, use_jax
    return assemble_shift_target_jax(
        w_xi,
        t_shift,
        ell_all,
        offset,
        Tl_all,
        Tp_all,
        Cnm,
        float(wdm.dF),
    )


def _assemble_shift_fixed_dispatch(wdm, w_xi, delta, ell_all, offset, Tl_vec, Tp_vec, Cnm, use_jax):
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
    )