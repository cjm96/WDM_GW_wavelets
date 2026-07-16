"""Experimental analytic-parity WDM variable-shift kernel.

This module is intentionally separate from the production dispatcher.

It removes the deterministic checkerboard ``Cnm`` array and the associated
row/lag/frequency gather from the lag-blocked target-mode kernel.  The exact
products of ``Cnm`` and the existing parity/sideband phase factors are replaced
by analytic signs depending only on:

- target-row parity,
- frequency-bin parity,
- lag parity and lag modulo four.

The numerical operator is unchanged apart from ordinary floating-point
reassociation.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


def _normalize_precision(precision: str | None) -> str:
    if precision is None:
        return "complex128"

    key = str(precision).lower()

    if key in ("complex128", "float64", "c128", "64"):
        return "complex128"

    if key in ("complex64", "float32", "c64", "32"):
        return "complex64"

    raise ValueError(
        "precision must be complex128/float64 or complex64/float32."
    )


def _pm_one_from_integer(values, real_dtype):
    """Return ``(-1)**values`` as a real JAX array."""

    values = jnp.asarray(values, dtype=jnp.int64)

    return jnp.where(
        jnp.mod(values, 2) == 0,
        jnp.asarray(1.0, dtype=real_dtype),
        jnp.asarray(-1.0, dtype=real_dtype),
    )


def _make_shift_target_chunked_lagblock_analytic_parity_impl(
    *,
    complex_dtype,
    real_dtype,
):
    """Create one analytic-parity row-chunked, lag-blocked kernel."""

    def impl(
        w_xi,
        t_shift,
        ell_all,
        offset,
        Tl_all,
        Tp_all,
        dF,
        row_chunk_size,
        lag_block_size,
    ):
        w_xi = jnp.asarray(w_xi, dtype=complex_dtype)
        t_shift = jnp.asarray(t_shift, dtype=real_dtype)
        ell_all = jnp.asarray(ell_all, dtype=jnp.int64)
        Tl_all = jnp.asarray(Tl_all, dtype=complex_dtype)
        Tp_all = jnp.asarray(Tp_all, dtype=complex_dtype)
        dF = jnp.asarray(dF, dtype=real_dtype)

        Nt, Nm = w_xi.shape
        n_lag = ell_all.shape[0]

        m_full = jnp.arange(Nm, dtype=real_dtype)
        one_j = jnp.asarray(1j, dtype=complex_dtype)
        two_pi = jnp.asarray(2.0 * np.pi, dtype=real_dtype)
        pi = jnp.asarray(np.pi, dtype=real_dtype)

        ph_m_all = jnp.exp(
            one_j
            * two_pi
            * (m_full[None, :] * dF)
            * t_shift[:, None]
        )

        half_bin_phase = jnp.exp(
            one_j
            * pi
            * dF
            * t_shift
        )

        ph_mid_all = (
            ph_m_all[:, :-1]
            * half_bin_phase[:, None]
        )

        n_chunks = (
            Nt + row_chunk_size - 1
        ) // row_chunk_size

        Nt_pad = n_chunks * row_chunk_size

        out0 = jnp.zeros(
            (Nt_pad, Nm),
            dtype=complex_dtype,
        )

        zero_col = jnp.zeros(
            (row_chunk_size, 1),
            dtype=complex_dtype,
        )

        row_offsets = jnp.arange(
            row_chunk_size,
            dtype=jnp.int64,
        )

        lag_offsets = jnp.arange(
            lag_block_size,
            dtype=jnp.int64,
        )

        # Sideband frequency parity uses m=0,...,Nm-2.
        sideband_frequency_sign = _pm_one_from_integer(
            jnp.arange(max(Nm - 1, 0), dtype=jnp.int64),
            real_dtype,
        )

        n_lag_blocks = (
            n_lag + lag_block_size - 1
        ) // lag_block_size

        def chunk_body(chunk_id, out_acc):
            start = chunk_id * row_chunk_size
            rows = start + row_offsets

            valid_row = rows < Nt
            rows_safe = jnp.clip(
                rows,
                0,
                Nt - 1,
            )

            row_sign = _pm_one_from_integer(
                rows_safe,
                real_dtype,
            )

            ph_m = ph_m_all[
                rows_safe,
                :,
            ]

            ph_mid = ph_mid_all[
                rows_safe,
                :,
            ]

            out_chunk0 = jnp.zeros(
                (row_chunk_size, Nm),
                dtype=complex_dtype,
            )

            def lag_block_body(
                lag_block_id,
                chunk_out,
            ):
                lag_start = (
                    lag_block_id
                    * lag_block_size
                )

                lag_indices = (
                    lag_start
                    + lag_offsets
                )

                valid_lag = (
                    lag_indices < n_lag
                )

                lag_indices_safe = jnp.clip(
                    lag_indices,
                    0,
                    n_lag - 1,
                )

                ell_block = ell_all[
                    lag_indices_safe
                ]

                lag_even = (
                    jnp.mod(ell_block, 2)
                    == 0
                )

                # For even ell, b_even = (-1)^(ell/2).
                even_half_sign = _pm_one_from_integer(
                    jnp.floor_divide(
                        ell_block,
                        2,
                    ),
                    real_dtype,
                )

                # For odd ell, b_odd = (-1)^((ell-1)/2).
                odd_half_sign = _pm_one_from_integer(
                    jnp.floor_divide(
                        ell_block - 1,
                        2,
                    ),
                    real_dtype,
                )

                j_neg_block = (
                    -ell_block
                    + offset
                )

                nprime = (
                    rows[:, None]
                    + ell_block[None, :]
                )

                valid_n = (
                    valid_row[:, None]
                    & valid_lag[None, :]
                    & (nprime >= 0)
                    & (nprime < Nt)
                )

                nprime_safe = jnp.clip(
                    nprime,
                    0,
                    Nt - 1,
                )

                Tl_row_block = (
                    Tl_all[rows_safe, :]
                    [:, j_neg_block]
                )

                Tp_row_block = (
                    Tp_all[rows_safe, :]
                    [:, j_neg_block]
                )

                w_n_block = w_xi[
                    nprime_safe,
                    :,
                ]

                # ---------------------------------------------------------
                # Carrier
                #
                # Existing factor:
                #   parity(ell,m) * conj(C[p,m]) * C[p+ell,m]
                #
                # Exact simplification:
                #   1                      for even ell
                #   i * (-1)^p            for odd ell
                # ---------------------------------------------------------

                carrier_base = (
                    ph_m[:, None, :]
                    * Tl_row_block[:, :, None]
                )

                carrier_real_even = jnp.real(
                    carrier_base
                )

                carrier_real_odd = (
                    -row_sign[:, None, None]
                    * jnp.imag(carrier_base)
                )

                carrier_coefficient = jnp.where(
                    lag_even[None, :, None],
                    carrier_real_even,
                    carrier_real_odd,
                )

                main = (
                    carrier_coefficient
                    * w_n_block
                )

                main = jnp.where(
                    valid_n[:, :, None],
                    main,
                    jnp.zeros_like(main),
                )

                main_sum = jnp.sum(
                    main,
                    axis=1,
                )

                def add_sidebands(main_acc):
                    # -----------------------------------------------------
                    # Sidebands
                    #
                    # All exact checkerboard/phase products are ±i.
                    # For z complex:
                    #   Re(i*s*z) = -s*Im(z), s in {+1,-1}.
                    #
                    # Even ell:
                    #   low: -i*(-1)^(p+m+ell/2)
                    #   up:  +i*(-1)^(p+m+ell/2)
                    #
                    # Odd ell:
                    #   low = up = i*(-1)^(m+(ell-1)/2)
                    # -----------------------------------------------------

                    sideband_base = (
                        ph_mid[:, None, :]
                        * Tp_row_block[:, :, None]
                    )

                    sideband_imag = jnp.imag(
                        sideband_base
                    )

                    even_common_sign = (
                        row_sign[:, None, None]
                        * even_half_sign[None, :, None]
                        * sideband_frequency_sign[
                            None,
                            None,
                            :,
                        ]
                    )

                    odd_common_sign = (
                        odd_half_sign[None, :, None]
                        * sideband_frequency_sign[
                            None,
                            None,
                            :,
                        ]
                    )

                    low_i_sign = jnp.where(
                        lag_even[None, :, None],
                        -even_common_sign,
                        odd_common_sign,
                    )

                    up_i_sign = jnp.where(
                        lag_even[None, :, None],
                        even_common_sign,
                        odd_common_sign,
                    )

                    low_coefficient = (
                        -low_i_sign
                        * sideband_imag
                    )

                    up_coefficient = (
                        -up_i_sign
                        * sideband_imag
                    )

                    low = (
                        low_coefficient
                        * w_n_block[:, :, :-1]
                    )

                    up = (
                        up_coefficient
                        * w_n_block[:, :, 1:]
                    )

                    low = jnp.where(
                        valid_n[:, :, None],
                        low,
                        jnp.zeros_like(low),
                    )

                    up = jnp.where(
                        valid_n[:, :, None],
                        up,
                        jnp.zeros_like(up),
                    )

                    low_sum = jnp.sum(
                        low,
                        axis=1,
                    )

                    up_sum = jnp.sum(
                        up,
                        axis=1,
                    )

                    low_pad = jnp.concatenate(
                        (
                            zero_col,
                            low_sum,
                        ),
                        axis=1,
                    )

                    up_pad = jnp.concatenate(
                        (
                            up_sum,
                            zero_col,
                        ),
                        axis=1,
                    )

                    return (
                        main_acc
                        + low_pad
                        + up_pad
                    )

                chunk_next = jax.lax.cond(
                    Nm > 1,
                    add_sidebands,
                    lambda value: value,
                    main_sum,
                )

                return (
                    chunk_out
                    + chunk_next
                )

            out_chunk = jax.lax.fori_loop(
                0,
                n_lag_blocks,
                lag_block_body,
                out_chunk0,
            )

            return jax.lax.dynamic_update_slice(
                out_acc,
                out_chunk,
                (start, 0),
            )

        out_padded = jax.lax.fori_loop(
            0,
            n_chunks,
            chunk_body,
            out0,
        )

        return out_padded[:Nt, :]

    return impl


_analytic_parity_impl_c128 = (
    _make_shift_target_chunked_lagblock_analytic_parity_impl(
        complex_dtype=jnp.complex128,
        real_dtype=jnp.float64,
    )
)

_analytic_parity_impl_c64 = (
    _make_shift_target_chunked_lagblock_analytic_parity_impl(
        complex_dtype=jnp.complex64,
        real_dtype=jnp.float32,
    )
)


def _make_batch_impl(
    *,
    single_job_impl,
    complex_dtype,
):
    """Create the batched analytic-parity diagnostic kernel."""

    def impl(
        w_xi_batch,
        t_shift_batch,
        ell_all,
        offset,
        Tl_batch,
        Tp_batch,
        dF,
        row_chunk_size,
        lag_block_size,
    ):
        B, Nt, Nm = w_xi_batch.shape

        out0 = jnp.zeros(
            (B, Nt, Nm),
            dtype=complex_dtype,
        )

        def body(job_index, out):
            shifted = single_job_impl(
                w_xi_batch[job_index],
                t_shift_batch[job_index],
                ell_all,
                offset,
                Tl_batch[job_index],
                Tp_batch[job_index],
                dF,
                row_chunk_size,
                lag_block_size,
            )

            return out.at[
                job_index,
                :,
                :,
            ].set(shifted)

        return jax.lax.fori_loop(
            0,
            B,
            body,
            out0,
        )

    return impl


_batch_analytic_parity_impl_c128 = _make_batch_impl(
    single_job_impl=_analytic_parity_impl_c128,
    complex_dtype=jnp.complex128,
)

_batch_analytic_parity_impl_c64 = _make_batch_impl(
    single_job_impl=_analytic_parity_impl_c64,
    complex_dtype=jnp.complex64,
)


_batch_analytic_parity_core_c128 = jax.jit(
    _batch_analytic_parity_impl_c128,
    static_argnums=(3, 7, 8),
)

_batch_analytic_parity_core_c64 = jax.jit(
    _batch_analytic_parity_impl_c64,
    static_argnums=(3, 7, 8),
)


def assemble_shift_target_batch_chunked_lagblock_analytic_parity_jax(
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    dF,
    row_chunk_size=128,
    lag_block_size=1,
    precision="complex128",
    return_device=False,
):
    """Run the experimental batched analytic-parity shift."""

    precision = _normalize_precision(
        precision
    )

    row_chunk_size = int(
        row_chunk_size
    )

    lag_block_size = int(
        lag_block_size
    )

    if row_chunk_size < 1:
        raise ValueError(
            "row_chunk_size must be >= 1."
        )

    if lag_block_size < 1:
        raise ValueError(
            "lag_block_size must be >= 1."
        )

    input_shape = tuple(
        np.shape(w_xi_batch)
    )

    if len(input_shape) != 3:
        raise ValueError(
            "w_xi_batch must have shape "
            "(num_jobs, Nt, Nm)."
        )

    if precision == "complex64":
        out = _batch_analytic_parity_core_c64(
            jnp.asarray(
                w_xi_batch,
                dtype=jnp.complex64,
            ),
            jnp.asarray(
                t_shift_batch,
                dtype=jnp.float32,
            ),
            jnp.asarray(
                ell_all,
                dtype=jnp.int64,
            ),
            int(offset),
            jnp.asarray(
                Tl_batch,
                dtype=jnp.complex64,
            ),
            jnp.asarray(
                Tp_batch,
                dtype=jnp.complex64,
            ),
            jnp.asarray(
                dF,
                dtype=jnp.float32,
            ),
            row_chunk_size,
            lag_block_size,
        )

        return (
            out
            if return_device
            else np.asarray(out)
        )

    out = _batch_analytic_parity_core_c128(
        jnp.asarray(
            w_xi_batch,
            dtype=jnp.complex128,
        ),
        jnp.asarray(
            t_shift_batch,
            dtype=jnp.float64,
        ),
        jnp.asarray(
            ell_all,
            dtype=jnp.int64,
        ),
        int(offset),
        jnp.asarray(
            Tl_batch,
            dtype=jnp.complex128,
        ),
        jnp.asarray(
            Tp_batch,
            dtype=jnp.complex128,
        ),
        jnp.asarray(
            dF,
            dtype=jnp.float64,
        ),
        row_chunk_size,
        lag_block_size,
    )

    return (
        out
        if return_device
        else np.asarray(out)
    )
