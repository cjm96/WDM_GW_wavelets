"""Experimental mask-reuse kernel for WDM variable time shifts.

This module is intentionally separate from the production dispatcher.  It
tests one narrow hypothesis in the current row-chunked, lag-blocked kernel:

    mask the gathered waveform block once and reuse it for the carrier,
    lower-sideband, and upper-sideband contributions.

It also explicitly reuses the conjugated target-row Cnm array.  The production
implementation remains unchanged.
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


def _make_shift_target_chunked_lagblock_maskreuse_impl(
    *,
    complex_dtype,
    real_dtype,
):
    """Create one row-chunked, lag-blocked mask-reuse kernel."""

    def impl(
        w_xi,
        t_shift,
        ell_all,
        offset,
        Tl_all,
        Tp_all,
        Cnm,
        dF,
        row_chunk_size,
        lag_block_size,
    ):
        w_xi = jnp.asarray(w_xi, dtype=complex_dtype)
        t_shift = jnp.asarray(t_shift, dtype=real_dtype)
        ell_all = jnp.asarray(ell_all, dtype=jnp.int64)
        Tl_all = jnp.asarray(Tl_all, dtype=complex_dtype)
        Tp_all = jnp.asarray(Tp_all, dtype=complex_dtype)
        Cnm = jnp.asarray(Cnm, dtype=complex_dtype)
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

        minus1_to_m = jnp.where(
            (jnp.arange(Nm) % 2) == 0,
            jnp.asarray(1.0, dtype=real_dtype),
            jnp.asarray(-1.0, dtype=real_dtype),
        )

        ell_even = (ell_all % 2) == 0

        parity_all = jnp.where(
            ell_even[:, None],
            jnp.ones((n_lag, Nm), dtype=real_dtype),
            minus1_to_m[None, :],
        )

        neg_i = jnp.asarray(-1j, dtype=complex_dtype)
        pos_i = jnp.asarray(1j, dtype=complex_dtype)

        low_phase = jnp.power(neg_i, -ell_all)
        up_phase = jnp.power(pos_i, -ell_all)

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

            # Explicitly compute the conjugated target rows once and reuse
            # them for all three contributions.
            cp_conj_rows = jnp.conj(
                Cnm[rows_safe, :]
            )

            carrier_prefac = (
                cp_conj_rows
                * ph_m_all[rows_safe, :]
            )

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

                parity_block = parity_all[
                    lag_indices_safe
                ]

                j_neg_block = (
                    -ell_block
                    + offset
                )

                low_phase_block = low_phase[
                    lag_indices_safe
                ]

                up_phase_block = up_phase[
                    lag_indices_safe
                ]

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

                Cn_block = Cnm[
                    nprime_safe,
                    :,
                ]

                w_n_block = w_xi[
                    nprime_safe,
                    :,
                ]

                # Candidate change:
                # apply the row/lag validity mask once to the shared
                # waveform block, then reuse the masked values in the
                # carrier and both sideband expressions.
                w_n_block = jnp.where(
                    valid_n[:, :, None],
                    w_n_block,
                    jnp.zeros_like(w_n_block),
                )

                main = (
                    parity_block[None, :, :]
                    * carrier_prefac[:, None, :]
                    * Cn_block
                    * Tl_row_block[:, :, None]
                ).real * w_n_block

                main_sum = jnp.sum(
                    main,
                    axis=1,
                )

                def add_sidebands(main_acc):
                    low = (
                        parity_block[:, :-1]
                        [None, :, :]
                        * low_phase_block[
                            None,
                            :,
                            None,
                        ]
                        * cp_conj_rows[
                            :,
                            1:,
                        ][:, None, :]
                        * Cn_block[
                            :,
                            :,
                            :-1,
                        ]
                        * Tp_row_block[
                            :,
                            :,
                            None,
                        ]
                        * ph_mid[
                            :,
                            None,
                            :,
                        ]
                    ).real * w_n_block[
                        :,
                        :,
                        :-1,
                    ]

                    up = (
                        parity_block[:, 1:]
                        [None, :, :]
                        * up_phase_block[
                            None,
                            :,
                            None,
                        ]
                        * cp_conj_rows[
                            :,
                            :-1,
                        ][:, None, :]
                        * Cn_block[
                            :,
                            :,
                            1:,
                        ]
                        * Tp_row_block[
                            :,
                            :,
                            None,
                        ]
                        * ph_mid[
                            :,
                            None,
                            :,
                        ]
                    ).real * w_n_block[
                        :,
                        :,
                        1:,
                    ]

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


_maskreuse_impl_c128 = (
    _make_shift_target_chunked_lagblock_maskreuse_impl(
        complex_dtype=jnp.complex128,
        real_dtype=jnp.float64,
    )
)

_maskreuse_impl_c64 = (
    _make_shift_target_chunked_lagblock_maskreuse_impl(
        complex_dtype=jnp.complex64,
        real_dtype=jnp.float32,
    )
)


def _make_batch_impl(
    *,
    single_job_impl,
    complex_dtype,
):
    """Create the batched diagnostic kernel."""

    def impl(
        w_xi_batch,
        t_shift_batch,
        ell_all,
        offset,
        Tl_batch,
        Tp_batch,
        Cnm,
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
                Cnm,
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


_batch_maskreuse_impl_c128 = _make_batch_impl(
    single_job_impl=_maskreuse_impl_c128,
    complex_dtype=jnp.complex128,
)

_batch_maskreuse_impl_c64 = _make_batch_impl(
    single_job_impl=_maskreuse_impl_c64,
    complex_dtype=jnp.complex64,
)


_batch_maskreuse_core_c128 = jax.jit(
    _batch_maskreuse_impl_c128,
    static_argnums=(3, 8, 9),
)

_batch_maskreuse_core_c64 = jax.jit(
    _batch_maskreuse_impl_c64,
    static_argnums=(3, 8, 9),
)


def assemble_shift_target_batch_chunked_lagblock_maskreuse_jax(
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    dF,
    row_chunk_size=128,
    lag_block_size=1,
    precision="complex128",
    return_device=False,
):
    """Run the experimental batched mask-reuse shift.

    This wrapper mirrors the production
    ``assemble_shift_target_batch_chunked_lagblock_jax`` API so the two
    kernels can be benchmarked using identical prepared inputs.
    """

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
        out = _batch_maskreuse_core_c64(
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
                Cnm,
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

    out = _batch_maskreuse_core_c128(
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
            Cnm,
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
