"""Lean experimental analytic-parity WDM shift kernel.

Version 2 keeps the compact complex arithmetic structure of the production
row-chunked, lag-blocked kernel while eliminating the deterministic ``Cnm``
checkerboard array.

The exact products involving ``Cnm`` and the existing parity/sideband phase
factors reduce to compact row-, lag-, and frequency-parity factors.  Unlike
the first experiment, this implementation does not create shared full-size
real/imaginary coefficient tensors.  It applies the compact analytic factors
directly inside the carrier, lower-sideband, and upper-sideband expressions.

The production dispatcher and plan are intentionally unchanged.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


ANALYTIC_PARITY_EXPERIMENT_VERSION = 2


def _normalize_precision(precision: str | None) -> str:
    """Normalize precision labels to the two supported complex dtypes."""

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
    """Create one lean analytic-parity lag-blocked shift kernel."""

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

        one_j = jnp.asarray(1j, dtype=complex_dtype)
        one_complex = jnp.asarray(1.0 + 0.0j, dtype=complex_dtype)
        two_pi = jnp.asarray(2.0 * np.pi, dtype=real_dtype)
        pi = jnp.asarray(np.pi, dtype=real_dtype)

        m_full = jnp.arange(Nm, dtype=real_dtype)

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

        # All compact parity data are built once outside the row/lag loops.
        row_sign_all = _pm_one_from_integer(
            jnp.arange(Nt, dtype=jnp.int64),
            real_dtype,
        )

        lag_even_all = (
            jnp.mod(ell_all, 2) == 0
        )

        even_half_sign_all = _pm_one_from_integer(
            jnp.floor_divide(ell_all, 2),
            real_dtype,
        )

        odd_half_sign_all = _pm_one_from_integer(
            jnp.floor_divide(ell_all - 1, 2),
            real_dtype,
        )

        sideband_frequency_sign = _pm_one_from_integer(
            jnp.arange(max(Nm - 1, 0), dtype=jnp.int64),
            real_dtype,
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

            row_sign = row_sign_all[
                rows_safe
            ]

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

                lag_even = lag_even_all[
                    lag_indices_safe
                ]

                even_half_sign = even_half_sign_all[
                    lag_indices_safe
                ]

                odd_half_sign = odd_half_sign_all[
                    lag_indices_safe
                ]

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
                # parity(ell,m) * conj(C[p,m]) * C[p+ell,m]
                #
                #   = 1                    for even ell
                #   = i * (-1)^p          for odd ell
                #
                # The select is only (row, lag, 1), not a full
                # (row, lag, frequency) real/imaginary selection.
                # ---------------------------------------------------------

                carrier_factor = jnp.where(
                    lag_even[None, :, None],
                    one_complex,
                    one_j
                    * row_sign[:, None, None],
                )

                main = (
                    carrier_factor
                    * ph_m[:, None, :]
                    * Tl_row_block[:, :, None]
                ).real * w_n_block

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
                    # The exact eliminated checkerboard factors are:
                    #
                    # even ell:
                    #   low = -i*(-1)^(p+m+ell/2)
                    #   up  = +i*(-1)^(p+m+ell/2)
                    #
                    # odd ell:
                    #   low = up = i*(-1)^(m+(ell-1)/2)
                    #
                    # Keep row/lag signs compact and multiply them directly
                    # into each production-like complex expression.  Do not
                    # share a full-size sideband_base/imaginary tensor.
                    # -----------------------------------------------------

                    even_row_lag_sign = (
                        row_sign[:, None]
                        * even_half_sign[None, :]
                    )

                    low_row_lag_sign = jnp.where(
                        lag_even[None, :],
                        -even_row_lag_sign,
                        odd_half_sign[None, :],
                    )

                    up_row_lag_sign = jnp.where(
                        lag_even[None, :],
                        even_row_lag_sign,
                        odd_half_sign[None, :],
                    )

                    low = (
                        one_j
                        * low_row_lag_sign[:, :, None]
                        * sideband_frequency_sign[
                            None,
                            None,
                            :,
                        ]
                        * ph_mid[:, None, :]
                        * Tp_row_block[:, :, None]
                    ).real * w_n_block[:, :, :-1]

                    up = (
                        one_j
                        * up_row_lag_sign[:, :, None]
                        * sideband_frequency_sign[
                            None,
                            None,
                            :,
                        ]
                        * ph_mid[:, None, :]
                        * Tp_row_block[:, :, None]
                    ).real * w_n_block[:, :, 1:]

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
    """Create the batched analytic-parity kernel."""

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
    """Run the lean experimental batched analytic-parity shift."""

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