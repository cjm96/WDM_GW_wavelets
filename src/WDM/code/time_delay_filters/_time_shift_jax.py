"""JAX kernels for WDM time-delay operators.

The module deliberately exposes only two target-mode implementations:

``reference``
    A readable checkerboard implementation used for regression tests and
    high-precision validation.

``production``
    The row/lag-blocked analytic-parity implementation used in repeated
    response evaluations.  It avoids the full ``C_nm`` checkerboard and keeps
    real coefficient arrays real throughout the large gather/reduction path.

Fixed-delay and non-target variable-delay modes remain reference-only because
those paths are not used by the production LISA response.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Reference checkerboard implementation
# ---------------------------------------------------------------------------


def _map_rows_range(row_fn, start, stop, Nm, use_vmap):
    """Evaluate ``row_fn`` for rows in ``[start, stop)``."""

    n_rows = max(0, int(stop) - int(start))
    if n_rows == 0:
        return jnp.zeros((0, Nm), dtype=jnp.complex128)
    if use_vmap:
        return jax.vmap(row_fn)(jnp.arange(start, stop, dtype=jnp.int32))

    def body(i, out):
        return out.at[i, :].set(row_fn(start + i))

    out0 = jnp.zeros((n_rows, Nm), dtype=jnp.complex128)
    return jax.lax.fori_loop(0, n_rows, body, out0)


def _phase_m(delta_vec, Nm, dF):
    """Return ``exp(i 2 pi m dF delta)`` for all requested delays."""

    m = jnp.arange(Nm, dtype=jnp.float64)
    return jnp.exp(2j * jnp.pi * (m[None, :] * dF) * delta_vec[:, None])


def _phase_mid(delta_vec, Nm, dF):
    """Return half-bin phase factors for adjacent-frequency couplings."""

    m = jnp.arange(max(Nm - 1, 0), dtype=jnp.float64)
    return jnp.exp(
        2j * jnp.pi * ((m[None, :] + 0.5) * dF) * delta_vec[:, None]
    )


def _accumulate_target_row(
    p,
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    dF,
    parity,
    low_phase,
    up_phase,
    interior,
):
    """Reference accumulation for one target row."""

    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]
    j_neg = -ell_all + offset

    Cp_conj = jnp.conj(Cnm[p, :])
    delta_p = t_shift[p]
    Tl_j = Tl_all[p, j_neg]
    Tp_j = Tp_all[p, j_neg]

    m_full = jnp.arange(Nm, dtype=jnp.float64)
    ph_m_row = jnp.exp(2j * jnp.pi * (m_full * dF) * delta_p)
    carrier_prefactor = Cp_conj * ph_m_row

    if Nm > 1:
        ph_mid_row = ph_m_row[:-1] * jnp.exp(1j * jnp.pi * dF * delta_p)
        cp_conj_hi = Cp_conj[1:]
        cp_conj_lo = Cp_conj[:-1]
    else:
        ph_mid_row = jnp.zeros((0,), dtype=jnp.complex128)
        cp_conj_hi = jnp.zeros((0,), dtype=jnp.complex128)
        cp_conj_lo = jnp.zeros((0,), dtype=jnp.complex128)

    zero_head = jnp.zeros((1,), dtype=jnp.complex128)
    zero_tail = jnp.zeros((1,), dtype=jnp.complex128)

    def lag_body(i, row_state):
        ell = ell_all[i]
        n = p + ell

        def add_row(state):
            Cn = Cnm[n, :]
            w_n = w_xi[n, :]

            carrier = (
                parity[i, :] * carrier_prefactor * Cn * Tl_j[i]
            ).real * w_n

            def add_sidebands(_):
                common = Tp_j[i] * ph_mid_row
                low = (
                    parity[i, :-1]
                    * low_phase[i]
                    * cp_conj_hi
                    * Cn[:-1]
                    * common
                ).real * w_n[:-1]
                up = (
                    parity[i, 1:]
                    * up_phase[i]
                    * cp_conj_lo
                    * Cn[1:]
                    * common
                ).real * w_n[1:]
                return jnp.concatenate((zero_head, low)) + jnp.concatenate(
                    (up, zero_tail)
                )

            sidebands = jax.lax.cond(
                Nm > 1,
                add_sidebands,
                lambda _: jnp.zeros((Nm,), dtype=jnp.complex128),
                operand=None,
            )
            return state + carrier + sidebands

        if interior:
            return add_row(row_state)
        return jax.lax.cond(
            (n >= 0) & (n < Nt),
            add_row,
            lambda state: state,
            row_state,
        )

    row0 = jnp.zeros((Nm,), dtype=jnp.complex128)
    return jax.lax.fori_loop(0, n_lag, lag_body, row0)


def _assemble_shift_target_reference_impl(
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    dF,
    use_vmap,
):
    """Reference target-mode variable-delay assembly."""

    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]

    minus1_to_m = jnp.where(
        (jnp.arange(Nm) % 2) == 0,
        1.0,
        -1.0,
    )
    ell_even = (ell_all % 2) == 0
    parity = jnp.where(
        ell_even[:, None],
        jnp.ones((n_lag, Nm), dtype=jnp.float64),
        minus1_to_m[None, :],
    )
    low_phase = ((-1j) ** (-ell_all)).astype(jnp.complex128)
    up_phase = ((+1j) ** (-ell_all)).astype(jnp.complex128)

    left = min(offset, Nt)
    right = max(left, Nt - offset)

    def row_fn(p):
        return _accumulate_target_row(
            p,
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            Cnm,
            dF,
            parity,
            low_phase,
            up_phase,
            False,
        )

    def row_fn_interior(p):
        return _accumulate_target_row(
            p,
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            Cnm,
            dF,
            parity,
            low_phase,
            up_phase,
            True,
        )

    if left == Nt:
        return _map_rows_range(row_fn, 0, Nt, Nm, use_vmap)

    return jnp.concatenate(
        (
            _map_rows_range(row_fn, 0, left, Nm, use_vmap),
            _map_rows_range(row_fn_interior, left, right, Nm, use_vmap),
            _map_rows_range(row_fn, right, Nt, Nm, use_vmap),
        ),
        axis=0,
    )


_assemble_shift_target_reference_core = jax.jit(
    _assemble_shift_target_reference_impl,
    static_argnums=(3, 8),
)


def _assemble_shift_target_batch_reference_impl(
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    dF,
):
    """Reference target-mode assembly for a small validation batch."""

    B, Nt, Nm = w_xi_batch.shape
    out0 = jnp.zeros((B, Nt, Nm), dtype=jnp.complex128)

    def body(job_index, out):
        shifted = _assemble_shift_target_reference_impl(
            w_xi_batch[job_index],
            t_shift_batch[job_index],
            ell_all,
            offset,
            Tl_batch[job_index],
            Tp_batch[job_index],
            Cnm,
            dF,
            False,
        )
        return out.at[job_index, :, :].set(shifted)

    return jax.lax.fori_loop(0, B, body, out0)


_assemble_shift_target_batch_reference_core = jax.jit(
    _assemble_shift_target_batch_reference_impl,
    static_argnums=(3,),
)


# ---------------------------------------------------------------------------
# Fixed and non-target reference paths
# ---------------------------------------------------------------------------


def _assemble_shift_fixed_impl(
    w_xi,
    delta,
    ell_all,
    offset,
    Tl_vec,
    Tp_vec,
    Cnm,
    dF,
    use_vmap,
):
    """Reference fixed-delay assembly."""

    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]
    j_neg = -ell_all + offset

    minus1_to_m = jnp.where(
        (jnp.arange(Nm) % 2) == 0,
        1.0,
        -1.0,
    )
    ell_even = (ell_all % 2) == 0
    parity = jnp.where(
        ell_even[:, None],
        jnp.ones((n_lag, Nm), dtype=jnp.float64),
        minus1_to_m[None, :],
    )
    low_phase = ((-1j) ** (-ell_all)).astype(jnp.complex128)
    up_phase = ((+1j) ** (-ell_all)).astype(jnp.complex128)

    Tl_j = Tl_vec[j_neg]
    Tp_j = Tp_vec[j_neg]
    m_full = jnp.arange(Nm, dtype=jnp.float64)
    ph_m = jnp.exp(2j * jnp.pi * (m_full * dF) * delta)
    ph_mid = (
        ph_m[:-1] * jnp.exp(1j * jnp.pi * dF * delta)
        if Nm > 1
        else jnp.zeros((0,), dtype=jnp.complex128)
    )

    left = min(offset, Nt)
    right = max(left, Nt - offset)
    zero_head = jnp.zeros((1,), dtype=jnp.complex128)
    zero_tail = jnp.zeros((1,), dtype=jnp.complex128)

    def row_accum(p, interior):
        Cp_conj = jnp.conj(Cnm[p, :])
        carrier_prefactor = Cp_conj * ph_m

        def lag_body(i, row_state):
            ell = ell_all[i]
            n = p + ell

            def add_row(state):
                Cn = Cnm[n, :]
                w_n = w_xi[n, :]
                carrier = (
                    parity[i, :] * carrier_prefactor * Cn * Tl_j[i]
                ).real * w_n

                def add_sidebands(_):
                    common = Tp_j[i] * ph_mid
                    low = (
                        parity[i, :-1]
                        * low_phase[i]
                        * Cp_conj[1:]
                        * Cn[:-1]
                        * common
                    ).real * w_n[:-1]
                    up = (
                        parity[i, 1:]
                        * up_phase[i]
                        * Cp_conj[:-1]
                        * Cn[1:]
                        * common
                    ).real * w_n[1:]
                    return jnp.concatenate((zero_head, low)) + jnp.concatenate(
                        (up, zero_tail)
                    )

                return state + carrier + jax.lax.cond(
                    Nm > 1,
                    add_sidebands,
                    lambda _: jnp.zeros((Nm,), dtype=jnp.complex128),
                    operand=None,
                )

            if interior:
                return add_row(row_state)
            return jax.lax.cond(
                (n >= 0) & (n < Nt),
                add_row,
                lambda state: state,
                row_state,
            )

        return jax.lax.fori_loop(
            0,
            n_lag,
            lag_body,
            jnp.zeros((Nm,), dtype=jnp.complex128),
        )

    def row_fn(p):
        return row_accum(p, False)

    def row_fn_interior(p):
        return row_accum(p, True)

    if left == Nt:
        return _map_rows_range(row_fn, 0, Nt, Nm, use_vmap)

    return jnp.concatenate(
        (
            _map_rows_range(row_fn, 0, left, Nm, use_vmap),
            _map_rows_range(row_fn_interior, left, right, Nm, use_vmap),
            _map_rows_range(row_fn, right, Nt, Nm, use_vmap),
        ),
        axis=0,
    )


_assemble_shift_fixed_core = jax.jit(
    _assemble_shift_fixed_impl,
    static_argnums=(3, 8),
)


def _assemble_shift_variable_mode_impl(
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    dF,
    mode,
):
    """Reference source/midpoint variable-delay assembly."""

    Nt, Nm = w_xi.shape
    j_neg = -ell_all + offset
    minus1_to_m = jnp.where(
        (jnp.arange(Nm) % 2) == 0,
        1.0,
        -1.0,
    )

    def row_fn(p):
        n_all = p + ell_all
        valid = (n_all >= 0) & (n_all < Nt)
        n_safe = jnp.clip(n_all, 0, Nt - 1)
        valid_f = valid.astype(jnp.float64)[:, None]

        Cp = Cnm[p, :]
        Cn = Cnm[n_safe, :]
        w_n = w_xi[n_safe, :]

        def source_values(_):
            return (
                t_shift[n_safe],
                Tl_all[n_safe, j_neg],
                Tp_all[n_safe, j_neg],
            )

        def midpoint_values(_):
            kf = 0.5 * (p + n_all)
            k0 = jnp.floor(kf).astype(jnp.int32)
            k1 = jnp.clip(k0 + 1, 0, Nt - 1)
            frac = kf - k0.astype(jnp.float64)
            return (
                (1.0 - frac) * t_shift[k0] + frac * t_shift[k1],
                (1.0 - frac) * Tl_all[k0, j_neg]
                + frac * Tl_all[k1, j_neg],
                (1.0 - frac) * Tp_all[k0, j_neg]
                + frac * Tp_all[k1, j_neg],
            )

        delta_vec, Tl_j, Tp_j = jax.lax.switch(
            mode,
            (source_values, midpoint_values),
            operand=None,
        )

        parity = jnp.where(
            ((ell_all % 2) == 0)[:, None],
            jnp.ones((1, Nm), dtype=jnp.float64),
            minus1_to_m[None, :],
        )

        ph_m = _phase_m(delta_vec, Nm, dF)
        carrier_kernel = (
            parity
            * jnp.conj(Cp)[None, :]
            * Cn
            * Tl_j[:, None]
            * ph_m
        )
        row = jnp.sum(valid_f * carrier_kernel.real * w_n, axis=0)

        def add_sidebands(row_in):
            ph_mid = _phase_mid(delta_vec, Nm, dF)
            low_factor = ((-1j) ** (-ell_all)) * Tp_j
            up_factor = ((+1j) ** (-ell_all)) * Tp_j

            low_kernel = (
                parity[:, :-1]
                * low_factor[:, None]
                * jnp.conj(Cp[1:])[None, :]
                * Cn[:, :-1]
                * ph_mid
            )
            up_kernel = (
                parity[:, 1:]
                * up_factor[:, None]
                * jnp.conj(Cp[:-1])[None, :]
                * Cn[:, 1:]
                * ph_mid
            )
            low = jnp.sum(valid_f * low_kernel.real * w_n[:, :-1], axis=0)
            up = jnp.sum(valid_f * up_kernel.real * w_n[:, 1:], axis=0)
            return row_in.at[1:].add(low).at[:-1].add(up)

        return jax.lax.cond(Nm > 1, add_sidebands, lambda value: value, row)

    return _map_rows_range(row_fn, 0, Nt, Nm, False)


_assemble_shift_variable_mode_core = jax.jit(
    _assemble_shift_variable_mode_impl,
    static_argnums=(3, 8),
)


# ---------------------------------------------------------------------------
# Production analytic-parity implementation
# ---------------------------------------------------------------------------


def _normalize_precision(precision):
    """Return the canonical production precision label."""

    if precision is None:
        return "complex64"
    key = str(precision).lower()
    if key in ("complex64", "float32", "c64", "32"):
        return "complex64"
    if key in ("complex128", "float64", "c128", "64"):
        return "complex128"
    raise ValueError(
        "precision must be complex64/float32 or complex128/float64."
    )


def _normalize_production_variant(variant):
    """Return the canonical production-kernel layout variant."""

    if variant is None:
        return "baseline"
    key = str(variant).lower()
    if key in ("baseline", "default"):
        return "baseline"
    if key in ("reordered", "contiguous_lag"):
        return "reordered"
    raise ValueError(
        "assembly_variant must be 'baseline' or 'reordered'."
    )


def _analytic_parity_pm_one(values, real_dtype):
    """Return ``(-1)**values`` without constructing ``C_nm``."""

    values = jnp.asarray(values, dtype=jnp.int32)
    return jnp.where(
        jnp.mod(values, 2) == 0,
        jnp.asarray(1.0, dtype=real_dtype),
        jnp.asarray(-1.0, dtype=real_dtype),
    )


def _make_shift_target_production_impl(
    *,
    complex_dtype,
    real_dtype,
    coefficient_dtype,
):
    """Create one analytic-parity row/lag-blocked production kernel."""

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
        kernels_reordered,
    ):
        # Real physical WDM coefficients remain real.  Complex coefficient
        # support is retained for the generic public API and regression tests.
        w_xi = jnp.asarray(w_xi, dtype=coefficient_dtype)
        t_shift = jnp.asarray(t_shift, dtype=real_dtype)
        ell_all = jnp.asarray(ell_all, dtype=jnp.int32)
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
        half_bin_phase = jnp.exp(one_j * pi * dF * t_shift)
        ph_mid_all = ph_m_all[:, :-1] * half_bin_phase[:, None]

        row_sign_all = _analytic_parity_pm_one(
            jnp.arange(Nt, dtype=jnp.int32),
            real_dtype,
        )
        lag_even_all = jnp.mod(ell_all, 2) == 0
        even_half_sign_all = _analytic_parity_pm_one(
            jnp.floor_divide(ell_all, 2),
            real_dtype,
        )
        odd_half_sign_all = _analytic_parity_pm_one(
            jnp.floor_divide(ell_all - 1, 2),
            real_dtype,
        )
        sideband_frequency_sign = _analytic_parity_pm_one(
            jnp.arange(max(Nm - 1, 0), dtype=jnp.int32),
            real_dtype,
        )

        n_chunks = (Nt + row_chunk_size - 1) // row_chunk_size
        Nt_pad = n_chunks * row_chunk_size
        n_lag_blocks = (n_lag + lag_block_size - 1) // lag_block_size
        if kernels_reordered:
            n_lag_padded = n_lag_blocks * lag_block_size
            lag_padding = n_lag_padded - n_lag
            Tl_apply = jnp.pad(Tl_all, ((0, 0), (0, lag_padding)))
            Tp_apply = jnp.pad(Tp_all, ((0, 0), (0, lag_padding)))

        out0 = jnp.zeros((Nt_pad, Nm), dtype=coefficient_dtype)
        row_offsets = jnp.arange(row_chunk_size, dtype=jnp.int32)
        lag_offsets = jnp.arange(lag_block_size, dtype=jnp.int32)

        def chunk_body(chunk_id, out_acc):
            start = chunk_id * row_chunk_size
            rows = start + row_offsets
            valid_row = rows < Nt
            rows_safe = jnp.clip(rows, 0, Nt - 1)

            row_sign = row_sign_all[rows_safe]
            ph_m = ph_m_all[rows_safe, :]
            ph_mid = ph_mid_all[rows_safe, :]
            if kernels_reordered:
                Tl_rows = Tl_apply[rows_safe, :]
                Tp_rows = Tp_apply[rows_safe, :]
            out_chunk0 = jnp.zeros(
                (row_chunk_size, Nm),
                dtype=coefficient_dtype,
            )

            def lag_block_body(lag_block_id, chunk_out):
                lag_start = lag_block_id * lag_block_size
                lag_indices = lag_start + lag_offsets
                valid_lag = lag_indices < n_lag
                lag_indices_safe = jnp.clip(lag_indices, 0, n_lag - 1)

                ell_block = ell_all[lag_indices_safe]
                lag_even = lag_even_all[lag_indices_safe]
                even_half_sign = even_half_sign_all[lag_indices_safe]
                odd_half_sign = odd_half_sign_all[lag_indices_safe]

                nprime = rows[:, None] + ell_block[None, :]
                valid_n = (
                    valid_row[:, None]
                    & valid_lag[None, :]
                    & (nprime >= 0)
                    & (nprime < Nt)
                )
                nprime_safe = jnp.clip(nprime, 0, Nt - 1)

                if kernels_reordered:
                    Tl_block = jax.lax.dynamic_slice(
                        Tl_rows,
                        (0, lag_start),
                        (row_chunk_size, lag_block_size),
                    )
                    Tp_block = jax.lax.dynamic_slice(
                        Tp_rows,
                        (0, lag_start),
                        (row_chunk_size, lag_block_size),
                    )
                else:
                    j_neg_block = -ell_block + offset
                    Tl_block = Tl_all[rows_safe, :][:, j_neg_block]
                    Tp_block = Tp_all[rows_safe, :][:, j_neg_block]
                w_block = w_xi[nprime_safe, :]

                # Exact checkerboard carrier product:
                #   1 for even lag; i*(-1)^p for odd lag.
                carrier_factor = jnp.where(
                    lag_even[None, :, None],
                    one_complex,
                    one_j * row_sign[:, None, None],
                )
                carrier = (
                    carrier_factor
                    * ph_m[:, None, :]
                    * Tl_block[:, :, None]
                ).real * w_block
                carrier = jnp.where(
                    valid_n[:, :, None],
                    carrier,
                    jnp.zeros_like(carrier),
                )
                block_sum = jnp.sum(carrier, axis=1)

                def add_sidebands(block_acc):
                    # Exact adjacent-bin checkerboard products reduce to +/-i
                    # multiplied by row, lag and frequency signs.
                    even_row_lag_sign = (
                        row_sign[:, None] * even_half_sign[None, :]
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
                        * sideband_frequency_sign[None, None, :]
                        * ph_mid[:, None, :]
                        * Tp_block[:, :, None]
                    ).real * w_block[:, :, :-1]
                    up = (
                        one_j
                        * up_row_lag_sign[:, :, None]
                        * sideband_frequency_sign[None, None, :]
                        * ph_mid[:, None, :]
                        * Tp_block[:, :, None]
                    ).real * w_block[:, :, 1:]

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
                    low_sum = jnp.sum(low, axis=1)
                    up_sum = jnp.sum(up, axis=1)

                    # Accumulate only into the valid adjacent-frequency slices.
                    # This replaces the previous full-width zero-padding arrays.
                    updated = block_acc.at[:, 1:].add(low_sum)
                    return updated.at[:, :-1].add(up_sum)

                block_sum = jax.lax.cond(
                    Nm > 1,
                    add_sidebands,
                    lambda value: value,
                    block_sum,
                )
                return chunk_out + block_sum

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

        out_padded = jax.lax.fori_loop(0, n_chunks, chunk_body, out0)
        return out_padded[:Nt, :]

    return impl


# Four compiled variants keep the large coefficient/output arrays in their
# natural real or complex dtype while retaining the requested kernel precision.
_prod_c64_complex_impl = _make_shift_target_production_impl(
    complex_dtype=jnp.complex64,
    real_dtype=jnp.float32,
    coefficient_dtype=jnp.complex64,
)
_prod_c64_real_impl = _make_shift_target_production_impl(
    complex_dtype=jnp.complex64,
    real_dtype=jnp.float32,
    coefficient_dtype=jnp.float32,
)
_prod_c128_complex_impl = _make_shift_target_production_impl(
    complex_dtype=jnp.complex128,
    real_dtype=jnp.float64,
    coefficient_dtype=jnp.complex128,
)
_prod_c128_real_impl = _make_shift_target_production_impl(
    complex_dtype=jnp.complex128,
    real_dtype=jnp.float64,
    coefficient_dtype=jnp.float64,
)

_prod_c64_complex_core = jax.jit(
    _prod_c64_complex_impl,
    static_argnums=(3, 7, 8, 9),
)
_prod_c64_real_core = jax.jit(
    _prod_c64_real_impl,
    static_argnums=(3, 7, 8, 9),
)
_prod_c128_complex_core = jax.jit(
    _prod_c128_complex_impl,
    static_argnums=(3, 7, 8, 9),
)
_prod_c128_real_core = jax.jit(
    _prod_c128_real_impl,
    static_argnums=(3, 7, 8, 9),
)


def _make_shift_target_batch_production_impl(*, single_job_impl, output_dtype):
    """Create a batched production kernel from one single-job kernel."""

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
        kernels_reordered,
    ):
        B, Nt, Nm = w_xi_batch.shape
        out0 = jnp.zeros((B, Nt, Nm), dtype=output_dtype)

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
                kernels_reordered,
            )
            return out.at[job_index, :, :].set(shifted)

        return jax.lax.fori_loop(0, B, body, out0)

    return impl


_prod_batch_c64_complex_core = jax.jit(
    _make_shift_target_batch_production_impl(
        single_job_impl=_prod_c64_complex_impl,
        output_dtype=jnp.complex64,
    ),
    static_argnums=(3, 7, 8, 9),
)
_prod_batch_c64_real_core = jax.jit(
    _make_shift_target_batch_production_impl(
        single_job_impl=_prod_c64_real_impl,
        output_dtype=jnp.float32,
    ),
    static_argnums=(3, 7, 8, 9),
)
_prod_batch_c128_complex_core = jax.jit(
    _make_shift_target_batch_production_impl(
        single_job_impl=_prod_c128_complex_impl,
        output_dtype=jnp.complex128,
    ),
    static_argnums=(3, 7, 8, 9),
)
_prod_batch_c128_real_core = jax.jit(
    _make_shift_target_batch_production_impl(
        single_job_impl=_prod_c128_real_impl,
        output_dtype=jnp.float64,
    ),
    static_argnums=(3, 7, 8, 9),
)


def _input_is_complex(values) -> bool:
    return bool(jnp.issubdtype(jnp.asarray(values).dtype, jnp.complexfloating))


# ---------------------------------------------------------------------------
# Public wrappers used by the high-level dispatcher
# ---------------------------------------------------------------------------


def assemble_shift_target_production_jax(
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    dF,
    *,
    row_chunk_size=128,
    lag_block_size=1,
    precision="complex64",
    assembly_variant="baseline",
    return_device=False,
):
    """Apply one production target-mode shift.

    Real inputs return real outputs; complex inputs retain complex support.
    """

    precision = _normalize_precision(precision)
    assembly_variant = _normalize_production_variant(assembly_variant)
    kernels_reordered = assembly_variant == "reordered"
    row_chunk_size = int(row_chunk_size)
    lag_block_size = int(lag_block_size)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")
    if lag_block_size < 1:
        raise ValueError("lag_block_size must be >= 1.")

    is_complex = _input_is_complex(w_xi)
    ell = jnp.asarray(ell_all, dtype=jnp.int32)

    if precision == "complex64":
        coefficient_dtype = jnp.complex64 if is_complex else jnp.float32
        core = _prod_c64_complex_core if is_complex else _prod_c64_real_core
        out = core(
            jnp.asarray(w_xi, dtype=coefficient_dtype),
            jnp.asarray(t_shift, dtype=jnp.float32),
            ell,
            int(offset),
            jnp.asarray(Tl_all, dtype=jnp.complex64),
            jnp.asarray(Tp_all, dtype=jnp.complex64),
            jnp.asarray(dF, dtype=jnp.float32),
            row_chunk_size,
            lag_block_size,
            kernels_reordered,
        )
    else:
        coefficient_dtype = jnp.complex128 if is_complex else jnp.float64
        core = _prod_c128_complex_core if is_complex else _prod_c128_real_core
        out = core(
            jnp.asarray(w_xi, dtype=coefficient_dtype),
            jnp.asarray(t_shift, dtype=jnp.float64),
            ell,
            int(offset),
            jnp.asarray(Tl_all, dtype=jnp.complex128),
            jnp.asarray(Tp_all, dtype=jnp.complex128),
            jnp.asarray(dF, dtype=jnp.float64),
            row_chunk_size,
            lag_block_size,
            kernels_reordered,
        )

    return out if return_device else np.asarray(out)


def assemble_shift_target_batch_production_jax(
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    dF,
    *,
    row_chunk_size=128,
    lag_block_size=1,
    precision="complex64",
    assembly_variant="baseline",
    return_device=False,
):
    """Apply a production target-mode shift to a job batch."""

    precision = _normalize_precision(precision)
    assembly_variant = _normalize_production_variant(assembly_variant)
    kernels_reordered = assembly_variant == "reordered"
    row_chunk_size = int(row_chunk_size)
    lag_block_size = int(lag_block_size)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")
    if lag_block_size < 1:
        raise ValueError("lag_block_size must be >= 1.")

    is_complex = _input_is_complex(w_xi_batch)
    ell = jnp.asarray(ell_all, dtype=jnp.int32)

    if precision == "complex64":
        coefficient_dtype = jnp.complex64 if is_complex else jnp.float32
        core = (
            _prod_batch_c64_complex_core
            if is_complex
            else _prod_batch_c64_real_core
        )
        out = core(
            jnp.asarray(w_xi_batch, dtype=coefficient_dtype),
            jnp.asarray(t_shift_batch, dtype=jnp.float32),
            ell,
            int(offset),
            jnp.asarray(Tl_batch, dtype=jnp.complex64),
            jnp.asarray(Tp_batch, dtype=jnp.complex64),
            jnp.asarray(dF, dtype=jnp.float32),
            row_chunk_size,
            lag_block_size,
            kernels_reordered,
        )
    else:
        coefficient_dtype = jnp.complex128 if is_complex else jnp.float64
        core = (
            _prod_batch_c128_complex_core
            if is_complex
            else _prod_batch_c128_real_core
        )
        out = core(
            jnp.asarray(w_xi_batch, dtype=coefficient_dtype),
            jnp.asarray(t_shift_batch, dtype=jnp.float64),
            ell,
            int(offset),
            jnp.asarray(Tl_batch, dtype=jnp.complex128),
            jnp.asarray(Tp_batch, dtype=jnp.complex128),
            jnp.asarray(dF, dtype=jnp.float64),
            row_chunk_size,
            lag_block_size,
            kernels_reordered,
        )

    return out if return_device else np.asarray(out)


def assemble_shift_target_reference_jax(
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    dF,
    *,
    return_device=False,
):
    """Apply one high-precision reference target-mode shift."""

    out = _assemble_shift_target_reference_core(
        jnp.asarray(w_xi, dtype=jnp.complex128),
        jnp.asarray(t_shift, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int32),
        int(offset),
        jnp.asarray(Tl_all, dtype=jnp.complex128),
        jnp.asarray(Tp_all, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        False,
    )
    return out if return_device else np.asarray(out)


def assemble_shift_target_batch_reference_jax(
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    dF,
    *,
    return_device=False,
):
    """Apply the high-precision reference target-mode shift to a batch."""

    out = _assemble_shift_target_batch_reference_core(
        jnp.asarray(w_xi_batch, dtype=jnp.complex128),
        jnp.asarray(t_shift_batch, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int32),
        int(offset),
        jnp.asarray(Tl_batch, dtype=jnp.complex128),
        jnp.asarray(Tp_batch, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
    )
    return out if return_device else np.asarray(out)


def assemble_shift_fixed_jax(
    w_xi,
    delta,
    ell_all,
    offset,
    Tl_vec,
    Tp_vec,
    Cnm,
    dF,
):
    """Apply the retained reference fixed-delay operator."""

    out = _assemble_shift_fixed_core(
        jnp.asarray(w_xi, dtype=jnp.complex128),
        jnp.asarray(delta, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int32),
        int(offset),
        jnp.asarray(Tl_vec, dtype=jnp.complex128),
        jnp.asarray(Tp_vec, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        False,
    )
    return np.asarray(out)


def assemble_shift_variable_mode_jax(
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    dF,
    delta_mode,
):
    """Apply the retained reference source/midpoint operator."""

    mode_map = {"source": 0, "midpoint": 1}
    if delta_mode not in mode_map:
        raise ValueError("delta_mode must be 'source' or 'midpoint'.")

    out = _assemble_shift_variable_mode_core(
        jnp.asarray(w_xi, dtype=jnp.complex128),
        jnp.asarray(t_shift, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int32),
        int(offset),
        jnp.asarray(Tl_all, dtype=jnp.complex128),
        jnp.asarray(Tp_all, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        int(mode_map[delta_mode]),
    )
    return np.asarray(out)
