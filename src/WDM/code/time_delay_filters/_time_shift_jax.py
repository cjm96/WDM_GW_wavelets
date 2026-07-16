"""JAX assembly kernels for WDM time-shift operators.

This module contains low-level numerical kernels used by
``time_shift_fast.py``. It intentionally avoids higher-level concerns such as
kernel-size selection, caching, and input pre-processing.
"""

import numpy as np

import jax
import jax.numpy as jnp


def _map_rows_range(row_fn, start, stop, Nm, use_vmap):
    """Evaluate ``row_fn`` for rows ``[start, stop)`` with optional ``vmap``.

    Parameters
    ----------
    row_fn : callable
        Function mapping row index ``p`` to a ``(Nm,)`` row.
    start, stop : int
        Half-open row index interval.
    Nm : int
        Number of frequency bins (row width).
    use_vmap : bool
        When true, evaluate rows with ``jax.vmap``. When false, use
        ``lax.fori_loop`` row-by-row to reduce peak memory usage.
    """
    n_rows = max(0, int(stop) - int(start))
    if n_rows == 0:
        return jnp.zeros((0, Nm), dtype=jnp.complex128)
    if use_vmap:
        return jax.vmap(row_fn)(jnp.arange(start, stop))

    def body(i, arr):
        p = start + i
        return arr.at[i, :].set(row_fn(p))

    out0 = jnp.zeros((n_rows, Nm), dtype=jnp.complex128)
    return jax.lax.fori_loop(0, n_rows, body, out0)


def _phase_m(delta_vec, Nm, dF):
    """Return full-bin phase factors ``exp(i 2 pi m dF delta)``.

    Parameters
    ----------
    delta_vec : jax.Array
        Vector of delays (typically indexed by lag ell).
    Nm : int
        Number of frequency bins.
    dF : float
        Frequency spacing.
    """
    m = jnp.arange(Nm)
    return jnp.exp(2j * jnp.pi * (m[None, :] * dF) * delta_vec[:, None])


def _phase_mid(delta_vec, Nm, dF):
    """Return half-bin phase factors for sideband coupling terms.

    Uses ``m + 0.5`` bins needed by the off-diagonal terms in Eq.(34)-style
    assembly.
    """
    m = jnp.arange(max(Nm - 1, 0))
    return jnp.exp(2j * jnp.pi * ((m[None, :] + 0.5) * dF) * delta_vec[:, None])




def _accumulate_target_row(p, w_xi, t_shift, ell_all, offset, Tl_all, Tp_all, Cnm, dF, parity, low_phase, up_phase, interior):
    """Accumulate one target-mode row directly over lag index.

    This avoids materializing the dense carrier and sideband coefficient tensors
    used by the previous implementation. When ``interior`` is true, the bounds
    checks for valid lag rows are skipped because every shifted source row is
    known to be in range.
    """
    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]
    j_neg = (-ell_all) + offset

    Cp = Cnm[p, :]
    Cp_conj = jnp.conj(Cp)
    delta_p = t_shift[p]
    Tl_j = Tl_all[p, j_neg]
    Tp_j = Tp_all[p, j_neg]
    m_full = jnp.arange(Nm, dtype=jnp.float64)
    ph_m_row = jnp.exp(2j * jnp.pi * (m_full * dF) * delta_p)
    carrier_prefac = Cp_conj * ph_m_row

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

        def skip_row(r):
            return r

        def add_row(r):
            Cn = Cnm[n, :]
            w_n = w_xi[n, :]

            carrier = (parity[i, :] * carrier_prefac * Cn * Tl_j[i]).real * w_n

            def add_sidebands(_):
                mid_common = Tp_j[i] * ph_mid_row
                low_vals = (
                    parity[i, :-1]
                    * low_phase[i]
                    * cp_conj_hi
                    * Cn[:-1]
                    * mid_common
                ).real * w_n[:-1]
                up_vals = (
                    parity[i, 1:]
                    * up_phase[i]
                    * cp_conj_lo
                    * Cn[1:]
                    * mid_common
                ).real * w_n[1:]
                return jnp.concatenate((zero_head, low_vals)) + jnp.concatenate((up_vals, zero_tail))

            sidebands = jax.lax.cond(
                Nm > 1,
                add_sidebands,
                lambda _: jnp.zeros((Nm,), dtype=jnp.complex128),
                operand=None,
            )
            return r + carrier + sidebands

        if interior:
            return add_row(row_state)
        return jax.lax.cond((n >= 0) & (n < Nt), add_row, skip_row, row_state)

    row0 = jnp.zeros((Nm,), dtype=jnp.complex128)
    return jax.lax.fori_loop(0, n_lag, lag_body, row0)


def _assemble_shift_target_impl(w_xi, t_shift, ell_all, offset, Tl_all, Tp_all, Cnm, dF, use_vmap):
    """JIT kernel for target-mode variable-delay assembly.

    For each output row ``p``, the delay and Tl/Tp vectors are taken from the
    target row ``p``.
    """
    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]
    minus1_to_m = jnp.where((jnp.arange(Nm) % 2) == 0, 1.0, -1.0)
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

    boundary_left = _map_rows_range(row_fn, 0, left, Nm, use_vmap)
    interior_rows = _map_rows_range(row_fn_interior, left, right, Nm, use_vmap)
    boundary_right = _map_rows_range(row_fn, right, Nt, Nm, use_vmap)
    return jnp.concatenate((boundary_left, interior_rows, boundary_right), axis=0)


_assemble_shift_target_core = jax.jit(_assemble_shift_target_impl, static_argnums=(3, 8))


def _assemble_shift_fixed_impl(w_xi, delta, ell_all, offset, Tl_vec, Tp_vec, Cnm, dF, use_vmap):
    """JIT kernel for fixed-delay assembly with precomputed Tl/Tp vectors."""
    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]
    j_neg = (-ell_all) + offset

    minus1_to_m = jnp.where((jnp.arange(Nm) % 2) == 0, 1.0, -1.0)
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
    ph_m_row = jnp.exp(2j * jnp.pi * (m_full * dF) * delta)
    if Nm > 1:
        ph_mid_row = ph_m_row[:-1] * jnp.exp(1j * jnp.pi * dF * delta)
    else:
        ph_mid_row = jnp.zeros((0,), dtype=jnp.complex128)

    left = min(offset, Nt)
    right = max(left, Nt - offset)

    zero_head = jnp.zeros((1,), dtype=jnp.complex128)
    zero_tail = jnp.zeros((1,), dtype=jnp.complex128)

    def row_accum(p, interior):
        Cp = Cnm[p, :]
        Cp_conj = jnp.conj(Cp)
        carrier_prefac = Cp_conj * ph_m_row
        if Nm > 1:
            cp_conj_hi = Cp_conj[1:]
            cp_conj_lo = Cp_conj[:-1]
        else:
            cp_conj_hi = jnp.zeros((0,), dtype=jnp.complex128)
            cp_conj_lo = jnp.zeros((0,), dtype=jnp.complex128)

        def lag_body(i, row_state):
            ell = ell_all[i]
            n = p + ell

            def skip_row(r):
                return r

            def add_row(r):
                Cn = Cnm[n, :]
                w_n = w_xi[n, :]

                carrier = (parity[i, :] * carrier_prefac * Cn * Tl_j[i]).real * w_n

                def add_sidebands(_):
                    mid_common = Tp_j[i] * ph_mid_row
                    low_vals = (
                        parity[i, :-1]
                        * low_phase[i]
                        * cp_conj_hi
                        * Cn[:-1]
                        * mid_common
                    ).real * w_n[:-1]
                    up_vals = (
                        parity[i, 1:]
                        * up_phase[i]
                        * cp_conj_lo
                        * Cn[1:]
                        * mid_common
                    ).real * w_n[1:]
                    return jnp.concatenate((zero_head, low_vals)) + jnp.concatenate((up_vals, zero_tail))

                sidebands = jax.lax.cond(
                    Nm > 1,
                    add_sidebands,
                    lambda _: jnp.zeros((Nm,), dtype=jnp.complex128),
                    operand=None,
                )
                return r + carrier + sidebands

            if interior:
                return add_row(row_state)
            return jax.lax.cond((n >= 0) & (n < Nt), add_row, skip_row, row_state)

        row0 = jnp.zeros((Nm,), dtype=jnp.complex128)
        return jax.lax.fori_loop(0, n_lag, lag_body, row0)

    def row_fn(p):
        return row_accum(p, False)

    def row_fn_interior(p):
        return row_accum(p, True)

    if left == Nt:
        return _map_rows_range(row_fn, 0, Nt, Nm, use_vmap)

    boundary_left = _map_rows_range(row_fn, 0, left, Nm, use_vmap)
    interior_rows = _map_rows_range(row_fn_interior, left, right, Nm, use_vmap)
    boundary_right = _map_rows_range(row_fn, right, Nt, Nm, use_vmap)
    return jnp.concatenate((boundary_left, interior_rows, boundary_right), axis=0)


_assemble_shift_fixed_core = jax.jit(_assemble_shift_fixed_impl, static_argnums=(3, 8))


def _assemble_shift_variable_mode_impl(w_xi, t_shift, ell_all, offset, Tl_all, Tp_all, Cnm, dF, mode, use_vmap):
    """JIT kernel for variable-delay assembly with selectable delay mode.

    Parameters
    ----------
    mode : int
        Encoded delay mode: ``0=target``, ``1=source``, ``2=midpoint``.

    Notes
    -----
    The target-mode fast path uses a separate kernel with a more efficient row
    accumulation strategy, so this routine is used only for ``source`` and
    ``midpoint`` dispatch.
    """
    Nt, Nm = w_xi.shape
    j_neg = (-ell_all) + offset
    minus1_to_m = jnp.where((jnp.arange(Nm) % 2) == 0, 1.0, -1.0)

    def row_fn(p):
        n_all = p + ell_all
        valid = (n_all >= 0) & (n_all < Nt)
        n_clipped = jnp.clip(n_all, 0, Nt - 1)
        valid_f = valid.astype(jnp.float64)[:, None]

        Cp = Cnm[p, :]
        Cn = Cnm[n_clipped, :]
        w_n = w_xi[n_clipped, :]

        # mode: 0=target, 1=source, 2=midpoint
        def mode_target(_):
            delta_vec = jnp.full_like(ell_all, t_shift[p], dtype=jnp.float64)
            Tl_j = Tl_all[p, j_neg]
            Tp_j = Tp_all[p, j_neg]
            return delta_vec, Tl_j, Tp_j

        def mode_source(_):
            delta_vec = t_shift[n_clipped]
            Tl_j = Tl_all[n_clipped, j_neg]
            Tp_j = Tp_all[n_clipped, j_neg]
            return delta_vec, Tl_j, Tp_j

        def mode_midpoint(_):
            kf = 0.5 * (p + n_all)
            k0 = jnp.floor(kf).astype(jnp.int64)
            k1 = jnp.clip(k0 + 1, 0, Nt - 1)
            w = kf - k0.astype(jnp.float64)
            delta_vec = (1.0 - w) * t_shift[k0] + w * t_shift[k1]
            Tl_j = (1.0 - w) * Tl_all[k0, j_neg] + w * Tl_all[k1, j_neg]
            Tp_j = (1.0 - w) * Tp_all[k0, j_neg] + w * Tp_all[k1, j_neg]
            return delta_vec, Tl_j, Tp_j

        delta_vec, Tl_j, Tp_j = jax.lax.switch(mode, (mode_target, mode_source, mode_midpoint), operand=None)

        ell_even = (ell_all % 2) == 0
        parity = jnp.where(ell_even[:, None], jnp.ones((1, Nm), dtype=jnp.float64), minus1_to_m[None, :])

        ph_m = _phase_m(delta_vec, Nm, dF)
        Kc = parity * jnp.conj(Cp)[None, :] * Cn * Tl_j[:, None] * ph_m
        row = jnp.sum(valid_f * Kc.real * w_n, axis=0)

        def add_sidebands(row_in):
            ph_mid = _phase_mid(delta_vec, Nm, dF)
            low_factor = ((-1j) ** (-ell_all)) * Tp_j
            up_factor = ((+1j) ** (-ell_all)) * Tp_j

            K_low = (
                parity[:, :-1]
                * low_factor[:, None]
                * jnp.conj(Cp[1:])[None, :]
                * Cn[:, :-1]
                * ph_mid
            )
            low_acc = jnp.sum(valid_f * K_low.real * w_n[:, :-1], axis=0)

            K_up = (
                parity[:, 1:]
                * up_factor[:, None]
                * jnp.conj(Cp[:-1])[None, :]
                * Cn[:, 1:]
                * ph_mid
            )
            up_acc = jnp.sum(valid_f * K_up.real * w_n[:, 1:], axis=0)

            row_out = row_in.at[1:].add(low_acc)
            row_out = row_out.at[:-1].add(up_acc)
            return row_out

        row = jax.lax.cond(Nm > 1, add_sidebands, lambda r: r, row)
        return row

    return _map_rows_range(row_fn, 0, Nt, Nm, use_vmap)


_assemble_shift_variable_mode_core = jax.jit(_assemble_shift_variable_mode_impl, static_argnums=(8, 9))


def _normalize_chunked_precision(precision):
    """Normalize precision labels to canonical complex dtype names."""
    if precision is None:
        return "complex128"
    key = str(precision).lower()
    if key in ("complex128", "float64", "c128", "64"):
        return "complex128"
    if key in ("complex64", "float32", "c64", "32"):
        return "complex64"
    raise ValueError("precision must be complex128/float64 or complex64/float32.")


def _assemble_shift_target_chunked_impl_c128(
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    dF,
    row_chunk_size,
):
    """Chunked lag-first target-mode assembly (complex128/float64)."""
    w_xi = jnp.asarray(w_xi, dtype=jnp.complex128)
    t_shift = jnp.asarray(t_shift, dtype=jnp.float64)
    ell_all = jnp.asarray(ell_all, dtype=jnp.int64)
    Tl_all = jnp.asarray(Tl_all, dtype=jnp.complex128)
    Tp_all = jnp.asarray(Tp_all, dtype=jnp.complex128)
    Cnm = jnp.asarray(Cnm, dtype=jnp.complex128)
    dF = jnp.asarray(dF, dtype=jnp.float64)

    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]

    m_full = jnp.arange(Nm, dtype=jnp.float64)
    one_j = jnp.asarray(1j, dtype=jnp.complex128)
    two_pi = jnp.asarray(2.0 * np.pi, dtype=jnp.float64)
    pi = jnp.asarray(np.pi, dtype=jnp.float64)

    # Full-bin phases:
    # exp(i 2 pi m dF Delta)
    ph_m_all = jnp.exp(
        one_j
        * two_pi
        * (m_full[None, :] * dF)
        * t_shift[:, None]
    )

    # Half-bin correction:
    # exp(i pi dF Delta)
    half_bin_phase = jnp.exp(
        one_j
        * pi
        * dF
        * t_shift
    )

    # exp[i 2 pi (m + 1/2) dF Delta]
    # = exp(i 2 pi m dF Delta) exp(i pi dF Delta)
    ph_mid_all = (
        ph_m_all[:, :-1]
        * half_bin_phase[:, None]
    )

    minus1_to_m = jnp.where(
        (jnp.arange(Nm) % 2) == 0,
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(-1.0, dtype=jnp.float64),
    )
    ell_even = (ell_all % 2) == 0
    parity_all = jnp.where(
        ell_even[:, None],
        jnp.ones((n_lag, Nm), dtype=jnp.float64),
        minus1_to_m[None, :],
    )

    neg_i = jnp.asarray(-1j, dtype=jnp.complex128)
    pos_i = jnp.asarray(1j, dtype=jnp.complex128)
    low_phase = jnp.power(neg_i, -ell_all)
    up_phase = jnp.power(pos_i, -ell_all)

    n_chunks = (Nt + row_chunk_size - 1) // row_chunk_size
    Nt_pad = n_chunks * row_chunk_size

    out0 = jnp.zeros((Nt_pad, Nm), dtype=jnp.complex128)
    zero_col = jnp.zeros((row_chunk_size, 1), dtype=jnp.complex128)

    def chunk_body(chunk_id, out_acc):
        start = chunk_id * row_chunk_size
        rows = start + jnp.arange(row_chunk_size, dtype=jnp.int64)
        valid_row = rows < Nt
        rows_safe = jnp.clip(rows, 0, Nt - 1)

        carrier_prefac = jnp.conj(Cnm[rows_safe, :]) * ph_m_all[rows_safe, :]
        ph_mid = ph_mid_all[rows_safe, :]

        out_chunk0 = jnp.zeros((row_chunk_size, Nm), dtype=jnp.complex128)

        def lag_body(i, chunk_out):
            ell = ell_all[i]
            j_neg = -ell + offset
            nprime = rows + ell
            valid_n = valid_row & (nprime >= 0) & (nprime < Nt)
            nprime_safe = jnp.clip(nprime, 0, Nt - 1)

            parity = parity_all[i]
            Tl_row = Tl_all[rows_safe, j_neg]
            Tp_row = Tp_all[rows_safe, j_neg]
            Cn = Cnm[nprime_safe, :]
            w_n = w_xi[nprime_safe, :]

            main = (parity[None, :] * carrier_prefac * Cn * Tl_row[:, None]).real * w_n
            main = jnp.where(valid_n[:, None], main, jnp.zeros_like(main))

            def add_sidebands(main_acc):
                low = (
                    parity[:-1][None, :]
                    * low_phase[i]
                    * jnp.conj(Cnm[rows_safe, 1:])
                    * Cn[:, :-1]
                    * Tp_row[:, None]
                    * ph_mid
                ).real * w_n[:, :-1]

                up = (
                    parity[1:][None, :]
                    * up_phase[i]
                    * jnp.conj(Cnm[rows_safe, :-1])
                    * Cn[:, 1:]
                    * Tp_row[:, None]
                    * ph_mid
                ).real * w_n[:, 1:]

                low_pad = jnp.concatenate((zero_col, low), axis=1)
                up_pad = jnp.concatenate((up, zero_col), axis=1)
                side = low_pad + up_pad
                side = jnp.where(valid_n[:, None], side, jnp.zeros_like(side))
                return main_acc + side

            chunk_next = jax.lax.cond(Nm > 1, add_sidebands, lambda x: x, main)
            return chunk_out + chunk_next

        out_chunk = jax.lax.fori_loop(0, n_lag, lag_body, out_chunk0)
        return jax.lax.dynamic_update_slice(out_acc, out_chunk, (start, 0))

    out_padded = jax.lax.fori_loop(0, n_chunks, chunk_body, out0)
    return out_padded[:Nt, :]


def _assemble_shift_target_chunked_impl_c64(
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    dF,
    row_chunk_size,
):
    """Chunked lag-first target-mode assembly (complex64/float32)."""
    w_xi = jnp.asarray(w_xi, dtype=jnp.complex64)
    t_shift = jnp.asarray(t_shift, dtype=jnp.float32)
    ell_all = jnp.asarray(ell_all, dtype=jnp.int64)
    Tl_all = jnp.asarray(Tl_all, dtype=jnp.complex64)
    Tp_all = jnp.asarray(Tp_all, dtype=jnp.complex64)
    Cnm = jnp.asarray(Cnm, dtype=jnp.complex64)
    dF = jnp.asarray(dF, dtype=jnp.float32)

    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]

    m_full = jnp.arange(Nm, dtype=jnp.float32)
    one_j = jnp.asarray(1j, dtype=jnp.complex64)
    two_pi = jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
    pi = jnp.asarray(np.pi, dtype=jnp.float32)

    # Full-bin phases:
    # exp(i 2 pi m dF Delta)
    ph_m_all = jnp.exp(
        one_j
        * two_pi
        * (m_full[None, :] * dF)
        * t_shift[:, None]
    )

    # Half-bin correction:
    # exp(i pi dF Delta)
    half_bin_phase = jnp.exp(
        one_j
        * pi
        * dF
        * t_shift
    )

    # exp[i 2 pi (m + 1/2) dF Delta]
    # = exp(i 2 pi m dF Delta) exp(i pi dF Delta)
    ph_mid_all = (
        ph_m_all[:, :-1]
        * half_bin_phase[:, None]
    )
    minus1_to_m = jnp.where(
        (jnp.arange(Nm) % 2) == 0,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(-1.0, dtype=jnp.float32),
    )
    ell_even = (ell_all % 2) == 0
    parity_all = jnp.where(
        ell_even[:, None],
        jnp.ones((n_lag, Nm), dtype=jnp.float32),
        minus1_to_m[None, :],
    )

    neg_i = jnp.asarray(-1j, dtype=jnp.complex64)
    pos_i = jnp.asarray(1j, dtype=jnp.complex64)
    low_phase = jnp.power(neg_i, -ell_all)
    up_phase = jnp.power(pos_i, -ell_all)

    n_chunks = (Nt + row_chunk_size - 1) // row_chunk_size
    Nt_pad = n_chunks * row_chunk_size

    out0 = jnp.zeros((Nt_pad, Nm), dtype=jnp.complex64)
    zero_col = jnp.zeros((row_chunk_size, 1), dtype=jnp.complex64)

    def chunk_body(chunk_id, out_acc):
        start = chunk_id * row_chunk_size
        rows = start + jnp.arange(row_chunk_size, dtype=jnp.int64)
        valid_row = rows < Nt
        rows_safe = jnp.clip(rows, 0, Nt - 1)

        carrier_prefac = jnp.conj(Cnm[rows_safe, :]) * ph_m_all[rows_safe, :]
        ph_mid = ph_mid_all[rows_safe, :]

        out_chunk0 = jnp.zeros((row_chunk_size, Nm), dtype=jnp.complex64)

        def lag_body(i, chunk_out):
            ell = ell_all[i]
            j_neg = -ell + offset
            nprime = rows + ell
            valid_n = valid_row & (nprime >= 0) & (nprime < Nt)
            nprime_safe = jnp.clip(nprime, 0, Nt - 1)

            parity = parity_all[i]
            Tl_row = Tl_all[rows_safe, j_neg]
            Tp_row = Tp_all[rows_safe, j_neg]
            Cn = Cnm[nprime_safe, :]
            w_n = w_xi[nprime_safe, :]

            main = (parity[None, :] * carrier_prefac * Cn * Tl_row[:, None]).real * w_n
            main = jnp.where(valid_n[:, None], main, jnp.zeros_like(main))

            def add_sidebands(main_acc):
                low = (
                    parity[:-1][None, :]
                    * low_phase[i]
                    * jnp.conj(Cnm[rows_safe, 1:])
                    * Cn[:, :-1]
                    * Tp_row[:, None]
                    * ph_mid
                ).real * w_n[:, :-1]

                up = (
                    parity[1:][None, :]
                    * up_phase[i]
                    * jnp.conj(Cnm[rows_safe, :-1])
                    * Cn[:, 1:]
                    * Tp_row[:, None]
                    * ph_mid
                ).real * w_n[:, 1:]

                low_pad = jnp.concatenate((zero_col, low), axis=1)
                up_pad = jnp.concatenate((up, zero_col), axis=1)
                side = low_pad + up_pad
                side = jnp.where(valid_n[:, None], side, jnp.zeros_like(side))
                return main_acc + side

            chunk_next = jax.lax.cond(Nm > 1, add_sidebands, lambda x: x, main)
            return chunk_out + chunk_next

        out_chunk = jax.lax.fori_loop(0, n_lag, lag_body, out_chunk0)
        return jax.lax.dynamic_update_slice(out_acc, out_chunk, (start, 0))

    out_padded = jax.lax.fori_loop(0, n_chunks, chunk_body, out0)
    return out_padded[:Nt, :]


_assemble_shift_target_chunked_core_c128 = jax.jit(
    _assemble_shift_target_chunked_impl_c128,
    static_argnums=(3, 8),
)

_assemble_shift_target_chunked_core_c64 = jax.jit(
    _assemble_shift_target_chunked_impl_c64,
    static_argnums=(3, 8),
)


def _assemble_shift_target_batch_chunked_impl_c128(
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    dF,
    row_chunk_size,
):
    """Batch chunked target-mode assembly (complex128/float64)."""
    B, Nt, Nm = w_xi_batch.shape
    out0 = jnp.zeros((B, Nt, Nm), dtype=jnp.complex128)

    def body(i, out):
        row = _assemble_shift_target_chunked_impl_c128(
            w_xi_batch[i],
            t_shift_batch[i],
            ell_all,
            offset,
            Tl_batch[i],
            Tp_batch[i],
            Cnm,
            dF,
            row_chunk_size,
        )
        return out.at[i, :, :].set(row)

    return jax.lax.fori_loop(0, B, body, out0)


def _assemble_shift_target_batch_chunked_impl_c64(
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    dF,
    row_chunk_size,
):
    """Batch chunked target-mode assembly (complex64/float32)."""
    B, Nt, Nm = w_xi_batch.shape
    out0 = jnp.zeros((B, Nt, Nm), dtype=jnp.complex64)

    def body(i, out):
        row = _assemble_shift_target_chunked_impl_c64(
            w_xi_batch[i],
            t_shift_batch[i],
            ell_all,
            offset,
            Tl_batch[i],
            Tp_batch[i],
            Cnm,
            dF,
            row_chunk_size,
        )
        return out.at[i, :, :].set(row)

    return jax.lax.fori_loop(0, B, body, out0)


_assemble_shift_target_batch_chunked_core_c128 = jax.jit(
    _assemble_shift_target_batch_chunked_impl_c128,
    static_argnums=(3, 8),
)

_assemble_shift_target_batch_chunked_core_c64 = jax.jit(
    _assemble_shift_target_batch_chunked_impl_c64,
    static_argnums=(3, 8),
)

def _combine_weighted_source_rows(
    source_coefficients,
    source_weights,
    row_indices,
):
    """Combine source rows without materialising a full job-major batch.

    ``source_coefficients`` has shape ``(num_sources, Nt, Nm)`` and
    ``source_weights`` has shape ``(num_sources, Nt, Nm)`` for one shift job.
    ``row_indices`` may be one- or multi-dimensional; the returned array has
    shape ``row_indices.shape + (Nm,)``.
    """

    source_rows = source_coefficients[:, row_indices, :]
    weight_rows = source_weights[:, row_indices, :]

    combined = weight_rows[0] * source_rows[0]

    def source_body(source_index, value):
        return value + weight_rows[source_index] * source_rows[source_index]

    return jax.lax.fori_loop(
        1,
        source_coefficients.shape[0],
        source_body,
        combined,
    )


def _make_shift_target_weighted_chunked_impl(
    *,
    complex_dtype,
    real_dtype,
):
    """Create one weighted-source row-chunked target-mode kernel."""

    def impl(
        source_coefficients,
        source_weights,
        t_shift,
        ell_all,
        offset,
        Tl_all,
        Tp_all,
        Cnm,
        dF,
        row_chunk_size,
    ):
        # Preserve the source/weight input precision during modulation.  The
        # combined rows are cast only after the weighted sum, matching the
        # materialised route: combine first, then cast for shift assembly.
        source_coefficients = jnp.asarray(source_coefficients)
        source_weights = jnp.asarray(source_weights)
        t_shift = jnp.asarray(t_shift, dtype=real_dtype)
        ell_all = jnp.asarray(ell_all, dtype=jnp.int64)
        Tl_all = jnp.asarray(Tl_all, dtype=complex_dtype)
        Tp_all = jnp.asarray(Tp_all, dtype=complex_dtype)
        Cnm = jnp.asarray(Cnm, dtype=complex_dtype)
        dF = jnp.asarray(dF, dtype=real_dtype)

        Nt = t_shift.shape[0]
        Nm = source_coefficients.shape[-1]
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

        n_chunks = (Nt + row_chunk_size - 1) // row_chunk_size
        Nt_pad = n_chunks * row_chunk_size

        out0 = jnp.zeros((Nt_pad, Nm), dtype=complex_dtype)
        zero_col = jnp.zeros((row_chunk_size, 1), dtype=complex_dtype)

        # A target-row chunk can read source rows up to ``offset`` rows on
        # either side. Build one weighted halo per chunk and reuse it for all
        # lags, rather than recomputing the polarization combination inside
        # every lag iteration.
        row_offsets = jnp.arange(row_chunk_size, dtype=jnp.int64)
        halo_size = row_chunk_size + 2 * offset
        halo_offsets = jnp.arange(halo_size, dtype=jnp.int64)

        def chunk_body(chunk_id, out_acc):
            start = chunk_id * row_chunk_size
            rows = start + row_offsets
            valid_row = rows < Nt
            rows_safe = jnp.clip(rows, 0, Nt - 1)

            carrier_prefac = (
                jnp.conj(Cnm[rows_safe, :])
                * ph_m_all[rows_safe, :]
            )
            ph_mid = ph_mid_all[rows_safe, :]

            # Required source rows for this target chunk span
            # [start - offset, start + row_chunk_size - 1 + offset].
            # Clipped boundary entries are harmless because ``valid_n`` masks
            # their contributions in the lag loop.
            halo_start = start - offset
            halo_rows = halo_start + halo_offsets
            halo_rows_safe = jnp.clip(halo_rows, 0, Nt - 1)

            weighted_halo = _combine_weighted_source_rows(
                source_coefficients,
                source_weights,
                halo_rows_safe,
            ).astype(complex_dtype)

            out_chunk0 = jnp.zeros(
                (row_chunk_size, Nm),
                dtype=complex_dtype,
            )

            def lag_body(i, chunk_out):
                ell = ell_all[i]
                j_neg = -ell + offset
                nprime = rows + ell
                valid_n = (
                    valid_row
                    & (nprime >= 0)
                    & (nprime < Nt)
                )
                nprime_safe = jnp.clip(nprime, 0, Nt - 1)

                parity = parity_all[i]
                Tl_row = Tl_all[rows_safe, j_neg]
                Tp_row = Tp_all[rows_safe, j_neg]
                Cn = Cnm[nprime_safe, :]

                # Reuse the weighted source halo built once for this row
                # chunk. Since
                #   nprime = start + row_offset + ell
                # and
                #   halo_start = start - offset,
                # the corresponding halo index is row_offset + ell + offset.
                local_source_rows = row_offsets + ell + offset
                w_n = weighted_halo[local_source_rows, :]

                main = (
                    parity[None, :]
                    * carrier_prefac
                    * Cn
                    * Tl_row[:, None]
                ).real * w_n
                main = jnp.where(
                    valid_n[:, None],
                    main,
                    jnp.zeros_like(main),
                )

                def add_sidebands(main_acc):
                    low = (
                        parity[:-1][None, :]
                        * low_phase[i]
                        * jnp.conj(Cnm[rows_safe, 1:])
                        * Cn[:, :-1]
                        * Tp_row[:, None]
                        * ph_mid
                    ).real * w_n[:, :-1]

                    up = (
                        parity[1:][None, :]
                        * up_phase[i]
                        * jnp.conj(Cnm[rows_safe, :-1])
                        * Cn[:, 1:]
                        * Tp_row[:, None]
                        * ph_mid
                    ).real * w_n[:, 1:]

                    low_pad = jnp.concatenate(
                        (zero_col, low),
                        axis=1,
                    )
                    up_pad = jnp.concatenate(
                        (up, zero_col),
                        axis=1,
                    )
                    side = low_pad + up_pad
                    side = jnp.where(
                        valid_n[:, None],
                        side,
                        jnp.zeros_like(side),
                    )
                    return main_acc + side

                chunk_next = jax.lax.cond(
                    Nm > 1,
                    add_sidebands,
                    lambda value: value,
                    main,
                )
                return chunk_out + chunk_next

            out_chunk = jax.lax.fori_loop(
                0,
                n_lag,
                lag_body,
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


_assemble_shift_target_weighted_chunked_impl_c128 = (
    _make_shift_target_weighted_chunked_impl(
        complex_dtype=jnp.complex128,
        real_dtype=jnp.float64,
    )
)

_assemble_shift_target_weighted_chunked_impl_c64 = (
    _make_shift_target_weighted_chunked_impl(
        complex_dtype=jnp.complex64,
        real_dtype=jnp.float32,
    )
)


def _make_shift_target_batch_weighted_chunked_impl(
    *,
    single_job_impl,
    complex_dtype,
):
    """Create the batched weighted-source row-chunked kernel."""

    def impl(
        source_coefficients,
        source_weights_batch,
        t_shift_batch,
        ell_all,
        offset,
        Tl_batch,
        Tp_batch,
        Cnm,
        dF,
        row_chunk_size,
    ):
        B = source_weights_batch.shape[0]
        Nt = source_coefficients.shape[1]
        Nm = source_coefficients.shape[2]
        out0 = jnp.zeros((B, Nt, Nm), dtype=complex_dtype)

        def body(i, out):
            shifted = single_job_impl(
                source_coefficients,
                source_weights_batch[i],
                t_shift_batch[i],
                ell_all,
                offset,
                Tl_batch[i],
                Tp_batch[i],
                Cnm,
                dF,
                row_chunk_size,
            )
            return out.at[i, :, :].set(shifted)

        return jax.lax.fori_loop(0, B, body, out0)

    return impl


_assemble_shift_target_batch_weighted_chunked_impl_c128 = (
    _make_shift_target_batch_weighted_chunked_impl(
        single_job_impl=(
            _assemble_shift_target_weighted_chunked_impl_c128
        ),
        complex_dtype=jnp.complex128,
    )
)

_assemble_shift_target_batch_weighted_chunked_impl_c64 = (
    _make_shift_target_batch_weighted_chunked_impl(
        single_job_impl=(
            _assemble_shift_target_weighted_chunked_impl_c64
        ),
        complex_dtype=jnp.complex64,
    )
)


_assemble_shift_target_batch_weighted_chunked_core_c128 = jax.jit(
    _assemble_shift_target_batch_weighted_chunked_impl_c128,
    static_argnums=(4, 9),
)

_assemble_shift_target_batch_weighted_chunked_core_c64 = jax.jit(
    _assemble_shift_target_batch_weighted_chunked_impl_c64,
    static_argnums=(4, 9),
)


def assemble_shift_target_batch_weighted_chunked_jax(
    source_coefficients,
    source_weights_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    dF,
    row_chunk_size=128,
    precision="complex128",
    return_device=False,
):
    """Shift job-specific weighted combinations of shared source arrays.

    Parameters
    ----------
    source_coefficients : array-like, shape (num_sources, Nt, Nm)
        Shared source coefficient grids.
    source_weights_batch : array-like, shape (num_jobs, num_sources, Nt, Nm)
        Job-specific weights applied to the shared sources.

    Notes
    -----
    The full ``(num_jobs, Nt, Nm)`` weighted coefficient batch is never
    constructed. Weighted source rows are formed inside each target-row/lag
    block immediately before shift accumulation.
    """

    precision = _normalize_chunked_precision(precision)
    row_chunk_size = int(row_chunk_size)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")

    source_coefficients = jnp.asarray(source_coefficients)
    source_weights_batch = jnp.asarray(source_weights_batch)

    if source_coefficients.ndim != 3:
        raise ValueError(
            "source_coefficients must have shape "
            "(num_sources, Nt, Nm)."
        )
    if source_weights_batch.ndim != 4:
        raise ValueError(
            "source_weights_batch must have shape "
            "(num_jobs, num_sources, Nt, Nm)."
        )

    num_sources, Nt, Nm = source_coefficients.shape
    num_jobs = source_weights_batch.shape[0]
    expected_weights = (num_jobs, num_sources, Nt, Nm)
    if tuple(source_weights_batch.shape) != expected_weights:
        raise ValueError(
            f"Expected source_weights_batch shape {expected_weights}, "
            f"got {tuple(source_weights_batch.shape)}."
        )
    if num_sources < 1:
        raise ValueError("At least one weighted source is required.")

    if precision == "complex64":
        out = _assemble_shift_target_batch_weighted_chunked_core_c64(
            source_coefficients,
            source_weights_batch,
            jnp.asarray(t_shift_batch, dtype=jnp.float32),
            jnp.asarray(ell_all, dtype=jnp.int64),
            int(offset),
            jnp.asarray(Tl_batch, dtype=jnp.complex64),
            jnp.asarray(Tp_batch, dtype=jnp.complex64),
            jnp.asarray(Cnm, dtype=jnp.complex64),
            jnp.asarray(dF, dtype=jnp.float32),
            row_chunk_size,
        )
        return out if return_device else np.asarray(out)

    out = _assemble_shift_target_batch_weighted_chunked_core_c128(
        source_coefficients,
        source_weights_batch,
        jnp.asarray(t_shift_batch, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_batch, dtype=jnp.complex128),
        jnp.asarray(Tp_batch, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        row_chunk_size,
    )
    return out if return_device else np.asarray(out)



def _make_shift_target_weighted_chunked_lagblock_impl(
    *,
    complex_dtype,
    real_dtype,
):
    """Create one weighted-source row-chunked, lag-blocked kernel."""

    def impl(
        source_coefficients,
        source_weights,
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
        # Preserve source/weight precision during modulation, then cast the
        # combined halo to the requested shift-assembly precision.
        source_coefficients = jnp.asarray(source_coefficients)
        source_weights = jnp.asarray(source_weights)
        t_shift = jnp.asarray(t_shift, dtype=real_dtype)
        ell_all = jnp.asarray(ell_all, dtype=jnp.int64)
        Tl_all = jnp.asarray(Tl_all, dtype=complex_dtype)
        Tp_all = jnp.asarray(Tp_all, dtype=complex_dtype)
        Cnm = jnp.asarray(Cnm, dtype=complex_dtype)
        dF = jnp.asarray(dF, dtype=real_dtype)

        Nt = t_shift.shape[0]
        Nm = source_coefficients.shape[-1]
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
        ph_mid_all = ph_m_all[:, :-1] * half_bin_phase[:, None]

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

        n_chunks = (Nt + row_chunk_size - 1) // row_chunk_size
        Nt_pad = n_chunks * row_chunk_size

        out0 = jnp.zeros((Nt_pad, Nm), dtype=complex_dtype)
        zero_col = jnp.zeros((row_chunk_size, 1), dtype=complex_dtype)

        row_offsets = jnp.arange(row_chunk_size, dtype=jnp.int64)
        halo_size = row_chunk_size + 2 * offset
        halo_offsets = jnp.arange(halo_size, dtype=jnp.int64)

        def chunk_body(chunk_id, out_acc):
            start = chunk_id * row_chunk_size
            rows = start + row_offsets
            valid_row = rows < Nt
            rows_safe = jnp.clip(rows, 0, Nt - 1)

            carrier_prefac = (
                jnp.conj(Cnm[rows_safe, :])
                * ph_m_all[rows_safe, :]
            )
            ph_mid = ph_mid_all[rows_safe, :]

            # Build the weighted source only once for all source rows that can
            # be reached by this target-row chunk. Every lag block reuses it.
            halo_start = start - offset
            halo_rows = halo_start + halo_offsets
            halo_rows_safe = jnp.clip(halo_rows, 0, Nt - 1)
            weighted_halo = _combine_weighted_source_rows(
                source_coefficients,
                source_weights,
                halo_rows_safe,
            ).astype(complex_dtype)

            out_chunk0 = jnp.zeros(
                (row_chunk_size, Nm),
                dtype=complex_dtype,
            )

            def lag_block_body(lag_block_id, chunk_out):
                lag_start = lag_block_id * lag_block_size
                lag_indices = lag_start + jnp.arange(
                    lag_block_size,
                    dtype=jnp.int64,
                )
                valid_lag = lag_indices < n_lag
                lag_indices_safe = jnp.clip(
                    lag_indices,
                    0,
                    n_lag - 1,
                )

                ell_block = ell_all[lag_indices_safe]
                parity_block = parity_all[lag_indices_safe]
                j_neg_block = -ell_block + offset
                low_phase_block = low_phase[lag_indices_safe]
                up_phase_block = up_phase[lag_indices_safe]

                nprime = rows[:, None] + ell_block[None, :]
                valid_n = (
                    valid_row[:, None]
                    & valid_lag[None, :]
                    & (nprime >= 0)
                    & (nprime < Nt)
                )
                nprime_safe = jnp.clip(nprime, 0, Nt - 1)

                Tl_row_block = Tl_all[rows_safe, :][:, j_neg_block]
                Tp_row_block = Tp_all[rows_safe, :][:, j_neg_block]
                Cn_block = Cnm[nprime_safe, :]

                # nprime - halo_start = row_offset + ell + offset.
                local_source_rows = (
                    row_offsets[:, None]
                    + ell_block[None, :]
                    + offset
                )
                w_n_block = weighted_halo[local_source_rows, :]

                main = (
                    parity_block[None, :, :]
                    * carrier_prefac[:, None, :]
                    * Cn_block
                    * Tl_row_block[:, :, None]
                ).real * w_n_block
                main = jnp.where(
                    valid_n[:, :, None],
                    main,
                    jnp.zeros_like(main),
                )
                main_sum = jnp.sum(main, axis=1)

                def add_sidebands(main_acc):
                    low = (
                        parity_block[:, :-1][None, :, :]
                        * low_phase_block[None, :, None]
                        * jnp.conj(Cnm[rows_safe, 1:])[:, None, :]
                        * Cn_block[:, :, :-1]
                        * Tp_row_block[:, :, None]
                        * ph_mid[:, None, :]
                    ).real * w_n_block[:, :, :-1]

                    up = (
                        parity_block[:, 1:][None, :, :]
                        * up_phase_block[None, :, None]
                        * jnp.conj(Cnm[rows_safe, :-1])[:, None, :]
                        * Cn_block[:, :, 1:]
                        * Tp_row_block[:, :, None]
                        * ph_mid[:, None, :]
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

                    low_sum = jnp.sum(low, axis=1)
                    up_sum = jnp.sum(up, axis=1)

                    low_pad = jnp.concatenate(
                        (zero_col, low_sum),
                        axis=1,
                    )
                    up_pad = jnp.concatenate(
                        (up_sum, zero_col),
                        axis=1,
                    )
                    return main_acc + low_pad + up_pad

                chunk_next = jax.lax.cond(
                    Nm > 1,
                    add_sidebands,
                    lambda value: value,
                    main_sum,
                )
                return chunk_out + chunk_next

            n_lag_blocks = (
                n_lag + lag_block_size - 1
            ) // lag_block_size
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


_assemble_shift_target_weighted_chunked_lagblock_impl_c128 = (
    _make_shift_target_weighted_chunked_lagblock_impl(
        complex_dtype=jnp.complex128,
        real_dtype=jnp.float64,
    )
)

_assemble_shift_target_weighted_chunked_lagblock_impl_c64 = (
    _make_shift_target_weighted_chunked_lagblock_impl(
        complex_dtype=jnp.complex64,
        real_dtype=jnp.float32,
    )
)


def _make_shift_target_batch_weighted_chunked_lagblock_impl(
    *,
    single_job_impl,
    complex_dtype,
):
    """Create the batched weighted-source lag-blocked kernel."""

    def impl(
        source_coefficients,
        source_weights_batch,
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
        B = source_weights_batch.shape[0]
        Nt = source_coefficients.shape[1]
        Nm = source_coefficients.shape[2]
        out0 = jnp.zeros((B, Nt, Nm), dtype=complex_dtype)

        def body(i, out):
            shifted = single_job_impl(
                source_coefficients,
                source_weights_batch[i],
                t_shift_batch[i],
                ell_all,
                offset,
                Tl_batch[i],
                Tp_batch[i],
                Cnm,
                dF,
                row_chunk_size,
                lag_block_size,
            )
            return out.at[i, :, :].set(shifted)

        return jax.lax.fori_loop(0, B, body, out0)

    return impl


_assemble_shift_target_batch_weighted_chunked_lagblock_impl_c128 = (
    _make_shift_target_batch_weighted_chunked_lagblock_impl(
        single_job_impl=(
            _assemble_shift_target_weighted_chunked_lagblock_impl_c128
        ),
        complex_dtype=jnp.complex128,
    )
)

_assemble_shift_target_batch_weighted_chunked_lagblock_impl_c64 = (
    _make_shift_target_batch_weighted_chunked_lagblock_impl(
        single_job_impl=(
            _assemble_shift_target_weighted_chunked_lagblock_impl_c64
        ),
        complex_dtype=jnp.complex64,
    )
)


_assemble_shift_target_batch_weighted_chunked_lagblock_core_c128 = jax.jit(
    _assemble_shift_target_batch_weighted_chunked_lagblock_impl_c128,
    static_argnums=(4, 9, 10),
)

_assemble_shift_target_batch_weighted_chunked_lagblock_core_c64 = jax.jit(
    _assemble_shift_target_batch_weighted_chunked_lagblock_impl_c64,
    static_argnums=(4, 9, 10),
)


def assemble_shift_target_batch_weighted_chunked_lagblock_jax(
    source_coefficients,
    source_weights_batch,
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
    """Shift weighted shared sources with row-chunked lag blocking."""

    precision = _normalize_chunked_precision(precision)
    row_chunk_size = int(row_chunk_size)
    lag_block_size = int(lag_block_size)

    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")
    if lag_block_size < 1:
        raise ValueError("lag_block_size must be >= 1.")

    source_coefficients = jnp.asarray(source_coefficients)
    source_weights_batch = jnp.asarray(source_weights_batch)

    if source_coefficients.ndim != 3:
        raise ValueError(
            "source_coefficients must have shape "
            "(num_sources, Nt, Nm)."
        )
    if source_weights_batch.ndim != 4:
        raise ValueError(
            "source_weights_batch must have shape "
            "(num_jobs, num_sources, Nt, Nm)."
        )

    num_sources, Nt, Nm = source_coefficients.shape
    num_jobs = source_weights_batch.shape[0]
    expected_weights = (num_jobs, num_sources, Nt, Nm)

    if tuple(source_weights_batch.shape) != expected_weights:
        raise ValueError(
            f"Expected source_weights_batch shape {expected_weights}, "
            f"got {tuple(source_weights_batch.shape)}."
        )
    if num_sources < 1:
        raise ValueError("At least one weighted source is required.")

    if precision == "complex64":
        out = (
            _assemble_shift_target_batch_weighted_chunked_lagblock_core_c64(
                source_coefficients,
                source_weights_batch,
                jnp.asarray(t_shift_batch, dtype=jnp.float32),
                jnp.asarray(ell_all, dtype=jnp.int64),
                int(offset),
                jnp.asarray(Tl_batch, dtype=jnp.complex64),
                jnp.asarray(Tp_batch, dtype=jnp.complex64),
                jnp.asarray(Cnm, dtype=jnp.complex64),
                jnp.asarray(dF, dtype=jnp.float32),
                row_chunk_size,
                lag_block_size,
            )
        )
        return out if return_device else np.asarray(out)

    out = _assemble_shift_target_batch_weighted_chunked_lagblock_core_c128(
        source_coefficients,
        source_weights_batch,
        jnp.asarray(t_shift_batch, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_batch, dtype=jnp.complex128),
        jnp.asarray(Tp_batch, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        row_chunk_size,
        lag_block_size,
    )
    return out if return_device else np.asarray(out)

def _assemble_shift_target_chunked_lagblock_impl_c128(
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
    """Chunked lag-blocked target-mode assembly (complex128/float64).
    
    Processes rows in chunks and lags in blocks within each chunk.
    """
    w_xi = jnp.asarray(w_xi, dtype=jnp.complex128)
    t_shift = jnp.asarray(t_shift, dtype=jnp.float64)
    ell_all = jnp.asarray(ell_all, dtype=jnp.int64)
    Tl_all = jnp.asarray(Tl_all, dtype=jnp.complex128)
    Tp_all = jnp.asarray(Tp_all, dtype=jnp.complex128)
    Cnm = jnp.asarray(Cnm, dtype=jnp.complex128)
    dF = jnp.asarray(dF, dtype=jnp.float64)

    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]

    m_full = jnp.arange(Nm, dtype=jnp.float64)
    one_j = jnp.asarray(1j, dtype=jnp.complex128)
    two_pi = jnp.asarray(2.0 * np.pi, dtype=jnp.float64)
    pi = jnp.asarray(np.pi, dtype=jnp.float64)

    # Full-bin phases:
    # exp(i 2 pi m dF Delta)
    ph_m_all = jnp.exp(
        one_j
        * two_pi
        * (m_full[None, :] * dF)
        * t_shift[:, None]
    )

    # Half-bin correction:
    # exp(i pi dF Delta)
    half_bin_phase = jnp.exp(
        one_j
        * pi
        * dF
        * t_shift
    )

    # exp[i 2 pi (m + 1/2) dF Delta]
    # = exp(i 2 pi m dF Delta) exp(i pi dF Delta)
    ph_mid_all = (
        ph_m_all[:, :-1]
        * half_bin_phase[:, None]
    )

    minus1_to_m = jnp.where(
        (jnp.arange(Nm) % 2) == 0,
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(-1.0, dtype=jnp.float64),
    )
    ell_even = (ell_all % 2) == 0
    parity_all = jnp.where(
        ell_even[:, None],
        jnp.ones((n_lag, Nm), dtype=jnp.float64),
        minus1_to_m[None, :],
    )

    neg_i = jnp.asarray(-1j, dtype=jnp.complex128)
    pos_i = jnp.asarray(1j, dtype=jnp.complex128)
    low_phase = jnp.power(neg_i, -ell_all)
    up_phase = jnp.power(pos_i, -ell_all)

    n_chunks = (Nt + row_chunk_size - 1) // row_chunk_size
    Nt_pad = n_chunks * row_chunk_size

    out0 = jnp.zeros((Nt_pad, Nm), dtype=jnp.complex128)
    zero_col = jnp.zeros((row_chunk_size, 1), dtype=jnp.complex128)

    def chunk_body(chunk_id, out_acc):
        start = chunk_id * row_chunk_size
        rows = start + jnp.arange(row_chunk_size, dtype=jnp.int64)
        valid_row = rows < Nt
        rows_safe = jnp.clip(rows, 0, Nt - 1)

        carrier_prefac = jnp.conj(Cnm[rows_safe, :]) * ph_m_all[rows_safe, :]
        ph_mid = ph_mid_all[rows_safe, :]

        out_chunk0 = jnp.zeros((row_chunk_size, Nm), dtype=jnp.complex128)

        def lag_block_body(lag_block_id, chunk_out):
            lag_start = lag_block_id * lag_block_size
            lag_indices = lag_start + jnp.arange(lag_block_size, dtype=jnp.int64)
            valid_lag = lag_indices < n_lag
            lag_indices_safe = jnp.clip(lag_indices, 0, n_lag - 1)

            ell_block = ell_all[lag_indices_safe]
            parity_block = parity_all[lag_indices_safe]
            j_neg_block = -ell_block + offset
            low_phase_block = low_phase[lag_indices_safe]
            up_phase_block = up_phase[lag_indices_safe]

            nprime = rows[:, None] + ell_block[None, :]
            valid_n = valid_row[:, None] & valid_lag[None, :] & (nprime >= 0) & (nprime < Nt)
            nprime_safe = jnp.clip(nprime, 0, Nt - 1)

            Tl_row_block = Tl_all[rows_safe, :][:, j_neg_block]
            Tp_row_block = Tp_all[rows_safe, :][:, j_neg_block]
            Cn_block = Cnm[nprime_safe, :]
            w_n_block = w_xi[nprime_safe, :]

            main = (parity_block[None, :, :] * carrier_prefac[:, None, :] * Cn_block * Tl_row_block[:, :, None]).real * w_n_block
            main = jnp.where(valid_n[:, :, None], main, jnp.zeros_like(main))
            main_sum = jnp.sum(main, axis=1)

            def add_sidebands(main_acc):
                low = (
                    parity_block[:, :-1][None, :, :]
                    * low_phase_block[None, :, None]
                    * jnp.conj(Cnm[rows_safe, 1:])[:, None, :]
                    * Cn_block[:, :, :-1]
                    * Tp_row_block[:, :, None]
                    * ph_mid[:, None, :]
                ).real * w_n_block[:, :, :-1]

                up = (
                    parity_block[:, 1:][None, :, :]
                    * up_phase_block[None, :, None]
                    * jnp.conj(Cnm[rows_safe, :-1])[:, None, :]
                    * Cn_block[:, :, 1:]
                    * Tp_row_block[:, :, None]
                    * ph_mid[:, None, :]
                ).real * w_n_block[:, :, 1:]

                low = jnp.where(valid_n[:, :, None], low, jnp.zeros_like(low))
                up = jnp.where(valid_n[:, :, None], up, jnp.zeros_like(up))

                low_sum = jnp.sum(low, axis=1)
                up_sum = jnp.sum(up, axis=1)

                low_pad = jnp.concatenate((zero_col, low_sum), axis=1)
                up_pad = jnp.concatenate((up_sum, zero_col), axis=1)
                side = low_pad + up_pad
                return main_acc + side

            chunk_next = jax.lax.cond(Nm > 1, add_sidebands, lambda x: x, main_sum)
            return chunk_out + chunk_next

        n_lag_blocks = (n_lag + lag_block_size - 1) // lag_block_size
        out_chunk = jax.lax.fori_loop(0, n_lag_blocks, lag_block_body, out_chunk0)
        return jax.lax.dynamic_update_slice(out_acc, out_chunk, (start, 0))

    out_padded = jax.lax.fori_loop(0, n_chunks, chunk_body, out0)
    return out_padded[:Nt, :]


def _assemble_shift_target_chunked_lagblock_impl_c64(
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
    """Chunked lag-blocked target-mode assembly (complex64/float32).
    
    Processes rows in chunks and lags in blocks within each chunk.
    """
    w_xi = jnp.asarray(w_xi, dtype=jnp.complex64)
    t_shift = jnp.asarray(t_shift, dtype=jnp.float32)
    ell_all = jnp.asarray(ell_all, dtype=jnp.int64)
    Tl_all = jnp.asarray(Tl_all, dtype=jnp.complex64)
    Tp_all = jnp.asarray(Tp_all, dtype=jnp.complex64)
    Cnm = jnp.asarray(Cnm, dtype=jnp.complex64)
    dF = jnp.asarray(dF, dtype=jnp.float32)

    Nt, Nm = w_xi.shape
    n_lag = ell_all.shape[0]

    m_full = jnp.arange(Nm, dtype=jnp.float32)
    one_j = jnp.asarray(1j, dtype=jnp.complex64)
    two_pi = jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
    pi = jnp.asarray(np.pi, dtype=jnp.float32)

    # Full-bin phases:
    # exp(i 2 pi m dF Delta)
    ph_m_all = jnp.exp(
        one_j
        * two_pi
        * (m_full[None, :] * dF)
        * t_shift[:, None]
    )

    # Half-bin correction:
    # exp(i pi dF Delta)
    half_bin_phase = jnp.exp(
        one_j
        * pi
        * dF
        * t_shift
    )

    # exp[i 2 pi (m + 1/2) dF Delta]
    # = exp(i 2 pi m dF Delta) exp(i pi dF Delta)
    ph_mid_all = (
        ph_m_all[:, :-1]
        * half_bin_phase[:, None]
    )
    minus1_to_m = jnp.where(
        (jnp.arange(Nm) % 2) == 0,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(-1.0, dtype=jnp.float32),
    )
    ell_even = (ell_all % 2) == 0
    parity_all = jnp.where(
        ell_even[:, None],
        jnp.ones((n_lag, Nm), dtype=jnp.float32),
        minus1_to_m[None, :],
    )

    neg_i = jnp.asarray(-1j, dtype=jnp.complex64)
    pos_i = jnp.asarray(1j, dtype=jnp.complex64)
    low_phase = jnp.power(neg_i, -ell_all)
    up_phase = jnp.power(pos_i, -ell_all)

    n_chunks = (Nt + row_chunk_size - 1) // row_chunk_size
    Nt_pad = n_chunks * row_chunk_size

    out0 = jnp.zeros((Nt_pad, Nm), dtype=jnp.complex64)
    zero_col = jnp.zeros((row_chunk_size, 1), dtype=jnp.complex64)

    def chunk_body(chunk_id, out_acc):
        start = chunk_id * row_chunk_size
        rows = start + jnp.arange(row_chunk_size, dtype=jnp.int64)
        valid_row = rows < Nt
        rows_safe = jnp.clip(rows, 0, Nt - 1)

        carrier_prefac = jnp.conj(Cnm[rows_safe, :]) * ph_m_all[rows_safe, :]
        ph_mid = ph_mid_all[rows_safe, :]

        out_chunk0 = jnp.zeros((row_chunk_size, Nm), dtype=jnp.complex64)

        def lag_block_body(lag_block_id, chunk_out):
            lag_start = lag_block_id * lag_block_size
            lag_indices = lag_start + jnp.arange(lag_block_size, dtype=jnp.int64)
            valid_lag = lag_indices < n_lag
            lag_indices_safe = jnp.clip(lag_indices, 0, n_lag - 1)

            ell_block = ell_all[lag_indices_safe]
            parity_block = parity_all[lag_indices_safe]
            j_neg_block = -ell_block + offset
            low_phase_block = low_phase[lag_indices_safe]
            up_phase_block = up_phase[lag_indices_safe]

            nprime = rows[:, None] + ell_block[None, :]
            valid_n = valid_row[:, None] & valid_lag[None, :] & (nprime >= 0) & (nprime < Nt)
            nprime_safe = jnp.clip(nprime, 0, Nt - 1)

            Tl_row_block = Tl_all[rows_safe, :][:, j_neg_block]
            Tp_row_block = Tp_all[rows_safe, :][:, j_neg_block]
            Cn_block = Cnm[nprime_safe, :]
            w_n_block = w_xi[nprime_safe, :]

            main = (parity_block[None, :, :] * carrier_prefac[:, None, :] * Cn_block * Tl_row_block[:, :, None]).real * w_n_block
            main = jnp.where(valid_n[:, :, None], main, jnp.zeros_like(main))
            main_sum = jnp.sum(main, axis=1)

            def add_sidebands(main_acc):
                low = (
                    parity_block[:, :-1][None, :, :]
                    * low_phase_block[None, :, None]
                    * jnp.conj(Cnm[rows_safe, 1:])[:, None, :]
                    * Cn_block[:, :, :-1]
                    * Tp_row_block[:, :, None]
                    * ph_mid[:, None, :]
                ).real * w_n_block[:, :, :-1]

                up = (
                    parity_block[:, 1:][None, :, :]
                    * up_phase_block[None, :, None]
                    * jnp.conj(Cnm[rows_safe, :-1])[:, None, :]
                    * Cn_block[:, :, 1:]
                    * Tp_row_block[:, :, None]
                    * ph_mid[:, None, :]
                ).real * w_n_block[:, :, 1:]

                low = jnp.where(valid_n[:, :, None], low, jnp.zeros_like(low))
                up = jnp.where(valid_n[:, :, None], up, jnp.zeros_like(up))

                low_sum = jnp.sum(low, axis=1)
                up_sum = jnp.sum(up, axis=1)

                low_pad = jnp.concatenate((zero_col, low_sum), axis=1)
                up_pad = jnp.concatenate((up_sum, zero_col), axis=1)
                side = low_pad + up_pad
                return main_acc + side

            chunk_next = jax.lax.cond(Nm > 1, add_sidebands, lambda x: x, main_sum)
            return chunk_out + chunk_next

        n_lag_blocks = (n_lag + lag_block_size - 1) // lag_block_size
        out_chunk = jax.lax.fori_loop(0, n_lag_blocks, lag_block_body, out_chunk0)
        return jax.lax.dynamic_update_slice(out_acc, out_chunk, (start, 0))

    out_padded = jax.lax.fori_loop(0, n_chunks, chunk_body, out0)
    return out_padded[:Nt, :]


_assemble_shift_target_chunked_lagblock_core_c128 = jax.jit(
    _assemble_shift_target_chunked_lagblock_impl_c128,
    static_argnums=(3, 8, 9),
)

_assemble_shift_target_chunked_lagblock_core_c64 = jax.jit(
    _assemble_shift_target_chunked_lagblock_impl_c64,
    static_argnums=(3, 8, 9),
)


def _assemble_shift_target_batch_chunked_lagblock_impl_c128(
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
    """Batch chunked lag-blocked target-mode assembly (complex128/float64)."""
    B, Nt, Nm = w_xi_batch.shape
    out0 = jnp.zeros((B, Nt, Nm), dtype=jnp.complex128)

    def body(i, out):
        row = _assemble_shift_target_chunked_lagblock_impl_c128(
            w_xi_batch[i],
            t_shift_batch[i],
            ell_all,
            offset,
            Tl_batch[i],
            Tp_batch[i],
            Cnm,
            dF,
            row_chunk_size,
            lag_block_size,
        )
        return out.at[i, :, :].set(row)

    return jax.lax.fori_loop(0, B, body, out0)


def _assemble_shift_target_batch_chunked_lagblock_impl_c64(
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
    """Batch chunked lag-blocked target-mode assembly (complex64/float32)."""
    B, Nt, Nm = w_xi_batch.shape
    out0 = jnp.zeros((B, Nt, Nm), dtype=jnp.complex64)

    def body(i, out):
        row = _assemble_shift_target_chunked_lagblock_impl_c64(
            w_xi_batch[i],
            t_shift_batch[i],
            ell_all,
            offset,
            Tl_batch[i],
            Tp_batch[i],
            Cnm,
            dF,
            row_chunk_size,
            lag_block_size,
        )
        return out.at[i, :, :].set(row)

    return jax.lax.fori_loop(0, B, body, out0)


_assemble_shift_target_batch_chunked_lagblock_core_c128 = jax.jit(
    _assemble_shift_target_batch_chunked_lagblock_impl_c128,
    static_argnums=(3, 8, 9),
)

_assemble_shift_target_batch_chunked_lagblock_core_c64 = jax.jit(
    _assemble_shift_target_batch_chunked_lagblock_impl_c64,
    static_argnums=(3, 8, 9),
)


def assemble_shift_target_chunked_lagblock_jax(
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    dF,
    row_chunk_size=128,
    lag_block_size=1,
    precision="complex128",
):
    """Assemble target-mode variable-delay shift using chunked lag-blocked kernels.
    
    Parameters
    ----------
    row_chunk_size : int
        Number of rows to process together.
    lag_block_size : int
        Number of lags to process together within each row chunk.
    precision : {"complex128", "complex64", "float64", "float32"}
        Internal arithmetic precision.
    
    Returns
    -------
    numpy.ndarray
        Shifted WDM coefficients with the same shape as ``w_xi``.
    """
    precision = _normalize_chunked_precision(precision)
    row_chunk_size = int(row_chunk_size)
    lag_block_size = int(lag_block_size)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")
    if lag_block_size < 1:
        raise ValueError("lag_block_size must be >= 1.")

    if precision == "complex64":
        out = _assemble_shift_target_chunked_lagblock_core_c64(
            jnp.asarray(w_xi, dtype=jnp.complex64),
            jnp.asarray(t_shift, dtype=jnp.float32),
            jnp.asarray(ell_all, dtype=jnp.int64),
            int(offset),
            jnp.asarray(Tl_all, dtype=jnp.complex64),
            jnp.asarray(Tp_all, dtype=jnp.complex64),
            jnp.asarray(Cnm, dtype=jnp.complex64),
            jnp.asarray(dF, dtype=jnp.float32),
            row_chunk_size,
            lag_block_size,
        )
        return np.asarray(out)

    out = _assemble_shift_target_chunked_lagblock_core_c128(
        jnp.asarray(w_xi, dtype=jnp.complex128),
        jnp.asarray(t_shift, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_all, dtype=jnp.complex128),
        jnp.asarray(Tp_all, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        row_chunk_size,
        lag_block_size,
    )
    return np.asarray(out)


def assemble_shift_target_batch_chunked_lagblock_jax(
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
    """Assemble batched target-mode shifts using chunked lag-blocked kernels.
    
    Parameters
    ----------
    row_chunk_size : int
        Number of rows to process together.
    lag_block_size : int
        Number of lags to process together within each row chunk.
    precision : {"complex128", "complex64", "float64", "float32"}
        Internal arithmetic precision.
    
    Returns
    -------
    numpy.ndarray
        Shifted WDM coefficients with shape ``(B, Nt, Nm)``.
    """
    precision = _normalize_chunked_precision(precision)
    row_chunk_size = int(row_chunk_size)
    lag_block_size = int(lag_block_size)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")
    if lag_block_size < 1:
        raise ValueError("lag_block_size must be >= 1.")

    if precision == "complex64":
        out = _assemble_shift_target_batch_chunked_lagblock_core_c64(
            jnp.asarray(w_xi_batch, dtype=jnp.complex64),
            jnp.asarray(t_shift_batch, dtype=jnp.float32),
            jnp.asarray(ell_all, dtype=jnp.int64),
            int(offset),
            jnp.asarray(Tl_batch, dtype=jnp.complex64),
            jnp.asarray(Tp_batch, dtype=jnp.complex64),
            jnp.asarray(Cnm, dtype=jnp.complex64),
            jnp.asarray(dF, dtype=jnp.float32),
            row_chunk_size,
            lag_block_size,
        )
        return out if return_device else np.asarray(out)

    out = _assemble_shift_target_batch_chunked_lagblock_core_c128(
        jnp.asarray(w_xi_batch, dtype=jnp.complex128),
        jnp.asarray(t_shift_batch, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_batch, dtype=jnp.complex128),
        jnp.asarray(Tp_batch, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        row_chunk_size,
        lag_block_size,
    )
    return out if return_device else np.asarray(out)


def _assemble_shift_target_batch_chunked_lagblock_jobblock_impl_c128(
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
    job_block_size,
):
    """Batch chunked lag-blocked target-mode assembly with job-block vectorization (complex128/float64)."""
    B, Nt, Nm = w_xi_batch.shape
    job_block_size = int(max(1, job_block_size))
    
    # Pad batch to be a multiple of job_block_size for clean reshaping
    B_padded = ((B + job_block_size - 1) // job_block_size) * job_block_size
    if B_padded > B:
        pad_width = ((0, B_padded - B), (0, 0), (0, 0))
        w_xi_batch = jnp.pad(w_xi_batch, pad_width, mode='constant', constant_values=0.0)
        t_shift_batch = jnp.pad(t_shift_batch, ((0, B_padded - B), (0, 0)), mode='constant', constant_values=0.0)
        Tl_batch = jnp.pad(Tl_batch, ((0, B_padded - B), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
        Tp_batch = jnp.pad(Tp_batch, ((0, B_padded - B), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
    
    # Reshape to separate job blocks (n_job_blocks, job_block_size, ...)
    n_job_blocks = B_padded // job_block_size
    w_xi_blocks = w_xi_batch.reshape(n_job_blocks, job_block_size, Nt, Nm)
    t_shift_blocks = t_shift_batch.reshape(n_job_blocks, job_block_size, Nt)
    Tl_blocks = Tl_batch.reshape(n_job_blocks, job_block_size, Nt, -1)
    Tp_blocks = Tp_batch.reshape(n_job_blocks, job_block_size, Nt, -1)
    
    def process_one_block(w_xi_jb, t_shift_jb, Tl_jb, Tp_jb):
        """Process one job block: vmap over job_block_size jobs."""
        def one_job(w_xi, t_shift, Tl_all, Tp_all):
            return _assemble_shift_target_chunked_lagblock_impl_c128(
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
            )
        
        return jax.vmap(one_job)(w_xi_jb, t_shift_jb, Tl_jb, Tp_jb)
    
    # vmap over job blocks to process in parallel
    out_blocks = jax.vmap(process_one_block)(w_xi_blocks, t_shift_blocks, Tl_blocks, Tp_blocks)
    # out_blocks shape: (n_job_blocks, job_block_size, Nt, Nm)
    
    # Reshape back to (B_padded, Nt, Nm)
    result = out_blocks.reshape(B_padded, Nt, Nm)
    
    # Return only the original batch (strip padding)
    return result[:B, :, :]


def _assemble_shift_target_batch_chunked_lagblock_jobblock_impl_c64(
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
    job_block_size,
):
    """Batch chunked lag-blocked target-mode assembly with job-block vectorization (complex64/float32)."""
    B, Nt, Nm = w_xi_batch.shape
    job_block_size = int(max(1, job_block_size))
    
    # Pad batch to be a multiple of job_block_size for clean reshaping
    B_padded = ((B + job_block_size - 1) // job_block_size) * job_block_size
    if B_padded > B:
        pad_width = ((0, B_padded - B), (0, 0), (0, 0))
        w_xi_batch = jnp.pad(w_xi_batch, pad_width, mode='constant', constant_values=0.0)
        t_shift_batch = jnp.pad(t_shift_batch, ((0, B_padded - B), (0, 0)), mode='constant', constant_values=0.0)
        Tl_batch = jnp.pad(Tl_batch, ((0, B_padded - B), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
        Tp_batch = jnp.pad(Tp_batch, ((0, B_padded - B), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
    
    # Reshape to separate job blocks (n_job_blocks, job_block_size, ...)
    n_job_blocks = B_padded // job_block_size
    w_xi_blocks = w_xi_batch.reshape(n_job_blocks, job_block_size, Nt, Nm)
    t_shift_blocks = t_shift_batch.reshape(n_job_blocks, job_block_size, Nt)
    Tl_blocks = Tl_batch.reshape(n_job_blocks, job_block_size, Nt, -1)
    Tp_blocks = Tp_batch.reshape(n_job_blocks, job_block_size, Nt, -1)
    
    def process_one_block(w_xi_jb, t_shift_jb, Tl_jb, Tp_jb):
        """Process one job block: vmap over job_block_size jobs."""
        def one_job(w_xi, t_shift, Tl_all, Tp_all):
            return _assemble_shift_target_chunked_lagblock_impl_c64(
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
            )
        
        return jax.vmap(one_job)(w_xi_jb, t_shift_jb, Tl_jb, Tp_jb)
    
    # vmap over job blocks to process in parallel
    out_blocks = jax.vmap(process_one_block)(w_xi_blocks, t_shift_blocks, Tl_blocks, Tp_blocks)
    # out_blocks shape: (n_job_blocks, job_block_size, Nt, Nm)
    
    # Reshape back to (B_padded, Nt, Nm)
    result = out_blocks.reshape(B_padded, Nt, Nm)
    
    # Return only the original batch (strip padding)
    return result[:B, :, :]


_assemble_shift_target_batch_chunked_lagblock_jobblock_core_c128 = jax.jit(
    _assemble_shift_target_batch_chunked_lagblock_jobblock_impl_c128,
    static_argnums=(3, 8, 9, 10),
)

_assemble_shift_target_batch_chunked_lagblock_jobblock_core_c64 = jax.jit(
    _assemble_shift_target_batch_chunked_lagblock_jobblock_impl_c64,
    static_argnums=(3, 8, 9, 10),
)


def assemble_shift_target_batch_chunked_lagblock_jobblock_jax(
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
    job_block_size=1,
    precision="complex128",
    return_device=False,
):
    """Assemble batched target-mode shifts using job-block vectorized lag-blocked kernels.
    
    This backend fuses multiple jobs into a single JAX vmap computation to
    improve memory efficiency and reduce kernel overhead.
    
    Parameters
    ----------
    job_block_size : int
        Number of jobs to process together in one vmap batch.
    row_chunk_size : int
        Number of rows to process together within each job.
    lag_block_size : int
        Number of lags to process together within each row chunk.
    precision : {"complex128", "complex64", "float64", "float32"}
        Internal arithmetic precision.
    
    Returns
    -------
    numpy.ndarray
        Shifted WDM coefficients with shape ``(B, Nt, Nm)``.
    """
    precision = _normalize_chunked_precision(precision)
    row_chunk_size = int(row_chunk_size)
    lag_block_size = int(lag_block_size)
    job_block_size = int(max(1, job_block_size))
    
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")
    if lag_block_size < 1:
        raise ValueError("lag_block_size must be >= 1.")
    if job_block_size < 1:
        raise ValueError("job_block_size must be >= 1.")

    if precision == "complex64":
        out = _assemble_shift_target_batch_chunked_lagblock_jobblock_core_c64(
            jnp.asarray(w_xi_batch, dtype=jnp.complex64),
            jnp.asarray(t_shift_batch, dtype=jnp.float32),
            jnp.asarray(ell_all, dtype=jnp.int64),
            int(offset),
            jnp.asarray(Tl_batch, dtype=jnp.complex64),
            jnp.asarray(Tp_batch, dtype=jnp.complex64),
            jnp.asarray(Cnm, dtype=jnp.complex64),
            jnp.asarray(dF, dtype=jnp.float32),
            row_chunk_size,
            lag_block_size,
            job_block_size,
        )
        return out if return_device else np.asarray(out)

    out = _assemble_shift_target_batch_chunked_lagblock_jobblock_core_c128(
        jnp.asarray(w_xi_batch, dtype=jnp.complex128),
        jnp.asarray(t_shift_batch, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_batch, dtype=jnp.complex128),
        jnp.asarray(Tp_batch, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        row_chunk_size,
        lag_block_size,
        job_block_size,
    )
    return out if return_device else np.asarray(out)


def assemble_shift_target_jax(w_xi, t_shift, ell_all, offset, Tl_all, Tp_all, Cnm, dF, assembly_vmap=False):
    """Assemble target-mode variable-delay shift using JAX kernels.

    Returns
    -------
    numpy.ndarray
        Shifted WDM coefficients with the same shape as ``w_xi``.
    """
    out = _assemble_shift_target_core(
        jnp.asarray(w_xi, dtype=jnp.complex128),
        jnp.asarray(t_shift, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_all, dtype=jnp.complex128),
        jnp.asarray(Tp_all, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        float(dF),
        bool(assembly_vmap),
    )
    return np.asarray(out)


def assemble_shift_target_chunked_jax(
    w_xi,
    t_shift,
    ell_all,
    offset,
    Tl_all,
    Tp_all,
    Cnm,
    dF,
    row_chunk_size=128,
    precision="complex128",
):
    """Assemble target-mode variable-delay shift using chunked lag-first kernels."""
    precision = _normalize_chunked_precision(precision)
    row_chunk_size = int(row_chunk_size)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")

    if precision == "complex64":
        out = _assemble_shift_target_chunked_core_c64(
            jnp.asarray(w_xi, dtype=jnp.complex64),
            jnp.asarray(t_shift, dtype=jnp.float32),
            jnp.asarray(ell_all, dtype=jnp.int64),
            int(offset),
            jnp.asarray(Tl_all, dtype=jnp.complex64),
            jnp.asarray(Tp_all, dtype=jnp.complex64),
            jnp.asarray(Cnm, dtype=jnp.complex64),
            jnp.asarray(dF, dtype=jnp.float32),
            row_chunk_size,
        )
        return np.asarray(out)

    out = _assemble_shift_target_chunked_core_c128(
        jnp.asarray(w_xi, dtype=jnp.complex128),
        jnp.asarray(t_shift, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_all, dtype=jnp.complex128),
        jnp.asarray(Tp_all, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        row_chunk_size,
    )
    return np.asarray(out)


def _assemble_shift_target_batch_impl(w_xi_batch, t_shift_batch, ell_all, offset, Tl_batch, Tp_batch, Cnm, dF, assembly_vmap):
    """JIT kernel for batched target-mode variable-delay assembly.

    Parameters
    ----------
    w_xi_batch : jax.Array
        Batch of WDM arrays with shape ``(B, Nt, Nm)``.
    t_shift_batch : jax.Array
        Batch of delay arrays with shape ``(B, Nt)``.
    Tl_batch, Tp_batch : jax.Array
        Batch of precomputed ``Tl/Tp`` arrays with shape ``(B, Nt, n_lag)``.
    assembly_vmap : bool
        Controls row mapping strategy inside each item (see
        ``assemble_shift_target_jax``).
    """
    B, Nt, Nm = w_xi_batch.shape

    if assembly_vmap:
        def one_item(w_xi, t_shift, Tl_all, Tp_all):
            return _assemble_shift_target_impl(
                w_xi,
                t_shift,
                ell_all,
                offset,
                Tl_all,
                Tp_all,
                Cnm,
                dF,
                True,
            )

        return jax.vmap(one_item)(w_xi_batch, t_shift_batch, Tl_batch, Tp_batch)

    def body(i, out):
        row = _assemble_shift_target_impl(
            w_xi_batch[i],
            t_shift_batch[i],
            ell_all,
            offset,
            Tl_batch[i],
            Tp_batch[i],
            Cnm,
            dF,
            False,
        )
        return out.at[i, :, :].set(row)

    out0 = jnp.zeros((B, Nt, Nm), dtype=jnp.complex128)
    return jax.lax.fori_loop(0, B, body, out0)


_assemble_shift_target_batch_core = jax.jit(_assemble_shift_target_batch_impl, static_argnums=(3, 8))


def assemble_shift_target_batch_jax(
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    dF,
    assembly_vmap=False,
    return_device=False,
):
    """Assemble batched target-mode variable-delay shifts using JAX kernels.

    Returns
    -------
    numpy.ndarray
        Shifted WDM coefficients with shape ``(B, Nt, Nm)``.
    """
    out = _assemble_shift_target_batch_core(
        jnp.asarray(w_xi_batch, dtype=jnp.complex128),
        jnp.asarray(t_shift_batch, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_batch, dtype=jnp.complex128),
        jnp.asarray(Tp_batch, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        float(dF),
        bool(assembly_vmap),
    )
    return out if return_device else np.asarray(out)


def assemble_shift_target_batch_chunked_jax(
    w_xi_batch,
    t_shift_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    dF,
    row_chunk_size=128,
    precision="complex128",
    return_device=False,
):
    """Assemble batched target-mode shifts using chunked lag-first kernels."""
    precision = _normalize_chunked_precision(precision)
    row_chunk_size = int(row_chunk_size)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")

    if precision == "complex64":
        out = _assemble_shift_target_batch_chunked_core_c64(
            jnp.asarray(w_xi_batch, dtype=jnp.complex64),
            jnp.asarray(t_shift_batch, dtype=jnp.float32),
            jnp.asarray(ell_all, dtype=jnp.int64),
            int(offset),
            jnp.asarray(Tl_batch, dtype=jnp.complex64),
            jnp.asarray(Tp_batch, dtype=jnp.complex64),
            jnp.asarray(Cnm, dtype=jnp.complex64),
            jnp.asarray(dF, dtype=jnp.float32),
            row_chunk_size,
        )
        return out if return_device else np.asarray(out)

    out = _assemble_shift_target_batch_chunked_core_c128(
        jnp.asarray(w_xi_batch, dtype=jnp.complex128),
        jnp.asarray(t_shift_batch, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_batch, dtype=jnp.complex128),
        jnp.asarray(Tp_batch, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        jnp.asarray(dF, dtype=jnp.float64),
        row_chunk_size,
    )
    return out if return_device else np.asarray(out)


def assemble_shift_fixed_jax(w_xi, delta, ell_all, offset, Tl_vec, Tp_vec, Cnm, dF, assembly_vmap=False):
    """Assemble fixed-delay shift using JAX kernels.

    Returns
    -------
    numpy.ndarray
        Shifted WDM coefficients with the same shape as ``w_xi``.
    """
    out = _assemble_shift_fixed_core(
        jnp.asarray(w_xi, dtype=jnp.complex128),
        jnp.asarray(delta, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_vec, dtype=jnp.complex128),
        jnp.asarray(Tp_vec, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        float(dF),
        bool(assembly_vmap),
    )
    return np.asarray(out)


def assemble_shift_variable_mode_jax(w_xi, t_shift, ell_all, offset, Tl_all, Tp_all, Cnm, dF, delta_mode, assembly_vmap=False):
    """Assemble variable-delay shift for ``target``, ``source``, or ``midpoint``.

    Parameters
    ----------
    delta_mode : str
        Delay-selection mode. Must be one of ``'target'``, ``'source'``, or
        ``'midpoint'``.
    """
    mode_map = {"target": 0, "source": 1, "midpoint": 2}
    if delta_mode not in mode_map:
        raise ValueError("delta_mode must be 'target', 'source', or 'midpoint'")

    out = _assemble_shift_variable_mode_core(
        jnp.asarray(w_xi, dtype=jnp.complex128),
        jnp.asarray(t_shift, dtype=jnp.float64),
        jnp.asarray(ell_all, dtype=jnp.int64),
        int(offset),
        jnp.asarray(Tl_all, dtype=jnp.complex128),
        jnp.asarray(Tp_all, dtype=jnp.complex128),
        jnp.asarray(Cnm, dtype=jnp.complex128),
        float(dF),
        int(mode_map[delta_mode]),
        bool(assembly_vmap),
    )
    return np.asarray(out)

# ---------------------------------------------------------------------------
# Experimental prephased lag-blocked backend
# ---------------------------------------------------------------------------


def _make_shift_target_chunked_lagblock_prephased_impl(
    *,
    complex_dtype,
    real_dtype,
):
    """Create one lag-blocked kernel that receives cached phase arrays."""

    def impl(
        w_xi,
        ph_m_all,
        half_bin_phase,
        ell_all,
        offset,
        Tl_all,
        Tp_all,
        Cnm,
        row_chunk_size,
        lag_block_size,
    ):
        w_xi = jnp.asarray(w_xi, dtype=complex_dtype)
        ph_m_all = jnp.asarray(ph_m_all, dtype=complex_dtype)
        half_bin_phase = jnp.asarray(
            half_bin_phase,
            dtype=complex_dtype,
        )
        ell_all = jnp.asarray(ell_all, dtype=jnp.int64)
        Tl_all = jnp.asarray(Tl_all, dtype=complex_dtype)
        Tp_all = jnp.asarray(Tp_all, dtype=complex_dtype)
        Cnm = jnp.asarray(Cnm, dtype=complex_dtype)

        Nt, Nm = w_xi.shape
        n_lag = ell_all.shape[0]

        expected_phase_shape = (Nt, Nm)
        if ph_m_all.shape != expected_phase_shape:
            raise ValueError(
                "ph_m_all must have shape "
                f"{expected_phase_shape}; got {ph_m_all.shape}."
            )
        if half_bin_phase.shape != (Nt,):
            raise ValueError(
                "half_bin_phase must have shape "
                f"{(Nt,)}; got {half_bin_phase.shape}."
            )

        # The sideband phase is cheap to form from the two cached factors and
        # avoids storing a second nearly full-size complex phase grid.
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

        n_chunks = (Nt + row_chunk_size - 1) // row_chunk_size
        Nt_pad = n_chunks * row_chunk_size

        out0 = jnp.zeros((Nt_pad, Nm), dtype=complex_dtype)
        zero_col = jnp.zeros(
            (row_chunk_size, 1),
            dtype=complex_dtype,
        )
        row_offsets = jnp.arange(row_chunk_size, dtype=jnp.int64)
        lag_offsets = jnp.arange(lag_block_size, dtype=jnp.int64)

        def chunk_body(chunk_id, out_acc):
            start = chunk_id * row_chunk_size
            rows = start + row_offsets
            valid_row = rows < Nt
            rows_safe = jnp.clip(rows, 0, Nt - 1)

            carrier_prefac = (
                jnp.conj(Cnm[rows_safe, :])
                * ph_m_all[rows_safe, :]
            )
            ph_mid = ph_mid_all[rows_safe, :]

            out_chunk0 = jnp.zeros(
                (row_chunk_size, Nm),
                dtype=complex_dtype,
            )

            def lag_block_body(lag_block_id, chunk_out):
                lag_start = lag_block_id * lag_block_size
                lag_indices = lag_start + lag_offsets
                valid_lag = lag_indices < n_lag
                lag_indices_safe = jnp.clip(
                    lag_indices,
                    0,
                    n_lag - 1,
                )

                ell_block = ell_all[lag_indices_safe]
                parity_block = parity_all[lag_indices_safe]
                j_neg_block = -ell_block + offset
                low_phase_block = low_phase[lag_indices_safe]
                up_phase_block = up_phase[lag_indices_safe]

                nprime = rows[:, None] + ell_block[None, :]
                valid_n = (
                    valid_row[:, None]
                    & valid_lag[None, :]
                    & (nprime >= 0)
                    & (nprime < Nt)
                )
                nprime_safe = jnp.clip(nprime, 0, Nt - 1)

                Tl_row_block = Tl_all[rows_safe, :][
                    :, j_neg_block
                ]
                Tp_row_block = Tp_all[rows_safe, :][
                    :, j_neg_block
                ]
                Cn_block = Cnm[nprime_safe, :]
                w_n_block = w_xi[nprime_safe, :]

                main = (
                    parity_block[None, :, :]
                    * carrier_prefac[:, None, :]
                    * Cn_block
                    * Tl_row_block[:, :, None]
                ).real * w_n_block
                main = jnp.where(
                    valid_n[:, :, None],
                    main,
                    jnp.zeros_like(main),
                )
                main_sum = jnp.sum(main, axis=1)

                def add_sidebands(main_acc):
                    low = (
                        parity_block[:, :-1][None, :, :]
                        * low_phase_block[None, :, None]
                        * jnp.conj(Cnm[rows_safe, 1:])[:, None, :]
                        * Cn_block[:, :, :-1]
                        * Tp_row_block[:, :, None]
                        * ph_mid[:, None, :]
                    ).real * w_n_block[:, :, :-1]

                    up = (
                        parity_block[:, 1:][None, :, :]
                        * up_phase_block[None, :, None]
                        * jnp.conj(Cnm[rows_safe, :-1])[:, None, :]
                        * Cn_block[:, :, 1:]
                        * Tp_row_block[:, :, None]
                        * ph_mid[:, None, :]
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

                    low_sum = jnp.sum(low, axis=1)
                    up_sum = jnp.sum(up, axis=1)
                    low_pad = jnp.concatenate(
                        (zero_col, low_sum),
                        axis=1,
                    )
                    up_pad = jnp.concatenate(
                        (up_sum, zero_col),
                        axis=1,
                    )
                    return main_acc + low_pad + up_pad

                chunk_next = jax.lax.cond(
                    Nm > 1,
                    add_sidebands,
                    lambda value: value,
                    main_sum,
                )
                return chunk_out + chunk_next

            n_lag_blocks = (
                n_lag + lag_block_size - 1
            ) // lag_block_size
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


_assemble_shift_target_chunked_lagblock_prephased_impl_c128 = (
    _make_shift_target_chunked_lagblock_prephased_impl(
        complex_dtype=jnp.complex128,
        real_dtype=jnp.float64,
    )
)
_assemble_shift_target_chunked_lagblock_prephased_impl_c64 = (
    _make_shift_target_chunked_lagblock_prephased_impl(
        complex_dtype=jnp.complex64,
        real_dtype=jnp.float32,
    )
)


def _make_shift_target_batch_chunked_lagblock_prephased_impl(
    *,
    single_job_impl,
    complex_dtype,
):
    """Create the batched prephased lag-blocked kernel."""

    def impl(
        w_xi_batch,
        ph_m_batch,
        half_bin_phase_batch,
        ell_all,
        offset,
        Tl_batch,
        Tp_batch,
        Cnm,
        row_chunk_size,
        lag_block_size,
    ):
        B, Nt, Nm = w_xi_batch.shape
        out0 = jnp.zeros((B, Nt, Nm), dtype=complex_dtype)

        def body(i, out):
            shifted = single_job_impl(
                w_xi_batch[i],
                ph_m_batch[i],
                half_bin_phase_batch[i],
                ell_all,
                offset,
                Tl_batch[i],
                Tp_batch[i],
                Cnm,
                row_chunk_size,
                lag_block_size,
            )
            return out.at[i, :, :].set(shifted)

        return jax.lax.fori_loop(0, B, body, out0)

    return impl


_assemble_shift_target_batch_chunked_lagblock_prephased_impl_c128 = (
    _make_shift_target_batch_chunked_lagblock_prephased_impl(
        single_job_impl=(
            _assemble_shift_target_chunked_lagblock_prephased_impl_c128
        ),
        complex_dtype=jnp.complex128,
    )
)
_assemble_shift_target_batch_chunked_lagblock_prephased_impl_c64 = (
    _make_shift_target_batch_chunked_lagblock_prephased_impl(
        single_job_impl=(
            _assemble_shift_target_chunked_lagblock_prephased_impl_c64
        ),
        complex_dtype=jnp.complex64,
    )
)


_assemble_shift_target_batch_chunked_lagblock_prephased_core_c128 = jax.jit(
    _assemble_shift_target_batch_chunked_lagblock_prephased_impl_c128,
    static_argnums=(4, 8, 9),
)
_assemble_shift_target_batch_chunked_lagblock_prephased_core_c64 = jax.jit(
    _assemble_shift_target_batch_chunked_lagblock_prephased_impl_c64,
    static_argnums=(4, 8, 9),
)


def assemble_shift_target_batch_chunked_lagblock_prephased_jax(
    w_xi_batch,
    ph_m_batch,
    half_bin_phase_batch,
    ell_all,
    offset,
    Tl_batch,
    Tp_batch,
    Cnm,
    row_chunk_size=128,
    lag_block_size=1,
    precision="complex128",
    return_device=False,
):
    """Assemble batched lag-blocked shifts from cached phase arrays.

    This experimental wrapper mirrors
    :func:`assemble_shift_target_batch_chunked_lagblock_jax`, except the
    waveform-independent full-bin and half-bin phase factors are supplied by
    the prepared plan instead of reconstructed inside every application.
    """

    precision = _normalize_chunked_precision(precision)
    row_chunk_size = int(row_chunk_size)
    lag_block_size = int(lag_block_size)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")
    if lag_block_size < 1:
        raise ValueError("lag_block_size must be >= 1.")

    if precision == "complex64":
        out = (
            _assemble_shift_target_batch_chunked_lagblock_prephased_core_c64(
                jnp.asarray(w_xi_batch, dtype=jnp.complex64),
                jnp.asarray(ph_m_batch, dtype=jnp.complex64),
                jnp.asarray(
                    half_bin_phase_batch,
                    dtype=jnp.complex64,
                ),
                jnp.asarray(ell_all, dtype=jnp.int64),
                int(offset),
                jnp.asarray(Tl_batch, dtype=jnp.complex64),
                jnp.asarray(Tp_batch, dtype=jnp.complex64),
                jnp.asarray(Cnm, dtype=jnp.complex64),
                row_chunk_size,
                lag_block_size,
            )
        )
        return out if return_device else np.asarray(out)

    out = (
        _assemble_shift_target_batch_chunked_lagblock_prephased_core_c128(
            jnp.asarray(w_xi_batch, dtype=jnp.complex128),
            jnp.asarray(ph_m_batch, dtype=jnp.complex128),
            jnp.asarray(
                half_bin_phase_batch,
                dtype=jnp.complex128,
            ),
            jnp.asarray(ell_all, dtype=jnp.int64),
            int(offset),
            jnp.asarray(Tl_batch, dtype=jnp.complex128),
            jnp.asarray(Tp_batch, dtype=jnp.complex128),
            jnp.asarray(Cnm, dtype=jnp.complex128),
            row_chunk_size,
            lag_block_size,
        )
    )
    return out if return_device else np.asarray(out)

# ---------------------------------------------------------------------------
# Experimental interior-row/source-halo lag-blocked backend
# ---------------------------------------------------------------------------


def _make_shift_target_chunked_lagblock_interior_impl(
    *,
    complex_dtype,
    real_dtype,
):
    """Create a lag-blocked kernel with an interior source-halo fast path.

    The production lag-blocked kernel applies row/source clipping and validity
    masks to every row chunk.  For a full chunk wholly contained in
    ``[offset, Nt - offset)``, every lagged source row is known to be valid.
    Such chunks use one contiguous source/Cnm halo and omit row-boundary
    clipping and masks. Boundary and padded chunks retain the validated general
    calculation.
    """

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
        ph_mid_all = ph_m_all[:, :-1] * half_bin_phase[:, None]

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

        n_chunks = (Nt + row_chunk_size - 1) // row_chunk_size
        Nt_pad = n_chunks * row_chunk_size

        out0 = jnp.zeros((Nt_pad, Nm), dtype=complex_dtype)
        zero_col = jnp.zeros(
            (row_chunk_size, 1),
            dtype=complex_dtype,
        )
        row_offsets = jnp.arange(row_chunk_size, dtype=jnp.int64)
        lag_offsets = jnp.arange(lag_block_size, dtype=jnp.int64)
        n_lag_blocks = (
            n_lag + lag_block_size - 1
        ) // lag_block_size

        # Static source-halo shape used only for full interior chunks.
        halo_size = row_chunk_size + 2 * offset

        def chunk_body(chunk_id, out_acc):
            start = chunk_id * row_chunk_size
            rows = start + row_offsets

            is_full_interior = (
                (start >= offset)
                & (start + row_chunk_size <= Nt - offset)
            )

            def general_chunk(_):
                """Validated boundary/padded calculation."""

                valid_row = rows < Nt
                rows_safe = jnp.clip(rows, 0, Nt - 1)
                carrier_prefac = (
                    jnp.conj(Cnm[rows_safe, :])
                    * ph_m_all[rows_safe, :]
                )
                ph_mid = ph_mid_all[rows_safe, :]

                out_chunk0 = jnp.zeros(
                    (row_chunk_size, Nm),
                    dtype=complex_dtype,
                )

                def lag_block_body(lag_block_id, chunk_out):
                    lag_start = lag_block_id * lag_block_size
                    lag_indices = lag_start + lag_offsets
                    valid_lag = lag_indices < n_lag
                    lag_indices_safe = jnp.clip(
                        lag_indices,
                        0,
                        n_lag - 1,
                    )

                    ell_block = ell_all[lag_indices_safe]
                    parity_block = parity_all[lag_indices_safe]
                    j_neg_block = -ell_block + offset
                    low_phase_block = low_phase[lag_indices_safe]
                    up_phase_block = up_phase[lag_indices_safe]

                    nprime = rows[:, None] + ell_block[None, :]
                    valid_n = (
                        valid_row[:, None]
                        & valid_lag[None, :]
                        & (nprime >= 0)
                        & (nprime < Nt)
                    )
                    nprime_safe = jnp.clip(nprime, 0, Nt - 1)

                    Tl_row_block = Tl_all[rows_safe, :][
                        :, j_neg_block
                    ]
                    Tp_row_block = Tp_all[rows_safe, :][
                        :, j_neg_block
                    ]
                    Cn_block = Cnm[nprime_safe, :]
                    w_n_block = w_xi[nprime_safe, :]

                    main = (
                        parity_block[None, :, :]
                        * carrier_prefac[:, None, :]
                        * Cn_block
                        * Tl_row_block[:, :, None]
                    ).real * w_n_block
                    main = jnp.where(
                        valid_n[:, :, None],
                        main,
                        jnp.zeros_like(main),
                    )
                    main_sum = jnp.sum(main, axis=1)

                    def add_sidebands(main_acc):
                        low = (
                            parity_block[:, :-1][None, :, :]
                            * low_phase_block[None, :, None]
                            * jnp.conj(Cnm[rows_safe, 1:])[:, None, :]
                            * Cn_block[:, :, :-1]
                            * Tp_row_block[:, :, None]
                            * ph_mid[:, None, :]
                        ).real * w_n_block[:, :, :-1]

                        up = (
                            parity_block[:, 1:][None, :, :]
                            * up_phase_block[None, :, None]
                            * jnp.conj(Cnm[rows_safe, :-1])[:, None, :]
                            * Cn_block[:, :, 1:]
                            * Tp_row_block[:, :, None]
                            * ph_mid[:, None, :]
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

                        low_sum = jnp.sum(low, axis=1)
                        up_sum = jnp.sum(up, axis=1)
                        low_pad = jnp.concatenate(
                            (zero_col, low_sum),
                            axis=1,
                        )
                        up_pad = jnp.concatenate(
                            (up_sum, zero_col),
                            axis=1,
                        )
                        return main_acc + low_pad + up_pad

                    chunk_next = jax.lax.cond(
                        Nm > 1,
                        add_sidebands,
                        lambda value: value,
                        main_sum,
                    )
                    return chunk_out + chunk_next

                return jax.lax.fori_loop(
                    0,
                    n_lag_blocks,
                    lag_block_body,
                    out_chunk0,
                )

            def interior_chunk(_):
                """Fast path for one full target chunk and its source halo."""

                carrier_prefac = (
                    jnp.conj(Cnm[rows, :])
                    * ph_m_all[rows, :]
                )
                ph_mid = ph_mid_all[rows, :]

                halo_start = start - offset
                source_halo = jax.lax.dynamic_slice(
                    w_xi,
                    (halo_start, 0),
                    (halo_size, Nm),
                )
                cnm_halo = jax.lax.dynamic_slice(
                    Cnm,
                    (halo_start, 0),
                    (halo_size, Nm),
                )

                out_chunk0 = jnp.zeros(
                    (row_chunk_size, Nm),
                    dtype=complex_dtype,
                )

                def lag_block_body(lag_block_id, chunk_out):
                    lag_start = lag_block_id * lag_block_size
                    lag_indices = lag_start + lag_offsets
                    valid_lag = lag_indices < n_lag
                    lag_indices_safe = jnp.clip(
                        lag_indices,
                        0,
                        n_lag - 1,
                    )

                    ell_block = ell_all[lag_indices_safe]
                    parity_block = parity_all[lag_indices_safe]
                    j_neg_block = -ell_block + offset
                    low_phase_block = low_phase[lag_indices_safe]
                    up_phase_block = up_phase[lag_indices_safe]

                    local_source_rows = (
                        row_offsets[:, None]
                        + ell_block[None, :]
                        + offset
                    )
                    Cn_block = cnm_halo[local_source_rows, :]
                    w_n_block = source_halo[local_source_rows, :]

                    Tl_row_block = Tl_all[rows, :][
                        :, j_neg_block
                    ]
                    Tp_row_block = Tp_all[rows, :][
                        :, j_neg_block
                    ]

                    main = (
                        parity_block[None, :, :]
                        * carrier_prefac[:, None, :]
                        * Cn_block
                        * Tl_row_block[:, :, None]
                    ).real * w_n_block
                    main = jnp.where(
                        valid_lag[None, :, None],
                        main,
                        jnp.zeros_like(main),
                    )
                    main_sum = jnp.sum(main, axis=1)

                    def add_sidebands(main_acc):
                        low = (
                            parity_block[:, :-1][None, :, :]
                            * low_phase_block[None, :, None]
                            * jnp.conj(Cnm[rows, 1:])[:, None, :]
                            * Cn_block[:, :, :-1]
                            * Tp_row_block[:, :, None]
                            * ph_mid[:, None, :]
                        ).real * w_n_block[:, :, :-1]

                        up = (
                            parity_block[:, 1:][None, :, :]
                            * up_phase_block[None, :, None]
                            * jnp.conj(Cnm[rows, :-1])[:, None, :]
                            * Cn_block[:, :, 1:]
                            * Tp_row_block[:, :, None]
                            * ph_mid[:, None, :]
                        ).real * w_n_block[:, :, 1:]

                        low = jnp.where(
                            valid_lag[None, :, None],
                            low,
                            jnp.zeros_like(low),
                        )
                        up = jnp.where(
                            valid_lag[None, :, None],
                            up,
                            jnp.zeros_like(up),
                        )

                        low_sum = jnp.sum(low, axis=1)
                        up_sum = jnp.sum(up, axis=1)
                        low_pad = jnp.concatenate(
                            (zero_col, low_sum),
                            axis=1,
                        )
                        up_pad = jnp.concatenate(
                            (up_sum, zero_col),
                            axis=1,
                        )
                        return main_acc + low_pad + up_pad

                    chunk_next = jax.lax.cond(
                        Nm > 1,
                        add_sidebands,
                        lambda value: value,
                        main_sum,
                    )
                    return chunk_out + chunk_next

                return jax.lax.fori_loop(
                    0,
                    n_lag_blocks,
                    lag_block_body,
                    out_chunk0,
                )

            out_chunk = jax.lax.cond(
                is_full_interior,
                interior_chunk,
                general_chunk,
                operand=None,
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


_assemble_shift_target_chunked_lagblock_interior_impl_c128 = (
    _make_shift_target_chunked_lagblock_interior_impl(
        complex_dtype=jnp.complex128,
        real_dtype=jnp.float64,
    )
)
_assemble_shift_target_chunked_lagblock_interior_impl_c64 = (
    _make_shift_target_chunked_lagblock_interior_impl(
        complex_dtype=jnp.complex64,
        real_dtype=jnp.float32,
    )
)


def _make_shift_target_batch_chunked_lagblock_interior_impl(
    *,
    single_job_impl,
    complex_dtype,
):
    """Create the batched interior-fast-path lag-blocked kernel."""

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
        out0 = jnp.zeros((B, Nt, Nm), dtype=complex_dtype)

        def body(i, out):
            shifted = single_job_impl(
                w_xi_batch[i],
                t_shift_batch[i],
                ell_all,
                offset,
                Tl_batch[i],
                Tp_batch[i],
                Cnm,
                dF,
                row_chunk_size,
                lag_block_size,
            )
            return out.at[i, :, :].set(shifted)

        return jax.lax.fori_loop(0, B, body, out0)

    return impl


_assemble_shift_target_batch_chunked_lagblock_interior_impl_c128 = (
    _make_shift_target_batch_chunked_lagblock_interior_impl(
        single_job_impl=(
            _assemble_shift_target_chunked_lagblock_interior_impl_c128
        ),
        complex_dtype=jnp.complex128,
    )
)
_assemble_shift_target_batch_chunked_lagblock_interior_impl_c64 = (
    _make_shift_target_batch_chunked_lagblock_interior_impl(
        single_job_impl=(
            _assemble_shift_target_chunked_lagblock_interior_impl_c64
        ),
        complex_dtype=jnp.complex64,
    )
)


_assemble_shift_target_batch_chunked_lagblock_interior_core_c128 = jax.jit(
    _assemble_shift_target_batch_chunked_lagblock_interior_impl_c128,
    static_argnums=(3, 8, 9),
)
_assemble_shift_target_batch_chunked_lagblock_interior_core_c64 = jax.jit(
    _assemble_shift_target_batch_chunked_lagblock_interior_impl_c64,
    static_argnums=(3, 8, 9),
)


def assemble_shift_target_batch_chunked_lagblock_interior_jax(
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
    """Experimental batched shift with an interior source-halo fast path.

    This does not replace the production lag-blocked wrapper.  If the chosen
    row chunk is too large to leave even one full interior chunk, the function
    falls back to the existing production wrapper.
    """

    precision = _normalize_chunked_precision(precision)
    row_chunk_size = int(row_chunk_size)
    lag_block_size = int(lag_block_size)
    offset = int(offset)

    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")
    if lag_block_size < 1:
        raise ValueError("lag_block_size must be >= 1.")
    if offset < 0:
        raise ValueError("offset must be >= 0.")

    shape = tuple(np.shape(w_xi_batch))
    if len(shape) != 3:
        raise ValueError(
            "w_xi_batch must have shape (num_jobs, Nt, Nm)."
        )
    Nt = int(shape[1])

    # No full target chunk can use the interior halo in this case. Avoid
    # compiling a dynamic_slice whose static halo size exceeds the operand.
    if row_chunk_size + 2 * offset > Nt:
        return assemble_shift_target_batch_chunked_lagblock_jax(
            w_xi_batch,
            t_shift_batch,
            ell_all,
            offset,
            Tl_batch,
            Tp_batch,
            Cnm,
            dF,
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            precision=precision,
            return_device=return_device,
        )

    if precision == "complex64":
        out = (
            _assemble_shift_target_batch_chunked_lagblock_interior_core_c64(
                jnp.asarray(w_xi_batch, dtype=jnp.complex64),
                jnp.asarray(t_shift_batch, dtype=jnp.float32),
                jnp.asarray(ell_all, dtype=jnp.int64),
                offset,
                jnp.asarray(Tl_batch, dtype=jnp.complex64),
                jnp.asarray(Tp_batch, dtype=jnp.complex64),
                jnp.asarray(Cnm, dtype=jnp.complex64),
                jnp.asarray(dF, dtype=jnp.float32),
                row_chunk_size,
                lag_block_size,
            )
        )
        return out if return_device else np.asarray(out)

    out = (
        _assemble_shift_target_batch_chunked_lagblock_interior_core_c128(
            jnp.asarray(w_xi_batch, dtype=jnp.complex128),
            jnp.asarray(t_shift_batch, dtype=jnp.float64),
            jnp.asarray(ell_all, dtype=jnp.int64),
            offset,
            jnp.asarray(Tl_batch, dtype=jnp.complex128),
            jnp.asarray(Tp_batch, dtype=jnp.complex128),
            jnp.asarray(Cnm, dtype=jnp.complex128),
            jnp.asarray(dF, dtype=jnp.float64),
            row_chunk_size,
            lag_block_size,
        )
    )
    return out if return_device else np.asarray(out)

