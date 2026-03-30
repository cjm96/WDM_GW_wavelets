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


def assemble_shift_target_batch_jax(w_xi_batch, t_shift_batch, ell_all, offset, Tl_batch, Tp_batch, Cnm, dF, assembly_vmap=False):
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
    return np.asarray(out)


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
