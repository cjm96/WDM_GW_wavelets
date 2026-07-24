"""High-level construction and application of WDM time-delay filters.

The maintained target-mode API has one production assembly and one reference
assembly.  Delay-kernel construction remains available in exact and
interpolated forms.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from ._time_shift_assembly import (
    _assemble_shift_fixed_dispatch,
    _assemble_shift_target_batch_dispatch,
    _assemble_shift_target_dispatch,
    _assemble_shift_variable_mode_dispatch,
)

_KERNEL_WDM_CACHE = {}
_KERNEL_PRECOMP_CACHE = {}
_CNM_PARITY_CACHE = {}

def _normalise_kernel_kwargs(kernel_kwargs):
    """Return a dictionary of kernel keyword arguments.

    Parameters
    ----------
    kernel_kwargs : dict or None
        Optional keyword arguments forwarded to kernel-WDM construction.

    Returns
    -------
    dict
        Empty dict when input is ``None``, otherwise a shallow copy.
    """
    if kernel_kwargs is None:
        return {}
    return dict(kernel_kwargs)

def _inherit_kernel_wdm_kwargs(wdm_data, kernel_kwargs):
    """
    Ensure kernel-WDM construction inherits WDM window/basis parameters from
    the data WDM object unless the caller explicitly overrides them.
    """
    out = _normalise_kernel_kwargs(kernel_kwargs)

    if "A_frac" not in out and hasattr(wdm_data, "A_frac"):
        out["A_frac"] = float(getattr(wdm_data, "A_frac"))

    return out

def _normalize_assembly_precision(precision):
    if precision is None:
        return "complex64"
    key = str(precision).lower()
    if key in ("complex128", "float64", "c128", "64"):
        return "complex128"
    if key in ("complex64", "float32", "c64", "32"):
        return "complex64"
    raise ValueError("assembly_precision must be complex128/float64 or complex64/float32.")

def _validate_row_chunk_size(row_chunk_size):
    try:
        row_chunk_size = int(row_chunk_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("row_chunk_size must be a positive integer.") from exc
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be >= 1.")
    return row_chunk_size

def _validate_lag_block_size(lag_block_size):
    try:
        lag_block_size = int(lag_block_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("lag_block_size must be a positive integer.") from exc
    if lag_block_size < 1:
        raise ValueError("lag_block_size must be >= 1.")
    return lag_block_size

def _kernel_cache_key(wdm_data, Nker, Nf, d, q, calc_m0, extra_kwargs):
    """Build a hashable cache key for kernel WDM objects/precomputes."""
    dt = float(getattr(wdm_data, "dt"))
    key_kwargs = tuple(sorted((str(k), repr(v)) for k, v in extra_kwargs.items()))
    return (
        dt,
        int(Nker),
        int(Nf),
        int(d),
        int(q),
        bool(calc_m0),
        key_kwargs,
    )

def choose_Nker(L_eff, Nf, safety=1.02, require_even_Ntker=True, require_even_Nker=True):
    """
    Choose Nker such that:
      - Nker >= safety*(2*L_eff*Nf + 1)
      - Nker divisible by Nf  (so Ntker = Nker/Nf is integer)
      - Ntker even (optional)
      - Nker even (optional)
    """
    N_need = int(np.ceil(safety * (2 * L_eff * Nf + 1)))

    # start from the next multiple of Nf
    Ntker = int(np.ceil(N_need / Nf))   # this is Nker/Nf
    if require_even_Ntker and (Ntker % 2 == 1):
        Ntker += 1

    Nker = Ntker * Nf

    if require_even_Nker and (Nker % 2 == 1):
        # if Nf is odd, Ntker even does not guarantee Nker even; bump Ntker by 1
        Ntker += 1
        if require_even_Ntker and (Ntker % 2 == 1):
            Ntker += 1
        Nker = Ntker * Nf

    return Nker

def _W1_from_windowFD_halfshift_fft_centered(window_FD_centered, dF, dt):
    """
    Compute Phi(f - 0.5 dF) * Phi(f + 0.5 dF) using FFT, centered ordering.
    """
    window_FD_centered = np.nan_to_num(window_FD_centered, nan=0.0, posinf=0.0, neginf=0.0)

    N = window_FD_centered.size
    half = 0.5 * dF

    Phi_u = np.fft.ifftshift(window_FD_centered)
    phi_u = np.fft.ifft(Phi_u)

    phi_c = np.fft.fftshift(phi_u)
    t_c = (np.arange(N) - (N // 2)) * dt

    phi_minus_c = phi_c * np.exp(+2j * np.pi * half * t_c)
    phi_plus_c  = phi_c * np.exp(-2j * np.pi * half * t_c)

    Phi_minus_u = np.fft.fft(np.fft.ifftshift(phi_minus_c))
    Phi_plus_u  = np.fft.fft(np.fft.ifftshift(phi_plus_c))

    W1_u = Phi_minus_u * Phi_plus_u
    return np.fft.fftshift(W1_u)

def _Cnm_parity(Nt, Nm, dtype=np.complex128):
    """Create the parity matrix used in Eq.(34)-style coefficient assembly."""
    n = np.arange(Nt)[:, None]
    m = np.arange(Nm)[None, :]
    even = ((n + m) % 2) == 0
    C = np.empty((Nt, Nm), dtype=dtype)
    C[even] = 1.0 + 0.0j
    C[~even] = 0.0 + 1.0j
    return C

def _get_Cnm_parity(Nt, Nm, dtype=np.complex128):
    """Return cached parity matrix for ``(Nt, Nm, dtype)``."""
    key = (int(Nt), int(Nm), np.dtype(dtype).str)
    cached = _CNM_PARITY_CACHE.get(key)
    if cached is not None:
        return cached
    C = _Cnm_parity(Nt, Nm, dtype=dtype)
    _CNM_PARITY_CACHE[key] = C
    return C

def _sample_TlTp_nonwrapping_from_ifft(a0_u, a1_u, ell_all, Nf, Nker):
    """
    Interpret ifft outputs as signed-lag kernels via fftshift,
    then sample at k = ell*Nf without modulo wrapping.
    """
    a0_c = np.fft.fftshift(a0_u)
    a1_c = np.fft.fftshift(a1_u)

    k_req = ell_all * Nf

    k_min = -(Nker // 2)
    k_max = (Nker // 2) - 1 if (Nker % 2 == 0) else (Nker // 2)

    valid = (k_req >= k_min) & (k_req <= k_max)
    if not np.all(valid):
        max_ell = int(np.max(np.abs(ell_all)))
        k_need = max_ell * Nf
        N_need = 2 * k_need + 1
        bad_ells = ell_all[~valid]
        raise ValueError(
            "Non-wrapping Tl/Tp sampling failed: requested lags exceed kernel signed-lag support.\n"
            f"  Nker={Nker}, Nf={Nf}, max|ell|={max_ell}, max|k|={k_need}\n"
            f"  Need roughly Nker >= {N_need}.\n"
            f"  Example out-of-range ells: {bad_ells[:10]}{' ...' if bad_ells.size>10 else ''}"
        )

    idx = (k_req + (Nker // 2)).astype(np.int64)
    return a0_c[idx], a1_c[idx]

def _build_kernel_wdm_like(wdm_data, Nker, Nf=None, d=None, q=None, calc_m0=True, **extra_kwargs):
    """
    Construct a WDM object on a larger Nker grid, using the same window-definition
    parameters as wdm_data. In WDM_GW_wavelets, dF depends on dt and Nf only,
    so dF stays consistent as long as dt and Nf match.

    You can override Nf/d/q via arguments if they are not stored on wdm_data.
    """
    # Import inside to avoid import issues in your package structure
    import WDM

    dt = float(getattr(wdm_data, "dt"))
    if Nf is None:
        Nf = int(getattr(wdm_data, "Nf", None))
        if Nf is None:
            raise ValueError("Nf not found on wdm_data; pass Nf explicitly.")
    if d is None:
        d = getattr(wdm_data, "d", None)
    if d is None:
        raise ValueError("d not found on wdm_data; pass d explicitly (e.g. d=4).")
    if q is None:
        q = 1 if int(Nker/Nf) == 2 else int(0.5 * Nker/Nf)
    if q is None:
        # Many of your constructors used q explicitly; require it if absent
        raise ValueError("q not found on wdm_data; pass q explicitly (e.g. q=0.5*Nt).")

    extra_kwargs = _inherit_kernel_wdm_kwargs(wdm_data, extra_kwargs)

    cache_key = _kernel_cache_key(
        wdm_data=wdm_data,
        Nker=Nker,
        Nf=Nf,
        d=d,
        q=q,
        calc_m0=calc_m0,
        extra_kwargs=extra_kwargs,
    )
    cached = _KERNEL_WDM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Build kernel transform object
    wdm_ker = WDM.WDM.WDM_transform(dt=dt, Nf=int(Nf), N=int(Nker), q=int(q), calc_m0=calc_m0, d=int(d), **extra_kwargs)
    _KERNEL_WDM_CACHE[cache_key] = wdm_ker
    return wdm_ker

def _get_kernel_precomputes(wdm, Nker, Nf, kernel_kwargs):
    """Return cached kernel-frequency arrays and products needed by shifts.

    The cached payload contains the FFT-grid frequencies and the precomputed
    ``W0`` and ``W1`` window products shared across repeated shift operations
    for the same kernel-WDM configuration.
    """
    kernel_kwargs = _inherit_kernel_wdm_kwargs(wdm, kernel_kwargs)
    wdm_ker = _build_kernel_wdm_like(wdm, Nker, Nf=Nf, calc_m0=True, **kernel_kwargs)

    if abs(float(wdm_ker.dF) - float(wdm.dF)) > 1e-30:
        raise ValueError(f"dF mismatch: wdm_ker.dF={wdm_ker.dF} vs wdm.dF={wdm.dF}. Check dt/Nf consistency.")

    pre_key = (
        _kernel_cache_key(
            wdm_data=wdm,
            Nker=Nker,
            Nf=Nf,
            d=getattr(wdm_ker, "d"),
            q=getattr(wdm_ker, "q"),
            calc_m0=True,
            extra_kwargs=kernel_kwargs,
        ),
        float(wdm_ker.dF),
        float(wdm_ker.dt),
        int(wdm_ker.N),
    )

    cached = _KERNEL_PRECOMP_CACHE.get(pre_key)
    if cached is not None:
        return wdm_ker, cached["freqs_u"], cached["W0_u"], cached["W1_u"], cached["scale"]

    Phi = np.nan_to_num(wdm_ker.window_FD, nan=0.0, posinf=0.0, neginf=0.0)
    W0 = np.abs(Phi) ** 2
    W1 = _W1_from_windowFD_halfshift_fft_centered(Phi, wdm_ker.dF, wdm_ker.dt)

    freqs_u = np.fft.ifftshift(wdm_ker.freqs)
    W0_u = np.fft.ifftshift(W0)
    W1_u = np.fft.ifftshift(W1)
    scale = wdm_ker.df * int(wdm_ker.N)

    _KERNEL_PRECOMP_CACHE[pre_key] = {
        "freqs_u": freqs_u,
        "W0_u": W0_u,
        "W1_u": W1_u,
        "scale": scale,
    }
    return wdm_ker, freqs_u, W0_u, W1_u, scale

def _infer_Nf(wdm, Nt, Nf):
    """Infer ``Nf`` from the WDM object when the caller does not provide it."""
    if Nf is not None:
        return int(Nf)
    if hasattr(wdm, "Nf"):
        return int(wdm.Nf)
    Nf = int(wdm.N) // Nt
    if Nf * Nt != int(wdm.N):
        raise ValueError(f"Cannot infer Nf from wdm.N={wdm.N} and Nt={Nt}")
    return Nf

def _resolve_ell_range(Nt, L_trunc):
    """Return the symmetric lag range and effective truncation level."""
    L_max = Nt - 1
    if L_trunc is None:
        L_eff = L_max
    else:
        L_eff = int(min(L_max, max(0, int(L_trunc))))
    ell_all = np.arange(-L_eff, L_eff + 1)
    return ell_all, L_eff

def _build_signed_lag_idx(ell_all, Nf, Nker_i):
    """Map lag indices to centered IFFT sample indices without wrapping."""
    k_req = ell_all * Nf
    k_min = -(Nker_i // 2)
    k_max = (Nker_i // 2) - 1 if (Nker_i % 2 == 0) else (Nker_i // 2)
    valid = (k_req >= k_min) & (k_req <= k_max)
    if not np.all(valid):
        max_ell = int(np.max(np.abs(ell_all)))
        k_need = max_ell * Nf
        N_need = 2 * k_need + 1
        bad_ells = ell_all[~valid]
        raise ValueError(
            "Non-wrapping Tl/Tp sampling failed: requested lags exceed kernel signed-lag support.\n"
            f"  Nker={Nker_i}, Nf={Nf}, max|ell|={max_ell}, max|k|={k_need}\n"
            f"  Need roughly Nker >= {N_need}.\n"
            f"  Example out-of-range ells: {bad_ells[:10]}{' ...' if bad_ells.size>10 else ''}"
        )
    return (k_req + (Nker_i // 2)).astype(np.int64)

def _build_TlTp_from_shift_matrix(t_shift_mat, freqs_u, W0_u, W1_u, scale, idx):
    phase_u_all = np.exp(-2j * np.pi * t_shift_mat[:, :, None] * freqs_u[None, None, :])
    a0_all = np.fft.ifft(phase_u_all * W0_u[None, None, :], axis=2) * scale
    a1_all = np.fft.ifft(phase_u_all * W1_u[None, None, :], axis=2) * scale
    a0_c_all = np.fft.fftshift(a0_all, axes=2)
    a1_c_all = np.fft.fftshift(a1_all, axes=2)
    Tl_all = a0_c_all[:, :, idx]
    Tp_all = a1_c_all[:, :, idx]
    return Tl_all, Tp_all

def _build_TlTp_from_shift_matrix_interp(
    t_shift_mat,
    freqs_u,
    W0_u,
    W1_u,
    scale,
    idx,
    interp_points=64,
    interp_pad=0.0,
    interp_kind="linear",
):
    """Approximate ``Tl/Tp`` by interpolation on a delay grid.

    Parameters
    ----------
    t_shift_mat : ndarray
        Delay matrix with shape ``(B, Nt)``.
    freqs_u, W0_u, W1_u, scale, idx : ndarray or float
        Precomputed kernel-frequency arrays and lag indices.
    interp_points : int, optional
        Number of delay grid points.
    interp_pad : float, optional
        Fractional padding applied to the delay span
        ``[min(t_shift), max(t_shift)]``.
    interp_kind : {"linear", "cubic"}, optional
        Interpolation kernel. Cubic uses a Catmull-Rom-style stencil.

    Notes
    -----
    This function is an approximation relative to exact FFT-based ``Tl/Tp``
    construction. Accuracy depends on ``interp_points``, ``interp_pad``, and
    delay distribution.
    """
    t_shift_mat = np.asarray(t_shift_mat, dtype=np.float64)
    if t_shift_mat.ndim != 2:
        raise ValueError(f"Expected t_shift_mat to have shape (B, Nt), got ndim={t_shift_mat.ndim}")

    flat_shift = t_shift_mat.reshape(-1)
    if flat_shift.size == 0:
        raise ValueError("Interpolation requested with empty shift matrix.")

    n_grid = max(2, int(interp_points))
    t_min = float(np.min(flat_shift))
    t_max = float(np.max(flat_shift))

    if not np.isfinite(t_min) or not np.isfinite(t_max):
        raise ValueError("Non-finite delays detected in t_shift_mat.")

    if t_max == t_min:
        return _build_TlTp_from_shift_matrix(t_shift_mat, freqs_u, W0_u, W1_u, scale, idx)

    span = t_max - t_min
    pad = float(interp_pad) * span
    grid_min = t_min - pad
    grid_max = t_max + pad

    if grid_max <= grid_min:
        return _build_TlTp_from_shift_matrix(t_shift_mat, freqs_u, W0_u, W1_u, scale, idx)

    shift_grid = np.linspace(grid_min, grid_max, n_grid, dtype=np.float64)
    Tl_grid, Tp_grid = _build_TlTp_from_shift_matrix(
        shift_grid[:, None], freqs_u, W0_u, W1_u, scale, idx
    )
    Tl_grid = Tl_grid[:, 0, :]
    Tp_grid = Tp_grid[:, 0, :]

    step = (grid_max - grid_min) / float(n_grid - 1)
    positions = (flat_shift - grid_min) / step
    positions = np.clip(positions, 0.0, float(n_grid - 1))
    idx0 = np.floor(positions).astype(np.int64)
    idx0 = np.clip(idx0, 0, n_grid - 2)
    frac = positions - idx0

    if interp_kind == "linear":
        frac_col = frac[:, None]
        Tl_flat = (1.0 - frac_col) * Tl_grid[idx0, :] + frac_col * Tl_grid[idx0 + 1, :]
        Tp_flat = (1.0 - frac_col) * Tp_grid[idx0, :] + frac_col * Tp_grid[idx0 + 1, :]
    elif interp_kind == "cubic":
        if n_grid < 4:
            raise ValueError("Cubic interpolation requires interp_points >= 4.")

        i1 = idx0
        i0 = np.clip(i1 - 1, 0, n_grid - 1)
        i2 = np.clip(i1 + 1, 0, n_grid - 1)
        i3 = np.clip(i1 + 2, 0, n_grid - 1)

        t = frac[:, None]
        t2 = t * t
        t3 = t2 * t

        p0_Tl = Tl_grid[i0, :]
        p1_Tl = Tl_grid[i1, :]
        p2_Tl = Tl_grid[i2, :]
        p3_Tl = Tl_grid[i3, :]

        p0_Tp = Tp_grid[i0, :]
        p1_Tp = Tp_grid[i1, :]
        p2_Tp = Tp_grid[i2, :]
        p3_Tp = Tp_grid[i3, :]

        Tl_flat = 0.5 * (
            2.0 * p1_Tl
            + (-p0_Tl + p2_Tl) * t
            + (2.0 * p0_Tl - 5.0 * p1_Tl + 4.0 * p2_Tl - p3_Tl) * t2
            + (-p0_Tl + 3.0 * p1_Tl - 3.0 * p2_Tl + p3_Tl) * t3
        )
        Tp_flat = 0.5 * (
            2.0 * p1_Tp
            + (-p0_Tp + p2_Tp) * t
            + (2.0 * p0_Tp - 5.0 * p1_Tp + 4.0 * p2_Tp - p3_Tp) * t2
            + (-p0_Tp + 3.0 * p1_Tp - 3.0 * p2_Tp + p3_Tp) * t3
        )
    else:
        raise ValueError("interp_kind must be 'linear' or 'cubic'.")

    out_shape = t_shift_mat.shape + (idx.shape[0],)
    Tl_all = Tl_flat.reshape(out_shape)
    Tp_all = Tp_flat.reshape(out_shape)
    return Tl_all, Tp_all


def _normalize_assembly_backend(backend):
    """Return the canonical maintained backend name.

    A small set of historical names is accepted as a migration aid.  They no
    longer select separate kernels.
    """

    if backend is None:
        return "production"
    key = str(backend).lower()
    if key in (
        "production",
        "lagfirst_chunked_lagblock",
        "lagblock",
        "lagfirst_lagblock",
        "auto",
    ):
        return "production"
    if key in (
        "reference",
        "legacy",
        "row",
        "lagfirst_row",
        "vmap",
        "lagfirst_chunked",
        "chunked",
    ):
        return "reference"
    raise ValueError(
        "assembly_backend must be 'production' or 'reference'."
    )


def _prepare_tl_tp(
    delays,
    *,
    mode,
    freqs_u,
    W0_u,
    W1_u,
    scale,
    signed_lag_idx,
    interp_points,
    interp_pad,
    interp_kind,
):
    """Build exact or interpolated delay kernels for a delay matrix."""

    if mode == "exact":
        return _build_TlTp_from_shift_matrix(
            delays,
            freqs_u,
            W0_u,
            W1_u,
            scale,
            signed_lag_idx,
        )
    if mode == "interp":
        return _build_TlTp_from_shift_matrix_interp(
            delays,
            freqs_u,
            W0_u,
            W1_u,
            scale,
            signed_lag_idx,
            interp_points=interp_points,
            interp_pad=interp_pad,
            interp_kind=interp_kind,
        )
    raise ValueError("tl_tp_mode must be 'exact' or 'interp'.")


def wdm_time_shift_variable(
    wdm,
    w_xi,
    t_shift,
    Nf=None,
    delta_mode="target",
    L_trunc=None,
    Nker=None,
    safety=1.02,
    kernel_kwargs=None,
    tl_tp_mode="exact",
    tl_tp_interp_points=64,
    tl_tp_interp_pad=0.0,
    tl_tp_interp_kind="linear",
    assembly_backend="production",
    assembly_precision="complex64",
    row_chunk_size=128,
    lag_block_size=1,
):
    """Apply one variable-delay WDM operator.

    Target mode supports the maintained ``production`` and ``reference``
    assemblies.  Source and midpoint modes are retained only as high-precision
    reference calculations.
    """

    w_xi = np.asarray(w_xi)
    if w_xi.ndim != 2:
        raise ValueError("w_xi must have shape (Nt, Nm).")
    Nt, Nm = w_xi.shape

    t_shift = np.asarray(t_shift, dtype=np.float64)
    if t_shift.shape != (Nt,):
        raise ValueError(
            f"Expected t_shift shape {(Nt,)}, got {t_shift.shape}."
        )

    Nf = _infer_Nf(wdm, Nt, Nf)
    ell_all, offset = _resolve_ell_range(Nt, L_trunc)
    if Nker is None:
        Nker = choose_Nker(
            offset,
            Nf,
            safety=safety,
            require_even_Ntker=True,
            require_even_Nker=True,
        )

    wdm_kernel, freqs_u, W0_u, W1_u, scale = _get_kernel_precomputes(
        wdm=wdm,
        Nker=Nker,
        Nf=Nf,
        kernel_kwargs=kernel_kwargs,
    )
    signed_lag_idx = _build_signed_lag_idx(
        ell_all,
        Nf,
        int(wdm_kernel.N),
    )
    Tl_matrix, Tp_matrix = _prepare_tl_tp(
        t_shift[None, :],
        mode=tl_tp_mode,
        freqs_u=freqs_u,
        W0_u=W0_u,
        W1_u=W1_u,
        scale=scale,
        signed_lag_idx=signed_lag_idx,
        interp_points=tl_tp_interp_points,
        interp_pad=tl_tp_interp_pad,
        interp_kind=tl_tp_interp_kind,
    )
    Tl_all = Tl_matrix[0]
    Tp_all = Tp_matrix[0]

    if delta_mode != "target":
        if delta_mode not in ("source", "midpoint"):
            raise ValueError(
                "delta_mode must be 'target', 'source', or 'midpoint'."
            )
        if assembly_backend not in (None, "reference"):
            warnings.warn(
                "source/midpoint modes use the reference assembly; "
                "assembly_backend was ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
        Cnm = _get_Cnm_parity(Nt, Nm, dtype=np.complex128)
        return _assemble_shift_variable_mode_dispatch(
            wdm,
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            Cnm,
            delta_mode=delta_mode,
        )

    backend = _normalize_assembly_backend(assembly_backend)
    precision = _normalize_assembly_precision(assembly_precision)
    row_chunk_size = _validate_row_chunk_size(row_chunk_size)
    lag_block_size = _validate_lag_block_size(lag_block_size)
    Cnm = (
        _get_Cnm_parity(Nt, Nm, dtype=np.complex128)
        if backend == "reference"
        else None
    )

    return _assemble_shift_target_dispatch(
        wdm,
        w_xi,
        t_shift,
        ell_all,
        offset,
        Tl_all,
        Tp_all,
        Cnm=Cnm,
        assembly_backend=backend,
        assembly_precision=precision,
        row_chunk_size=row_chunk_size,
        lag_block_size=lag_block_size,
    )


def wdm_time_shift_variable_batch(
    wdm,
    shift_jobs,
    Nf=None,
    L_trunc=None,
    Nker=None,
    safety=1.02,
    kernel_kwargs=None,
    batch_chunk=32,
    tl_tp_mode="exact",
    tl_tp_interp_points=64,
    tl_tp_interp_pad=0.0,
    tl_tp_interp_kind="linear",
    assembly_backend="production",
    assembly_precision="complex64",
    row_chunk_size=128,
    lag_block_size=1,
    profile_shift_stages=False,
    profile_label=None,
):
    """Apply target-mode variable shifts to a sequence of jobs.

    Each job is a ``(coefficients, delay)`` pair.  Jobs are prepared and
    assembled in fixed-size chunks to bound memory and compilation shapes.
    """

    shift_jobs = list(shift_jobs)
    if not shift_jobs:
        empty_profile = {
            "n_jobs": 0,
            "batch_chunk": None,
            "assembly_backend": None,
            "assembly_precision": None,
            "row_chunk_size": None,
            "lag_block_size": None,
            "tl_tp_mode": None,
            "tl_tp_interp_points": None,
            "total_s": 0.0,
            "tl_tp_prepare_s": 0.0,
            "jax_assembly_s": 0.0,
            "postprocess_s": 0.0,
            "other_s": 0.0,
            "label": profile_label,
        }
        return ([], empty_profile) if profile_shift_stages else []

    first_coefficients, _ = shift_jobs[0]
    first_coefficients = np.asarray(first_coefficients)
    if first_coefficients.ndim != 2:
        raise ValueError("Each coefficient array must have shape (Nt, Nm).")
    Nt, Nm = first_coefficients.shape

    Nf = _infer_Nf(wdm, Nt, Nf)
    ell_all, offset = _resolve_ell_range(Nt, L_trunc)
    if Nker is None:
        Nker = choose_Nker(
            offset,
            Nf,
            safety=safety,
            require_even_Ntker=True,
            require_even_Nker=True,
        )

    wdm_kernel, freqs_u, W0_u, W1_u, scale = _get_kernel_precomputes(
        wdm=wdm,
        Nker=Nker,
        Nf=Nf,
        kernel_kwargs=kernel_kwargs,
    )
    signed_lag_idx = _build_signed_lag_idx(
        ell_all,
        Nf,
        int(wdm_kernel.N),
    )

    backend = _normalize_assembly_backend(assembly_backend)
    precision = _normalize_assembly_precision(assembly_precision)
    row_chunk_size = _validate_row_chunk_size(row_chunk_size)
    lag_block_size = _validate_lag_block_size(lag_block_size)
    Cnm = (
        _get_Cnm_parity(Nt, Nm, dtype=np.complex128)
        if backend == "reference"
        else None
    )

    coefficients = []
    delays = []
    for job_index, (job_coefficients, job_delays) in enumerate(shift_jobs):
        job_coefficients = np.asarray(job_coefficients)
        job_delays = np.asarray(job_delays, dtype=np.float64)
        if job_coefficients.shape != (Nt, Nm):
            raise ValueError(
                f"Job {job_index}: expected coefficient shape {(Nt, Nm)}, "
                f"got {job_coefficients.shape}."
            )
        if job_delays.shape != (Nt,):
            raise ValueError(
                f"Job {job_index}: expected delay shape {(Nt,)}, "
                f"got {job_delays.shape}."
            )
        coefficients.append(job_coefficients)
        delays.append(job_delays)

    chunk_size = (
        len(shift_jobs)
        if batch_chunk is None
        else min(len(shift_jobs), max(1, int(batch_chunk)))
    )
    outputs = [None] * len(shift_jobs)

    total_started = time.perf_counter()
    tl_tp_seconds = 0.0
    assembly_seconds = 0.0
    postprocess_seconds = 0.0

    for start in range(0, len(shift_jobs), chunk_size):
        stop = min(start + chunk_size, len(shift_jobs))
        coefficient_batch = np.stack(coefficients[start:stop], axis=0)
        delay_batch = np.stack(delays[start:stop], axis=0)

        prepare_started = time.perf_counter()
        Tl_batch, Tp_batch = _prepare_tl_tp(
            delay_batch,
            mode=tl_tp_mode,
            freqs_u=freqs_u,
            W0_u=W0_u,
            W1_u=W1_u,
            scale=scale,
            signed_lag_idx=signed_lag_idx,
            interp_points=tl_tp_interp_points,
            interp_pad=tl_tp_interp_pad,
            interp_kind=tl_tp_interp_kind,
        )
        tl_tp_seconds += time.perf_counter() - prepare_started

        assembly_started = time.perf_counter()
        shifted = _assemble_shift_target_batch_dispatch(
            wdm,
            coefficient_batch,
            delay_batch,
            ell_all,
            offset,
            Tl_batch,
            Tp_batch,
            Cnm=Cnm,
            assembly_backend=backend,
            assembly_precision=precision,
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
        )
        assembly_seconds += time.perf_counter() - assembly_started

        postprocess_started = time.perf_counter()
        for local_index in range(stop - start):
            outputs[start + local_index] = shifted[local_index]
        postprocess_seconds += time.perf_counter() - postprocess_started

    total_seconds = time.perf_counter() - total_started
    if not profile_shift_stages:
        return outputs

    profile = {
        "n_jobs": len(shift_jobs),
        "batch_chunk": chunk_size,
        "assembly_backend": backend,
        "assembly_precision": precision,
        "row_chunk_size": row_chunk_size,
        "lag_block_size": lag_block_size,
        "tl_tp_mode": tl_tp_mode,
        "tl_tp_interp_points": int(tl_tp_interp_points),
        "total_s": float(total_seconds),
        "tl_tp_prepare_s": float(tl_tp_seconds),
        "jax_assembly_s": float(assembly_seconds),
        "postprocess_s": float(postprocess_seconds),
        "other_s": float(
            total_seconds
            - tl_tp_seconds
            - assembly_seconds
            - postprocess_seconds
        ),
        "label": profile_label,
    }
    return outputs, profile


def wdm_time_shift_fixed_batch(
    wdm,
    shift_jobs,
    Nf=None,
    L_trunc=None,
    Nker=None,
    safety=1.02,
    kernel_kwargs=None,
):
    """Apply the retained fixed-delay reference operator to several jobs."""

    shift_jobs = list(shift_jobs)
    if not shift_jobs:
        return []

    first_coefficients, _ = shift_jobs[0]
    first_coefficients = np.asarray(first_coefficients)
    Nt, Nm = first_coefficients.shape
    Nf = _infer_Nf(wdm, Nt, Nf)
    ell_all, offset = _resolve_ell_range(Nt, L_trunc)

    if Nker is None:
        Nker = choose_Nker(
            offset,
            Nf,
            safety=safety,
            require_even_Ntker=True,
            require_even_Nker=True,
        )

    wdm_kernel, freqs_u, W0_u, W1_u, scale = _get_kernel_precomputes(
        wdm=wdm,
        Nker=Nker,
        Nf=Nf,
        kernel_kwargs=kernel_kwargs,
    )
    signed_lag_idx = _build_signed_lag_idx(
        ell_all,
        Nf,
        int(wdm_kernel.N),
    )
    Cnm = _get_Cnm_parity(Nt, Nm, dtype=np.complex128)

    coefficients = []
    deltas = []
    for job_index, (job_coefficients, delta) in enumerate(shift_jobs):
        job_coefficients = np.asarray(job_coefficients)
        if job_coefficients.shape != (Nt, Nm):
            raise ValueError(
                f"Job {job_index}: expected coefficient shape {(Nt, Nm)}, "
                f"got {job_coefficients.shape}."
            )
        coefficients.append(job_coefficients)
        deltas.append(float(delta))

    deltas = np.asarray(deltas, dtype=np.float64)
    phase = np.exp(-2j * np.pi * deltas[:, None] * freqs_u[None, :])
    a0 = np.fft.ifft(phase * W0_u[None, :], axis=1) * scale
    a1 = np.fft.ifft(phase * W1_u[None, :], axis=1) * scale
    a0 = np.fft.fftshift(a0, axes=1)
    a1 = np.fft.fftshift(a1, axes=1)
    Tl_all = a0[:, signed_lag_idx]
    Tp_all = a1[:, signed_lag_idx]

    return [
        _assemble_shift_fixed_dispatch(
            wdm,
            coefficients[index],
            deltas[index],
            ell_all,
            offset,
            Tl_all[index],
            Tp_all[index],
            Cnm,
        )
        for index in range(len(shift_jobs))
    ]


def wdm_time_shift_fixed(
    wdm,
    w_xi,
    delta,
    Nf=None,
    L_trunc=None,
    Nker=None,
    safety=1.02,
    kernel_kwargs=None,
):
    """Apply one fixed-delay reference operator."""

    return wdm_time_shift_fixed_batch(
        wdm,
        [(w_xi, delta)],
        Nf=Nf,
        L_trunc=L_trunc,
        Nker=Nker,
        safety=safety,
        kernel_kwargs=kernel_kwargs,
    )[0]
