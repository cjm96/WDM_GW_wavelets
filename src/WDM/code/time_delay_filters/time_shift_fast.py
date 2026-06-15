import warnings
import time

import numpy as np

try:
    import jax.numpy as jnp
except Exception:
    jnp = None

try:
    from ._time_shift_assembly import (
        _assemble_shift_fixed_dispatch,
        _assemble_shift_target_dispatch,
        _assemble_shift_target_batch_dispatch,
        _assemble_shift_variable_mode_dispatch,
    )
    _JAX_AVAILABLE = True
except Exception:
    _assemble_shift_fixed_dispatch = None
    _assemble_shift_target_dispatch = None
    _assemble_shift_target_batch_dispatch = None
    _assemble_shift_variable_mode_dispatch = None
    _JAX_AVAILABLE = False


_KERNEL_WDM_CACHE = {}
_KERNEL_PRECOMP_CACHE = {}
_CNM_PARITY_CACHE = {}
_USE_JAX_ASSEMBLY = _JAX_AVAILABLE


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


def _resolve_use_jax(use_jax=None):
    """Resolve JAX usage for assembly.

    The public ``use_jax`` argument is kept for backward compatibility, but the
    NumPy assembly backend has been retired. All active assembly paths require
    JAX and return ``True`` when available.
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX is required for time_shift_fast assembly paths.")
    # Kept for API compatibility with older call sites; all paths are JAX now.
    _ = use_jax
    return True


def _normalize_assembly_precision(precision):
    if precision is None:
        return "complex64"
    key = str(precision).lower()
    if key in ("complex128", "float64", "c128", "64"):
        return "complex128"
    if key in ("complex64", "float32", "c64", "32"):
        return "complex64"
    raise ValueError("assembly_precision must be complex128/float64 or complex64/float32.")


def _normalize_assembly_backend(backend):
    if backend is None:
        return None
    key = str(backend).lower()
    if key in ("lagfirst_chunked", "chunked", "auto"):
        return "lagfirst_chunked"
    if key in ("lagfirst_chunked_lagblock", "lagblock", "lagfirst_lagblock"):
        return "lagfirst_chunked_lagblock"
    if key in (
        "lagfirst_chunked_lagblock_jobblock",
        "lagblock_jobblock",
        "jobblock",
        "lagfirst_lagblock_jobblock",
    ):
        return "lagfirst_chunked_lagblock_jobblock"
    if key in ("legacy", "row", "lagfirst_row"):
        return "legacy"
    if key == "vmap":
        return "vmap"
    raise ValueError(
        "assembly_backend must be lagfirst_chunked, lagfirst_chunked_lagblock, "
        "lagfirst_chunked_lagblock_jobblock, legacy, row, lagfirst_row, vmap, or auto."
    )


def _resolve_assembly_backend(assembly_backend, assembly_vmap):
    if assembly_backend is None:
        if assembly_vmap is True:
            return "vmap"
        return "lagfirst_chunked"
    return _normalize_assembly_backend(assembly_backend)


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


def _build_TlTp_from_shift_matrix_interp_jax(
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
    """Approximate ``Tl/Tp`` via delay-grid interpolation using JAX array ops.

    Notes
    -----
    This path keeps FFT-based grid construction unchanged, but performs the
    interpolation gather/blend with JAX arrays. It is intended as an optional
    optimization path for JAX-heavy workflows.
    """
    if jnp is None:
        raise RuntimeError("JAX interpolation backend requested but jax is not available.")

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
    Tl_grid_np, Tp_grid_np = _build_TlTp_from_shift_matrix(
        shift_grid[:, None], freqs_u, W0_u, W1_u, scale, idx
    )
    Tl_grid = jnp.asarray(Tl_grid_np[:, 0, :], dtype=jnp.complex128)
    Tp_grid = jnp.asarray(Tp_grid_np[:, 0, :], dtype=jnp.complex128)

    step = (grid_max - grid_min) / float(n_grid - 1)
    pos = (jnp.asarray(flat_shift, dtype=jnp.float64) - grid_min) / step
    pos = jnp.clip(pos, 0.0, float(n_grid - 1))
    idx0 = jnp.floor(pos).astype(jnp.int64)
    idx0 = jnp.clip(idx0, 0, n_grid - 2)
    frac = pos - idx0.astype(jnp.float64)

    if interp_kind == "linear":
        frac_col = frac[:, None]
        Tl_flat = (1.0 - frac_col) * Tl_grid[idx0, :] + frac_col * Tl_grid[idx0 + 1, :]
        Tp_flat = (1.0 - frac_col) * Tp_grid[idx0, :] + frac_col * Tp_grid[idx0 + 1, :]
    elif interp_kind == "cubic":
        if n_grid < 4:
            raise ValueError("Cubic interpolation requires interp_points >= 4.")
        i1 = idx0
        i0 = jnp.clip(i1 - 1, 0, n_grid - 1)
        i2 = jnp.clip(i1 + 1, 0, n_grid - 1)
        i3 = jnp.clip(i1 + 2, 0, n_grid - 1)

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
    Tl_all = np.asarray(Tl_flat).reshape(out_shape)
    Tp_all = np.asarray(Tp_flat).reshape(out_shape)
    return Tl_all, Tp_all


def _resolve_interp_backend(tl_tp_interp_backend, use_jax):
    """Resolve interpolation backend name for ``tl_tp_mode='interp'``.

    Parameters
    ----------
    tl_tp_interp_backend : str
        One of ``"numpy"``, ``"jax"``, or ``"auto"``.
    use_jax : bool
        Effective JAX usage flag for assembly.

    Returns
    -------
    str
        Resolved backend name, either ``"numpy"`` or ``"jax"``.
    """
    backend = str(tl_tp_interp_backend).lower()
    if backend == "auto":
        return "jax" if (_JAX_AVAILABLE and bool(use_jax)) else "numpy"
    if backend not in ("numpy", "jax"):
        raise ValueError("tl_tp_interp_backend must be 'numpy', 'jax', or 'auto'.")
    return backend


def wdm_time_shift_variable(
    wdm, w_xi, t_shift, Nf=None, delta_mode="target",
    L_trunc=None, Nker=None, safety=1.02,
    # if your wdm constructor needs extra window kwargs, pass them here:
    kernel_kwargs=None,
    use_jax=None,
    tl_tp_mode="exact",
    tl_tp_interp_points=64,
    tl_tp_interp_pad=0.0,
    tl_tp_interp_kind="linear",
    tl_tp_interp_backend="numpy",
    assembly_backend=None,
    assembly_precision="complex64",
    row_chunk_size=128,
    lag_block_size=1,
    assembly_vmap=None,
):
    """
    Apply variable-delay WDM time shifting with optional interpolated ``Tl/Tp``.

    Parameters
    ----------
    delta_mode : {"target", "source", "midpoint"}
        Row-selection mode for delay usage in assembly.
    L_trunc : int or None
        Truncation of lag range. If ``None``, uses full ``[-Nt+1, Nt-1]``.
    tl_tp_mode : {"exact", "interp"}
        ``"exact"`` builds ``Tl/Tp`` by direct FFT evaluation.
        ``"interp"`` builds them from a delay grid interpolation.
    tl_tp_interp_points : int
        Number of delay grid points when ``tl_tp_mode='interp'``.
    tl_tp_interp_pad : float
        Fractional padding for interpolation grid bounds.
    tl_tp_interp_kind : {"linear", "cubic"}
        Interpolation kernel used when ``tl_tp_mode='interp'``.
    tl_tp_interp_backend : {"numpy", "jax", "auto"}
        Backend used for interpolation gather/blend in ``tl_tp_mode='interp'``.
        ``"numpy"`` preserves legacy behavior. ``"jax"`` uses JAX arrays for
        interpolation arithmetic. ``"auto"`` selects JAX when available.
    assembly_backend : {"lagfirst_chunked", "lagfirst_chunked_lagblock", "legacy", "row", "lagfirst_row", "vmap", "auto"}
        Selects the target-mode assembly backend. When omitted, the default
        is ``"lagfirst_chunked"`` (fastest dense backend). ``"lagfirst_chunked_lagblock"``
        (alias ``"lagblock"``) processes lags in blocks for potential improvements.
        ``"legacy"``/``"row"`` keep the older row-first implementation. ``"vmap"`` 
        uses the older vmap-based route.
    assembly_precision : {"complex64", "complex128", "float32", "float64"}
        Precision used for the chunked backend. ``"complex64"`` is the fast
        mode; ``"complex128"`` is the faithful exact mode.
    row_chunk_size : int
        Row chunk size used by the chunked backend (default 128).
    lag_block_size : int
        Lag block size used by the lag-blocked backend (default 1).
    assembly_vmap : bool or None
        Legacy flag selecting vmap assembly when ``assembly_backend`` is not
        provided. When ``assembly_backend`` is set, this flag is ignored.

    Notes
    -----
    ``tl_tp_mode='interp'`` is approximate and should be validated against
    ``tl_tp_mode='exact'`` for target tolerances.
    """
    Nt, Nm = w_xi.shape
    use_jax = _resolve_use_jax(use_jax=use_jax)
    Nf = _infer_Nf(wdm, Nt, Nf)

    # Effective ell range
    ell_all, offset = _resolve_ell_range(Nt, L_trunc)

    if Nker is None:
        Nker = choose_Nker(offset, Nf, safety=safety, require_even_Ntker=True, require_even_Nker=True)

    t_shift = np.asarray(t_shift, dtype=float)
    if t_shift.shape[0] != Nt:
        raise ValueError(f"Expected t_shift length {Nt}, got {t_shift.shape[0]}.")

    wdm_ker, freqs_u, W0_u, W1_u, scale = _get_kernel_precomputes(
        wdm=wdm,
        Nker=Nker,
        Nf=Nf,
        kernel_kwargs=kernel_kwargs,
    )

    Nker_i = int(wdm_ker.N)
    idx = _build_signed_lag_idx(ell_all, Nf, Nker_i)

    if tl_tp_mode == "exact":
        Tl_all_mat, Tp_all_mat = _build_TlTp_from_shift_matrix(
            t_shift[None, :], freqs_u, W0_u, W1_u, scale, idx
        )
    elif tl_tp_mode == "interp":
        backend = _resolve_interp_backend(tl_tp_interp_backend, use_jax=use_jax)
        if backend == "jax":
            Tl_all_mat, Tp_all_mat = _build_TlTp_from_shift_matrix_interp_jax(
                t_shift[None, :],
                freqs_u,
                W0_u,
                W1_u,
                scale,
                idx,
                interp_points=tl_tp_interp_points,
                interp_pad=tl_tp_interp_pad,
                interp_kind=tl_tp_interp_kind,
            )
        else:
            Tl_all_mat, Tp_all_mat = _build_TlTp_from_shift_matrix_interp(
                t_shift[None, :],
                freqs_u,
                W0_u,
                W1_u,
                scale,
                idx,
                interp_points=tl_tp_interp_points,
                interp_pad=tl_tp_interp_pad,
                interp_kind=tl_tp_interp_kind,
            )
    else:
        raise ValueError("tl_tp_mode must be 'exact' or 'interp'.")

    Tl_all = Tl_all_mat[0]
    Tp_all = Tp_all_mat[0]

    resolved_backend = _resolve_assembly_backend(assembly_backend, assembly_vmap)
    precision = _normalize_assembly_precision(assembly_precision)
    row_chunk_size = _validate_row_chunk_size(row_chunk_size)
    lag_block_size = _validate_lag_block_size(lag_block_size)

    if delta_mode == "target":
        cnm_dtype = np.complex64 if precision == "complex64" else np.complex128
        Cnm = _get_Cnm_parity(Nt, Nm, dtype=cnm_dtype)
        return _assemble_shift_target_dispatch(
            wdm,
            w_xi,
            t_shift,
            ell_all,
            offset,
            Tl_all,
            Tp_all,
            Cnm=Cnm,
            use_jax=use_jax,
            assembly_backend=resolved_backend,
            assembly_precision=precision,
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            assembly_vmap=assembly_vmap,
        )

    if resolved_backend == "lagfirst_chunked":
        warnings.warn(
            "lagfirst_chunked backend only applies to delta_mode='target'; using legacy path instead.",
            RuntimeWarning,
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
        Cnm=Cnm,
        use_jax=use_jax,
        delta_mode=delta_mode,
        assembly_vmap=assembly_vmap,
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
    use_jax=None,
    tl_tp_mode="exact",
    tl_tp_interp_points=64,
    tl_tp_interp_pad=0.0,
    tl_tp_interp_kind="linear",
    tl_tp_interp_backend="numpy",
    assembly_backend=None,
    assembly_precision="complex64",
    row_chunk_size=128,
    lag_block_size=1,
    job_block_size=1,
    assembly_vmap=None,
    jax_pad_last_chunk=False,
    # Profiling hooks (opt-in, backwards compatible)
    profile_shift_stages=False,
    profile_label=None,
):
    """Apply variable target-mode WDM time shifts for multiple jobs at once.

    Parameters
    ----------
    wdm : object
        WDM transform object used to define dF/dt and basis sizes.
    shift_jobs : sequence[tuple[ndarray, ndarray]]
        Sequence of ``(w_xi, t_shift)`` pairs. Each ``w_xi`` is ``(Nt, Nm)`` and
        each ``t_shift`` is ``(Nt,)``.
    Nf : int or None, optional
        WDM frequency bins. Inferred from ``wdm`` when omitted.
    L_trunc : int or None, optional
        Lag truncation parameter.
    Nker : int or None, optional
        Kernel grid length. Chosen automatically when omitted.
    safety : float, optional
        Safety factor used for automatic ``Nker`` sizing.
    kernel_kwargs : dict or None, optional
        Extra kwargs forwarded to kernel-WDM construction.
    batch_chunk : int, optional
        Number of jobs processed together in one FFT batch.
    tl_tp_mode : {"exact", "interp"}, optional
        Method used to build ``Tl/Tp``. ``"exact"`` performs direct FFT-based
        evaluation for each requested delay. ``"interp"`` precomputes a delay
        grid and interpolates in delay.
    tl_tp_interp_points : int, optional
        Number of delay grid points used when ``tl_tp_mode='interp'``.
    tl_tp_interp_pad : float, optional
        Fractional padding added to the [min, max] delay range when
        ``tl_tp_mode='interp'``.
    tl_tp_interp_kind : {"linear", "cubic"}, optional
        Interpolation kernel used when ``tl_tp_mode='interp'``.
    tl_tp_interp_backend : {"numpy", "jax", "auto"}, optional
        Backend used for interpolation gather/blend in ``tl_tp_mode='interp'``.
        ``"numpy"`` preserves legacy behavior. ``"jax"`` uses JAX arrays for
        interpolation arithmetic. ``"auto"`` selects JAX when available.
    assembly_backend : {"lagfirst_chunked", "lagfirst_chunked_lagblock", "legacy", "row", "lagfirst_row", "vmap", "auto"}
        Selects the target-mode assembly backend. When omitted, the default is
        ``"lagfirst_chunked"``. ``"lagfirst_chunked_lagblock"`` (alias ``"lagblock"``)
        processes lags in blocks. ``"legacy"``/``"row"`` keep the older path.
    assembly_precision : {"complex64", "complex128", "float32", "float64"}
        Precision used by the chunked backend.
    row_chunk_size : int, optional
        Row chunk size used by the chunked backend.
    lag_block_size : int, optional
        Lag block size used by the lag-blocked backend (default 1).
    assembly_vmap : bool or None, optional
        Legacy flag selecting vmap assembly when ``assembly_backend`` is not
        provided.
    jax_pad_last_chunk : bool, optional
        When using JAX with ``assembly_vmap=True``, pad the final short chunk to
        ``batch_chunk`` rows by repeating the last row so every chunk keeps the
        same shape. This can reduce JIT retracing in some workloads.

    Notes
    -----
    Interpolated mode is approximate relative to exact ``Tl/Tp`` construction.

    Returns
    -------
    list[ndarray]
        Shifted WDM arrays in the same order as ``shift_jobs``.
    """
    shift_jobs = list(shift_jobs)
    use_jax = _resolve_use_jax(use_jax=use_jax)
    if len(shift_jobs) == 0:
        if profile_shift_stages:
            profile = {
                "n_jobs": 0,
                "batch_chunk": None,
                "job_block_size": None,
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
            return [], profile
        return []

    w0, s0 = shift_jobs[0]
    Nt, Nm = np.asarray(w0).shape
    Nf = _infer_Nf(wdm, Nt, Nf)
    ell_all, offset = _resolve_ell_range(Nt, L_trunc)

    if Nker is None:
        Nker = choose_Nker(offset, Nf, safety=safety, require_even_Ntker=True, require_even_Nker=True)

    wdm_ker, freqs_u, W0_u, W1_u, scale = _get_kernel_precomputes(
        wdm=wdm,
        Nker=Nker,
        Nf=Nf,
        kernel_kwargs=kernel_kwargs,
    )
    idx = _build_signed_lag_idx(ell_all, Nf, int(wdm_ker.N))
    resolved_backend = _resolve_assembly_backend(assembly_backend, assembly_vmap)
    precision = _normalize_assembly_precision(assembly_precision)
    row_chunk_size = _validate_row_chunk_size(row_chunk_size)
    lag_block_size = _validate_lag_block_size(lag_block_size)
    cnm_dtype = np.complex64 if precision == "complex64" else np.complex128
    Cnm = _get_Cnm_parity(Nt, Nm, dtype=cnm_dtype)

    w_arrays = []
    t_arrays = []
    for i, (w_xi, t_shift) in enumerate(shift_jobs):
        w_xi = np.asarray(w_xi)
        t_shift = np.asarray(t_shift, dtype=float)
        if w_xi.shape != (Nt, Nm):
            raise ValueError(f"Job {i}: expected w_xi shape {(Nt, Nm)}, got {w_xi.shape}")
        if t_shift.shape[0] != Nt:
            raise ValueError(f"Job {i}: expected t_shift length {Nt}, got {t_shift.shape[0]}")
        w_arrays.append(w_xi)
        t_arrays.append(t_shift)

    out = [None] * len(shift_jobs)
    chunk = len(shift_jobs) if batch_chunk is None else max(1, int(batch_chunk))
    interp_backend = _resolve_interp_backend(tl_tp_interp_backend, use_jax=use_jax)

    # Profiling accumulators
    if profile_shift_stages:
        prof_n_jobs = len(shift_jobs)
        prof_batch_chunk = chunk
        prof_job_block_size = job_block_size
        prof_assembly_backend = resolved_backend
        prof_assembly_precision = precision
        prof_row_chunk_size = row_chunk_size
        prof_lag_block_size = lag_block_size
        prof_tl_tp_mode = tl_tp_mode
        prof_tl_tp_interp_points = tl_tp_interp_points
        prof_tl_tp_prepare_s = 0.0
        prof_jax_assembly_s = 0.0
        prof_postprocess_s = 0.0
        prof_other_s = 0.0
        prof_chunk_profiles = []
        t_total_start = time.perf_counter()

    for k0 in range(0, len(shift_jobs), chunk):
        k1 = min(k0 + chunk, len(shift_jobs))
        t_shift_mat = np.stack(t_arrays[k0:k1], axis=0)
        w_chunk = np.stack(w_arrays[k0:k1], axis=0)
        true_batch = k1 - k0

        should_pad = (
            bool(use_jax)
            and bool(assembly_vmap)
            and bool(jax_pad_last_chunk)
            and (len(shift_jobs) > chunk)
            and (true_batch < chunk)
        )
        if should_pad:
            pad_rows = chunk - true_batch
            t_pad = np.repeat(t_shift_mat[-1:, :], pad_rows, axis=0)
            w_pad = np.repeat(w_chunk[-1:, :, :], pad_rows, axis=0)
            t_shift_work = np.concatenate((t_shift_mat, t_pad), axis=0)
            w_work = np.concatenate((w_chunk, w_pad), axis=0)
        else:
            t_shift_work = t_shift_mat
            w_work = w_chunk

        if tl_tp_mode == "exact":
            t0 = time.perf_counter() if profile_shift_stages else None
            Tl_batch, Tp_batch = _build_TlTp_from_shift_matrix(
                t_shift_work, freqs_u, W0_u, W1_u, scale, idx
            )
            if profile_shift_stages:
                prof_tl_tp_prepare_s += time.perf_counter() - t0
        elif tl_tp_mode == "interp":
            if interp_backend == "jax":
                t0 = time.perf_counter() if profile_shift_stages else None
                Tl_batch, Tp_batch = _build_TlTp_from_shift_matrix_interp_jax(
                    t_shift_work,
                    freqs_u,
                    W0_u,
                    W1_u,
                    scale,
                    idx,
                    interp_points=tl_tp_interp_points,
                    interp_pad=tl_tp_interp_pad,
                    interp_kind=tl_tp_interp_kind,
                )
                if profile_shift_stages:
                    prof_tl_tp_prepare_s += time.perf_counter() - t0
            else:
                t0 = time.perf_counter() if profile_shift_stages else None
                Tl_batch, Tp_batch = _build_TlTp_from_shift_matrix_interp(
                    t_shift_work,
                    freqs_u,
                    W0_u,
                    W1_u,
                    scale,
                    idx,
                    interp_points=tl_tp_interp_points,
                    interp_pad=tl_tp_interp_pad,
                    interp_kind=tl_tp_interp_kind,
                )
                if profile_shift_stages:
                    prof_tl_tp_prepare_s += time.perf_counter() - t0
        else:
            raise ValueError("tl_tp_mode must be 'exact' or 'interp'.")
        t1 = time.perf_counter() if profile_shift_stages else None
        shifted_chunk = _assemble_shift_target_batch_dispatch(
            wdm,
            w_work,
            t_shift_work,
            ell_all,
            offset,
            Tl_batch,
            Tp_batch,
            Cnm=Cnm,
            use_jax=use_jax,
            assembly_backend=resolved_backend,
            assembly_precision=precision,
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            job_block_size=job_block_size,
            assembly_vmap=assembly_vmap,
        )
        if profile_shift_stages:
            # Ensure JAX kernels complete before stopping the timer
            try:
                for arr in np.asarray(shifted_chunk):
                    if hasattr(arr, "block_until_ready"):
                        arr.block_until_ready()
            except Exception:
                pass
            prof_jax_assembly_s += time.perf_counter() - t1
        for j in range(true_batch):
            t2 = time.perf_counter() if profile_shift_stages else None
            out[k0 + j] = shifted_chunk[j]
            if profile_shift_stages:
                prof_postprocess_s += time.perf_counter() - t2

    if profile_shift_stages:
        t_total_end = time.perf_counter()
        prof_total_s = t_total_end - t_total_start
        prof_other_s = prof_total_s - (prof_tl_tp_prepare_s + prof_jax_assembly_s + prof_postprocess_s)
        profile = {
            "n_jobs": prof_n_jobs,
            "batch_chunk": prof_batch_chunk,
            "job_block_size": int(prof_job_block_size),
            "assembly_backend": str(prof_assembly_backend),
            "assembly_precision": str(prof_assembly_precision),
            "row_chunk_size": int(prof_row_chunk_size),
            "lag_block_size": int(prof_lag_block_size),
            "tl_tp_mode": str(prof_tl_tp_mode),
            "tl_tp_interp_points": int(prof_tl_tp_interp_points),
            "total_s": float(prof_total_s),
            "tl_tp_prepare_s": float(prof_tl_tp_prepare_s),
            "jax_assembly_s": float(prof_jax_assembly_s),
            "postprocess_s": float(prof_postprocess_s),
            "other_s": float(prof_other_s),
            "label": profile_label,
        }
        return out, profile

    return out


def wdm_time_shift_fixed_batch(
    wdm,
    shift_jobs,
    Nf=None,
    L_trunc=None,
    Nker=None,
    safety=1.02,
    kernel_kwargs=None,
    use_jax=None,
    assembly_vmap=False,
):
    """Apply fixed WDM time shifts for multiple jobs using shared kernel precomputes.

    Parameters
    ----------
    wdm : object
        WDM transform object used to define dF/dt and basis sizes.
    shift_jobs : sequence[tuple[ndarray, float]]
        Sequence of ``(w_xi, delta)`` pairs.
    Nf : int or None, optional
        WDM frequency bins. Inferred from ``wdm`` when omitted.
    L_trunc : int or None, optional
        Lag truncation parameter.
    Nker : int or None, optional
        Kernel grid length. Chosen automatically when omitted.
    safety : float, optional
        Safety factor used for automatic ``Nker`` sizing.
    kernel_kwargs : dict or None, optional
        Extra kwargs forwarded to kernel-WDM construction.
    assembly_vmap : bool, optional
        If ``True``, JAX assembly maps rows with ``vmap``. ``False`` (default)
        reduces peak memory usage at the cost of some throughput.

    Returns
    -------
    list[ndarray]
        Shifted WDM arrays in the same order as ``shift_jobs``.
    """
    shift_jobs = list(shift_jobs)
    use_jax = _resolve_use_jax(use_jax=use_jax)
    if len(shift_jobs) == 0:
        return []

    w0, _ = shift_jobs[0]
    Nt, Nm = np.asarray(w0).shape
    Nf = _infer_Nf(wdm, Nt, Nf)
    ell_all, offset = _resolve_ell_range(Nt, L_trunc)

    if Nker is None:
        Nker = choose_Nker(offset, Nf, safety=safety, require_even_Ntker=True, require_even_Nker=True)

    wdm_ker, freqs_u, W0_u, W1_u, scale = _get_kernel_precomputes(
        wdm=wdm,
        Nker=Nker,
        Nf=Nf,
        kernel_kwargs=kernel_kwargs,
    )
    idx = _build_signed_lag_idx(ell_all, Nf, int(wdm_ker.N))
    Cnm = _get_Cnm_parity(Nt, Nm, dtype=np.complex128)

    w_arrays = []
    deltas = []
    for i, (w_xi, delta) in enumerate(shift_jobs):
        w_xi = np.asarray(w_xi)
        if w_xi.shape != (Nt, Nm):
            raise ValueError(f"Job {i}: expected w_xi shape {(Nt, Nm)}, got {w_xi.shape}")
        w_arrays.append(w_xi)
        deltas.append(float(delta))

    deltas = np.asarray(deltas, dtype=float)
    phase_u_all = np.exp(-2j * np.pi * deltas[:, None] * freqs_u[None, :])
    a0_all = np.fft.ifft(phase_u_all * W0_u[None, :], axis=1) * scale
    a1_all = np.fft.ifft(phase_u_all * W1_u[None, :], axis=1) * scale

    a0_c_all = np.fft.fftshift(a0_all, axes=1)
    a1_c_all = np.fft.fftshift(a1_all, axes=1)
    Tl_all = a0_c_all[:, idx]
    Tp_all = a1_c_all[:, idx]

    out = [None] * len(shift_jobs)
    for i in range(len(shift_jobs)):
        out[i] = _assemble_shift_fixed_dispatch(
            wdm,
            w_arrays[i],
            deltas[i],
            ell_all,
            offset,
            Tl_all[i],
            Tp_all[i],
            Cnm=Cnm,
            use_jax=use_jax,
            assembly_vmap=assembly_vmap,
        )

    return out

def wdm_time_shift_fixed(wdm, w_xi, delta, Nf=None,
                         L_trunc=None, Nker=None, safety=1.02,
                         kernel_kwargs=None, use_jax=None):
    """Apply a constant time shift to WDM coefficients.

    Uses the same Eq.(34)-style assembly as ``wdm_time_shift_variable`` but with
    a scalar delay ``delta`` shared across all time rows.
    """
    Nt, Nm = w_xi.shape
    use_jax = _resolve_use_jax(use_jax=use_jax)

    if Nf is None:
        if hasattr(wdm, "Nf"):
            Nf = int(wdm.Nf)
        else:
            Nf = int(wdm.N) // Nt
            if Nf * Nt != int(wdm.N):
                raise ValueError(f"Cannot infer Nf from wdm.N={wdm.N} and Nt={Nt}")

    L_max = Nt - 1
    if L_trunc is None:
        L_eff = L_max
    else:
        L_eff = int(min(L_max, max(0, int(L_trunc))))

    ell_all = np.arange(-L_eff, L_eff + 1)
    offset = L_eff

    if Nker is None:
        Nker = choose_Nker(L_eff, Nf, safety=safety,
                       require_even_Ntker=True,
                       require_even_Nker=True)

    wdm_ker, freqs_u, W0_u, W1_u, scale = _get_kernel_precomputes(
        wdm=wdm,
        Nker=Nker,
        Nf=Nf,
        kernel_kwargs=kernel_kwargs,
    )
    delta = float(delta)

    phase_u = np.exp(-2j * np.pi * freqs_u * delta)
    a0 = np.fft.ifft(W0_u * phase_u) * scale
    a1 = np.fft.ifft(W1_u * phase_u) * scale

    Tl_vec, Tp_vec = _sample_TlTp_nonwrapping_from_ifft(a0, a1, ell_all, Nf, int(wdm_ker.N))

    Cnm = _get_Cnm_parity(Nt, Nm, dtype=np.complex128)

    return _assemble_shift_fixed_dispatch(
        wdm,
        w_xi,
        delta,
        ell_all,
        offset,
        Tl_vec,
        Tp_vec,
        Cnm=Cnm,
        use_jax=use_jax,
    )