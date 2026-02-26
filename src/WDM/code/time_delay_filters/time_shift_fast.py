import numpy as np

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

# ============================================================
# 1) W1 helper (yours; with NaN safety)
# ============================================================
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


# ============================================================
# 2) Cnm parity helper (unchanged)
# ============================================================
def _Cnm_parity(Nt, Nm, dtype=np.complex128):
    n = np.arange(Nt)[:, None]
    m = np.arange(Nm)[None, :]
    even = ((n + m) % 2) == 0
    C = np.empty((Nt, Nm), dtype=dtype)
    C[even] = 1.0 + 0.0j
    C[~even] = 0.0 + 1.0j
    return C


# ============================================================
# 3) Kernel-length + non-wrapping sampling helpers
# ============================================================
def _ceil_to_multiple(x, m):
    return int(np.ceil(x / m) * m)

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


# ============================================================
# 4) Build kernel WDM object (same dt, Nf, d, q, etc.; bigger N)
# ============================================================
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

    # Build kernel transform object
    wdm_ker = WDM.WDM.WDM_transform(dt=dt, Nf=int(Nf), N=int(Nker), q=int(q), calc_m0=calc_m0, d=int(d), **extra_kwargs)
    return wdm_ker


# ============================================================
# 5) Variable time shift (kernel WDM + optional L_trunc)
# ============================================================
def wdm_time_shift_variable(
    wdm, w_xi, t_shift, Nf=None, delta_mode="target",
    L_trunc=None, Nker=None, safety=1.02,
    # if your wdm constructor needs extra window kwargs, pass them here:
    kernel_kwargs=None,
):
    """
    Same Eq.(34) operator you had, but Tl/Tp are computed on a kernel WDM grid
    of length Nker (bigger df, same dF) and sampled non-wrapping.

    L_trunc:
      If provided, use ell in [-L_eff,+L_eff] where L_eff=min(Nt-1,L_trunc).
      If None, use full ell range [-Nt+1, Nt-1].
    """
    Nt, Nm = w_xi.shape

    if Nf is None:
        if hasattr(wdm, "Nf"):
            Nf = int(wdm.Nf)
        else:
            # fallback to data-grid inference
            Nf = int(wdm.N) // Nt
            if Nf * Nt != int(wdm.N):
                raise ValueError(f"Cannot infer Nf from wdm.N={wdm.N} and Nt={Nt}")

    # Effective ell range
    L_max = Nt - 1
    if L_trunc is None:
        L_eff = L_max
    else:
        L_eff = int(min(L_max, max(0, int(L_trunc))))

    ell_all = np.arange(-L_eff, L_eff + 1)
    offset = L_eff

    # Choose kernel length
    if Nker is None:
        Nker = choose_Nker(L_eff, Nf, safety=safety,
                       require_even_Ntker=True,
                       require_even_Nker=True)

    if kernel_kwargs is None:
        kernel_kwargs = {}

    # Build kernel WDM object with SAME dt, Nf, d, q => SAME dF
    wdm_ker = _build_kernel_wdm_like(wdm, Nker, Nf=Nf, calc_m0=True, **kernel_kwargs)

    # Sanity: dF must match the data WDM (in this library, it should)
    if abs(float(wdm_ker.dF) - float(wdm.dF)) > 1e-30:
        raise ValueError(f"dF mismatch: wdm_ker.dF={wdm_ker.dF} vs wdm.dF={wdm.dF}. Check dt/Nf consistency.")

    # Build spectra on kernel grid
    Phi = np.nan_to_num(wdm_ker.window_FD, nan=0.0, posinf=0.0, neginf=0.0)
    W0 = np.abs(Phi) ** 2                       # IMPORTANT: |Phi|^2
    W1 = _W1_from_windowFD_halfshift_fft_centered(Phi, wdm_ker.dF, wdm_ker.dt)

    freqs_u = np.fft.ifftshift(wdm_ker.freqs)
    W0_u    = np.fft.ifftshift(W0)
    W1_u    = np.fft.ifftshift(W1)

    # Same scaling convention as your original code, but on kernel grid
    scale = wdm_ker.df * int(wdm_ker.N)

    # Precompute Tl/Tp for each time bin
    Tl_all = np.empty((Nt, ell_all.size), dtype=np.complex128)
    Tp_all = np.empty((Nt, ell_all.size), dtype=np.complex128)

    for k in range(Nt):
        delta_k = float(t_shift[k])
        phase_u = np.exp(-2j * np.pi * freqs_u * delta_k)
        a0 = np.fft.ifft(W0_u * phase_u) * scale
        a1 = np.fft.ifft(W1_u * phase_u) * scale
        Tl_k, Tp_k = _sample_TlTp_nonwrapping_from_ifft(a0, a1, ell_all, Nf, int(wdm_ker.N))
        Tl_all[k, :] = Tl_k
        Tp_all[k, :] = Tp_k

    # Now apply your Eq.(34) assembly exactly as before
    Cnm = _Cnm_parity(Nt, Nm, dtype=np.complex128)

    m = np.arange(Nm)
    minus1_to_m = np.where((m % 2) == 0, 1.0, -1.0)
    ones_m = np.ones(Nm, dtype=minus1_to_m.dtype)

    w_target = np.zeros_like(w_xi, dtype=np.complex128)

    for p in range(Nt):
        Cp = Cnm[p, :]

        for ell in ell_all:
            n = p + ell
            if n < 0 or n >= Nt:
                continue

            j_neg = (-ell) + offset

            if delta_mode == "target":
                delta = float(t_shift[p])
                Tl_j = Tl_all[p, j_neg]
                Tp_j = Tp_all[p, j_neg]

            elif delta_mode == "source":
                delta = float(t_shift[n])
                Tl_j = Tl_all[n, j_neg]
                Tp_j = Tp_all[n, j_neg]

            elif delta_mode == "midpoint":
                kf = 0.5 * (p + n)
                k0 = int(np.floor(kf))
                k1 = min(k0 + 1, Nt - 1)
                w = kf - k0

                delta = (1.0 - w) * float(t_shift[k0]) + w * float(t_shift[k1])
                Tl_j = (1.0 - w) * Tl_all[k0, j_neg] + w * Tl_all[k1, j_neg]
                Tp_j = (1.0 - w) * Tp_all[k0, j_neg] + w * Tp_all[k1, j_neg]

            else:
                raise ValueError("delta_mode must be 'target', 'source', or 'midpoint'")

            parity_full = ones_m if (ell % 2 == 0) else minus1_to_m

            ph_m = np.exp(2j * np.pi * (m * wdm.dF) * delta)
            Kc = parity_full * np.conj(Cp) * Cnm[n, :] * Tl_j * ph_m
            w_target[p, :] += Kc.real * w_xi[n, :]

            if Nm <= 1:
                continue

            ms = m[:-1]
            ph_low = np.exp(2j * np.pi * ((ms + 0.5) * wdm.dF) * delta)
            K_low = (
                parity_full[:-1]
                * ((-1j) ** (-ell))
                * np.conj(Cnm[p, 1:]) * Cnm[n, :-1]
                * Tp_j
                * ph_low
            )
            w_target[p, 1:] += K_low.real * w_xi[n, :-1]

            ms2 = m[1:]
            ph_up = np.exp(2j * np.pi * ((ms2 - 0.5) * wdm.dF) * delta)
            K_up = (
                parity_full[1:]
                * ((+1j) ** (-ell))
                * np.conj(Cnm[p, :-1]) * Cnm[n, 1:]
                * Tp_j
                * ph_up
            )
            w_target[p, :-1] += K_up.real * w_xi[n, 1:]

    return w_target


# ============================================================
# 6) Fixed time shift (kernel WDM + optional L_trunc)
# ============================================================
def wdm_time_shift_fixed(wdm, w_xi, delta, Nf=None,
                         L_trunc=None, Nker=None, safety=1.02,
                         kernel_kwargs=None):
    Nt, Nm = w_xi.shape

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

    if kernel_kwargs is None:
        kernel_kwargs = {}

    wdm_ker = _build_kernel_wdm_like(wdm, Nker, Nf=Nf, calc_m0=True, **kernel_kwargs)

    if abs(float(wdm_ker.dF) - float(wdm.dF)) > 1e-30:
        raise ValueError(f"dF mismatch: wdm_ker.dF={wdm_ker.dF} vs wdm.dF={wdm.dF}.")

    Phi = np.nan_to_num(wdm_ker.window_FD, nan=0.0, posinf=0.0, neginf=0.0)
    W0 = np.abs(Phi) ** 2
    W1 = _W1_from_windowFD_halfshift_fft_centered(Phi, wdm_ker.dF, wdm_ker.dt)

    freqs_u = np.fft.ifftshift(wdm_ker.freqs)
    W0_u    = np.fft.ifftshift(W0)
    W1_u    = np.fft.ifftshift(W1)

    scale = wdm_ker.df * int(wdm_ker.N)
    delta = float(delta)

    phase_u = np.exp(-2j * np.pi * freqs_u * delta)
    a0 = np.fft.ifft(W0_u * phase_u) * scale
    a1 = np.fft.ifft(W1_u * phase_u) * scale

    Tl_vec, Tp_vec = _sample_TlTp_nonwrapping_from_ifft(a0, a1, ell_all, Nf, int(wdm_ker.N))

    Cnm = _Cnm_parity(Nt, Nm, dtype=np.complex128)

    m = np.arange(Nm)
    minus1_to_m = np.where((m % 2) == 0, 1.0, -1.0)
    ones_m = np.ones(Nm, dtype=minus1_to_m.dtype)

    ph_m   = np.exp(2j * np.pi * (m * wdm.dF) * delta)
    if Nm > 1:
        ms = m[:-1]
        ms2 = m[1:]
        ph_low = np.exp(2j * np.pi * ((ms  + 0.5) * wdm.dF) * delta)
        ph_up  = np.exp(2j * np.pi * ((ms2 - 0.5) * wdm.dF) * delta)

    w_target = np.zeros_like(w_xi, dtype=np.complex128)

    for p in range(Nt):
        Cp = Cnm[p, :]

        for ell in ell_all:
            n = p + ell
            if n < 0 or n >= Nt:
                continue

            j_neg = (-ell) + offset
            Tl_j = Tl_vec[j_neg]
            Tp_j = Tp_vec[j_neg]

            parity_full = ones_m if (ell % 2 == 0) else minus1_to_m

            Kc = parity_full * np.conj(Cp) * Cnm[n, :] * Tl_j * ph_m
            w_target[p, :] += Kc.real * w_xi[n, :]

            if Nm <= 1:
                continue

            K_low = (
                parity_full[:-1]
                * ((-1j) ** (-ell))
                * np.conj(Cnm[p, 1:]) * Cnm[n, :-1]
                * Tp_j
                * ph_low
            )
            w_target[p, 1:] += K_low.real * w_xi[n, :-1]

            K_up = (
                parity_full[1:]
                * ((+1j) ** (-ell))
                * np.conj(Cnm[p, :-1]) * Cnm[n, 1:]
                * Tp_j
                * ph_up
            )
            w_target[p, :-1] += K_up.real * w_xi[n, 1:]

    return w_target