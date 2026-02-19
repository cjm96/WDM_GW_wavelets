import numpy as np

def _W1_from_windowFD_halfshift_fft_centered(window_FD_centered, dF, dt):
    """
    Compute the offset Phi(f) functions from equation 43 of the time_delay_filters wiki.
    Using fft methods for highest accuracy and centred to agree with the range of frequences of WDM method.

    Parameters
    ----------
    window_FD_centered : ndarray 
        ndarray of Phi(f)
    dF : float
        resolution of frequency space in WDM domain
    dt : float
        difference between samples in TD

    Returns
    -------
    ___ : ndarray
        Phi(f - 0.5 dF) * Phi(f + 0.5 dF) 
    """
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
    """
    Compute the C_nm matrix as per equation (13) of the WDM_GW_wavelets/theory section. 

    Parameters
    ----------
    Nt : int
        dimension of time in WDM domain
    Nm : int
        dimension of frequency in WDM domain

    Returns
    -------
    C : ndarray
        Nt x Nm matrix of 1 and i 
    """
    n = np.arange(Nt)[:, None]
    m = np.arange(Nm)[None, :]
    even = ((n + m) % 2) == 0
    C = np.empty((Nt, Nm), dtype=dtype)
    C[even] = 1.0 + 0.0j
    C[~even] = 0.0 + 1.0j
    return C


def wdm_time_shift_variable(
    wdm, w_xi, t_shift, Nf=None, delta_mode="target"
):
    """
    Implements Eq. (34) orientation:
        w_out[n,m] = sum_{n',m'} w_in[n',m'] * X_{n n'; m m'}( t_shift[n] )
    where you typically pass t_shift[p] = -Delta(t_p) for a physical delay Delta.

    delta_mode chooses where delta is sampled (target/source/midpoint) as before.

    Parameters
    ----------
    wdm : object
        WDM transform object with fields: N, Nf (optional), window_FD, freqs, dF, df, dt
    w_xi : ndarray, shape (Nt, Nm)
        input WDM coefficients
    t_shift : ndarray, shape (Nt,)
        delta values per time bin (delta in g(t + delta)); for a physical delay x(t-Δ), pass -Δ
    Nf : int, optional
        frequency oversampling; defaults to wdm.Nf if present, else inferred
    delta_mode : {"target","source","midpoint"}
        where to sample delta (and corresponding Tl/Tp)

    Returns
    -------
    w_target : ndarray, shape (Nt, Nm)
        shifted coefficients implementing Eq. (34) orientation
    """
    Nt, Nm = w_xi.shape
    N = int(wdm.N)

    if Nf is None:
        if hasattr(wdm, "Nf"):
            Nf = int(wdm.Nf)
        else:
            Nf = N // Nt
            if Nf * Nt != N:
                raise ValueError(f"Cannot infer Nf: wdm.N={N} not divisible by Nt={Nt}")

    # Spectra in stored (centered) ordering
    W0 = wdm.window_FD**2
    W1 = _W1_from_windowFD_halfshift_fft_centered(wdm.window_FD, wdm.dF, wdm.dt)

    # Convert to DFT order for ifft
    freqs_u = np.fft.ifftshift(wdm.freqs)
    W0_u = np.fft.ifftshift(W0)
    W1_u = np.fft.ifftshift(W1)

    Cnm = _Cnm_parity(Nt, Nm, dtype=np.complex128)

    m = np.arange(Nm)
    minus1_to_m = np.where((m % 2) == 0, 1.0, -1.0)
    ones_m = np.ones(Nm, dtype=minus1_to_m.dtype)

    ell_all = np.arange(-(Nt - 1), (Nt - 1) + 1)     # length 2*Nt-1
    # index mapping: ell -> j, and -ell -> j_neg
    # since ell_all is ordered from -(Nt-1) ... +(Nt-1):
    # j = ell + (Nt-1), j_neg = (-ell) + (Nt-1) = (Nt-1) - ell
    offset = (Nt - 1)

    ell_idx = (ell_all * Nf) % N
    scale = wdm.df * N

    # ---------- PRECOMPUTE Tl/Tp for each k using delta = t_shift[k] ----------
    Tl_all = np.empty((Nt, ell_all.size), dtype=np.complex128)
    Tp_all = np.empty((Nt, ell_all.size), dtype=np.complex128)

    for k in range(Nt):
        delta_k = float(t_shift[k])
        phase_u = np.exp(-2j * np.pi * freqs_u * delta_k)
        a0 = np.fft.ifft(W0_u * phase_u) * scale
        a1 = np.fft.ifft(W1_u * phase_u) * scale
        Tl_all[k, :] = a0[ell_idx]   # corresponds to T_{ell}(delta_k)
        Tp_all[k, :] = a1[ell_idx]   # corresponds to T'_{ell}(delta_k)
    # ------------------------------------------------------------------------

    w_target = np.zeros_like(w_xi, dtype=np.complex128)

    for p in range(Nt):
        Cp = Cnm[p, :]  # C_{p,m}

        for j, ell in enumerate(ell_all):
            n = p + ell
            if n < 0 or n >= Nt:
                continue

            # For Eq.(34) we need l = p - n = -ell, so use j_neg
            j_neg = (-ell) + offset  # index where ell_all == -ell

            # Choose delta and Tl/Tp sampling index based on delta_mode
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
            # parity uses (-1)^{(p-n)m} = (-1)^{-ell*m} = (-1)^{ell*m}, so unchanged

            # Center term: C*_{p,m} C_{n,m} and T_{p-n} = T_{-ell}
            ph_m = np.exp(2j * np.pi * (m * wdm.dF) * delta)
            Kc = (
                parity_full
                * np.conj(Cp) * Cnm[n, :]    
                * Tl_j
                * ph_m
            )
            w_target[p, :] += Kc.real * w_xi[n, :]

            if Nm <= 1:
                continue

            # Lower: output m_out = m_in + 1 (use C*_{p,m_out} C_{n,m_in})
            ms = m[:-1]
            parity_ms = parity_full[:-1]
            ph_low = np.exp(2j * np.pi * ((ms + 0.5) * wdm.dF) * delta)
            K_low = (
                parity_ms
                * ((-1j) ** (-ell))                # (∓i)^{p-n} = (∓i)^{-ell}
                * np.conj(Cnm[p, 1:]) * Cnm[n, :-1] 
                * Tp_j
                * ph_low
            )
            w_target[p, 1:] += K_low.real * w_xi[n, :-1]

            # Upper: output m_out = m_in - 1 (use C*_{p,m_out} C_{n,m_in})
            ms2 = m[1:]
            parity_ms2 = parity_full[1:]
            ph_up = np.exp(2j * np.pi * ((ms2 - 0.5) * wdm.dF) * delta)
            K_up = (
                parity_ms2
                * ((+1j) ** (-ell))                 # (±i)^{p-n} = (±i)^{-ell}
                * np.conj(Cnm[p, :-1]) * Cnm[n, 1:]  
                * Tp_j
                * ph_up
            )
            w_target[p, :-1] += K_up.real * w_xi[n, 1:]

    return w_target

def wdm_time_shift_fixed(wdm, w_xi, delta, Nf=None):
    """
    Fixed (constant) time shift in the Eq.(34) orientation:
        w_out[n,m] = sum_{n',m'} w_in[n',m'] * X_{n n'; m m'}(delta)

    This is faster than the variable-δ version because Tl/Tp are computed once.
    """
    Nt, Nm = w_xi.shape
    N = int(wdm.N)

    if Nf is None:
        if hasattr(wdm, "Nf"):
            Nf = int(wdm.Nf)
        else:
            Nf = N // Nt
            if Nf * Nt != N:
                raise ValueError(f"Cannot infer Nf: wdm.N={N} not divisible by Nt={Nt}")

    # Spectra (centered ordering)
    W0 = wdm.window_FD**2
    W1 = _W1_from_windowFD_halfshift_fft_centered(wdm.window_FD, wdm.dF, wdm.dt)

    # Convert to DFT order
    freqs_u = np.fft.ifftshift(wdm.freqs)
    W0_u = np.fft.ifftshift(W0)
    W1_u = np.fft.ifftshift(W1)

    Cnm = _Cnm_parity(Nt, Nm, dtype=np.complex128)

    m = np.arange(Nm)
    minus1_to_m = np.where((m % 2) == 0, 1.0, -1.0)
    ones_m = np.ones(Nm, dtype=minus1_to_m.dtype)

    ell_all = np.arange(-(Nt - 1), (Nt - 1) + 1)     # ell = n - p
    offset = Nt - 1
    ell_idx = (ell_all * Nf) % N

    scale = wdm.df * N
    delta = float(delta)

    # ---- compute Tl(ell,delta) and Tp(ell,delta) ONCE ----
    phase_u = np.exp(-2j * np.pi * freqs_u * delta)
    a0 = np.fft.ifft(W0_u * phase_u) * scale
    a1 = np.fft.ifft(W1_u * phase_u) * scale
    Tl_vec = a0[ell_idx]   # Tl_vec[j] corresponds to ell_all[j]
    Tp_vec = a1[ell_idx]
    # ------------------------------------------------------

    # Phases only depend on m when delta is fixed
    ph_m   = np.exp(2j * np.pi * (m * wdm.dF) * delta)
    if Nm > 1:
        ms = m[:-1]
        ms2 = m[1:]
        ph_low = np.exp(2j * np.pi * ((ms  + 0.5) * wdm.dF) * delta)
        ph_up  = np.exp(2j * np.pi * ((ms2 - 0.5) * wdm.dF) * delta)

    w_target = np.zeros_like(w_xi, dtype=np.complex128)

    for p in range(Nt):
        Cp = Cnm[p, :]

        for j, ell in enumerate(ell_all):
            n = p + ell
            if n < 0 or n >= Nt:
                continue

            # Eq.(34) wants l = p - n = -ell
            j_neg = (-ell) + offset
            Tl_j = Tl_vec[j_neg]
            Tp_j = Tp_vec[j_neg]

            parity_full = ones_m if (ell % 2 == 0) else minus1_to_m

            # Center: C*_{p,m} C_{n,m} and T_{-ell}
            Kc = (
                parity_full
                * np.conj(Cp) * Cnm[n, :]
                * Tl_j
                * ph_m
            )
            w_target[p, :] += Kc.real * w_xi[n, :]

            if Nm <= 1:
                continue

            # Lower: output m_out = m_in + 1  (uses input m_in = m-1)
            K_low = (
                parity_full[:-1]
                * ((-1j) ** (-ell))
                * np.conj(Cnm[p, 1:]) * Cnm[n, :-1]
                * Tp_j
                * ph_low
            )
            w_target[p, 1:] += K_low.real * w_xi[n, :-1]

            # Upper: output m_out = m_in - 1 (uses input m_in = m+1)
            K_up = (
                parity_full[1:]
                * ((+1j) ** (-ell))
                * np.conj(Cnm[p, :-1]) * Cnm[n, 1:]
                * Tp_j
                * ph_up
            )
            w_target[p, :-1] += K_up.real * w_xi[n, 1:]

    return w_target

