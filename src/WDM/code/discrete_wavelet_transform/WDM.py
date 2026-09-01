import jax
import jax.numpy as jnp

from WDM.code.utils.Meyer import Meyer
from WDM.code.utils.utils import C_nm, overlapping_windows
from WDM.code.time_delay_filters.filters import (FILTER_TABLE_BLOCK_BYTES,
                                                 build_filter_tables)

from typing import Tuple
from functools import partial


class WDM_transform:
    r"""
    This class implements the WDM discrete wavelet transform.

    Attributes
    ----------
    dt : float
        The cadence, or time step, of the time series, :math:`\delta t`. 
        Equal to inverse of the sampling frequency.
    Nf : int
        Number of wavelet frequency bands, :math:`N_f`. Must be even. 
        This controls the time/frequency resolution of the wavelets. 
    N : int
        Length of the input time series, :math:`N`. Must be an even multiple of 
        :math:`N_f`.
    Nt : int
        Number of wavelet time bands, :math:`N_t`. Equal to :math:`N/N_f`. 
        Must be even.
    q : int
        Truncation parameter, :math:`q`. Formally the time domain wavelets have 
        infinite extent but in practice are truncated at :math:`\pm q \Delta T`.
        This must be an integer in the range :math:`1 \leq q \leq N_t/2`.
    d : int
        Steepness parameter for the Meyer window transition. 
        Must be a positive integer, :math:`d\geq 1`.
    A_frac : float
        Fraction of total bandwidth used for the flat-top response region.
        Must be in the range [0, 1].
    B_frac : float
        Fraction of total bandwidth used for the transition region. This is set
        based on A_frac so :math:`2A_{\mathrm{frac}}+B_{\mathrm{frac}}=1`.
    A : float
        Half-width of the flat-top response region in angular frequency (radians
        per unit time), :math:`A`. Satisfies :math:`\Delta \Omega = 2A + B`.
    B : float
        Width of the transition region in angular frequency (radians per unit 
        time), :math:`B`. Satisfies :math:`\Delta \Omega = 2A + B`.
    dF : float
        Frequency resolution of the wavelets, or the total wavelet 
        frequency bandwidth :math:`\Delta F = \frac{\Delta \Omega}{2 \pi}`.
    dT : float
        Time resolution of the wavelets. Related to the frequency 
        resolution by :math:`\Delta F \Delta T = \frac{1}{2}`.
    dOmega : float
        Angular Frequency resolution of the wavelets (radians per unit time), or
        total wavelet angular frequency bandwidth, :math:`\Delta \Omega = 2A+B`.
    T : float
        Total duration of the time series. Related to :math:`N` and 
        :math:`\delta t` by :math:`T = N \delta t`.
    df : float
        The frequency resolution of the time series, :math:`\delta f = 1/T`.
    f_s : float
        Sampling frequency of the time series, :math:`f_s = \frac{1}{\delta t}`.
    f_Ny : float
        Nyquist frequency (i.e. maximum frequency) of the time series,
        :math:`f_{\rm Ny} = \frac{1}{2 \delta t}`.
    K : int
        Window length in samples, :math:`K = 2 q N_f`. By definition, 
        this is always an even integer.
    times : jnp.ndarray
        The sample times of the time series, :math:`t_k = k \delta t` for 
        :math:`k\in\{0,1,\ldots,N-1\}`. Array shape=(N,).
    freqs : jnp.ndarray
        The sample frequencies of the time series, :math:`f_k = k \delta f` for
        :math:`k\in\{-N/2,N/2+1,\ldots,N/2-1\}`. Array shape=(N,).
        Note, the zero-frequency component is in the center of the spectrum.
    Cnm : jnp.ndarray 
        Coefficients :math:`C_{nm}` used for the wavelet transform. Equal to 1 
        if :math:`n+m` is even or :math:`i` if it's odd. Array shape=(N_t, N_f).
    calc_m0 : bool
        If this is set to False (default value) then the wavelet coefficients
        with :math:`m=0` are handled INCORRECTLY. This is faster. If these 
        coefficients are needed the initialise the class with `calc_m0=True`.
    window_TD : jnp.ndarray 
        The time-domain Meyer window function, :math:`\phi(t)`. 
        Array shape=(N,).
    window_FD : jnp.ndarray 
        The frequency-domain Meyer window function, :math:`\tilde{\Phi}(f)`. 
        Array shape=(N,), dtype=complex.
    cached_Gnm_basis : jnp.ndarray
        The frequency-domain wavelet basis :math:`\tilde{G}_{nm}(f)`.
        Array shape=(N, Nt, Nf).
    cached_gnm_basis : jnp.ndarray
        The time-domain wavelet basis :math:`g_{nm}(t)`. 
        Array shape=(N, Nt, Nf).
    jax_dtype : jnp.float64
        Use jax.config.update("jax_enable_x64", True).
    jax_dtype_int : jnp.int64
        Use jax.config.update("jax_enable_x64", True).
    """

    def __init__(self, 
                 dt : float,
                 Nf : int,
                 N : int,
                 q : int = 16,
                 d : int = 4,
                 A_frac : float = 0.25,
                 calc_m0 : bool = False) -> None:
        r"""
        Initialize the WDM_transform.

        Parameters
        ----------
        dt : float
            The time series cadence, or time step. 
        Nf : int
            Number of frequency bands, controls the time/frequency resolution.
        N : int
            Length of the input time series. Must be an even multiple of Nf.
        q : int
            Truncation parameter. Integer :math:`1 \leq q \leq N_t/2`. Optional.
        d : int
            Steepness parameter for the transition. Optional.
        A_frac : float
            Bandwidth fraction of flat-top response. Optional.
        calc_m0 : bool
            If False, then the wavelet calculations for the :math:`m=0` temrs 
            will be wrong; this has performance benefits. If True, then all 
            calculations will be correct, but this may be slower. Optional.

        Returns
        -------
        None
        """
        self.dt = float(dt)
        self.Nf = int(Nf)
        self.N = int(N)
        self.q = int(q)
        self.A_frac = float(A_frac)
        self.d = int(d)
        self.calc_m0 = bool(calc_m0)

        self.validate_parameters()

        # Derived parameters
        self.times = jnp.arange(self.N) * self.dt
        self.freqs = jnp.fft.fftshift(jnp.fft.fftfreq(self.N, d=self.dt))
        self.Nt = self.N // self.Nf
        self.T = self.N * self.dt
        self.df = 1. / self.T
        self.dF = 1. / ( 2. * self.dt * self.Nf )  
        self.dOmega = 2. * jnp.pi * self.dF
        self.dT = self.dt * self.Nf 
        self.f_s = 1. / self.dt
        self.f_Ny = 0.5 / self.dt
        self.B_frac = 1. - 2. * self.A_frac  
        self.A = self.A_frac * self.dOmega
        self.B = self.B_frac * self.dOmega
        self.K = 2 * self.q * self.Nf
        self.Cnm = jnp.where((jnp.arange(self.Nt)[:, jnp.newaxis]
                              + jnp.arange(self.Nf)[jnp.newaxis, :]) % 2 == 0,
                             1.0+0.0j, 1.0j)

        self.window_FD = self.build_frequency_domain_window()
        self.window_TD = self.build_time_domain_window()

        self._cached_Gnm_basis = None
        self._cached_gnm_basis = None

        if jax.config.read("jax_enable_x64"):
            self.jax_dtype = jnp.float64
            self.jax_dtype_int = jnp.int64
        else:
            self.jax_dtype = jnp.float32
            self.jax_dtype_int = jnp.int32

    def validate_parameters(self) -> None:
        r"""
        Validate the parameters provided to the WDM_transform __init__ method.
        Raises an AssertionError if any parameters are invalid.

        Returns
        -------
        None
        """
        assert self.dt > 0, \
                    f"dt must be positive, got {self.dt=}."

        assert self.Nf > 0 and self.Nf % 2 == 0, \
                    f"Nf must be a positive even integer, got {self.Nf=}."

        assert self.N > 0 and self.N % 2 == 0, \
                    f"Nt must be a positive even integer, got {self.N=}."

        assert self.N % self.Nf == 0 and ( self.N // self.Nf ) % 2 == 0, \
                    f"N must be even multiple of Nf, got {self.N=}, {self.Nf=}."

        Nt = self.N // self.Nf
        assert self.q >= 1 and self.q <= Nt//2, \
                    f"q must be integer in range 1<=q<={Nt//2}, got {self.q=}."

        assert 0. < self.A_frac < 1., \
                    f"A_frac must be in range 0<A_frac<1, got {self.A_frac=}."

        assert self.d >= 1, \
                    f"d must be a positive integer, got {self.d=}."

    def build_frequency_domain_window(self) -> jnp.ndarray:
        r"""
        Construct the frequency-domain window function :math:`\tilde{\Phi}(f)`.

        Note, the zero-frequency component is in the center of the spectrum. 

        Returns
        -------
        Phi : jnp.ndarray 
            Array of shape (N,). Complex-valued frequency-domain window. 
        """
        Phi = Meyer(2.*jnp.pi*self.freqs, self.d, self.A, self.B)
        return jnp.sqrt(2.*jnp.pi) * Phi

    def build_time_domain_window(self) -> jnp.ndarray:
        r"""
        Construct the time-domain window function :math:`\phi(t)`.

        This method builds the Meyer window in the frequency domain and applies
        an inverse FFT to obtain the corresponding time-domain window.

        Returns
        -------
        phi : jnp.ndarray 
            Array of shape (N,). Real-valued time-domain window. 
        """
        phi = jnp.fft.ifft(jnp.fft.ifftshift(self.window_FD)).real / self.dt
        return phi

    @partial(jax.jit, static_argnums=0)
    def check_indices(self, n : jnp.ndarray, m : jnp.ndarray) -> bool:
        r"""
        Check if the wavelet indices are in the valid range. 

        The `n` indices must satisfy :math:`0 \leq n < N_t` and the `m` indices
        must satisfy :math:`0 \leq m < N_f`.

        Parameters
        ----------
        n : jnp.ndarray
            Array of n indices, dtype=int. Wavelet time index.
        m : jnp.ndarray
            Array of m indices, dtype=int. Wavelet frequency index.

        Returns
        -------
        check : bool
            True if the all indices are valid, otherwise False.
        """
        n = jnp.asarray(n, self.jax_dtype_int)
        m = jnp.asarray(m, self.jax_dtype_int)

        n_test = jnp.all(jnp.logical_and(n>=0, n<self.Nt))
        m_test = jnp.all(jnp.logical_and(m>=0, m<self.Nf))

        check = jnp.logical_and(n_test, m_test)

        return check

    def wavelet_central_time_frequency(self, 
                                       n : jnp.ndarray, 
                                       m : jnp.ndarray) -> Tuple[jnp.ndarray, 
                                                                 jnp.ndarray]:
        r"""
        Compute the central time :math:`t_{nm}= n \Delta t` and the central 
        frequency :math:`f_{nm} = m \Delta f` of the wavelet :math:`g_{nm}(t)`.

        The case :math:`m=0` is special and is handled separately using

        .. math::

            t_{n0} = 2n \Delta t , 

        .. math::

            f_{n0} = \begin{cases} 0 & \mathrm{if}\; n<N_t/2 \\ 
                    f_{\mathrm{Ny}} & \mathrm{if}\; n\geq N_t/2 \end{cases} . 

        Parameters
        ----------
        n : jnp.ndarray
            Wavelet time index, dtype=int, shape=(num_n,). 
        m : jnp.ndarray
            Wavelet frequency index, dtype=int, shape=(num_m,). 

        Returns
        -------
        t_nm : jnp.ndarray
            Array of times, shape=(num_n, num_m). The wavelet central times.
        f_nm : jnp.ndarray
            Array of frequencies, shape=(num_n, num_m). The wavelet central 
            frequencies.
        """
        assert self.check_indices(n, m), f"Invalid indices: {n=} {m=}"

        return self.wavelet_central_time_frequency_compiled(n, m)

    @partial(jax.jit, static_argnums=0)
    def wavelet_central_time_frequency_compiled(self, 
                                       n : jnp.ndarray, 
                                       m : jnp.ndarray) -> Tuple[jnp.ndarray, 
                                                                jnp.ndarray]:
        """
        Compiled part of wavelet_central_time_frequency method.

        Parameters
        ----------
        n : jnp.ndarray
            Wavelet time index, dtype=int, shape=(num_n,). 
        m : jnp.ndarray
            Wavelet frequency index, dtype=int, shape=(num_m,). 

        Returns
        -------
        t_nm : jnp.ndarray
            Array of times, shape=(num_n, num_m). The wavelet central times.
        f_nm : jnp.ndarray
            Array of frequencies, shape=(num_n, num_m). The wavelet central 
            frequencies.
        """
        n = jnp.asarray(n, self.jax_dtype_int)  
        m = jnp.asarray(m, self.jax_dtype_int) 

        n_col = n[:, None] # (len(n), 1)
        m_row = m[None, :] # (1, len(m))

        mzero = (m_row == 0)

        t_nm = jnp.where(mzero,
                        2 * n_col * self.dT,
                        n_col * self.dT)

        f_m0 = jnp.where(n_col < (self.Nt // 2), 0.0, self.f_Ny) 
        f_nm = jnp.where(mzero, f_m0, m_row * self.dF)

        return t_nm, f_nm

    def Gnm(self, 
            n : int, 
            m : int,
            freq : jnp.ndarray = None) -> jnp.ndarray:
        r"""
        Compute the frequency-domain representation of the wavelets, 
        :math:`\tilde{G}_{nm}(f)`.

        This method computes the frequency-domain wavelet for a single choice 
        of :math:`n` and :math:`m` using the expressions below. If you instead
        want to compute the full wavelet basis for all :math:`n` and :math:`m`
        efficiently, use the `Gnm_basis` method.

        For :math:`m>0`, the wavelet is given by

        .. math::

            \tilde{G}_{nm}(f) = \frac{\exp(-2\pi i n f \Delta T)}{\sqrt{2}} 
                    \left( C_{nm}\tilde{\Phi}(f+m\Delta F)
                            + C^*_{nm}\tilde{\Phi}(f-m\Delta F) \right) .

        For the special case :math:`m=0`, the wavelet is given by

        .. math::

            \tilde{G}_{n0}(f) = \begin{cases} 
                \exp(-4\pi i n f \Delta T) \tilde{\Phi}(f)
                    & \mathrm{if}\; n<N_t/2 \\
                \frac{1}{2} \exp(-4\pi i n f \Delta T) \left( 
                            \tilde{\Phi}(f-f_{\rm Ny}) 
                                + \tilde{\Phi}(f+f_{\rm Ny}) \right) 
                        & \mathrm{if}\; n\geq N_t/2
            \end{cases}

        Parameters
        ----------
        n : int
            Wavelet time index.
        m : int
            Wavelet frequency index.
        freq : jnp.ndarray
            Frequencies at which to evaluate the wavelet. 
            If None, then defaults to self.freqs. Optional

        Returns
        -------
        Gnm : jnp.ndarray
            Complex array shaped like freq. The frequency-domain wavelet.
        """
        assert self.check_indices(n, m), f"Invalid indices: {n=} {m=}"

        k_vals = jnp.arange(self.N)

        if m > 0:
            Gnm = (1./jnp.sqrt(2.)) * \
                        jnp.exp(-1j*n*2.*jnp.pi*self.freqs*self.dT) * (
                            C_nm(n, m) *  
                                self.window_FD[(k_vals+m*self.Nt//2)%self.N] +
                            jnp.conj(C_nm(n, m)) * 
                                self.window_FD[(k_vals-m*self.Nt//2)%self.N]
                            )

        else: 
            if n < self.Nt // 2: # zero-frequency terms
                Gnm = jnp.exp(-1j*n*4.*jnp.pi*self.freqs*self.dT) * \
                            self.window_FD

            else: # Nyquist-frequency terms
                Gnm = 0.5 * jnp.exp(-1j*n*4.*jnp.pi*self.freqs*self.dT) * \
                        (self.window_FD[(k_vals+self.N//2)%self.N] + 
                         self.window_FD[(k_vals-self.N//2)%self.N]) 

        return Gnm
    
    def Gnm_dual(self,
                 n : int, 
                 m : int) -> jnp.ndarray:
        r""" 
        This method compute the frequency-domain dual basis wavelets 
        :math:`\hat{g}_{nm}(t)` using the following expressions,

        NOT IMPLEMENTED YET
        """
        raise NotImplementedError("Gnm_dual method not implemented yet.")
    
    @partial(jax.jit, static_argnums=0)
    def Gnm_basis(self) -> jnp.ndarray:
        r"""
        Efficient computation of frequency-domain wavelet basis 
        :math:`\tilde{G}_{nm}(f)`. Instead of calling the functions for 
        :math:`\tilde{G}_{nm}(f)` explicilty as is done in the `Gnm` method, 
        this function shifts indices of `window_FD`.

        The result is cached to speed up subsequent calls.

        Returns
        -------
        basis : jnp.ndarray 
            Array of shape (N, Nt, Nf). The time-domain wavelet basis.
            The first axis is frequency, the second is the wavelet time index,
            and the third is the wavelet frequency index.
        """
        if self._cached_Gnm_basis is not None:
            pass

        else:
            n_vals = jnp.arange(self.Nt)
            m_vals = jnp.arange(self.Nf)

            om = 2. * jnp.pi * self.freqs

            shift_up = (jnp.arange(self.N)[:,jnp.newaxis] +
                        m_vals[jnp.newaxis,:]*self.Nt//2) 
            shift_do = (jnp.arange(self.N)[:,jnp.newaxis] -
                        m_vals[jnp.newaxis,:]*self.Nt//2) 

            basis = (1./jnp.sqrt(2.)) * \
                        jnp.exp(-1j*n_vals[jnp.newaxis,:,jnp.newaxis]*\
                                om[:,jnp.newaxis,jnp.newaxis]*self.dT) * \
                            (self.Cnm[jnp.newaxis,:,:]*\
                              self.window_FD[shift_up%self.N][:,jnp.newaxis,:]+
                             jnp.conj(self.Cnm[jnp.newaxis,:,:])*\
                              self.window_FD[shift_do%self.N][:,jnp.newaxis,:])

            if self.calc_m0:
                # overwrite m=0 terms for n<Nt/2 (zero-frequency terms)
                n_vals = jnp.arange(self.Nt//2)

                f0_term = jnp.exp(-2j*n_vals[jnp.newaxis,:] * \
                                om[:,jnp.newaxis]*self.dT) * \
                                    self.window_FD[:,jnp.newaxis]

                basis = basis.at[:, n_vals, 0].set(f0_term)

                # overwrite m=0 terms for n>=Nt/2 (Nyquist-frequency terms)
                n_vals = jnp.arange(self.Nt//2, self.Nt)

                shift_up = (jnp.arange(self.N) + self.N//2) 
                shift_do = (jnp.arange(self.N) - self.N//2) 

                fNy_term = 0.5 * jnp.exp(-2j*n_vals[jnp.newaxis,:] * \
                                om[:,jnp.newaxis]*self.dT) * \
                            (self.window_FD[shift_up%self.N][:,jnp.newaxis] +
                                self.window_FD[shift_do%self.N][:,jnp.newaxis])

                basis = basis.at[:, n_vals, 0].set(fNy_term)

            self._cached_Gnm_basis = basis

        return self._cached_Gnm_basis
    
    def gnm(self, 
            n : int, 
            m : int) -> jnp.ndarray:
        r"""
        Compute the time-domain representation of the wavelets, 
        :math:`g_{nm}(t)`.

        This method computes the frequency-domain wavelets for a single choice
        of :math:`n` and :math:`m` and performs and inverse Fourier transform.
        If you instead want to compute the full wavelet basis for all :math:`n`
        and :math:`m` efficiently, use the `gnm_basis` method.

        Parameters
        ----------
        n : int
            Wavelet time index.
        m : int
            Wavelet frequency index.

        Returns
        -------
        gnm : jnp.ndarray 
            Array shape (N,). The time-domain wavelet.
        """
        assert self.check_indices(n, m), f"Invalid indices: {n=} {m=}"

        Gnm = self.Gnm(n, m)

        gnm = jnp.fft.ifft(jnp.fft.ifftshift(Gnm)).real / self.dt

        return gnm

    @partial(jax.jit, static_argnums=0)
    def gnm_dual(self, 
                 n : int, 
                 m : int) -> jnp.ndarray:
        r"""
        This method compute the time-domain dual basis wavelets 
        :math:`\hat{g}_{nm}(t)` using the following expressions,

        .. math::
            \hat{g}_{nm}(t) = \begin{cases}
                    \sqrt{2} (-1)^{nm}
                        \sin\left(\frac{\pi m t}{\Delta T}\right) 
                            \phi(t-n\Delta T) 
                                & \mathrm{if}\;n+m\;\mathrm{even} \\
                    \sqrt{2} 
                        \cos\left(\frac{\pi m t}{\Delta T}\right) 
                            \phi(t-n\Delta T) 
                                & \mathrm{if}\;n+m\;\mathrm{odd}
            \end{cases} .

        These expressions are used for all :math:`m`, including for :math:`m=0`. 
        Therefore, this only gives the correct results for :math:`m>0`.

        Parameters
        ----------
        n : int
            Wavelet time index.
        m : int
            Wavelet frequency index.

        Returns
        -------
        ghat_nm : jnp.ndarray 
            Array of shape (N,). The time-domain wavelet basis.
        """
        k_vals = jnp.arange(self.N)

        shift = ((n+m)%2) * jnp.pi/2.

        ghat_nm = jnp.sqrt(2.) * (-1)**(n*m) * \
                    jnp.sin(jnp.pi*m*k_vals/self.Nf + shift) * \
                        self.window_TD[(k_vals-n*self.Nf)%self.N]

        return ghat_nm
    
    @partial(jax.jit, static_argnums=0)
    def gnm_basis(self) -> jnp.ndarray:
        r"""
        Efficient computation of time-domain wavelet basis :math:`g_{nm}(f)`. 
        Instead of calling the functions for :math:`\tilde{G}_{nm}(f)` and 
        performing an inverse Fourier transform, as is done in the `gnm` method,
        this function shifts indices of `window_TD`.

        For :math:`m>0`, the wavelet is given by

        .. math::

            g_{nm}(t) = \begin{cases}
            \sqrt{2} (-1)^{mn} \cos\left(\frac{\pi m t}{\Delta T}\right) 
                \phi(t-n\Delta T) & \mathrm{if}\;n+m\;\mathrm{even} \\
            \sqrt{2} \sin\left(\frac{\pi m t}{\Delta T}\right) 
                \phi(t-n\Delta T) & \mathrm{if}\;n+m\;\mathrm{odd}
            \end{cases} .

        For the special case :math:`m=0`, the wavelet is given by

        .. math::

            g_{n0}(t) = \begin{cases}
                    \phi(t-2n\Delta T) & \mathrm{if}\;n<N_t/2 \\
                    \frac{1}{2} \exp(-4\pi i n f \Delta T) 
                        \left( \tilde{\Phi}(f-f_{\rm Ny}) 
                            + \tilde{\Phi}(f+f_{\rm Ny}) \right) 
                            & \mathrm{if}\; n\geq N_t/2
                \end{cases}.

        The result is cached to speed up subsequent calls.

        Returns
        -------
        basis : jnp.ndarray 
            Array of shape (N, Nt, Nf). The time-domain wavelet basis.
        """
        if self._cached_gnm_basis is not None:
            pass

        else:
            n_vals = jnp.arange(self.Nt)
            m_vals = jnp.arange(self.Nf)
            k_vals = jnp.arange(self.N)

            def temp_func(n, m):
                shift = ((n+m)%2) * jnp.pi/2.
                return jnp.sqrt(2.) * (-1)**(n*m) * \
                            jnp.cos(jnp.pi*m*k_vals/self.Nf-shift) * \
                                self.window_TD[(k_vals-n*self.Nf)%self.N]

            f_vmapped = jax.vmap(jax.vmap(temp_func, 
                                        in_axes=(None, 0)), 
                                in_axes=(0, None))

            basis = f_vmapped(n_vals, m_vals)
            basis = jnp.transpose(basis, (2, 0, 1))

            if self.calc_m0:
                # overwrite m=0 terms for n<Nt/2 (zero-frequency terms)
                n_vals = jnp.arange(self.Nt//2)

                f0_term = self.window_TD[(k_vals[:,jnp.newaxis]
                                    -2*n_vals[jnp.newaxis,:]*self.Nf)%self.N]

                basis = basis.at[:, n_vals, 0].set(f0_term)

                # overwrite m=0 terms for n>=Nt/2 (Nyquist-frequency terms)
                n_vals = jnp.arange(self.Nt//2, self.Nt)

                def temp_func(n):
                    return (-1)**(k_vals) * \
                            self.window_TD[(k_vals-2*n*self.Nf)%self.N]

                f_vmapped = jax.vmap(temp_func)

                fNy_term = f_vmapped(n_vals).T

                basis = basis.at[:, n_vals, 0].set(fNy_term)

            self._cached_gnm_basis = basis

        return self._cached_gnm_basis

    @partial(jax.jit, static_argnums=0)
    def short_fft(self, x : jnp.ndarray) -> jnp.ndarray:
        r"""
        The windowed short FFT of the input.

        The input time series is split into :math:`N_t` overlapping segments 
        each of length :math:`K` and with a hop interval of :math:`N_f` between
        their centres. Each of these segments is then windowed and FFT'd.

        .. math::

            X_n[j] = \sum_{k=-K/2}^{K/2-1} \exp(2\pi i kj/K) x[nN_f+k] \phi[k]

        Parameters
        ----------
        x : jnp.ndarray
            Array shape (N,). Input time series signal to be transformed.

        Returns
        -------
        windowed_fft : jnp.ndarray
            Array shape shape (Nt, K). Short FFT of the input, :math:`X_n[j]`.
        """
        x = jnp.asarray(x)

        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        windowed_fft = overlapping_windows(x, self.K, self.Nt, self.Nf)

        k_vals = jnp.arange(-self.K//2, self.K//2)
        sign = (-1)**jnp.arange(self.K)

        windowed_fft *= self.window_TD[k_vals%self.N]

        windowed_fft = jnp.fft.ifft(windowed_fft, axis=-1) * self.K * sign

        return windowed_fft
    
    @partial(jax.jit, static_argnums=0)
    def forward_transform_exact(self, x : jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        This method computes the wavelet coefficients using the exact expression

        .. math::

            w_{nm} = \delta t \sum_{k=0}^{N-1} g_{nm}[k] x[k] ,

        where the sum is over the whole time-domain signal (no truncation). 
        
        This method is slow but exact.

        Parameters
        ----------
        x : jnp.ndarray
            Array shape shape (N,). Input time-domain signal to be transformed.

        Returns
        -------
        w : jnp.ndarray
            Array shape shape (Nt, Nf). 
            WDM time-frequency-domain wavelet coefficients. 
        """
        x = jnp.asarray(x)

        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"

        gnm_basis = jnp.transpose(self.gnm_basis(), (1,2,0))

        w = jnp.sum(gnm_basis * x, axis=-1) * self.dt

        return w
    
    @partial(jax.jit, static_argnums=0)
    def forward_transform_truncated(self, x : jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        This method computes the wavelet coefficients using the truncated 
        expressions

        .. math::

            w_{n0} = \delta t\sum_{k=-K/2}^{K/2-1} 
                    g_{nm}[k + 2 n N_f] x[k + 2 n N_f] ,

        .. math::

            w_{nm} = \delta t\sum_{k=-K/2}^{K/2-1} 
                    g_{nm}[k + n N_f] x[k + n N_f] \quad \mathrm{for} \; m>0 ,

        where the sum is over the truncated window of length :math:`K=2qN_f`.

        In the above expressions, indices out of bounds of the array are 
        to be understood as wrapping around circularly.

        Parameters
        ----------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.

        Returns
        -------
        w : jnp.ndarray 
            Array shape (Nt, Nf). 
            WDM time-frequency-domain wavelet coefficients.

        Notes
        -----
        This method is slow. It is only intended to be used for testing and 
        debugging purposes. 
        """
        x = jnp.asarray(x)

        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        w = jnp.zeros((self.Nt, self.Nf), dtype=self.jax_dtype) 

        B = self.gnm_basis()

        k_vals = jnp.arange(-self.K//2, self.K//2)

        for n in range(self.Nt):
            for m in range(not self.calc_m0, self.Nf): # start at m=0 or 1 
                gnm = B[:, n, m]
                gnm_x = gnm[(k_vals+(1 if m>0 else 2)*n*self.Nf)%self.N] * \
                            x[(k_vals+(1 if m>0 else 2)*n*self.Nf)%self.N]
                w = w.at[n, m].set(self.dt*jnp.sum(gnm_x))

        return w

    @partial(jax.jit, static_argnums=0)
    def forward_transform_truncated_window(self, 
                                           x : jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        This method computes the wavelet coefficients using the truncated 
        expressions using the window function:

        .. math::

            w_{n0} = \delta t \begin{cases} 
                        \sum_{k=-K/2}^{K/2-1} x[k+2nN_f]\phi[k] 
                                & \mathrm{if}\;n<N_t/2 \\
                        \sum_{k=-K/2}^{K/2-1} (-1)^k x[k+2nN_f]\phi[k] 
                                & \mathrm{if}\;n\geq N_t/2 \\
                    \end{cases} ,

        .. math::

            w_{nm} = \sqrt{2}\delta t \, \mathrm{Re} \sum_{k=-K/2}^{K/2-1} 
                        C^*_{nm} \exp\left(\frac{i\pi km}{N_f}\right) 
                        x[k+nN_f] \phi[k] \quad \mathrm{for}\; m>0.

        Parameters
        ----------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.

        Returns
        -------
        w : jnp.ndarray 
            Array shape (Nt, Nf). 
            WDM time-frequency-domain wavelet coefficients.

        Notes
        -----
        This method is slow. It is only intended to be used for testing and 
        debugging purposes. 
        """
        x = jnp.asarray(x)

        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        w = jnp.zeros((self.Nt, self.Nf), dtype=self.jax_dtype) 

        n_vals = jnp.arange(self.Nt)
        m_vals = jnp.arange(self.Nf)
        k_vals = jnp.arange(-self.K//2, self.K//2)

        k_plus_n = (k_vals[:,jnp.newaxis]+n_vals[jnp.newaxis,:]*self.Nf)%self.N
        mk = m_vals[jnp.newaxis,jnp.newaxis,:]*k_vals[:,jnp.newaxis,jnp.newaxis]

        w = jnp.sqrt(2.) * self.dt * \
                jnp.sum(
                    jnp.conj(self.Cnm[jnp.newaxis,:,:]) * \
                    jnp.exp((1j)*jnp.pi*mk/self.Nf) * \
                    x[k_plus_n][:,:,jnp.newaxis] * \
                    self.window_TD[k_vals%self.N,jnp.newaxis,jnp.newaxis], 
                axis=0).real

        if self.calc_m0:
            # overwrite m=0 terms for n<Nt/2 (zero-frequency terms)
            n_vals = jnp.arange(self.Nt//2)

            k_plus_2n = (k_vals[:,jnp.newaxis]+2*n_vals[jnp.newaxis,:]*self.Nf)

            f0_term = self.dt * jnp.sum(
                            self.window_TD[k_vals%self.N, jnp.newaxis] * \
                            x[k_plus_2n%self.N],
                        axis=0)

            w = w.at[n_vals, 0].set(f0_term)

            # overwrite m=0 terms for n>=Nt/2 (Nyquist-frequency terms)
            n_vals = jnp.arange(self.Nt//2, self.Nt)

            fNy_term = self.dt * jnp.sum( 
                            (-1)**k_vals[:,jnp.newaxis] * \
                            self.window_TD[k_vals%self.N, jnp.newaxis] * \
                            x[k_plus_2n%self.N],
                        axis=0)

            w = w.at[n_vals, 0].set(fNy_term)

        return w
    
    @partial(jax.jit, static_argnums=0)
    def forward_transform_short_fft(self, x : jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        For the :math:`m>0` terms, the wavelet coefficients are calculated 
        using the following expression,

        .. math::

            w_{nm} = \sqrt{2} \delta t \, \mathrm{Re}\, C_{nm}^* X_n[mq] ,

        where the short FFT is defined as 

        .. math::

            X_n[j] = \sum_{k=-K/2}^{K/2-1} \exp(2\pi i kj/K) x[nN_f+k] \phi[k].

        The :math:`m=0` terms, if required, are calculated using the same method 
        as in `forward_transform_truncated_window`. 

        Parameters
        ----------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.

        Returns
        -------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency-domain wavelet coefficients.

        Notes
        -----
        This method is fairly fast. But `forward_transform_fft` is usually 
        faster. This is included for testing and debugging purposes.
        """
        x = jnp.asarray(x)

        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"

        X = self.short_fft(x)

        m_vals = jnp.arange(self.Nf)

        w = jnp.sqrt(2.) * self.dt * \
                    jnp.real( jnp.conj(self.Cnm) * X[:,(m_vals*self.q)%self.K] )

        k_vals = jnp.arange(-self.K//2, self.K//2)

        if self.calc_m0:
            # overwrite m=0 terms for n<Nt/2 (zero-frequency terms)
            n_vals = jnp.arange(self.Nt//2)

            k_plus_2n = (k_vals[:,jnp.newaxis]+2*n_vals[jnp.newaxis,:]*self.Nf)

            f0_term = self.dt * jnp.sum(
                            self.window_TD[k_vals%self.N, jnp.newaxis] * \
                            x[k_plus_2n%self.N],
                        axis=0)

            w = w.at[n_vals, 0].set(f0_term)

            # overwrite m=0 terms for n>=Nt/2 (Nyquist-frequency terms)
            n_vals = jnp.arange(self.Nt//2, self.Nt)

            fNy_term = self.dt * jnp.sum( 
                            (-1)**k_vals[:,jnp.newaxis] * \
                            self.window_TD[k_vals%self.N, jnp.newaxis] * \
                            x[k_plus_2n%self.N],
                        axis=0)

            w = w.at[n_vals, 0].set(fNy_term)

        return w
    
    @partial(jax.jit, static_argnums=0)
    def forward_transform_fft(self, x : jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        For the :math:`m>0` terms, the wavelet coefficients are calculated 
        using the following expression,

        .. math::

            w_{nm} = \frac{\sqrt{2}\delta t}{N} (-1)^{nm} \,\mathrm{Re}\, 
                \Big( C_{nm}^* x_m[n] \Big) 

        where

        .. math::

            x_m[n] = \sum_{l=-N_t/2}^{N_t/2-1} \exp\left(\frac{2\pi i nl}{N_t}
                        \right) \Phi[l] X[l-mN_t/2] .

        The :math:`m=0` terms, if required, are calculated using the same method
        as in `forward_transform_truncated_window`. 

        This is vectorised to allow for batch jobs computing the dwt for 
        multiple time series at once; note the shapes of the input and output 
        arrays.

        Parameters
        ----------
        x : jnp.ndarray 
            The time-domain signal. Array shape (..., N). 

        Returns
        -------
        w : jnp.ndarray 
            Wavelet coefficients. Array shape (..., Nt, Nf). 

        Notes
        -----
        This method is fast. Use this to perform discrete wavelet transforms for
        production analysis. This method is called by `self.dwt`.
        """
        x = jnp.asarray(x, dtype=self.jax_dtype)

        assert x.shape[-1:] == (self.N,), \
                f"Input signal must have shape({self.Nt}, {self.Nf}), " \
                f"got {x.shape[-1:]=}."

        leading = x.shape[:-1]

        l_vals = jnp.arange(-self.Nt//2, self.Nt//2)
        n_vals = jnp.arange(self.Nt)
        m_vals = jnp.arange(self.Nf)
        mask = l_vals[:,jnp.newaxis] - \
                m_vals[jnp.newaxis,:]*self.Nt//2

        X = jnp.fft.fft(x, axis=-1) * self.dt

        X = jnp.take(X, mask, axis=-1, mode='wrap')

        Phi = jnp.fft.ifftshift(self.window_FD)[*(jnp.newaxis,)*len(leading),
                                                l_vals,
                                                jnp.newaxis]

        x_mn = self.Nt * jnp.fft.ifft(Phi*X, axis=-2)

        w = jnp.sqrt(2.) * self.df * \
                (-1)**(n_vals[:,jnp.newaxis] * m_vals[jnp.newaxis,:]) * \
                    jnp.real( jnp.conj(self.Cnm[:,:]) * x_mn ) * \
                        (-1)**(n_vals[:,jnp.newaxis]) 

        k_vals = jnp.arange(-self.K//2, self.K//2)

        if self.calc_m0:
            # overwrite m=0 terms for n<Nt/2 (zero-frequency terms)
            n_vals = jnp.arange(self.Nt//2)

            k_plus_2n = (k_vals[:,jnp.newaxis]+2*n_vals[jnp.newaxis,:]*self.Nf)

            f0_term = self.dt * jnp.sum(
                            self.window_TD[k_vals%self.N, jnp.newaxis] * \
                            jnp.take(x, k_plus_2n, axis=-1, mode='wrap'),
                        axis=-2)

            w = w.at[..., n_vals, 0].set(f0_term)

            # overwrite m=0 terms for n>=Nt/2 (Nyquist-frequency terms)
            n_vals = jnp.arange(self.Nt//2, self.Nt)

            fNy_term = self.dt * jnp.sum( 
                            (-1)**k_vals[:,jnp.newaxis] * \
                            self.window_TD[k_vals%self.N, jnp.newaxis] * \
                            jnp.take(x, k_plus_2n, axis=-1, mode='wrap'),
                        axis=-2)

            w = w.at[..., n_vals, 0].set(fNy_term)

        return w
    
    @partial(jax.jit, static_argnums=0)
    def inverse_transform(self, w : jnp.ndarray) -> jnp.ndarray:
        r""" 
        Perform the inverse discrete wavelet transform. Transforms the wavelet 
        coefficients from the time-frequency domain into the time domain.

        This method computes the inverse dwt using the truncated wavelets.
        This is also vectorised to allow for batch jobs computing the idwt for 
        multiple sets of wavelet coefficients at once; note the shapes of the 
        input and output arrays.

        Parameters
        ----------
        w : jnp.ndarray 
            Wavelet coefficients. Array shape (..., Nt, Nf). 

        Returns
        -------
        x : jnp.ndarray 
            The time-domain signal. Array shape (..., N). 
        """
        w = jnp.asarray(w, dtype=self.jax_dtype)

        assert w.shape[-2:] == (self.Nt, self.Nf), \
                f"Input coefficients must have shape ({self.Nt}, {self.Nf}), " \
                f"got {w.shape[-2:].shape=}."

        leading = w.shape[:-2]

        x = jnp.zeros(leading+(self.N,), dtype=self.jax_dtype)

        @jax.jit
        def add_one_time(x, n):
            k_vals = jnp.arange(-self.K//2, self.K//2)
            indices = (k_vals+n*self.Nf)%self.N

            @jax.jit
            def add_one_freq(x, m):
                shift = ((n+m)%2) * jnp.pi/2.

                wavelet = jnp.sqrt(2.) * (-1)**(n*m) * \
                            jnp.cos(jnp.pi*m*indices/self.Nf-shift) * \
                                 self.window_TD[k_vals]

                coeff = jnp.atleast_1d(w[...,n,m])
                term  = coeff[..., None] * wavelet[None, ...] 
                updates_shape = x[..., indices].shape
                x = x.at[..., indices].add(jnp.reshape(term, updates_shape))
                return x

            x = jax.lax.fori_loop(1, # only sum over m>0
                                  self.Nf, 
                                  lambda m, acc: add_one_freq(acc, m), 
                                  x)
            return x

        x = jax.lax.fori_loop(0, 
                              self.Nt, 
                              lambda n, acc: add_one_time(acc, n), 
                              x)
        
        if self.calc_m0:
            # overwrite m=0 terms for n<Nt/2 (zero-frequency terms)
            n_vals = jnp.arange(self.Nt//2)

            @jax.jit
            def add_zero_freq(x, n):
                k_vals = jnp.arange(-self.K//2, self.K//2)
                wavelet = self.window_TD[k_vals]
                indices = (k_vals+2*n*self.Nf)%self.N
                coeff = jnp.atleast_1d(w[...,n,0])
                term  = coeff[..., None] * wavelet[None, ...] 
                updates_shape = x[..., indices].shape
                x = x.at[..., indices].add(jnp.reshape(term, updates_shape))
                return x

            x = jax.lax.fori_loop(0, 
                                  self.Nt//2,
                                  lambda n, acc: add_zero_freq(acc, n), 
                                  x)
            
            @jax.jit
            def add_Nyquist_freq(x, n):
                k_vals = jnp.arange(-self.K//2, self.K//2)
                wavelet = (-1)**(k_vals) * self.window_TD[k_vals]
                indices = (k_vals+2*n*self.Nf)%self.N
                coeff = jnp.atleast_1d(w[...,n,0])
                term  = coeff[..., None] * wavelet[None, ...] 
                updates_shape = x[..., indices].shape
                x = x.at[..., indices].add(jnp.reshape(term, updates_shape))
                return x

            x = jax.lax.fori_loop(self.Nt//2, 
                                  self.Nt,
                                  lambda n, acc: add_Nyquist_freq(acc, n), 
                                  x)

        return x

    def inverse_transform_exact(self, w : jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the inverse discrete wavelet transform. Transforms the wavelet 
        coefficients from the time-frequency domain into the time domain.

        This method computes the inverse dwt direcrtly using the expression

        .. math::

            x[k] = \sum_{n=0}^{N_t-1} \sum_{m=0}^{N_f-1} w_{nm} g_{nm}[k] .

        This method is slow and very memory inefficient. It is here
        mainly for testing. Consider using `inverse_transform` instead.

        Parameters
        ----------
        w : jnp.ndarray 
            Array shape (Nt, Nf). 
            WDM time-frequency-domain wavelet coefficients.

        Returns
        -------
        x : jnp.ndarray 
            Array shape (N,). The time-domain signal.
        """
        w = jnp.asarray(w, dtype=self.jax_dtype)

        assert w.shape == (self.Nt, self.Nf), \
                f"Input coefficients must have shape ({self.Nt}, {self.Nf}), " \
                f"got {w.shape=}."

        gnm_basis = self.gnm_basis()

        wg = w * gnm_basis

        wg = wg.reshape(wg.shape[0], -1)

        x = jnp.sum(wg, axis=-1)

        return x

    def dwt(self, x : jnp.ndarray) -> jnp.ndarray:
        r"""
        Forward discrete wavelet transform.

        Calls `self.fast_forward_transform`. Vectorised to allow for 
        transforming multiple time series at once.

        Parameters
        ----------
        x : jnp.ndarray
            Input time series. Array shape=(N,) or (..., N).

        Returns
        -------
        w : jnp.ndarray
            Wavelet coefficients. Array shape=(Nt, Nf) or (..., Nt, Nf).
        """
        x = jnp.asarray(x, dtype=self.jax_dtype)

        assert jnp.all(jnp.isreal(x)), "time series must be real."

        return self.forward_transform_fft(x)
    
    def idwt(self, w : jnp.ndarray) -> jnp.ndarray:
        r"""
        Inverse discrete wavelet transform.

        Calls `self.inverse_transform`. Vectorised to allow for transforming 
        multiple time series at once.

        Parameters
        ----------
        w : jnp.ndarray
            Wavelet coefficients. Array shape=(Nt, Nf) or (..., Nt, Nf).

        Returns
        -------
        x : jnp.ndarray
            Input time series. Array shape=(N,) or (..., N).
        """
        w = jnp.asarray(w, dtype=self.jax_dtype)

        assert jnp.all(jnp.isreal(w)), "wavelet coefficients must be real."

        return self.inverse_transform(w)

    def build_time_delay_filter_interpolants(self,
                                             max_lag_L : int,
                                             num_interp_points : int,
                                             max_bytes : int
                                                = FILTER_TABLE_BLOCK_BYTES
                                             ) -> None:
        r""" 
        If the user needs to do any time-shift operations involving the 
        WDM wavelets, then this function should be called first. It tabulates 
        the time-delay filter functions :math:`T_l(\delta)` and
        :math:`T'_l(\delta)` for :math:`l=-L,\ldots,L-1, L` (where L is the max 
        lag) on a uniform grid in the range 
        :math:`-\Delta T/2\leq \delta\leq \Delta T/2`. Subsequent lookups
        interpolate linearly between the tabulated points.

        Parameters
        ----------
        max_lag_L : int
            The maximum lag index, :math:`L`.
        num_interp_points : int
            The number of interpolation points in the range
            :math:`-\Delta T/2\leq \delta\leq \Delta T/2.`
        max_bytes : int
            Working-set budget, in bytes, for one frequency block of the
            tabulation, passed straight to `build_filter_tables`. The
            tabulation is blocked over frequency so that its peak allocation
            is bounded by this rather than by :math:`N`; lower it if the build
            still does not fit. Defaults to
            `filters.FILTER_TABLE_BLOCK_BYTES`, 256 MiB. Optional.

        Returns
        -------
        filter_tables : jnp.array
            dtype=float, shape=(2, 2L+1, num_interp_points). Index 0 is
            :math:`T'_l`, index 1 is :math:`T_l`. Also stored as
            `self.filter_tables`, but the returned value is what should be
            passed to the methods that use it - see the note below them.

        Notes
        -----
        The returned table is small - :math:`2(2L+1)` by `num_interp_points` -
        and independent of :math:`N`. Only the intermediates scale with the
        time series, and `build_filter_tables` bounds those, so this is safe to
        call on year-plus grids for any `num_interp_points` whose table itself
        fits in memory.
        """
        assert max_lag_L > 0, \
                        "Max lag must be positive"

        assert max_lag_L < self.Nt, \
                "Max lag can't be larger than number of time points"

        self.max_lag_L = int(max_lag_L)
        self.num_interp_points = int(num_interp_points)

        self.delta_interp_grid = jnp.linspace(-0.5*self.dT,
                                              +0.5*self.dT,
                                              self.num_interp_points)

        # shape (2, 2L+1, num_interp_points); index 0 is T', index 1 is T
        self.filter_tables = build_filter_tables(
                                jnp.arange(-self.max_lag_L, self.max_lag_L+1),
                                self.delta_interp_grid,
                                self.freqs,
                                self.window_FD,
                                self.dT,
                                self.dF,
                                self.df,
                                max_bytes=max_bytes)

        return self.filter_tables

    # Every method below takes `filter_tables` as an explicit argument rather
    # than reading `self.filter_tables`. This is deliberate, and it is a JAX
    # requirement, not a style choice.
    #
    # The body of a jitted function runs only ONCE, during tracing. Anything it
    # reads off `self` at that moment is frozen into the compiled code as a
    # literal. `self` is a static argument and a python object is compared by
    # identity, so rebuilding an attribute does not invalidate the compilation
    # cache. A jitted method that read `self.filter_tables` directly would
    # therefore keep returning the first build's numbers after
    # `build_time_delay_filter_interpolants` was called again - silently.
    #
    # Passing the tables in makes them an ordinary traced argument: JAX keys
    # the cache on their shape, so new values flow straight through and only a
    # change of `max_lag_L` or `num_interp_points` forces a recompile.
    #
    # `test_rebuilding_interpolants_takes_effect` guards this.

    def time_delay_filter_Tl(self,
                             filter_tables : jnp.array,
                             lag_index_l : jnp.array,
                             delta : jnp.array) -> jnp.array:
        r"""
        Fast, vectorised way of calling the time-delay filter function
        :math:`T_l(\delta)`, which interpolates the pre-built table.

        Parameters
        ----------
        filter_tables : jnp.array
            The tables returned by `build_time_delay_filter_interpolants`,
            dtype=float, shape=(2, 2L+1, num_interp_points).
        lag_index_l : jnp.array
            Array of lag indices, dtype=int, shape=(A,)
        delta : jnp.array
            Array of time delays, dtype=float, shape=(B,)

        Returns
        -------
        Tl : jnp.array
             Array of time-delay filters, dtype=float, shape=(A, B)
        """
        return self._interp_filter(filter_tables[1], lag_index_l, delta)

    def time_delay_filter_Tprimel(self,
                                  filter_tables : jnp.array,
                                  lag_index_l : jnp.array,
                                  delta : jnp.array) -> jnp.array:
        r"""
        Fast, vectorised way of calling the time-delay filter function
        :math:`T'_l(\delta)`, which interpolates the pre-built table.

        Parameters
        ----------
        filter_tables : jnp.array
            The tables returned by `build_time_delay_filter_interpolants`,
            dtype=float, shape=(2, 2L+1, num_interp_points).
        lag_index_l : jnp.array
            Array of lag indices, dtype=int, shape=(A,)
        delta : jnp.array
            Array of time delays, dtype=float, shape=(B,)

        Returns
        -------
        Tprimel : jnp.array
                Array of time-delay filters, dtype=float, shape=(A, B)
        """
        return self._interp_filter(filter_tables[0], lag_index_l, delta)

    @partial(jax.jit, static_argnums=0)
    def _interp_filter(self,
                       table : jnp.array,
                       lag_index_l : jnp.array,
                       delta : jnp.array) -> jnp.array:
        r"""
        Look up a pre-built time-delay filter table at arbitrary lags and 
        delays.

        The tabulation grid is uniform, so this is a gather followed by a 
        linear blend - there is no search. Delays outside 
        :math:`[-\Delta T/2, \Delta T/2]` are folded back in by shifting the 
        lag index; lags that then fall outside :math:`\pm L` contribute zero.

        The table is taken as an argument rather than read from `self` so that
        rebuilding it invalidates the compilation cache correctly.

        Parameters
        ----------
        table : jnp.ndarray
            A filter table, dtype=float, shape=(2L+1, num_interp_points).
        lag_index_l : jnp.ndarray
            Array of lag indices, dtype=int, shape=(A,)
        delta : jnp.ndarray
            Array of time delays, dtype=float, shape=(B,)

        Returns
        -------
        filt : jnp.ndarray
            Array of time-delay filters, dtype=float, shape=(A, B)
        """
        num_lags, num_interp_points = table.shape
        max_lag_L = (num_lags - 1)//2

        k, delta_wrapped = jnp.divmod(delta + 0.5*self.dT, self.dT)
        delta_wrapped = delta_wrapped - 0.5*self.dT

        row = lag_index_l[:, jnp.newaxis] \
                - k.astype(int)[jnp.newaxis, :] + max_lag_L
        in_range = (row >= 0) & (row <= 2*max_lag_L)
        row = jnp.clip(row, 0, 2*max_lag_L)

        u = (delta_wrapped + 0.5*self.dT)/self.dT * (num_interp_points - 1)
        i0 = jnp.clip(jnp.floor(u).astype(int), 0, num_interp_points - 2)
        frac = (u - i0)[jnp.newaxis, :]

        lo = table[row, i0[jnp.newaxis, :]]
        hi = table[row, i0[jnp.newaxis, :] + 1]

        return jnp.where(in_range, lo + frac*(hi - lo), 0.0)

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def time_delay_matrix_X(self, filter_tables, n, m, l, sigma,
                            delta) -> jnp.array:
        r"""
        Generate an array time delay matrix elements
        :math:`X_{n(n-l),m(m+\sigma)}(-\delta_n)` used in the variable time
        shifting method for fixed :math:'\sigma' and :math:'l' values.

        This is a direct transcription of the defining expressions and is
        deliberately left in that form: it is the reference against which
        `apply_variable_time_shift` is tested. The shift itself does not call
        this method.

        `l` and `sigma` are static, so the three frequency-lag cases are
        selected here in python rather than with `lax.switch` - which, being
        traced, would evaluate all three.

        Parameters
        ----------
        filter_tables : jnp.array
            The tables returned by `build_time_delay_filter_interpolants`,
            dtype=float, shape=(2, 2L+1, num_interp_points).
        n : jnp.array
            Time indices. Array, dtype=int, shape=(Nt,)
        m : jnp.array
            Freq indices. Array, dtype=int, shape=(Nf,)
        l : int
            Time lag index.
        sigma : int
            Freq lag. This should be :math:`0` or :math:`\pm 1`
        delta : jnp.array
            Array, dtype=float, shape=(Nt,)

        Returns
        -------
        X : jnp.array
            The X coefficients. Array, dtype=float, shape=(Nt, Nf)
        """
        n_ = n - l
        m_ = m + sigma

        # (-1)^{(n-n')m}, and i^{(n'+m')%2} / i^{(n+m)%2}
        alternating = (-1)**((n - n_)[:, jnp.newaxis] * m[jnp.newaxis, :])
        parity = jnp.conjugate((1j)**((n[:, jnp.newaxis]
                                       + m[jnp.newaxis, :]) % 2)) * \
                 (1j)**((n_[:, jnp.newaxis] + m_[jnp.newaxis, :]) % 2)

        if sigma == 0:
            scalar = 1.0
            offset = 0.0
            table = filter_tables[1]
        elif sigma == -1:
            scalar = (1j)**l
            offset = -0.5
            table = filter_tables[0]
        else:
            scalar = (-1j)**l
            offset = +0.5
            table = filter_tables[0]

        carrier = jnp.exp(-2*jnp.pi*(1j)*(m[jnp.newaxis, :] + offset)
                          * self.dF * delta[:, jnp.newaxis])

        filt = self._interp_filter(table, jnp.array([l]), -delta)

        X = alternating * scalar * carrier * parity \
                * filt[0][:, jnp.newaxis]

        return jnp.real(X)

    @partial(jax.jit, static_argnums=0, static_argnames=('lag_block',))
    def apply_variable_time_shift(self,
                                  filter_tables : jnp.array,
                                  wdm_coeff : jnp.array,
                                  delta : jnp.array,
                                  lag_block : int = None) -> jnp.array:
        r"""
        Perform the variable time shift operation on a grid of WDM coefficients
        by evaluating the sum

        .. math::

            \sum_{\substack{l \le |L| \\ \sigma = \{-1,0,1\} }} 
             \omega_{(n-l) (m+\sigma)} \, X_{n(n-l);m(m+\sigma)}(-\delta_n) .

        If the original grid represents the coefficients of a function
        :math:'f(t) = \sum_{nm} \omega_{nm} g_{nm}(t)'
        and we sample a time-shift :math:'\delta(t_n) = \delta_n' then the
        resulting shifted grid :math:'\tilde{\omega}_{nm}'
        is the equivalent to the WDM transform of :math:'f(t+\delta(t))'.

        Mind the sign. Shifting the *coefficients* is the active
        transformation and carries the opposite sign to shifting the *basis
        functions*, since
        :math:`\langle f(t-\delta), g_{nm}(t)\rangle
        = \langle f(t), g_{nm}(t+\delta)\rangle`. That is why the matrix
        elements above are evaluated at :math:`-\delta_n` while the output is
        the transform of :math:`f(t+\delta(t))`: pass the delay with the sign
        of the shift you want applied to the samples. Checked directly - an
        asymmetric pulse fed through with a constant :math:`\delta>0` comes
        back centred :math:`\delta` seconds *earlier*.

        Terms where :math:`n-l` or :math:`m+\sigma` fall outside the array
        limits are wrapped around periodically. It is the users reponsibility
        to ensure that the input array `wdm_coeff` is suitably zero padded.

        The matrix elements :math:`X` are the ones written out plainly in
        `time_delay_matrix_X`; this method evaluates the same sum in real
        arithmetic for speed. The two are checked against each other by
        `test_apply_variable_time_shift_matches_X_reference`, so read
        `time_delay_matrix_X` first if you want the expressions in their
        textbook form.

        The matrix elements are evaluated in real arithmetic - see
        `real_matrix_element` below for that rearrangement. The carrier phase
        :math:`\phi=2\pi(m+\sigma/2)\Delta F\delta_n` does not depend on
        :math:`l`, so it is built once outside the lag loop and the
        :math:`\sigma=\pm1` cases follow from it by angle addition.

        Parameters
        ----------
        filter_tables : jnp.array
            The tables returned by `build_time_delay_filter_interpolants`,
            dtype=float, shape=(2, 2L+1, num_interp_points).
        wdm_coeff : jnp.array
            WDM coefficient grid. Array, dtype=float, shape=(Nt,Nf)
        delta : jnp.array
            Array, dtype=float, shape=(Nt,)
        lag_block : int or None
            Number of lags handled per loop iteration. `None` (the default)
            unrolls the lag sum completely, which is fastest for small `L`
            because the lag index becomes a compile-time constant. For
            :math:`L \gtrsim 30` prefer a small block (8 is a reasonable
            start): the unrolled graph becomes large enough that scheduling
            cost, compile time and compile-time memory outweigh the benefit.

        Returns
        -------
        shifted_wdm_coeff : jnp.array
            Shifted wdm_coefficients. Array, dtype=float, shape=(Nt, Nf)
        """
        max_lag_L = (filter_tables.shape[1] - 1)//2

        n = jnp.arange(self.Nt)
        m = jnp.arange(self.Nf)

        # (-1)^n and (-1)^m, kept rank-1 so they never form a grid
        sign_n = jnp.where(n % 2 == 0, 1.0, -1.0)[:, jnp.newaxis]
        sign_m = jnp.where(m % 2 == 0, 1.0, -1.0)[jnp.newaxis, :]

        # cos/sin of the sigma=0 carrier; sigma=+-1 by angle addition with
        # the rank-1 half-bin phase
        theta = 2*jnp.pi*self.dF*jnp.outer(delta, m)
        cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
        half = jnp.pi*self.dF*delta[:, jnp.newaxis]
        cos_h, sin_h = jnp.cos(half), jnp.sin(half)
        carrier = {-1: (cos_t*cos_h + sin_t*sin_h, sin_t*cos_h - cos_t*sin_h),
                    0: (cos_t, sin_t),
                   +1: (cos_t*cos_h - sin_t*sin_h, sin_t*cos_h + cos_t*sin_h)}

        # every lag at once: shape (2L+1, Nt), no Nf axis
        lags = jnp.arange(-max_lag_L, max_lag_L + 1)
        Tl = self._interp_filter(filter_tables[1], lags, -delta)
        Tprimel = self._interp_filter(filter_tables[0], lags, -delta)
        delay_filter = {-1: Tprimel, 0: Tl, +1: Tprimel}

        # Cyclic extension by L rows and one column, so the wrapped read is a
        # contiguous slice rather than a gather. This is NOT zero padding: it
        # repeats the caller's own coefficients, exactly as the periodic wrap
        # would. Zero padding remains the caller's responsibility.
        cyclic_ext = jnp.concatenate([wdm_coeff[-max_lag_L:],
                                      wdm_coeff,
                                      wdm_coeff[:max_lag_L]], axis=0)
        cyclic_ext = jnp.concatenate([cyclic_ext[:, -1:],
                                      cyclic_ext,
                                      cyclic_ext[:, :1]], axis=1)

        # (1j)**l for l modulo 4, as (real, imaginary) pairs
        i_pow = jnp.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])

        def real_matrix_element(l, i, sigma):
            r"""
            :math:`X_{n(n-l);m(m+\sigma)}(-\delta_n)` in real arithmetic, for
            one lag and one frequency offset. Array, shape=(Nt, Nf).

            Parameters
            ----------
            l : int
                Time lag index, :math:`-L\leq l\leq L`.
            i : int
                Position of `l` in the tabulated range, :math:`i=l+L`. Used to
                index the pre-evaluated delay filters, which are stored over
                that range rather than over :math:`l` itself.
            sigma : int
                Frequency lag, :math:`0` or :math:`\pm1`.
            """
            # C = (+-i)**l, the scalar lag phase: sigma=-1 carries (1j)**l,
            # sigma=+1 its conjugate, and sigma=0 has no such factor at all.
            i_pow_re, i_pow_im = i_pow[l % 4]
            if sigma == 0:
                lag_phase_re, lag_phase_im = 1.0, 0.0
            else:
                lag_phase_re = i_pow_re
                lag_phase_im = i_pow_im if sigma == -1 else -i_pow_im

            # The parity factor conj(i**a) i**b, with a = (n+m) % 2 and
            # b = (n'+m') % 2, collapses to a choice of just two cases:
            #     (l+sigma) even ->  1
            #     (l+sigma) odd  ->  i (-1)**(n+m)
            # Taking the real part of C exp(-i phi) under each then gives
            #     even ->  lag_phase_re cos(phi) + lag_phase_im sin(phi)
            #     odd  -> -lag_phase_im cos(phi) + lag_phase_re sin(phi)
            # which is why nothing here needs to be complex.
            carrier_cos, carrier_sin = carrier[sigma]
            element = jnp.where(
                    ((l + sigma) % 2) != 0,
                    sign_n*sign_m*(-lag_phase_im*carrier_cos
                                   + lag_phase_re*carrier_sin),
                    lag_phase_re*carrier_cos + lag_phase_im*carrier_sin)

            # the two remaining factors: (-1)**(l m), and the delay filter
            alternating = jnp.where((l % 2) != 0, sign_m, 1.0)

            return element * alternating \
                        * delay_filter[sigma][i][:, jnp.newaxis]

        def body(i, acc):
            l = i - max_lag_L

            # omega_{(n-l)(m+sigma)}: the row shift depends on l and is shared
            # by all three sigma, so it is taken once here.
            rows = jax.lax.dynamic_slice_in_dim(cyclic_ext, max_lag_L - l,
                                                self.Nt, axis=0)

            for sigma in (-1, 0, 1):
                # offset by 1 because cyclic_ext carries one extra column at
                # each edge, so column m+sigma sits at index m+sigma+1
                coeff = jax.lax.slice_in_dim(rows, sigma + 1,
                                             sigma + 1 + self.Nf, axis=1)

                acc = acc + coeff*real_matrix_element(l, i, sigma)

            return acc

        acc = jnp.zeros((self.Nt, self.Nf), dtype=wdm_coeff.dtype)

        if lag_block is None:
            for i in range(2*max_lag_L + 1):
                acc = body(i, acc)
            return acc

        return jax.lax.fori_loop(0, 2*max_lag_L + 1, body, acc,
                                 unroll=lag_block)

    def __repr__(self) -> str:
        r"""
        String representation of the WDM_transform instance.

        Returns
        -------
        text : str
            A string representation of WDM_transform instance.
        """
        lines = []
        lines.append( (f"WDM_transform(Nf={self.Nf}, N={self.N}, q={self.q}, "
                f"d={self.d}, A_frac={self.A_frac}, calc_m0={self.calc_m0})") )
        lines.append( f"{self.Nt = } time cells" )
        lines.append( f"{self.Nf = } frequency cells" )
        lines.append( f"{self.dT = } time resolution" )
        lines.append( f"{self.dF = } frequency resolution" )
        lines.append( f"{self.dt = } time series cadence" )
        lines.append( f"{self.df = } time series fft frequency resolution" )
        lines.append( f"{self.K = } window length" )
        text = "\n".join(lines)
        return text

    def __call__(self, x : jnp.ndarray) -> jnp.ndarray:
        r"""
        Calls the forward transform self.dwt.
        """
        return self.dwt(x)
