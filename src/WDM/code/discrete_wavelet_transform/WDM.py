import jax
import jax.numpy as jnp
from matplotlib.pylab import indices

from jax.scipy.interpolate import RegularGridInterpolator

from WDM.code.utils.Meyer import Meyer
from WDM.code.utils.utils import C_nm, overlapping_windows
from WDM.code.time_delay_filters.filters import time_delay_filter_Tl
from WDM.code.time_delay_filters.filters import time_delay_filter_Tprimel

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
        self.Cnm = jnp.array([[ C_nm(n, m) for m in range(self.Nf)] 
                                            for n in range(self.Nt)])

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
                                             num_interp_points : int) -> None:
        r""" 
        If the user needs to do any time-shift operations involving the 
        WDM wavelets, then this function should be called first. It builds 
        interpolants for the time-delay filer functions :math:`T_l(\delta)` and
        :math:`T'_l(\delta)` for :math:`l=-L,\ldots,L-1, L` (where L is the max 
        lag) in the range :math:`-\Delta T/2\leq \delta\leq \Delta T/2`.

        Parameters
        ----------
        max_lag_L : int
            The maximum lag index, :math:`L`.
        num_interp_points : int
            The number of interpolation points in the range 
            :math:`-\Delta T/2\leq \delta\leq \Delta T/2.`

        Returns
        -------
        None
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

        # build Tl and Tlprime interpolants, store these functions in dicts
        Tl_interp = {}
        Tprimel_interp = {}

        for l in range(-max_lag_L, max_lag_L+1):
            # Tl interpolants
            data = jnp.zeros(self.num_interp_points)
            for delta_idx in range(self.num_interp_points):
                delta = self.delta_interp_grid[delta_idx]
                Tl_delta = time_delay_filter_Tl(l,
                                                delta,
                                                self.freqs,
                                                self.window_FD,
                                                self.dT,
                                                self.df)
                data = data.at[delta_idx].set(Tl_delta)
            Tl_interp[l] = RegularGridInterpolator(
                                (self.delta_interp_grid,), data)

            # Tlprime interpolants
            data = jnp.zeros(self.num_interp_points)  # Reset data array!
            for delta_idx in range(self.num_interp_points):
                delta = self.delta_interp_grid[delta_idx]
                Tlprime_delta = time_delay_filter_Tprimel(l,
                                                          delta,
                                                          self.freqs,
                                                          self.window_FD,
                                                          self.dT,
                                                          self.dF,
                                                          self.N,
                                                          self.df)
                data = data.at[delta_idx].set(Tlprime_delta)
            Tprimel_interp[l] = RegularGridInterpolator(
                                    (self.delta_interp_grid,), data)

        # store the functions in lists
        self.list_of_Tl_functions = [Tl_interp[l] 
                        for l in range(-self.max_lag_L, self.max_lag_L+1)]

        self.list_of_Tprimel_functions = [Tprimel_interp[l] 
                        for l in range(-self.max_lag_L, self.max_lag_L+1)]

        return None

    @partial(jax.jit, static_argnums=0)
    def time_delay_filter_Tl(self,
                             lag_index_l : jnp.array,
                             delta : jnp.array) -> jnp.array:
        r"""
        Fast, vectorised way of calling the time-delay filter function
        :math:`T_l(\delta)` which calls the pre-built interpolants.

        Parameters
        ----------
        lag_index_l : jnp.array
            Array of lag indices, dtype=int, shape=(A,)
        delta : jnp.array
            Array of time delays, dtype=float, shape=(B,)

        Returns
        -------
        Tl : jnp.array
             Array of time-delay filters, dtype=float, shape=(A, B)
        """
        k, delta_wrapped = jnp.divmod(delta+0.5*self.dT, self.dT)
        k = jnp.array(k, dtype=int)
        delta_wrapped = delta_wrapped - 0.5*self.dT

        def apply_func_single_lag(l_idx):
            # For a single lag index, evaluate across all deltas
            adjusted_indices = l_idx - k + self.max_lag_L

            # Check if indices are in valid range [0, 2*max_lag_L]
            in_range = (adjusted_indices >= 0) & (adjusted_indices <= 2*self.max_lag_L)

            # Clamp indices for safe evaluation (actual values will be masked)
            safe_indices = jnp.clip(adjusted_indices, 0, 2*self.max_lag_L)

            def eval_single_delta(safe_idx, dw, mask):
                result = jax.lax.switch(safe_idx,
                                       self.list_of_Tl_functions,
                                       jnp.array([dw]))[0]
                # Return 0 if out of range, otherwise return result
                return jnp.where(mask, result, 0.0)

            return jax.vmap(eval_single_delta)(safe_indices, delta_wrapped, in_range)

        Tl = jax.vmap(apply_func_single_lag)(lag_index_l)

        return Tl

    @partial(jax.jit, static_argnums=0)
    def time_delay_filter_Tprimel(self,
                                  lag_index_l : jnp.array,
                                  delta : jnp.array) -> jnp.array:
        r"""
        Fast, vectorised way of calling the time-delay filter function
        :math:`T'_l(\delta)` which calls the pre-built interpolants.

        Parameters
        ----------
        lag_index_l : jnp.array
            Array of lag indices, dtype=int, shape=(A,)
        delta : jnp.array
            Array of time delays, dtype=float, shape=(B,)

        Returns
        -------
        Tprimel : jnp.array
                Array of time-delay filters, dtype=float, shape=(A, B)
        """
        k, delta_wrapped = jnp.divmod(delta+0.5*self.dT, self.dT)
        k = jnp.array(k, dtype=int)
        delta_wrapped = delta_wrapped - 0.5*self.dT

        def apply_func_single_lag(l_idx):
            # For a single lag index, evaluate across all deltas
            adjusted_indices = l_idx - k + self.max_lag_L

            # Check if indices are in valid range [0, 2*max_lag_L]
            in_range = (adjusted_indices >= 0) & (adjusted_indices <= 2*self.max_lag_L)

            # Clamp indices for safe evaluation (actual values will be masked)
            safe_indices = jnp.clip(adjusted_indices, 0, 2*self.max_lag_L)

            def eval_single_delta(safe_idx, dw, mask):
                result = jax.lax.switch(safe_idx,
                                       self.list_of_Tprimel_functions,
                                       jnp.array([dw]))[0]
                # Return 0 if out of range, otherwise return result
                return jnp.where(mask, result, 0.0)

            return jax.vmap(eval_single_delta)(safe_indices, delta_wrapped, in_range)

        Tprimel = jax.vmap(apply_func_single_lag)(lag_index_l)

        return Tprimel

    def time_delay_matrix_X(self, n, m, l, sigma, delta) -> jnp.array:
        r"""
        Description.

        Generate an array time delay matrix elements :math:`X_{n(n-l),m(m+\sigma)}(-\delta_n)` 
        for fixed :math:'\sigma' and :math:'l' values.

        Parameters
        ----------
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
            The X coefficients. Array, dtype=floar, shape=(Nt, Nf)
        """
        sigma_idx = sigma + 1

        def case_minus_1(n, m, l, delta):
            # Code for sigma = -1
            n_ = n - l
            m_ = m - 1
            X = (-1)**( (n-n_)[:,jnp.newaxis] * m[jnp.newaxis,:]) * \
                (1j)**l * \
                jnp.exp(-2*jnp.pi*(1j)*(m[jnp.newaxis,:]-0.5)*self.dF*delta[:,jnp.newaxis]) * \
                jnp.conjugate((1j)**((n[:,jnp.newaxis] + m[jnp.newaxis,:])%2)) * \
                (1j)**((n_[:,jnp.newaxis] + m_[jnp.newaxis,:])%2) * \
                self.time_delay_filter_Tprimel(jnp.array([l]), -delta)[0][:,jnp.newaxis]
            return jnp.real(X)

        def case_0(n, m, l, delta):
            # Code for sigma = 0
            n_ = n - l
            m_ = m
            X = (-1)**( (n-n_)[:,jnp.newaxis] * m[jnp.newaxis,:]) * \
                jnp.exp(-2*jnp.pi*(1j)*m[jnp.newaxis,:]*self.dF*delta[:,jnp.newaxis]) * \
                jnp.conjugate((1j)**((n[:,jnp.newaxis] + m[jnp.newaxis,:])%2)) * \
                (1j)**((n_[:,jnp.newaxis] + m[jnp.newaxis,:])%2) * \
                self.time_delay_filter_Tl(jnp.array([l]), -delta)[0][:,jnp.newaxis]
            return jnp.real(X)

        def case_plus_1(n, m, l, delta):
            # Code for sigma = +1
            n_ = n - l
            m_ = m + 1
            X = (-1)**( (n-n_)[:,jnp.newaxis] * m[jnp.newaxis,:]) * \
                (-1j)**l * \
                jnp.exp(-2*jnp.pi*(1j)*(m[jnp.newaxis,:]+0.5)*self.dF*delta[:,jnp.newaxis]) * \
                jnp.conjugate((1j)**((n[:,jnp.newaxis] + m[jnp.newaxis,:])%2)) * \
                (1j)**((n_[:,jnp.newaxis] + m_[jnp.newaxis,:])%2) * \
                self.time_delay_filter_Tprimel(jnp.array([l]), -delta)[0][:,jnp.newaxis]
            return jnp.real(X)

        X = jax.lax.switch(sigma_idx, [case_minus_1, case_0, case_plus_1], n, m, l, delta)

        return X

    @partial(jax.jit, static_argnums=0, static_argnames=('unroll',))
    def apply_variable_time_shift(self, 
                                  wdm_coeff, 
                                  delta, 
                                  unroll: int=1) -> jnp.array:
        r"""
        Perform the variable time shift operation on a grid of WDM coefficients 
        by evaluating the sum

        .. math::

            '\sum_{\substack{l \le |L| \\ \sigma = \{-1,0,1\} }} 
             \omega_{(n-l) (m+\sigma)} \, X_{n(n-l);m(m+\sigma)}(-\delta_n) .

        If the original grid represents the coefficinets of a function 
        :math:'f(t) = \sum_{nm} \omega_{nm} g_{nm}(t)'
        and we sample a time-shift :math:'\delta(t_n) = \delta_n' then the 
        resulting shifted grid :math:'\tilde{\omega}_{nm}'
        is the equivalent to the WDM transform of :math:'f(t-\delta(t))'.

        Terms where :math:`n-l` or :math:`m+\sigma` fall outside the array
        limits are wrapped around periodically. It is the users reponsibility
        to ensure that the input array `wdm_coeff` is suitably zero padded.

        Parameters
        ----------
        wdm_coeff : jnp.array
            WDM coefficient grid. Array, dtype=float, shape=(Nt,Nf)
        delta : jnp.array
            Array, dtype=float, shape=(Nt,)
        unroll : int or bool
            Static unroll factor passed to ``lax.fori_loop`` over ``l``.

        Returns
        -------
        shifted_wdm_coeff : jnp.array
            Shifted wdm_coefficients. Array, dtype=float, shape=(Nt, Nf)
        """
        n_vals = jnp.arange(self.Nt)
        m_vals = jnp.arange(self.Nf)
        sigma_vals = jnp.array([-1, 0, 1])

        # X_{n(n-l), m(m+sigma)}(-delta_n) for all three sigma at once: vmap-ing
        # over sigma turns time_delay_matrix_X's internal lax.switch into a
        # single batched op (all three case_* branches evaluated together),
        # rather than three separately-traced calls to time_delay_matrix_X.
        X_all_sigma = jax.vmap(self.time_delay_matrix_X, in_axes=(None, None, None, 0, None))

        def body_fun(l, acc):
            X = X_all_sigma(n_vals, m_vals, l, sigma_vals, delta)  # (3, Nt, Nf)

            # wdm_coeff[(n-l) % Nt, (m+sigma) % Nf], read via circular shifts
            # rather than a padded gather. The row shift depends on l (traced,
            # varies every fori_loop step) so it is done once and shared by
            # all three sigma; the column shift only needs sigma, which is a
            # static Python int in each of the three unrolled terms, so each
            # of those rolls is cheap.
            row_shifted = jnp.roll(wdm_coeff, shift=l, axis=0)
            shifted = jnp.stack([
                jnp.roll(row_shifted, shift=-sigma, axis=1)
                for sigma in (-1, 0, 1)
            ])  # (3, Nt, Nf)

            return acc + jnp.sum(shifted * X, axis=0)

        init = jnp.zeros((self.Nt, self.Nf), dtype=wdm_coeff.dtype)
        shifted_wdm_coeff = jax.lax.fori_loop(
            -self.max_lag_L, self.max_lag_L + 1, body_fun, init, unroll=unroll
        )

        return shifted_wdm_coeff

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
