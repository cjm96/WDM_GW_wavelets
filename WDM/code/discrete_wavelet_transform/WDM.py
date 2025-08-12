import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from WDM.code.utils.Meyer import Meyer
from WDM.code.utils.utils import C_nm, overlapping_windows
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
        Sampling frequency of the time series, :math:`f_s = \frac{1}{2 dt}`.
    f_Ny : float
        Nyquist frequency (i.e. maximum frequency) of the time series,
        :math:`f_{\rm Ny} = \frac{1}{2 dt}`.
    K : int
        Window length in samples, :math:`K = 2 q N_f`. By definition, 
        this is always an even integer.
    times : jnp.ndarray
        The sample times of the time series, :math:`t_k = k \delta t` for 
        :math:`k\in\{0,1,\ldtots,N-1\}`. Array shape=(N,).
    freqs : jnp.ndarray
        The sample frequencies of the time series, :math:`f_k = k \delta f` for 
        :math:`k\in\{-N/2,N/2+1,\ldtots,N/2-1\}`. Array shape=(N,).
    Cnm : jnp.ndarray 
        Coefficients :math:`C_{nm}` used for the wavelet transform. Equal to 1 
        if :math:`n+m` is even or :math:`i` if it is odd. 
        Array shape=(N_t, N_f).
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
                 dt: float,
                 Nf: int,
                 N: int,
                 q: int = 16,
                 d: int = 4,
                 A_frac: float = 0.25,
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
            Truncation parameter (default: 16). This must be an integer in the 
            range :math:`1 \leq q \leq N_t/2`. Optional.
        A_frac : float
            Fraction of bandwidth used for flat-top response (default: 0.25). 
            Optional.
        d : int
            Steepness parameter for the window transition (default: 4).
            Optional.
        calc_m0 : bool
            If this is set to False then the all wavelet calculations for 
            :math:`m=0` will be handled incorrectly. This comes with some
            performance benefits. If it is set to True, then the calculations
            will be done correctly, but may be slower. Optional.

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

        Returns
        -------
        Phi : jnp.ndarray 
            Array of shape (N,). Real-valued time-domain window. 
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
    def check_indices(self, n: jnp.ndarray, m: jnp.ndarray) -> bool:
        r"""
        Check if the wavelet indices are in the valid range.

        Parameters
        ----------
        n : jnp.ndarray
            Array of n indices, dtype=int. Wavelet time index.
        m : jnp.ndarray
            Array of m indices, dtype=int. Wavelet frequency index.

        Returns
        -------
        check : bool
            True if the indices are all valid, False otherwise.
        """
        n = jnp.asarray(n, self.jax_dtype_int)
        m = jnp.asarray(m, self.jax_dtype_int)

        n_test = jnp.all(jnp.logical_and(n>=0, n<self.Nt))
        m_test = jnp.all(jnp.logical_and(m>=0, m<self.Nf))

        check = jnp.logical_and(n_test, m_test)

        return check

    def wavelet_central_time_frequency(self, 
                                       n: jnp.ndarray, 
                                       m: jnp.ndarray) -> Tuple[jnp.ndarray, 
                                                                jnp.ndarray]:
        r"""
        Compute the central time :math:`t_{nm}= n \Delta t` and the central 
        frequency :math:`f_{nm} = m \Delta f` of the wavelet :math:`g_{nm}(t)`.

        The case :math:`m=0` is special and is handled separately using

        .. math::

            t_{n0} = (2n\,\mathrm{mod}\,N_t) \Delta t , 

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
                                       n: jnp.ndarray, 
                                       m: jnp.ndarray) -> Tuple[jnp.ndarray, 
                                                                jnp.ndarray]:
        """
        Compiled part of wavelet_central_time_frequency method.
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
            n: int, 
            m: int,
            freq: jnp.ndarray = None) -> jnp.ndarray:
        r"""
        Compute the frequency-domain representation of the wavelets, 
        :math:`\tilde{G}_{nm}(f)`.

        This method computes the frequency-domain wavelet for a single choice 
        of :math:`n` and :math:`m` using the expressions below. If you instead
        want to compute the full wavelet basis for all :math:`n` and :math:`m`
        efficiently, use the `Gnm_basis` method.

        This method is slow.
        
        For :math:`m>0`, the wavelet is given by

        .. math::

            \tilde{G}_{nm}(f) = \exp(-2\pi i n f \Delta T) 
                    \left( C_{nm}\tilde{\Phi}(2\pi [f-m\Delta F])
                        + C^*_{nm}\tilde{\Phi}(2\pi [f+m\Delta F]) \right) .

        For the special case :math:`m=0`, the wavelet is given by

        .. math::

            \tilde{G}_{nm}(f) = \begin{cases}
                \exp(-4\pi i n f \Delta T) \tilde{\Phi}(2\pi f) 
                    & \mathrm{if}\; n<N_t/2 \\
                \exp(-4\pi i n f \Delta T) 
                    \left( \tilde{\Phi}(2\pi [f+N_f\Delta F]) + 
                        \tilde{\Phi}(2\pi [f-N_f\Delta F]) \right)
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

        if freq is None:
            freq = self.freqs
        else:
            freq = jnp.asarray(freq, dtype=self.jax_dtype)

        if m > 0:
            Gnm = jnp.sqrt(jnp.pi) * jnp.exp(-1j*n*2.*jnp.pi*freq*self.dT) * ( 
                    C_nm(n, m) * 
                    Meyer(2.*jnp.pi*(freq-m*self.dF), self.d, self.A, self.B) +
                    jnp.conj(C_nm(n, m)) * 
                    Meyer(2.*jnp.pi*(freq+m*self.dF), self.d, self.A, self.B) )

        else:
            if n < self.Nt // 2:
                Gnm = jnp.sqrt(2.*jnp.pi) * \
                        jnp.exp(-1j*n*4.*jnp.pi*freq*self.dT) * \
                            Meyer(2.*jnp.pi*freq, self.d, self.A, self.B)

            else:
                Gnm = jnp.sqrt(2.*jnp.pi) * \
                        jnp.exp(-1j*n*4.*jnp.pi*freq*self.dT) * \
                    (Meyer(2.*jnp.pi*(freq+self.f_Ny), self.d, self.A, self.B) + 
                     Meyer(2.*jnp.pi*(freq-self.f_Ny), self.d, self.A, self.B) )

        return Gnm
    
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
                              self.window_FD[shift_do%self.N][:,jnp.newaxis,:]+
                             jnp.conj(self.Cnm[jnp.newaxis,:,:])*\
                              self.window_FD[shift_up%self.N][:,jnp.newaxis,:])
            
            if self.calc_m0:
                # overwrite m=0 terms for n<Nt/2
                n_vals = jnp.arange(self.Nt//2)

                term = jnp.exp(-2j*n_vals[jnp.newaxis,:] * \
                                om[:,jnp.newaxis]*self.dT) * \
                                    self.window_FD[:,jnp.newaxis]

                basis = basis.at[:, n_vals, 0].set(term)

                # overwrite m=0 terms for n>=Nt/2
                n_vals = jnp.arange(self.Nt//2, self.Nt)

                shift_up = (jnp.arange(self.N) + self.N//2) 
                shift_do = (jnp.arange(self.N) - self.N//2) 

                term = 0.5 * jnp.exp(-2j*n_vals[jnp.newaxis,:] * \
                                om[:,jnp.newaxis]*self.dT) * \
                            (self.window_FD[shift_up%self.N][:,jnp.newaxis] +
                                self.window_FD[shift_do%self.N][:,jnp.newaxis])

                basis = basis.at[:, n_vals, 0].set(term)

            self._cached_Gnm_basis = basis

        return self._cached_Gnm_basis
    
    def gnm(self, 
            n: int, 
            m: int, 
            time: jnp.ndarray = None) -> jnp.ndarray:
        r"""
        Compute the time-domain representation of the wavelets, 
        :math:`g_{nm}(t)`.

        This method computes the frequency-domain wavelets for a single choice 
        of :math:`n` and :math:`m` and performs and inverse Fourier transform. 
        If you instead want to compute the full wavelet basis for all :math:`n` 
        and :math:`m` efficiently, use the `gnm_basis` method.

        This method is slow.

        Parameters
        ----------
        n : int
            Wavelet time index.
        m : int
            Wavelet frequency index.
        time : jnp.ndarray
            Times at which to evaluate the wavelet. 
            If None, then defaults to the self.times. Optional

        Returns
        -------
        gnm : jnp.ndarray 
            Array shape (N,). The time-domain wavelet.
        """
        assert self.check_indices(n, m), f"Invalid indices: {n=} {m=}"

        if time is None:
            time = self.times
        else:
            time = jnp.asarray(time, dtype=self.jax_dtype)

        dt = jnp.mean(jnp.diff(time))

        freq = jnp.fft.fftshift(jnp.fft.fftfreq(len(time), d=dt))

        Gnm = self.Gnm(n, m, freq=freq)

        gnm = jnp.fft.ifft(jnp.fft.ifftshift(Gnm)).real / self.dt

        return gnm
    
    @partial(jax.jit, static_argnums=0)
    def gnm_basis(self) -> jnp.ndarray:
        r"""
        Efficient computation of time-domain wavelet basis :math:`g_{nm}(f)`. 
        Instead of calling the functions for :math:`\tilde{G}_{nm}(f)` and 
        performing an inverse Fourier transform, as is done in the `gnm` method, 
        this function shifts indices of `window_TD`.
        
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
                return jnp.where((n + m) % 2 == 0, 
                                jnp.sqrt(2.) * (-1)**(n*m) * \
                                    jnp.cos(jnp.pi*m*k_vals/self.Nf) * \
                                     self.window_TD[(k_vals-n*self.Nf)%self.N], 
                                -jnp.sqrt(2.) * \
                                    jnp.sin(jnp.pi*m*k_vals/self.Nf) * \
                                     self.window_TD[(k_vals-n*self.Nf)%self.N])

            f_vmapped = jax.vmap(jax.vmap(temp_func, 
                                        in_axes=(None, 0)), 
                                in_axes=(0, None))

            basis = f_vmapped(n_vals, m_vals)
            basis = jnp.transpose(basis, (2, 0, 1))

            if self.calc_m0:
                # overwrite m=0 terms for n<Nt/2
                n_vals = jnp.arange(self.Nt//2)

                term = self.window_TD[(k_vals[:,jnp.newaxis]
                                    -2*n_vals[jnp.newaxis,:]*self.Nf)%self.N]

                basis = basis.at[:, n_vals, 0].set(term)

                # overwrite m=0 terms for n>=Nt/2
                n_vals = jnp.arange(self.Nt//2, self.Nt)

                def temp_func(n):
                    return jnp.where(n % 2 == 0, 
                                (-1)**(n*self.Nf) * \
                                    jnp.cos(jnp.pi*self.Nf*k_vals/self.Nf) * \
                                    self.window_TD[(k_vals-2*n*self.Nf)%self.N], 
                                    (-1)**k_vals * \
                                    self.window_TD[(k_vals-2*n*self.Nf)%self.N])

                f_vmapped = jax.vmap(temp_func)

                term = f_vmapped(n_vals).T

                basis = basis.at[:, n_vals, 0].set(term)

            self._cached_gnm_basis = basis

        return self._cached_gnm_basis

    def pad_signal(self, x: jnp.ndarray, where: str = 'end') -> jnp.ndarray:
        r"""
        The transform method requires the input time series signal to have a 
        specific length :math:`N`. This method can be used to zero-pad any 
        signal to the desired length.

        This function also returns a Boolean mask that can be used later to 
        recover arrays of the original length.

        Parameters
        ----------
        x : jnp.ndarray
            Input signal to be padded.
        where : str
            Where to add the padding. Options are 'end', 'start', or 'equal' 
            which puts the zero padding at the end of the signal, the start of 
            the signal, or equally at both ends respectively. Optional.

        Returns
        -------
        x_padded : jnp.ndarray
            Padded signal to length N, with zeros added at the end.
        mask : jnp.ndarray
            Boolean mask indicating the valid part of the padded signal.

        Notes
        -----
        The Boolean mask can be used to get back to the original signal; i.e.
        `x_padded[mask]` will recover the original signal, `x`.
        """
        n = len(x)
        padding_length = self.N - n

        mask = jnp.full(self.N, True, dtype=bool)

        if where == 'end':
            x_padded = jnp.pad(x, (0, padding_length), 
                               mode='constant', constant_values=0)
            mask = mask.at[n:].set(False)
        elif where == 'start':
            x_padded = jnp.pad(x, (padding_length, 0), 
                               mode='constant', constant_values=0)
            mask = mask.at[:padding_length].set(False)
        elif where == 'equal':
            a = padding_length // 2
            b = padding_length - a
            x_padded = jnp.pad(x, (a, b),
                               mode='constant', constant_values=0)
            mask = mask.at[:a].set(False)
            mask = mask.at[n + a:].set(False)
        else:
            raise ValueError(f"Invalid padding location {where=}.")

        return x_padded, mask
    
    @partial(jax.jit, static_argnums=0)
    def windowed_fft(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        The windowed short FFT of the input.

        The input time series is split into :math:`N_t` overlapping segments 
        each of length :math:`K` and with a hop interval of :math:`N_f` between 
        their centres. Each of these segments is then windowed and transformed

        .. math::

            X_n[j] = \sum_{k=-K/2}^{K/2-1} \exp(2\pi i kj/K) x[nN_f+k] \phi[k]

        Parameters
        ----------
        x : jnp.ndarray
            Array shape (N,). Input signal to be transformed.

        Returns
        -------
        windowed_fft : jnp.ndarray
            Array shape shape (Nt, K). Windowed FFT of the input signal.
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        windowed_fft = overlapping_windows(x, self.K, self.Nt, self.Nf)

        k_vals = jnp.arange(-self.K//2, self.K//2)

        windowed_fft *= self.window_TD[k_vals%self.N]

        windowed_fft = jnp.fft.ifftshift(windowed_fft, axes=-1)

        windowed_fft = jnp.fft.ifft(windowed_fft, axis=-1) * self.K

        return windowed_fft
    
    @partial(jax.jit, static_argnums=0)
    def forward_transform_exact(self, x: jnp.ndarray) -> jnp.ndarray:
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
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        gnm_basis = jnp.transpose(self.gnm_basis(), (1,2,0))

        w = jnp.sum(gnm_basis * x, axis=-1) * self.dt

        return w
    
    @partial(jax.jit, static_argnums=0)
    def forward_transform_truncated(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        This method computes the wavelet coefficients using the truncated 
        expressions

        .. math::

            w_{n0} = 2\pi\delta t\sum_{k=-K/2}^{K/2-1} 
                                    g_{nm}[k + 2 n N_f] x[k + 2 n N_f] .

        .. math::

            w_{nm} = 2\pi\delta t\sum_{k=-K/2}^{K/2-1} 
                                    g_{nm}[k + n N_f] x[k + n N_f] ,

        where the sum is over the truncated window of length :math:`K=2qN_f`.

        In the above expressions, indices out of bounds of the array are 
        to be understood as wrapping around circularly.

        This method is slow.

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
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        w = jnp.empty((self.Nt, self.Nf), dtype=self.jax_dtype) 

        B = self.gnm_basis()

        k_vals = jnp.arange(-self.K//2, self.K//2)

        for n in range(self.Nt):
            for m in range(self.Nf):
                gnm = B[:, n, m]
                gnm_x = gnm[(k_vals+(1 if m>0 else 2)*n*self.Nf)%self.N] * \
                            x[(k_vals+(1 if m>0 else 2)*n*self.Nf)%self.N]
                w = w.at[n, m].set(self.dt*jnp.sum(gnm_x))

        return w
    
    @partial(jax.jit, static_argnums=0)
    def forward_transform_short_fft(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        Parameters
        ----------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.

        Returns
        -------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency-domain wavelet coefficients.
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"

        X = self.windowed_fft(x)

        m_vals = jnp.arange(self.Nf)

        w = jnp.sqrt(2.) * self.dt * \
                    jnp.real( self.Cnm * X[:,(m_vals*self.q)%self.K] )
        
        if self.calc_m0:
            k_vals = jnp.arange(-self.K//2, self.K//2)
            for n in range(self.Nt):
                if n<self.Nt//2:
                    x_term = x[(k_vals + 2*n*self.Nf) % self.N]
                    phi_term = self.window_TD[k_vals % self.N]
                    term = self.dt*x_term*phi_term
                else:
                    x_term = x[(k_vals + 2*n*self.Nf) % self.N]
                    phi_term = self.window_TD[k_vals % self.N]
                    alt_term = (-1)**k_vals
                    term = self.dt*x_term*phi_term*alt_term
                w = w.at[n, 0].set(jnp.sum(term).real)

        return w
    
    @partial(jax.jit, static_argnums=0)
    def forward_transform_fft(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        Parameters
        ----------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.

        Returns
        -------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency-domain wavelet coefficients.
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        n_vals = jnp.arange(self.Nt)
        m_vals = jnp.arange(self.Nf)

        alt = (-1.)**(n_vals[:,jnp.newaxis]*m_vals[jnp.newaxis, :])

        term = jnp.fft.fft(jnp.fft.fftshift(x))

        term = overlapping_windows(term, self.Nt, self.Nf, self.Nt//2)

        l_vals = jnp.arange(-self.Nt//2, self.Nt//2)
        term = term * self.window_FD[l_vals%self.N]

        term = jnp.fft.fftshift(term, axes=-1)

        term = jnp.fft.fft(term, axis=-1) 

        w = jnp.sqrt(2.) * self.dt * alt * jnp.real( self.Cnm * term.T )

        if self.calc_m0:
            pass

        return w
    
    @partial(jax.jit, static_argnums=0)
    def inverse_transform(self, w : jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the inverse discrete wavelet transform. Transforms the wavelet 
        coefficients from the time-frequency domain into the time domain.

        This method computes the wavelet coefficients using the expression

        .. math::

            x[k] = \sum_{n=0}^{N_t-1} \sum_{m=0}^{N_f-1} w_{nm} g_{nm}[k] .

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
        
        x = jnp.zeros(self.N, dtype=self.jax_dtype)

        def add_one_band(x, n):
            def add_one_cell(x, m):
                m0_cond = jnp.where(m > 0, 1, 2) 
                k_vals = jnp.arange(self.N)
                indices = (k_vals-m0_cond*n*self.Nf)%self.N

                minusone_mn = (-1)**(n*m)
                minusone_nNf = (-1)**(n*self.Nf)
                minusone_k = (-1)**k_vals

                wavelet = jnp.where(m > 0,
                            jnp.where((n + m) % 2 == 0, 
                                jnp.sqrt(2.) * minusone_mn * \
                                    jnp.cos(jnp.pi*m*k_vals/self.Nf) * \
                                        self.window_TD[indices], 
                                -jnp.sqrt(2.) * \
                                    jnp.sin(jnp.pi*m*k_vals/self.Nf) * \
                                        self.window_TD[indices]),
                            jnp.where(n < self.Nt // 2, 
                                    self.window_TD[indices], 
                                    jnp.where(n % 2 == 0, 
                                        minusone_nNf * \
                                        jnp.cos(jnp.pi*self.Nf*k_vals/self.Nf)*\
                                        self.window_TD[indices], 
                                        minusone_k * \
                                        self.window_TD[indices])
                                )
                            )

                x = x + w[n, m] * wavelet
                return x
            
            x = jax.lax.fori_loop(0, 
                                  self.Nf, 
                                  lambda m, acc: add_one_cell(acc, m), 
                                  x)
            return x

        x = jax.lax.fori_loop(0, 
                              self.Nt, 
                              lambda n, acc: add_one_band(acc, n), 
                              x)

        return x
    
    def inverse_transform_old(self, w : jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the inverse discrete wavelet transform. Transforms the wavelet 
        coefficients from the time-frequency domain into the time domain.

        This method computes the wavelet coefficients using the expression

        .. math::

            x[k] = \sum_{n=0}^{N_t-1} \sum_{m=0}^{N_f-1} w_{nm} g_{nm}[k] .

        This method computes all the basis vectors :math:`g_{nm}[k]` together
        using the method `gnm_basis` and then sums them up. This is quite fast
        but uses O(N^2) memory. A more memory-efficient implementation is 
        provided in the method `inverse_transform`.

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

    def dwt(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Forward discrete wavelet transform.

        Calls self.fast_forward_transform.
        """
        return self.forward_transform_fft(x)
    
    def idwt(self, w: jnp.ndarray) -> jnp.ndarray:
        r"""
        Inverse discrete wavelet transform.

        Calls self.inverse_transform.
        """
        return self.inverse_transform(w)

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
        lines.append( f"{self.K = } window length" )
        text = "\n".join(lines)
        return text
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Calls the forward transform self.dwt.
        """
        return self.dwt(x)
    
    