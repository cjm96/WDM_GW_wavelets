import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from WDM.code.utils.Meyer import Meyer
from WDM.code.utils.utils import C_nm, overlapping_windows
from typing import Tuple


class WDM_transform:
    r"""
    This class implements the WDM discrete wavelet transform for the Meyer 
    window function.

    Attributes
    ----------
    dt : float
        The cadence, or time step, of the original time series (seconds). 
        Equal to inverse of the sampling frequency (Hertz).
    Nf : int
        Number of wavelet frequency bands. This controls the time/frequency 
        resolution; as :math:`N_f \rightarrow N/2` the wavelet expansion 
        approaches the  Fourier series, and as :math:`N_f \rightarrow 1` it 
        approaches the original time series.
    N : int
        Length of the input time series. Must be an even multiple of 
        :math:`N_f`.
    Nt : int
        Number of wavelet time bands. Equal to :math:`N/N_f`. This must be even.
    q : int
        Truncation parameter. Formally the time domain wavelet has infinite 
        extent, but in practice it is truncated at :math:`\pm q \Delta T`. 
        This must be an integer in the range :math:`1 \leq q \leq N_t/2`.
    A_frac : float
        Fraction of total bandwidth used for the flat-top response region.
        Must be in the range [0, 1].
    B_frac : float
        Fraction of total bandwidth used for the transition region. This is set
        based on A_frac so :math:`2A_{\mathrm{frac}}+B_{\mathrm{frac}}=1`.
    d : int
        Steepness parameter for the Meyer window transition. 
        Must be a positive integer, :math:`d\geq 1`.
    K : int
        Window length in samples (equal to :math:`2 q N_f`). By definition, 
        this is always an even integer.
    kvals : jnp.ndarray of shape (K,) 
        Array of integers from :math:`-K/2` to :math:`K/2-1` mod N. Used for 
        indexing arrays.
    dF : float
        Frequency resolution of the wavelets (Hertz), or the total wavelet 
        frequency bandwidth :math:`\Delta F = \frac{\Delta \Omega}{2 \pi}`.
    dT : float
        Time resolution of the wavelets (seconds). Related to the frequency 
        resolution by :math:`\Delta F \Delta T = \frac{1}{2}`.
    T : float
        Total duraion of the time series (seconds). Related to :math:`N` and 
        :math:`\delta t` by :math:`T = N \delta t`.
    Q : int
        Parity of the number of frequency bands :math:`N_f`. This is 0 if 
        :math:`N_f` is even, and 1 if :math:`N_f` is odd.
    dOmega : float
        Angular Frequency resolution of the wavelets (radians per second), or 
        the total wavelet angular frequency bandwidth 
        :math:`\Delta \Omega = 2A + B`.
    f_Ny : float
        Nyquist frequency (i.e. maximum frequency) of the original time series 
        (Hertz), equal to :math:`\frac{1}{2 dt}`.
    A : float
        Half-width of the flat-top response region in angular frequency 
        (radians per second).
    B : float
        Width of the transition region in angular frequency 
        (radians per second).
    window : jnp.ndarray
        Time-domain window of length :math:`K`.
    Cnm : jnp.ndarray of shape (Nt, Nf)
        Coefficients :math:`C_{nm}` used for the wavelet transform.
    """

    def __init__(self, 
                 dt: float,
                 Nf: int,
                 N: int,
                 q: int = 16,
                 d: int = 4,
                 A_frac: float = 0.25) -> None:
        r"""
        Initialize the WDM_transform.

        Parameters
        ----------
        dt : float
            The cadence, or time step (seconds). 
        Nf : int
            Number of frequency bands, controls the time/frequency resolution.
        N : int
            Length of the input time series. Must be an even multiple of Nf.
        q : int, optional
            Truncation parameter (default: 16). 
        A_frac : float, optional
            Fraction of bandwidth used for flat-top response (default: 0.25). 
        d : int, optional
            Steepness parameter for the window transition (default: 4).

        Returns
        -------
        None
        """
        # User provided parameters
        self.dt = float(dt)
        self.Nf = int(Nf)
        self.N = int(N)
        self.q = int(q)
        self.A_frac = float(A_frac)
        self.d = int(d)

        self.validate_parameters()
        
        # Derived parameters
        self.times = jnp.arange(self.N) * self.dt
        self.freqs = jnp.fft.fftfreq(self.N, d=self.dt)
        self.Nt = self.N // self.Nf
        self.T = self.N * self.dt
        self.Q = self.Nf % 2 
        self.dF = 1. / ( 2. * self.dt * self.Nf )  
        self.dOmega = 2. * jnp.pi * self.dF
        self.dT = self.dt * self.Nf 
        self.f_Ny = 0.5 / self.dt
        self.B_frac = 1. - 2. * self.A_frac  
        self.A = self.A_frac * self.dOmega
        self.B = self.B_frac * self.dOmega
        self.K = 2 * self.q * self.Nf
        self.kvals = jnp.arange(-self.K//2, self.K//2) % self.N
        self.Cnm = jnp.array([[C_nm(n, m) for m in range(self.Nf)] 
                              for n in range(self.Nt)])

        self.window = self.build_time_domain_window()
        self.window_FD = self.build_frequency_domain_window()

        if jax.config.read("jax_enable_x64"):
            self.jax_dtype = jnp.float64
        else:
            self.jax_dtype = jnp.float32

    def validate_parameters(self) -> None:
        r"""
        Validate the parameters provided to the WDM_transform __init__ method.
        Raises an AssertionError if any parameters are invalid.

        Returns
        -------
        None
        """
        assert self.dt > 0, \
                    f"dt must be positive, got {self.dt=}"
        
        assert self.Nf > 0, \
                    f"Nf must be a positive integer, got {self.Nf=}"
        
        assert self.Nf > 0, \
                    f"Nf must be a positive integer, got {self.Nf=}"
        
        assert self.N > 0, \
                    f"Nt must be a positive integer, got {self.N=}"
        
        assert self.N % self.Nf == 0 and ( self.N // self.Nf ) % 2 == 0, \
                    f"N must be even multiple of Nf, got {self.N=}, {self.Nf=}"
        
        assert self.q>=1, \
                    f"q must be a positive integer, got {self.q=}"
        
        Nt = self.N // self.Nf
        assert self.q<=Nt//2, \
                    f"q must be less than {Nt//2}, got {self.q=}"
        
        assert 0. < self.A_frac < 1., \
                    f"A_frac must be in [0, 1], got {self.A_frac=}"
        
        assert self.d>=1, \
                    f"d must be a positive integer, got {self.d=}"

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
        f = jnp.fft.fftfreq(self.N, d=self.dt) 
        Phi = Meyer(2.*jnp.pi*f, self.d, self.A, self.B)
        phi = jnp.fft.ifft(Phi).real
        return phi
    
    def build_frequency_domain_window(self) -> jnp.ndarray:
        r"""
        Construct the frequenct-domain window function :math:`\tilde{\Phi}(f)`.

        Returns
        -------
        Phi : jnp.ndarray 
            Array of shape (N,). Real-valued time-domain window. 
        """
        f = jnp.fft.fftfreq(self.N, d=self.dt) 
        Phi = Meyer(2.*jnp.pi*f, self.d, self.A, self.B)
        return Phi
    
    def Gnm(self, 
            n: int, 
            m: int,
            freq: jnp.ndarray = None) -> jnp.ndarray:
        r"""
        Compute the frequency-domain wavelets :math:`\tilde{g}_{nm}(f)`.

        Parameters
        ----------
        n : int
            Wavelet time index.
        m : int
            Wavelet frequency index.
        freq : jnp.ndarray
            Frequencies at which to evaluate the wavelet (Hertz). 
            If None, then defaults to the FFT frequencies. Optional

        Returns
        -------
        Gnm : jnp.ndarray
            Complex array shaped like freq. The frequency-domain wavelet.
        """
        if freq is None:
            freq = self.freqs
        else:
            freq = jnp.asarray(freq, dtype=self.jax_dtype)

        if m == 0:
            Gnm = jnp.exp(-1j*n*4.*jnp.pi*freq*self.dT) * \
                Meyer(2.*jnp.pi*freq, self.d, self.A, self.B)
        elif m == self.Nf:
            Q = self.Nf % 2
            Gnm = jnp.exp(-1j*(2*n+Q)*2.*jnp.pi*freq*self.dT) * \
                    (Meyer(2.*jnp.pi*(freq+self.f_Ny), self.d, self.A, self.B) + 
                     Meyer(2.*jnp.pi*(freq-self.f_Ny), self.d, self.A, self.B) )
        else:
            Gnm = (1/jnp.sqrt(2.)) * jnp.exp(-1j*n*2.*jnp.pi*freq*self.dT) * ( 
                    C_nm(n, m) * 
                    Meyer(2.*jnp.pi*(freq-m*self.dF), self.d, self.A, self.B) +
                    jnp.conj(C_nm(n, m)) * 
                    Meyer(2.*jnp.pi*(freq+m*self.dF), self.d, self.A, self.B) )
        
        return Gnm
    
    def gnm(self, 
            n: int, 
            m: int, 
            time: jnp.ndarray = None) -> jnp.ndarray:
        r"""
        Compute the time-domain wavelets :math:`g_{nm}(t)`.

        This function computes the inverse Fourier transform of the 
        frequency-domain wavelet, computed using the method Gnm.

        Parameters
        ----------
        n : int
            Wavelet time index.
        m : int
            Wavelet frequency index.
        time : jnp.ndarray
            Times at which to evaluate the wavelet (seconds). 
            If None, then defaults to the FFT values. Optional

        Returns
        -------
        gnm : jnp.ndarray 
            Array shape (N,). The time-domain wavelet.
        """
        if time is None:
            time = self.times
        else:
            time = jnp.asarray(time, dtype=self.jax_dtype)

        _dt = jnp.mean(jnp.diff(time))

        freq = jnp.fft.fftfreq(len(time), d=_dt)

        n_, m_ = n, m
        if m==0 and n>=self.Nt//2:
            n_ = n - self.Nt//2
            m_ = self.Nf

        Gnm = self.Gnm(n_, m_, freq=freq)

        gnm = jnp.fft.ifft(Gnm).real / _dt

        return gnm
    
    def pad_signal(self, x: jnp.ndarray, where: str = 'end') -> jnp.ndarray:
        r""" 
        The transform method requires the input time series signal to have a 
        specific length :math:`N`.

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
    
    def windowed_fft(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the windowed FFT of the input. (Uses unusual FFT conventions)

        The input time series is split into :math:`N_t` overlapping segments 
        each of length :math:`K` and with a hop interval of :math:`N_f` between 
        their centres. Each of these segments is then windowed and transformed 
        using the FFT.

        .. math::

            X_n[j] = \sum_{k=-K/2}^{K/2-1} \exp(2\pi i kj/K) x[nN_f+k] \phi[k]

        Parameters
        ----------
        x : jnp.ndarray
            Array shape (N,). Input signal to be transformed.

        Returns
        -------
        X : jnp.ndarray
            Array shape shape (Nt, K). Windowed FFT of the input signal.
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        X = overlapping_windows(x, self.K, self.Nt, self.Nf)

        X *= self.window[self.kvals]

        X = jnp.fft.ifftshift(X, axes=-1)

        X = jnp.fft.ifft(X, axis=-1) * self.K

        return X

    def forward_transform_exact(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        This method computes the wavelet coefficients using the exact expression

        .. math::

            w_{nm} = 2 \pi \delta t \sum_{k=0}^{N-1} g_{nm}[k] x[k] ,

        where the sum is over the whole time-domain signal (no truncation). The 
        time domain wavelets :math:`g_{nm}[k]` are computed using an iFFT. 
        
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

        Notes
        -----
        This is implemented using for loops. It is slow. It is only intended to 
        be used for testing and debugging purposes. 
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"

        w = jnp.empty((self.Nt, self.Nf), dtype=self.jax_dtype) 

        for n in range(self.Nt):
            for m in range(self.Nf):
                gnm = self.gnm(n, m)
                w = w.at[n, m].set(2.*jnp.pi*self.dt*jnp.sum(gnm*x))

        return w
    
    def inverse_transform_exact(self, w: jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the inverse discrete wavelet transform. Transforms the input
        signal from the time-frequency wavelet domain into the time domain.

        This method computes the inverse discrete wavelet transform using the 
        exact expression

        .. math::

            x[k] = \sum_{n=0}^{N_t-1} \sum_{m=0}^{N_f-1} w_{nm} g_{nm}[k] ,

        where the sum is over the whole time-domain signal (no truncation). The 
        time domain wavelets `g_{nm}[k]` are computed using an inverse FFT. 
        
        This method is slow but exact.

        Parameters
        ----------
        w : jnp.ndarray
            Array shape  shape (Nt, Nf). 
            WDM time-frequency-domain wavelet coefficients.

        Returns
        -------
        x : jnp.ndarray 
            Array shape shape (N,). 
            Input time-domain signal to be transformed.

        Notes
        -----
        This is implemented using for loops. It is slow. It is only intended to 
        be used for testing and debugging purposes. 
        """
        assert w.shape == (self.Nt, self.Nf), \
                    f"Input signal must have shape ({self.Nt}, {self.Nf}), " \
                    f"got {w.shape=}"
        
        x = jnp.zeros(self.N, dtype=self.jax_dtype) 

        for n in range(self.Nt):
            for m in range(self.Nf):
                x = x + w[n, m] * self.gnm(n, m)

        return x
    
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

        (In the above expressions, indices out of bounds of the array are 
        to be understood as wrapping around circularly.)

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

        for n in range(self.Nt):
            for m in range(self.Nf):
                gnm = self.gnm(n, m)
                window = jnp.array([gnm[(k+(1 if m>0 else 2)*n*self.Nf)%self.N]* 
                                     x[(k+(1 if m>0 else 2)*n*self.Nf)%self.N]
                                for k in range(-self.K//2, self.K//2)])
                w = w.at[n, m].set(2.*jnp.pi*self.dt*jnp.sum(window))

        return w
    
    def inverse_transform_truncated(self, w: jnp.ndarray) -> jnp.ndarray:
        r"""
        This is the same as the inverse_transform_exact method.

        Parameters
        ----------
        w : jnp.ndarray 
            Array shape (Nt, Nf). 
            WDM time-frequency-domain wavelet coefficients.

        Returns
        -------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.

        Notes
        -----
        This method is slow. It is only intended to be used for testing and 
        debugging purposes. 
        """
        x = self.inverse_transform_exact(w)
        return x
    
    def forward_transform_truncated_window(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform using the truncated sum 
        and the window function `self.window`.

        .. math::

            w_{nm} = 2\sqrt{2}\pi\delta t \mathrm{Re} \sum_{k=-K/2}^{K/2-1} 
                            C_{nm} \exp(i\pi km/N_f) 
                            x[k+nN_f] \phi[k] \quad \mathrm{for}\; m>0.

        The case :math:`m=0` is handled separately, with the following two 
        expressions:

        .. math::

            w_{n0} = 2\pi\delta t\sum_{k=-K/2}^{K/2-1} 
                            x[k+2nN_f] \phi[k] \quad \mathrm{for}\; n<N_t/2,

        .. math::

            w_{n0} = 2\pi\delta t\sum_{k=-K/2}^{K/2-1} (-1)^k x[k+(2n+Q)N_f]
                            \phi[k] \quad \mathrm{for}\; n\geq N_t/2.

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

        k_vals = jnp.arange(-self.K//2, self.K//2)

        for n in range(self.Nt):
            for m in range(self.Nf):
                if m==0:
                    if n<self.Nt//2:
                        x_term = x[(k_vals + 2*n*self.Nf) % self.N]
                        phi_term = self.window[k_vals % self.N]
                        norm = 2.*jnp.pi*self.dt
                        term = norm*x_term*phi_term
                    else:
                        x_term = x[(k_vals + (2*n+self.Q)*self.Nf) % self.N]
                        phi_term = self.window[k_vals % self.N]
                        alt_term = (-1)**k_vals
                        norm = 2.*jnp.pi*self.dt
                        term = norm*x_term*phi_term*alt_term
                else:
                    exp_term = jnp.exp((1j)*jnp.pi*k_vals*m/self.Nf)
                    x_term = x[(k_vals + n*self.Nf) % self.N]
                    phi_term = self.window[k_vals % self.N]
                    norm = 2.*jnp.sqrt(2.)*jnp.pi*self.dt*C_nm(n,m)
                    term = norm*exp_term*x_term*phi_term
                w = w.at[n, m].set(jnp.sum(term).real)

        return w

    def inverse_transform_truncated_window(self, w: jnp.ndarray) -> jnp.ndarray:
        r"""
        This is the same as the inverse_transform_exact method.

        Parameters
        ----------
        w : jnp.ndarray 
            Array shape (Nt, Nf). 
            WDM time-frequency-domain wavelet coefficients.

        Returns
        -------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.

        Notes
        -----
        This method is slow. It is only intended to be used for testing and 
        debugging purposes. 
        """
        x = self.inverse_transform_exact(w)
        return x
    
    def forward_transform_truncated_windowed_fft(self, x: jnp.ndarray,
                                        m0: bool = False) -> jnp.ndarray:
        r"""
        Perform the forward discrete wavelet transform using the windowed FFT
        of the input time series.

        For :math:`m>0`, the wavelet coefficients are computed using 

        .. math::

            w_{nm} = 2\pi \sqrt{2} \delta t \mathrm{Re} C_{nm} X_n[mq] , 
                        \quad \mathrm{for} \; m>0.

        This is quite fast. But it only works for the :math:`m>0` terms. If the
        :math:`m=0` terms are needed, then set `m0=True` and the method will
        compute them using the truncated window expressions. (This is slower.)

        Parameters
        ----------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.
        m0 : bool
            If True, then the :math:`m=0` terms are computed correctly.
            If False, then these terms will be incorrect.
            If these terms are not needed, then leave this at the default False 
            value for faster performance. Optional.

        Returns
        -------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency-domain wavelet coefficients.
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"

        X = self.windowed_fft(x)

        m_vals = jnp.arange(self.Nf)

        w = jnp.sqrt(2.) * 2. * jnp.pi * self.dt * \
                    jnp.real( self.Cnm * X[:,(m_vals*self.q)%self.K] )
        
        if m0:
            k_vals = jnp.arange(-self.K//2, self.K//2)
            for n in range(self.Nt):
                if n<self.Nt//2:
                    x_term = x[(k_vals + 2*n*self.Nf) % self.N]
                    phi_term = self.window[k_vals % self.N]
                    norm = 2.*jnp.pi*self.dt
                    term = norm*x_term*phi_term
                else:
                    x_term = x[(k_vals + (2*n+self.Q)*self.Nf) % self.N]
                    phi_term = self.window[k_vals % self.N]
                    alt_term = (-1)**k_vals
                    norm = 2.*jnp.pi*self.dt
                    term = norm*x_term*phi_term*alt_term
                w = w.at[n, 0].set(jnp.sum(term).real)

        return w
    
    def inverse_transform_truncated_windowed_fft(self, 
                                                 w: jnp.ndarray) -> jnp.ndarray:
        r"""
        This is the same as the inverse_transform_exact method.

        Parameters
        ----------
        w : jnp.ndarray 
            Array shape (Nt, Nf). 
            WDM time-frequency-domain wavelet coefficients.

        Returns
        -------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.

        Notes
        -----
        This method is slow. It is only intended to be used for testing and 
        debugging purposes. 
        """
        x = self.inverse_transform_exact(w)
        return x
    
    def forward_transform_truncated_fft(self, x: jnp.ndarray,
                                        m0: bool = False) -> jnp.ndarray:
        r"""
        Perform the forward 

        This is fast. But it only works for the :math:`m>0` terms. If the
        :math:`m=0` terms are needed, then set `m0=True` and the method will
        compute them using the truncated window expressions. (This is slower.)

        Parameters
        ----------
        x : jnp.ndarray 
            Array shape (N,). Input time-domain signal to be transformed.
        m0 : bool
            If True, then the :math:`m=0` terms are computed correctly.
            If False, then these terms will be incorrect.
            If these terms are not needed, then leave this at the default False 
            value for faster performance. Optional.

        Returns
        -------
        w : jnp.ndarray
            Array shape shape (Nt, Nf). 
            WDM time-frequency-domain wavelet coefficients.
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        n_vals = jnp.arange(self.Nt)
        m_vals = jnp.arange(self.Nf)
        alternate = (-1)**(n_vals[:,jnp.newaxis] * m_vals[jnp.newaxis,:])

        X = jnp.fft.ifftshift(x, axes=-1)

        X = jnp.fft.ifft(X, axis=-1)

        X = overlapping_windows(X, self.Nt, self.Nf, self.Nt//2) # (Nf, Nf)

        l_vals = jnp.arange(-self.Nt//2, self.Nt//2)

        X *= self.window_FD[l_vals % self.N]

        X = jnp.fft.ifftshift(X, axes=-1)

        X = jnp.fft.fft(X, axis=-1) 

        w = jnp.sqrt(2.) * 2. * jnp.pi * self.dt * alternate * \
                    jnp.real( self.Cnm * X.T )
        
        if m0:
            k_vals = jnp.arange(-self.K//2, self.K//2)
            for n in range(self.Nt):
                if n<self.Nt//2:
                    x_term = x[(k_vals + 2*n*self.Nf) % self.N]
                    phi_term = self.window[k_vals % self.N]
                    norm = 2.*jnp.pi*self.dt
                    term = norm*x_term*phi_term
                else:
                    x_term = x[(k_vals + (2*n+self.Q)*self.Nf) % self.N]
                    phi_term = self.window[k_vals % self.N]
                    alt_term = (-1)**k_vals
                    norm = 2.*jnp.pi*self.dt
                    term = norm*x_term*phi_term*alt_term
                w = w.at[n, 0].set(jnp.sum(term).real)

        return w

    def FWT(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Wrapper for the fast_forward_transform with a nice short name.

        FWT stands for Fast Wavelet Transform.
        """
        return self.fast_forward_transform(x)
    
    def IFWT(self, w: jnp.ndarray) -> jnp.ndarray:
        r"""
        Wrapper for the fast_inverse_transform with a nice short name.

        IFWT stands for Inverse Fast Wavelet Transform.
        """
        return self.fast_inverse_transform(w)

    def time_domain_plot(self, 
                         x: jnp.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        r"""
        Plot the time-domain signal.

        Parameters
        ----------
        x : jnp.ndarray
            Array shape (N,). Input time-domain signal to be plotted.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object.
        ax : matplotlib.axes.Axes
            The matplotlib Axes object where the wavelets were plotted.

        Notes
        -----
        This function does not call ``plt.show()``. The user is responsible
        for displaying or saving the plot.
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        fig, ax = plt.subplots()
        ax.plot(self.times, x)
        ax.set_xlabel(r'Time $t$')
        ax.set_ylabel(r'Signal $x(t)$')
        return fig, ax

    def frequency_domain_plot(self,
                              x: jnp.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        r"""
        Plot the frequency-domain signal.

        Parameters
        ----------
        x : jnp.ndarray
            Array shape (N,). Input time-domain signal to be plotted.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object.
        ax : matplotlib.axes.Axes
            The matplotlib Axes object where the wavelets were plotted.

        Notes
        -----
        This function does not call ``plt.show()``. The user is responsible
        for displaying or saving the plot.
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        data = jnp.abs(jnp.fft.fft(x))
        mask = self.freqs >= 0.

        fig, ax = plt.subplots()
        ax.loglog(self.freqs[mask], data[mask])
        ax.set_xlabel(r'Frequency $f$')
        ax.set_ylabel(r'Signal $|\tilde{X}(f)|$')
        return fig, ax

    def time_frequency_plot(self, 
                            w: jnp.ndarray, 
                            part='abs',
                            scale='linear') -> Tuple[plt.Figure, plt.Axes]:
        r"""
        Plot the time-frequency coefficients of the WDM transform.

        Parameters
        ----------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency coefficients to be plotted.
        part : str
            Part of the coefficients to plot. Options are 'abs' for magnitude, 
            'real', or 'imag'. Default is 'abs'. Optional.
        scale : str
            Scale of the colour axis of the plot. Passed to matplotlib. 
            Options are 'linear' or 'log'. Default is 'linear'. Logarithmic 
            scale should only be used with part='abs' otherwise problems with 
            negative values will occur. Optional.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object.
        ax : matplotlib.axes.Axes
            The matplotlib Axes object where the wavelets were plotted.

        Notes
        -----
        This function does not call ``plt.show()``. The user is responsible
        for displaying or saving the plot.
        """
        assert w.shape == (self.Nt, self.Nf), \
                    f"Input signal must have shape ({self.Nt}, {self.Nf}), " \
                    f"got {w.shape=}"

        if part == 'abs':
            data = jnp.abs(w)
        elif part == 'real':
            data = jnp.real(w)
        elif part == 'imag':
            data = jnp.imag(w)
        else:
            raise ValueError(f"Invalid {part=}. " + 
                             "Choose 'abs', 'real', or 'imag'.")

        fig, ax = plt.subplots()
        if scale == 'linear':
            im = ax.imshow(data.T, aspect='auto', origin='lower', 
                       extent=[0., self.T, 0., self.f_Ny], cmap='jet')
            fig.colorbar(im, label='Magnitude', ax=ax)
        elif scale == 'log':
            im = ax.imshow(jnp.log10(data).T, aspect='auto', origin='lower', 
                       extent=[0., self.T, 0., self.f_Ny], cmap='jet')
            fig.colorbar(im, label='log10 Magnitude', ax=ax)
        else:
            raise ValueError(f"Invalid {scale=}. Choose 'linear' or 'log'.")
        
        ax.set_xlabel(r'Time $t$')
        ax.set_ylabel(r'Frequency $f$')
        return fig, ax

    def __repr__(self) -> str:
        r"""
        String representation of the WDM_transform instance.

        Returns
        -------
        text : str
            A string representation of WDM_transform instance 
        """
        text = (f"WDM_transform(dt={self.dt}, Nf={self.Nf}, q={self.q}, "
                f"A_frac={self.A_frac}, B_frac={self.B_frac}, d={self.d})")
        return text
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Call method to perform fast forward discrete wavelet transform.

        Parameters
        ----------
        x : jnp.ndarray
            Array shape (N,). Input signal to be transformed. 
        
        Returns
        -------
        jnp.ndarray 
            Array shape (Nt, Nf). WDM time-frequency coefficients.
        """
        return self.FWT(x)