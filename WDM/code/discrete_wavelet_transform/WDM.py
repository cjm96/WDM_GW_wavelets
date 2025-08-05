import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from WDM.code.utils.Meyer import Meyer
from WDM.code.utils.utils import next_multiple, C_nm


if jax.config.read("jax_enable_x64"):
    jax_dtype = jnp.float64
else:
    jax_dtype = jnp.float32


class WDM_transform:
    """
    This class implements the WDM discrete wavelet transform for the Meyer 
    window function.

    Attributes
    ----------
    dt : float
        The cadence, or time step, of the original time series (seconds). 
        Equal to inverse of the sampling frequency (Hertz).
    Nf : int
        Number of wavelet frequency bands. This controls the time/frequency 
        resolution; as :math:`N_f \\rightarrow N/2` the wavelet expansion 
        approaches the  Fourier series, and as :math:`N_f \\rightarrow 1` it 
        approaches the original time series.
    N : int
        Length of the input time series. Must be an even multiple of 
        :math:`N_f`.
    Nt : int
        Number of wavelet time bands. Equal to :math:`N/N_f`.
    q : int
        Truncation parameter. Formally the time domain wavelet has infinite 
        extent, but in practice it is truncated at :math:`\\pm q \\Delta T`. 
    A_frac : float
        Fraction of total bandwidth used for the flat-top response region.
        Must be in the range [0, 1].
    B_frac : float
        Fraction of total bandwidth used for the transition region. This is set
        based on A_frac so :math:`2A_{\\mathrm{frac}}+B_{\\mathrm{frac}}=1`.
    d : int
        Steepness parameter for the Meyer window transition. 
        Must be a positive integer, :math:`d\\geq 1`.
    K : int
        Window length in samples (equal to :math:`2 q N_f`). By definition, 
        this is always an even integer.
    dF : float
        Frequency resolution of the wavelets (Hertz), or the total wavelet 
        frequency bandwidth :math:`\\Delta F = \\frac{\\Delta \\Omega}{2 \\pi}`.
    dT : float
        Time resolution of the wavelets (seconds). Related to the frequency 
        resolution by :math:`\\Delta F \\Delta T = \\frac{1}{2}`.
    T : float
        Total duraion of the time series (seconds). Related to :math:`N` and 
        :math:`\\delta t` by :math:`T = N \\delta t`.
    dOmega : float
        Angular Frequency resolution of the wavelets (radians per second), or 
        the total wavelet angular frequency bandwidth 
        :math:`\\Delta \\Omega = 2A + B`.
    f_Ny : float
        Nyquist frequency (i.e. maximum frequency) of the original time series 
        (Hertz), equal to :math:`\\frac{1}{2 dt}`.
    A : float
        Half-width of the flat-top response region in angular frequency 
        (radians per second).
    B : float
        Width of the transition region in angular frequency 
        (radians per second).
    window : jnp.ndarray
        Time-domain window of length :math:`K`.
    """

    def __init__(self, 
                 dt: float,
                 Nf: int,
                 N: int,
                 q: int = 16,
                 d: int = 4,
                 A_frac: float = 0.25) -> None:
        """
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
        self.dF = 1. / ( 2. * self.dt * self.Nf )  
        self.dOmega = 2. * jnp.pi * self.dF
        self.dT = self.dt * self.Nf 
        self.f_Ny = 0.5 / self.dt
        self.B_frac = 1. - 2. * self.A_frac  
        self.A = self.A_frac * self.dOmega
        self.B = self.B_frac * self.dOmega
        self.K = 2 * self.q * self.Nf

        self.window = self.build_time_domain_window()

    def validate_parameters(self) -> None:
        """
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
        
        assert 0. < self.A_frac < 1., \
                    f"A_frac must be in [0, 1], got {self.A_frac=}"
        
        assert self.d>=1, \
                    f"d must be a positive integer, got {self.d=}"

    def build_time_domain_window(self) -> jnp.ndarray:
        """
        Construct the time-domain window function :math:`\\phi(t)`.

        This method builds the Meyer window in the frequency domain and applies
        an inverse FFT to obtain the corresponding time-domain window.

        Returns
        -------
        phi : jnp.ndarray of shape (K,)
            Real-valued time-domain window. (This is not normalised.)
        """
        f = jnp.fft.fftfreq(self.K, d=self.dt) 
        Phi = Meyer(2.*jnp.pi*f, self.d, self.A, self.B)
        phi = jnp.fft.ifft(Phi).real
        return phi
    
    def Gnm(self, 
            n: int, 
            m: int,
            freq: jnp.ndarray = None) -> jnp.ndarray:
        """
        Compute the frequency-domain wavelets :math:`\\tilde{g}_{nm}(f)`.

        Parameters
        ----------
        n : int
            Wavelet time index.
        m : int
            Wavelet frequency index.
        freq : jnp.ndarray, optional
            Frequencies at which to evaluate the wavelet (Hertz). 
            If None, then defaults to the FFT frequencies.

        Returns
        -------
        Gnm : complex jnp.ndarray shaped like freq
            The frequency-domain wavelet.
        """
        if freq is None:
            freq = self.freqs
        else:
            freq = jnp.asarray(freq, dtype=jax_dtype)

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
        """
        Compute the time-domain wavelets :math:`g_{nm}(t)`.

        This function computes the inverse Fourier transform of the 
        frequency-domain wavelet, computed using the method Gnm.

        Parameters
        ----------
        n : int
            Wavelet time index.
        m : int
            Wavelet frequency index.
        time : jnp.ndarray, optional
            Times at which to evaluate the wavelet (seconds). 
            If None, then defaults to the FFT values.

        Returns
        -------
        gnm : jnp.ndarray of shape (N,)
            The time-domain wavelet.
        """
        if time is None:
            time = self.times
        else:
            time = jnp.asarray(time, dtype=jax_dtype)

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
        """ 
        The transform method requires the input time series signal to have a 
        specific length :math:`N`.

        This function also returns a Boolean mask that can be used later to 
        recover arrays of the original length.

        Parameters
        ----------
        x : jnp.ndarray
            Input signal to be padded.
        where : str, optional
            Where to add the padding. Options are 'end', 'start', or 'equal' 
            which puts the zero padding at the end of the signal, the start of 
            the signal, or equally at both ends respectively.

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
    
    def forward_transform_exact(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        This method computes the wavelet coefficients using the exact expression

        .. math::

            w_{nm} = 2\\pi\\delta t\\sum_{k=0}^{N-1} g_{nm}[k] x[k] ,

        where the sum is over the whole time-domain signal (no truncation). The 
        time domain wavelets `g_{nm}[k]` are computed using an inverse FFT. This 
        method is slow but exact.

        Parameters
        ----------
        x : jnp.ndarray of shape (N,)
            Input time-domain signal to be transformed.

        Returns
        -------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency-domain wavelet coefficients.

        Notes
        -----
        This is implemented using for loops. It is slow. It is only intended to 
        be used for testing and debugging purposes. 
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"

        w = jnp.zeros((self.Nt, self.Nf), dtype=jax_dtype) 

        for n in range(self.Nt):
            for m in range(self.Nf):
                gnm = self.gnm(n, m)
                w = w.at[n, m].set(2.*jnp.pi*self.dt*jnp.sum(gnm*x))

        return w
    
    def inverse_transform_exact(self, w: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the inverse discrete wavelet transform. Transforms the input
        signal from the time-frequency wavelet domain into the time domain.

        This method computes the inverse discrete wavelet transform using the 
        exact expression

        .. math::

            x[k] = \\sum_{n=0}^{N_t-1}\\sum_{m=0}^{N_f-1} w_{nm} g_{nm}[k] ,

        where the sum is over the whole time-domain signal (no truncation). The 
        time domain wavelets `g_{nm}[k]` are computed using an inverse FFT. This 
        method is slow but exact.

        Parameters
        ----------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency-domain wavelet coefficients.

        Returns
        -------
        x : jnp.ndarray of shape (N,)
            Input time-domain signal to be transformed.

        Notes
        -----
        This is implemented using for loops. It is slow. It is only intended to 
        be used for testing and debugging purposes. 
        """
        assert w.shape == (self.Nt, self.Nf), \
                    f"Input signal must have shape ({self.Nt}, {self.Nf}), " \
                    f"got {w.shape=}"
        
        x = jnp.zeros(self.N, dtype=jax_dtype) 

        for n in range(self.Nt):
            for m in range(self.Nf):
                x = x + w[n, m] * self.gnm(n, m)

        return x
    
    def forward_transform_truncated(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward discrete wavelet transform. Transforms the input
        signal from the time domain into the time-frequency domain.

        This method computes the wavelet coefficients using the truncated 
        expression

        .. math::

            w_{nm} = Eq13 ,

        where the sum is over the truncated window. This method is slow. 

        Parameters
        ----------
        x : jnp.ndarray of shape (N,)
            Input time-domain signal to be transformed.

        Returns
        -------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency-domain wavelet coefficients.

        Notes
        -----
        This method is slow. It is only intended to be used for testing and 
        debugging purposes. 
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"

        w = jnp.zeros((self.Nt, self.Nf), dtype=jax_dtype) 

        for n in range(self.Nt):
            for m in range(self.Nf):
                if m==0:
                    if n < self.Nt//2:
                        x_term = jnp.array([ x[(2*n*self.Nf + k)%self.N]
                                            for k in range(-self.K//2, self.K//2)])
                        phi_term = jnp.array([ self.window[k%self.K]
                                            for k in range(-self.K//2, self.K//2)])
                        all_terms = 2.*jnp.pi*self.dt*phi_term*x_term
                        w = w.at[n, m].set(jnp.sum(all_terms))
                    else:
                        Q = self.Nf % 2
                        alt_term = jnp.array([ (-1)**k
                                            for k in range(-self.K//2, self.K//2)])
                        x_term = jnp.array([ x[(2*n*self.Nf + Q*self.Nf + k)%self.N]
                                            for k in range(-self.K//2, self.K//2)])
                        phi_term = jnp.array([ self.window[k%self.K]
                                            for k in range(-self.K//2, self.K//2)])
                        all_terms = 2.*jnp.pi*self.dt*phi_term*x_term
                        w = w.at[n, m].set(jnp.sum(all_terms))
                else:
                    exp_term = jnp.array([ jnp.exp((1j) * jnp.pi * k * m / self.Nf)
                                        for k in range(-self.K//2, self.K//2)])
                    x_term = jnp.array([ x[(n*self.Nf + k)%self.N]
                                        for k in range(-self.K//2, self.K//2)])
                    phi_term = jnp.array([ self.window[k%self.K]
                                        for k in range(-self.K//2, self.K//2)])
                    all_terms = 2.*jnp.sqrt(2.)*jnp.pi*self.dt*C_nm(n,m)*phi_term*exp_term*x_term
                    w = w.at[n, m].set(jnp.sum(all_terms).real)

        return w
    
    def inverse_transform_truncated(self, w: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the inverse discrete wavelet transform. Transforms the input
        signal from the time-frequency wavelet domain into the time domain.

        This method computes the wavelet coefficients using the truncated 
        expression

        .. math::

            x[k] = ? ,

        where the sum is over... This method is slow.

        Parameters
        ----------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency-domain wavelet coefficients.

        Returns
        -------
        x : jnp.ndarray of shape (N,)
            Input time-domain signal to be transformed.

        Notes
        -----
        This method is slow. It is only intended to be used for testing and 
        debugging purposes. 
        """
        assert w.shape == (self.Nt, self.Nf), \
                    f"Input signal must have shape ({self.Nt}, {self.Nf}), " \
                    f"got {w.shape=}"
        
        x = jnp.zeros(self.N, dtype=jax_dtype) 

        #

        return x
    
    def fast_forward_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the fast forward discrete wavelet transform. 
        """
        pass

    def fast_inverse_transform(self, w: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the fast inverse discrete wavelet transform. 
        """
        pass

    def FWT(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Wrapper for the fast_forward_transform with a nice short name.

        FWT stands for Fast Wavelet Transform.
        """
        return self.fast_forward_transform(x)
    
    def IFWT(self, w: jnp.ndarray) -> jnp.ndarray:
        """
        Wrapper for the fast_inverse_transform with a nice short name.

        IFWT stands for Inverse Fast Wavelet Transform.
        """
        return self.fast_inverse_transform(w)

    def time_domain_plot(self, x: jnp.ndarray) -> None:
        """
        Plot the time-domain signal.

        Parameters
        ----------
        x : jnp.ndarray of shape (N,)
            Input time-domain signal to be plotted.

        Returns
        -------
        None
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        fig, ax = plt.subplots()
        ax.plot(self.times, x)
        ax.set_xlabel(r'Time $t$')
        ax.set_ylabel(r'Signal $x(t)$')
        plt.show()

    def frequency_domain_plot(self, x: jnp.ndarray) -> None:
        """
        Plot the frequency-domain signal.

        Parameters
        ----------
        x : jnp.ndarray of shape (N,)
            Input time-domain signal to be plotted.

        Returns
        -------
        None
        """
        assert x.shape == (self.N,), \
                    f"Input signal must have shape ({self.N},), got {x.shape=}"
        
        data = jnp.abs(jnp.fft.fft(x))
        mask = self.freqs >= 0.

        fig, ax = plt.subplots()
        ax.loglog(self.freqs[mask], data[mask])
        ax.set_xlabel(r'Frequency $f$')
        ax.set_ylabel(r'Signal $|\tilde{X}(f)|$')
        plt.show()

    def time_frequency_plot(self, w: jnp.ndarray, 
                            part='abs',
                            scale='linear') -> None:
        """
        Plot the time-frequency coefficients of the WDM transform.

        Parameters
        ----------
        w : jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency coefficients to be plotted.
        part : str, optional
            Part of the coefficients to plot. Options are 'abs' for magnitude, 
            'real', or 'imag'. Default is 'abs'.
        scale : str, optional
            Scale of the colour axis of the plot. Passed to matplotlib. 
            Options are 'linear' or 'log'. Default is 'linear'. Logarithmic 
            scale should only be used with part='abs' otherwise problems with 
            negative values will typically occur.

        Returns
        -------
        None
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
            raise ValueError(f"Invalid {part=}. Choose 'abs', 'real', or 'imag'.")

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
        plt.show()

    def __repr__(self) -> str:
        """
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
        """
        Call method to perform fast forward discrete wavelet transform.

        Parameters
        ----------
        x : jnp.ndarray of shape (N,)
            Input signal to be transformed. 
        
        Returns
        -------
        jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency coefficients.
        """
        return self.FWT(x)