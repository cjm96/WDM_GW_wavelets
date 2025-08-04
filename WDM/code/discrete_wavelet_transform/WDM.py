import jax
import jax.numpy as jnp
from WDM.code.utils.Meyer import Meyer
from WDM.code.utils.utils import next_multiple, C_nm


single = jnp.float32


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
        :math:`\delta t` by :math:`T = N \\delta t`.
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
        Time-domain window of length :math:`K`, normalized to unit L2 norm.
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
        """
        self.dt = float(dt)
        self.Nf = int(Nf)
        self.N = int(N)
        self.q = int(q)
        self.A_frac = float(A_frac)
        self.d = int(d)

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
                    f"A_frac must be in [0, 1], for {self.A_frac=}"
        assert self.d>=1, \
                    f"d must be a positive integer, got {self.d=}"
        
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

    def build_time_domain_window(self) -> jnp.ndarray:
        """
        Construct the time-domain window function :math:`\\phi(t)`.

        This method builds the Meyer window in the frequency domain and applies
        an inverse FFT to obtain the corresponding time-domain window.

        Returns
        -------
        phi : jnp.ndarray of shape (K,)
            Real-valued time-domain window, normalized to unit L2 norm.
        """
        f = jnp.fft.fftfreq(self.K, d=self.dt) 
        Phi = Meyer(2.*jnp.pi*f, self.d, self.A, self.B)
        phi = jnp.fft.ifft(Phi).real
        phi = phi / jnp.linalg.norm(phi)
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
            freq = jnp.asarray(freq, dtype=single)

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
            time = jnp.asarray(time, dtype=single)

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
        Call method to perform forward discrete wavelet transform.

        Parameters
        ----------
        x : jnp.ndarray of shape (N,)
            Input signal to be transformed. 
        
        Returns
        -------
        jnp.ndarray of shape (Nt, Nf)
            WDM time-frequency coefficients.
        """
        return None #self.transform(x)