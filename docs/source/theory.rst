Mathematical Background
=======================

This section describes the mathematical background to the WDM wavelet transform.

.. contents::
   :local:



Introduction
------------

The Wilson-Daubechies-Meyer (WDM) wavelet basis is widely used for gravitational wave (GW) data analysis.
While far from being the only available choice, the WDM basis wavelets have some properties that make 
them particularly suitable for this purpose: they are well separated in frequency (in fact, the wavelets have 
compact support in the frequency domain) which helps to make connections with other GW data analysis which 
is almost exclusively done in the frequency domain, and they provide uniform tiling in both time and frequency.
The WDM wavelets were first introduced to GW data analysis in Ref. [1]_ (see also Ref. [2]_) and are used in the
coherent WaveBurst (CWB; Refs. [3]_ and [4]_).



Regularised Incomplete Beta Function
------------------------------------

The WDM wavelets use the regularised (normalised) incomplete
beta function, :math:`\nu_d(x)`,

.. math::
   :name: eq:reg_incomplete_beta

   \nu_d(x) = \frac{ \int_0^x \mathrm{d}t \, t^{d-1} (1 - t)^{d-1} }
                         { \int_0^1 \mathrm{d}t \, t^{d-1} (1 - t)^{d-1} } .

This is defined for :math:`x \in [0, 1]` and acts as a smooth 
transition function (or compact sigmoid-like function) from 0 to 1.
The parameter :math:`d` controls the steepness of the transition;
see :numref:`fig-reg_incomplete_beta`.

The function :math:`\nu_d(x)` is implemented in :func:`WDM.code.utils.Meyer.nu_d`.

.. _fig-reg_incomplete_beta:

.. figure:: ../figures/reg_incomplete_beta.png
   :alt: reg_incomplete_beta
   :align: center
   :width: 70%

   The regularised incomplete beta function :math:`\nu_d(x)` for different values of :math:`d`.



Meyer Window
------------

The WDM wavelet transform is based on the Meyer window function, which is 
defined in the frequency domain as

.. math::
   :name: eq:Meyer_window

    \tilde{\Phi}(\omega) = \begin{cases}
        \frac{1}{\sqrt{\Delta\Omega}} & \text{if } |\omega| < A, \\
        \frac{1}{\sqrt{\Delta\Omega}} \cos\left(\frac{\pi}{2}\nu_d\left(\frac{|\omega| - A}{B}\right)\right) & \text{if } A \leq |\omega| \leq A + B \\
        0 & \text{if } |\omega| > A + B
    \end{cases} ,

where :math:`\omega=2\pi f` is the angular frequency, :math:`A` and :math:`B` are two positive angular frequency parameters
satisfying :math:`2A + B = \Delta\Omega`, and :math:`\Delta\Omega` is the wavelet bandwidth.
The parameter :math:`A` is the half-width of the flat-top response region while :math:`B` is the width of the transition region;
see :numref:`fig-Meyer_window`.

The function :math:`\tilde{\Phi}(\omega)` is implemented in :func:`WDM.code.utils.Meyer.Meyer`.

.. _fig-Meyer_window:

.. figure:: ../figures/Meyer_window.png
   :alt: Meyer_window
   :align: center
   :width: 90%

   The Meyer window function :math:`\Phi(\omega)` for different values of :math:`d`.
   The top panel shows the window in the frequency-domain, while the bottom panel shows it in the time-domain; :math:`\phi(t)=\mathrm{FT}^{-1}(\Phi(\omega))`, and where :math:`\Delta T = \pi/\Delta \Omega`.
   This plot uses :math:`A=\Delta \Omega/4`, :math:`B=\Delta \Omega/2`, and includes the case :math:`d=4` to match Fig.1 of Ref. [2]_.



WDM Wavelets
------------

Henceforth, we will work with time :math:`t` (e.g. in seconds) and frequency :math:`f` (Hertz) rather than angular frequency 
:math:`\omega=2\pi f`. This is inline with the rest of the GW data analysis community which tends to work with frequency.

The WDM wavelets form a complete basis for any time series.
Consider a time series with cadence :math:`\delta t`, duration :math:`T=N \delta t`, and maximum Nyquist frequency :math:`f_{\rm Ny} = \frac{1}{2\delta t}`. 
In order to define the WDM wavelet transform it ie necessary to choose a number of frequency bands :math:`N_f`.
We will assume that :math:`N_f` divides :math:`N` exactly (if not, then the time series can be padded as necessary), 
and :math:`N_t = N/N_f` is the number of time bands.
Other derived quantities that follow from this are  :math:`\Delta \Omega = 2\pi \Delta F` 
where :math:`\Delta F = \frac{1}{2 \delta t N_f }` and :math:`\Delta T \Delta F = \frac{1}{2}`.

The wavelet expansion of a time series :math:`x[k]` (where :math:`k\in\{0, 1, \ldots, N\}` indexes the time) is given by

.. math::
   :name: eq:wavelet_expansion

   x[k] = \sum_{n=0}^{N_t-1} \sum_{m=0}^{N_f-1} w_{nm} g_{nm}[k] ,

where :math:`w_{nm}` are the wavelet coefficients and :math:`n\in\{0, 1, \ldots, N_t-1\}` and 
:math:`m\in\{0, 1, \ldots, N_f-1\}` index the time and frequency respectively.

The WDM wavelets :math:`g_{nm}` are constructed from the Meyer window function. 
In the frequency-domain they are defined as

.. math::
   :name: eq:Gnm

    \tilde{G}_{nm}(f) = \begin{cases}
        \exp(-4\pi i n f \Delta T) \tilde{\Phi}(2\pi f) & m=0 \\
        \exp(-2\pi i n f \Delta T) \left( C_{nm}\tilde{\Phi}(2\pi [f-m\Delta F])
        +C^*_{nm}\tilde{\Phi}(2\pi [f+m\Delta F]) \right) & 0<m<N_f \\
        \exp(-2\pi i Q f \Delta T) \left( \tilde{\Phi}(2\pi [f+N_f\Delta F]) + \tilde{\Phi}(2\pi [f-N_f\Delta F]) \right) & m=N_f \\
    \end{cases} ,

where :math:`Q=2n+(N_f\,\mathrm{mod}\,2)` and the coefficients :math:`C_{nm}` are defined to be 1 is if :math:`n+m` 
is even, and :math:`i` if :math:`n+m` is odd and are implemented in :func:`WDM.code.utils.utils.C_nm`.

The WDM wavelets :math:`\tilde{G}_{nm}(\omega)` are implemented in :func:`WDM.code.discrete_wavelet_transform.WDM.WDM_transform.Gnm`.

To understand the wavelet definitions it is best to focus first on the middle case, :math:`0<m<N_f`.
The index :math:`n` describes a time shift by an amount :math:`\Delta T`.
The index :math:`m` describes a frequency shift of the wavelet by an amount :math:`m\Delta\Omega`.
Unfortunatelt, this doesn't quite holde for the cases :math:`m=0` and :math:`m=N_f` which are handled separately.

The WDM wavelets are plotted in the frequency domain in :numref:`fig-WDM_wavelets_FD`.

.. _fig-WDM_wavelets_FD:

.. figure:: ../figures/Gnm_spectra.png
   :alt: Gnm_spectra
   :align: center
   :width: 70%

   The :math:`d=4` WDM wavelets :math:`|\tilde{G}_{nm}(\omega)|` plotted in the frequency domain for 
   :math:`m=0, 1, 2,\ldots,N_f`. (The :math:`n` index only describes a time shift and has no effect on 
   this plot.) This plot was produced using :math:`N_f=16` to match Fig.2 of Ref. [1]_.

The wavelets in the time-domain, :math:`g_{nm}(t)`, are constructed by taking an inverse Fourier transform.
The wavelets :math:`g_{nm}(t)` are implemented in :func:`WDM.code.discrete_wavelet_transform.WDM.WDM_transform.gnm`.

Using :math:`N=512`, :math:`\delta t=1`, and :math:`N_f=16`, several examples of the time-domain WDM 
wavelets are plotted in :numref:`fig-WDM_wavelets_TD`, :numref:`fig-WDM_wavelets_TF` and :numref:`fig-WDM_wavelets_animate`.
Notice how the wavelets are well localised in frequency but much less so in time.

.. _fig-WDM_wavelets_TD:

.. figure:: ../figures/gnm_wavelets.png
   :alt: gnm_wavelets
   :align: center
   :width: 70%

   The WDM wavelets :math:`g_{nm}(t)` plotted in the time domain for a few selected values of :math:`n` and :math:`m`.

.. _fig-WDM_wavelets_TF:

.. figure:: ../figures/wavelets_TF.png
   :alt: wavelets_TF
   :align: center
   :width: 90%

   The WDM wavelets plotted in the time (top) and frequency (right) domains for selected values of :math:`n` and :math:`m`.
   The main plot shows a grid of time-frequency shaded to indicate where the corresponding wavelets have significant support.

.. _fig-WDM_wavelets_animate:

.. figure:: ../figures/wavelet_animation.gif
   :alt: wavelet_animation
   :align: center
   :width: 90%

   Animation looping through all the wavelets. Note that the :math:`m\in\{0,1,\ldots, N_f\}` index is related in a straightforward
   way to the central frequency of the wavelets. The :math:`n\in\{0,1,\ldots, N_t-1\}` index is USUALLY related to the central 
   time of the wavelet, except when :math:`m=0` or :math:`m=N_f` where the time shifting is more complicated.

The WDM wavelet basis has the following orthonomality property,

.. math::
   :name: eq:orthonorm

   2 \pi \delta t \sum_{k=0}^{N-1} g_{nm}[k] g_{n'm'}[k] = \delta_{nn'} \delta_{mm'} .



The Discrete WDM Wavelet Transform
----------------------------------

Hello.


The Discrete WDM Wavelet Inverse Transform
------------------------------------------

Hello.


Glossary 
--------

- :math:`t`: Time (e.g. seconds).
- :math:`f`: Frequency (e.g. Hertz).
- :math:`\omega`: Angular frequency (e.g. radians/second). Defined as :math:`\omega=2\pi f`.
- :math:`\delta t`: Time series cadence (seconds). Named ``dt`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`f_{\rm Ny}`: Nyquist frequency, or the maximum frequency (Hertz). Defined as :math:`f_{\rm Ny}=\frac{1}{2 \delta t}`. Named ``f_Ny`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`A`: With of flat-top response in the Meyer window (radians/second). Named ``A`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`B`: With of transition region in the Meyer window (radians/second). Named ``B`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`\Delta \Omega`: Angular frequency resolution of the wavelets (radians/second). Satisfies :math:`\Delta \Omega = 2A + B`. Named ``dOmega`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`\Delta F`: Frequency resolution of the wavelets (Hertz). Satisfies :math:`\Delta F = \frac{\Delta \Omega}{2\pi}`. Named ``dF`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`\Delta T`: Time resolution of the wavelets (seconds). Satisfies :math:`\Delta T \Delta F= \frac{1}{2}`. Named ``dT`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`d`: Steepness parameter for the Meyer window. Named ``d`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`q`: Truncation parameter for the Meyer window. Named ``q`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`N_f`: Number of frequency bands for the wavelets. Named ``N_f`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`N_t`: Number of time bands for the wavelets, must be even. Named ``N_t`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`. 
- :math:`N`: Number of points in the time series. Satisfies :math:`N = N_t N_f`. Named ``N`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`.
- :math:`T`: Duration of the time series (seconds). Satisfies :math:`T = N \delta t`. Named ``T`` in :func:`WDM_transform <WDM.code.discrete_wavelet_transform.WDM.WDM_transform>`.
- :math:`n`: Time index for the wavelets. In the range :math:`n\in\{0,1,\ldots, N_t-1\}`.
- :math:`m`: Frequency index for the wavelets. In the range :math:`m\in\{0,1,\ldots, N_f\}`.
   

References
----------

.. [1] V. Necula, S. Klimenko and G. Mitselmakher, *Transient analysis with fast Wilson-Daubechies time-frequency transform*, Journal of Physics: Conference Series 363 012032, 2012.  
       `DOI 10.1088/1742-6596/363/1/012032 <https://iopscience.iop.org/article/10.1088/1742-6596/363/1/012032>`_

.. [2] N. J. Cornish, *Time-Frequency Analysis of Gravitational Wave Data*, Physical Review D 102 124038, 2020.  
       `arXiv:2009.00043 <https://arxiv.org/abs/2009.00043>`_

.. [3] S. Klimenko, S. Mohanty, M. Rakhmanov & G. Mitselmakher, *Constraint likelihood analysis for a network of gravitational wave detectors*, Physical Review D 72, 122002, 2005.
       `arXiv:gr-qc/0508068 <https://arxiv.org/abs/gr-qc/0508068>`_

.. [4] S. Klimenko *et al.*, *Method for detection and reconstruction of gravitational wave transients with networks of advanced detectors*, Physical Review D 93, 042004, 2016.
       `arXiv:1511.05999 <https://arxiv.org/abs/1511.05999>`_

.. [5] Author, *Title*, Journal, Year.  
       `arXiv:0000.00000 <https://arxiv.org/abs/0000.00000>`_