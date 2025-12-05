==================
Time-Delay Filters
==================

This section describes the time shift properties of the WDM wavelets. 
This is based on the results in section 4 of Ref. [1]_ .


.. contents::
   :local:


Introduction
------------

The WDM wavelets :math:`\{g_{nm}(t)|n\in\{0,1,\ldots,N_t-f\}, m\in\{0,1,\ldots,N_f-1\}\}` 
form a complete basis, meaning that any time series can be expanded in terms of these wavelets. 
In particular, if we take a single wavelet and shift it in time, :math:`g_{n'm'}(t+\delta t)` 
we can rexpand this in terms of the original, unshifted wavelet basis.
The coefficients of this expansion are given by the overlap integrals

.. math::
    
    X_{nn';mm'}(\delta t) = \int \mathrm{d}t \, g_{nm}(t+\delta t) g_{n'm'}(t) . 

This can be written as an integral over frequency; 

.. math::
    
    X_{nn';mm'}(\delta t) = \int \mathrm{d}f \, \exp(-2\pi i f\delta t) \tilde{G}^*_{nm}(f) \tilde{G}_{n'm'}(f) .

Most of the :math:`X_{nn';mm'}(\delta t)` coefficients are zero because the wavelets :math:`\tilde{G}_{nm}(f)` 
have compact support in the frequency domain and the integral will vanish unless the wavelets overlap.;
the only non-zero coefficients are those for which :math:`m'=m` or :math:`m'=m\pm 1`.

The main results for the general case when :math:`m\neq 0` and :math:`m'\neq 0` are 

.. math::
    
    X_{nn';mm}(\delta t) = \mathrm{Re} \bigg\{ (-1)^{(n-n')m} \exp(2\pi i m \Delta F \delta t) 
                            C^*_{nm} C_{n'm} T_{n-n'}(\delta t) \bigg\} , 

.. math::

    X_{nn';m(m\pm 1)}(\delta t) = \mathrm{Re} \bigg\{ (-1)^{(n-n')m} (\mp i)^{n-n'} 
                            \exp\left(2\pi i \left(m\pm\frac{1}{2}\right) \Delta F \delta t\right) 
                            C^*_{nm} C_{n'(m\pm 1)} T'_{n-n'}(\delta t) \bigg\} ,

where the time-delay filters :math:`T_{\ell}(\delta t)` and :math:`T'_{\ell}(\delta t)` are
defined as

.. math::
    
    T_{\ell}(\delta t) = \int \mathrm{d}f \, \exp(2\pi i f (\ell \Delta T - \delta t)) |\tilde{\Phi}(f)|^2 , 

.. math::
    
    T'_{\ell}(\delta t) = \int \mathrm{d}f \, \exp(2\pi i f (\ell \Delta T - \delta t)) 
                        \tilde{\Phi}\left(f-\frac{1}{2}\Delta F\right)\tilde{\Phi}\left(f+\frac{1}{2}\Delta F\right) .

The time-delay filters :math:`T_{\ell}(\delta t)` and :math:`T'_{\ell}(\delta t)` are implemented in 
:func:`WDM.code.time_delay_filters.filters.time_delay_filter_Tl` and
:func:`WDM.code.time_delay_filters.filters.time_delay_filter_Tl_prime` respectively.

The full time-delay matrix elements :math:`X_{nn';mm'}(\delta t)` are implemented in
:func:`WDM.code.time_delay_filters.filters.time_delay_X`.

These functions can be precomputed and interpolated for efficient use later.
As can be seen from the plots in  :numref:`fig-time_delay-filters`, these time-delay filters
only need to be interpolated in the narrow range :math:`0\leq \delta t < 2\Delta T`.

.. _fig-time_delay-filters:

.. figure:: ../figures/time_delay_filters.png
   :alt: Time_Delay_Filters
   :align: center
   :width: 90%

   The time-delay filter functions :math:`T_\ell(\delta t)` (left) and 
   :math:`T'_\ell(\delta t)` (right) plotted as a function of :math:`\delta t`
   for several values of :math:`\ell`. 



Derivation
----------

This section is for a derivation of the results in Eqs. (40) to (43) above.

SEE HANDWRITTEN NOTES




References
----------

.. [1] V. Necula, S. Klimenko & G. Mitselmakher, *Transient analysis with fast Wilson-Daubechies time-frequency transform*, Journal of Physics: Conference Series 363 012032, 2012.  
       `DOI 10.1088/1742-6596/363/1/012032 <https://iopscience.iop.org/article/10.1088/1742-6596/363/1/012032>`_
