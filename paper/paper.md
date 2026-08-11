---
title: 'WDM_GW_Wavelets: A fast, JAX-based Python implementation of the Wilson-Daubechies-Meyer wavelet transform for the time-frequency analysis of gravitational wave data'
tags:
  - Python
  - JAX
  - astronomy
  - gravitational waves
  - time-frequency methods
  - time series data analysis
authors:
  - name: Christopher J. Moore
    orcid: 0000-0002-2527-0213
    affiliation: "1, 2, 3"
  - name: Tomasz Kinowski
    orcid: 0009-0004-9474-7654
    affiliation: "1, 2"
affiliations:
 - name: Institute of Astronomy, University of Cambridge, Madingley Road, Cambridge, CB3 0HA, UK
   index: 1
 - name: Kavli Institute for Cosmology, University of Cambridge, Madingley Road, Cambridge, CB3 0HA, UK
   index: 2
 - name: Department of Applied Mathematics and Theoretical Physics, Centre for Mathematical Sciences, University of Cambridge, Wilberforce Road, Cambridge, CB3 0WA, UK
   index: 3
date: 11 August 2026
bibliography: paper.bib
---

# Summary


Transient gravitational-wave (GW) signals observed by current ground-based interferometers are short; mostly $\sim 0.5-30\,\mathrm{s}$. 
These signals are usually conveniently analysed in the frequency domain where the instrumental noise (usually assumed to be stationary) is easily modelled.
The next generation of ground-based interferometers, as well as the upcoming LISA space-based interferometer, will observe much longer signals; several hours for ground-based instruments and years for space-based ones.
The instrumental noise will not remain stationary over these time scales.
Additionally, the detectors will move and rotate appreciably over these time scales, complicating the modelling of the instrument response.

These challenges motivate the development of time-frequency methods for GW data analysis.
The Wilson-Daubechies-Meyer (WDM) wavelet basis introduced by `@Necula:2012`, and previously used in a GW context by Coherent WaveBurst (cWB) (see, for example, `@Klimenko:2016` and `@Drago:2021`), has been identified as a promising set of basis functions for this purpose by `@Cornish:2020`. 
The WDM wavelets expand a time series signal $x(t)$ of length $N$ using a discrete, orthogonal basis of wave packets,
\begin{equation}
    x(t) = \sum_{n=0}^{N_t-1}\; \sum_{m=0}^{N_f-1} \, w_{nm} \, g_{nm}(t) \, ,
\end{equation}
where $N=N_tN_f$ and the basis functions $g_{nm}$ are localised in time around $t_n=n\Delta T$ and have compact support in frequency around $f_m=m\Delta F$. 
The quantities $\Delta T$ and $\Delta F$ are the wavelet time and frequency resolutions respectively; 
one may be chosen freely while the other is constrained by $\Delta T \Delta F = 1/2$.
The WDM basis maps the time series onto a regular grid of pixels that cover the time-frequency plane uniformly (see figure \autoref{fig:time_freq_plane}).
The discrete WDM wavelet transform can be written in a form that uses the windowed discrete Fourier transform and has a cost that scales as $\mathcal{O}(2qN\log(2qN_f))$ where $q$ is a small integer (typically $q=16$) that controls the truncation of the window function; see `@Cornish:2020`. 
This is comparable to the cost of a fast Fourier transform, $\mathcal{O}(N\log N)$, making the transformation relatively fast.

GW data analysis often requires introducing time delays between different detectors or data streams, either to synchronize them with respect to a particular sky location or to eliminate a common noise source. 
A key property of the WDM wavelets is the existence of analytic time-delay filters which allow time delays to be performed in the time-frequency domain. 

![Example WDM wavelets plotted in both time (top) and frequency (right) domains. The  time-frequency grid shows where the wavelets have support. Figure reproduced from the package documentation.\label{fig:time_freq_plane}](wavelets_TF.png)

# Statement of need

There are many different methods for time series data analysis in the time-frequency domain.
Similarly, there are many families of wavelets and different conventions within each family.
This can lead to difficulties when comparing results between codes and authors.
There is a need in the GW data analysis community for a single convention with a stable, publicly available implementation that is well-documented and has the mathematical details and conventions clearly described alongside the code.
For maximum utility, this should include the time-delay filters along with the transform itself.
Additionally, to allow for Bayesian data analysis to be performed in the time-frequency domain, there is a need for the wavelet transform to be fast and to be executable on multiple backends, including CPUs, GPUs, and TPUs.

`WDM_GW_Wavelets` is designed to meet these needs.
`WDM_GW_Wavelets` is a JAX-compatible Python package that implements the WDM wavelet transform and the associated time-delay filters and is accompanied by mathematical documentation describing the conventions used.

# State of the field


The WDM wavelets have been used previously in the context of GW data analysis (see, for example, `@Necula:2012`, `@Klimenko:2016` and @`Drago:2021`).
However, `@Cornish:2020` was the first to advocate the benefits of this particular wavelet family for performing GW data analysis in the LISA context. 
`@Cornish:2020` gives only brief mathematical details of the WDM wavelets (for example, it does not discuss the special handling needed for the $m=0$ coefficients which store the zero and Nyquist frequency components of the signal) and directs the reader to `@Necula:2012` for these details.
`@Cornish:2020` includes C code [GitHub](https://github.com/eXtremeGravityInstitute/WDM_Transform) for performing the wavelet transform.

Multiple groups are actively investigating the use of WDM wavelets for GW data analysis.
While this `WDM_GW_Wavelets` package was being developed, the preprints `@Johnson:2026` and `@Vajpeyi:2026` appeared. 
`@Vajpeyi:2026` is accompanied by a public package [GitHub](https://github.com/pywavelet/wdm_transform) that implements the WDM transform (but currently not the filters).

`WDM_GW_Wavelets` provides a well-documented and stable reference implementation that combines both the WDM transform and the time-delay filters with JAX compatibility in a single package.
`WDM_GW_Wavelets` also enables the use of Bayesian inference algorithms that require access to derivatives through JAX's automatic differentiation (autodiff) functionality.

# Software design

`WDM_GW_Wavelets`'s design is kept simple with the main functionality contained in a single class and with the methods aligned directly with the mathematical description in the [documentation](https://cjm96.github.io/WDM_GW_wavelets/).
This was done to make the definitions and conventions of the wavelets used as clear as possible and to allow for comparison with other authors and codes.
JAX is used throughout as this allows the code to be run on different backends (e.g. CPU, GPU or TPU) as well as allowing for JIT compilation and automatic differentiation to be used downstream in Bayesian inference.

# Research impact statement

`WDM_GW_Wavelets` is being actively used in the context of the UK LISA Ground Segment for the development of tools to address data analysis challenges for LISA, including the global fit. 
The first publication by the authors to use this package focuses on modelling the LISA response function to GWs (`@Kinowski:2026` and [GitHub](https://github.com/tomaszkinowski/WDM_LISAresponse).
`WDM_GW_Wavelets` has also been used to make comparisons with the results of other authors and codes (`@Vajpeyi:2026` and [GitHub](https://github.com/pywavelet/wdm_transform).

# AI usage disclosure

Parts of the `WDM_GW_Wavelets` codebase, particularly those associated with performance and JIT-compilation with JAX, were created with the assistance of the generative AI tool `Claude`. 
However, no code was generated wholesale by AI, and all code has been verified by human developers. 
Other than spelling and grammar checking, this paper was written without AI.

# Acknowledgements

We gratefully acknowledge many useful discussions on the topic of GW data 
analysis in the time-frequency domain with members of the UK LISA Ground 
Segment team:
Alberto Vecchio, Hannah Middleton, Geraint Pratten, Christian Chapman-Bird,
Ian Harry, Michael Williams, 
Graham Woan, Christopher Berry, Alexander (Ollie) Burke, and 
Leor Barack.

# References
