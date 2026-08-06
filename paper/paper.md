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
    affiliation: "1"
affiliations:
 - name: Institute of Astronomy, University of Cambridge, Madingley Road, Cambridge, CB3 0HA, UK
   index: 1
 - name: Kavli Institute for Cosmology, University of Cambridge, Madingley Road, Cambridge, CB3 0HA, UK
   index: 2
 - name: Department of Applied Mathematics and Theoretical Physics, Centre for Mathematical Sciences, University of Cambridge, Wilberforce Road, Cambridge, CB3 0WA, UK
   index: 3
date: 6 August 2026
bibliography: paper.bib
---

# Summary

The transient gravitational wave (GW) signals observed by the current 
generation of ground-based interferometers are short (mostly between one and a 
few tens of seconds). 
These signals are usually most convenietly analysed in the frequency doamin 
where the instrumental noise (which is assumed to be stationary) is easily 
modelled.
The next generation of ground-based interferometers, as well as the upcoming 
LISA space-based interferometer, will observe much longer signals (several 
hours for ground-based instruments and years for space-based ones).
The instrumental noise will not remain stationary over these time scales.
Additionally, the detectors will move and rotate appreciably over these time 
scales complicating the modelling of the response of the instrument.

For these reasons, there is a need to devlop time-frequency methods for GW data 
analysis.
The Wilson-Daubechies-Meyer (WDM) wavelet basis [@Necula:2012], previosuly used 
in a GW context by coherent wave burst (CWB) (see @Klimenko:2005 and 
Klimenko:2016), has been identified as a promising set of basis functions for 
this purpose @Cornish:2020. 
The WDM wavelets expand time series signals ussing a discrete, orthogonal 
basis of wavelet wavepackets; these map the time series onto a uniform grid of 
pixels that cover the time-frequency plane (see, for example, figure 
\autoref{fig:time_freq_plane}).

![Example WDM wavelets plotted in both time (top) and frequency (right) domain. The main plot shows a time-frequency grid shaded to indicate where the wavelets have support. Figure reproduced from the `WDM_GW_Wavelets` package documentation.\label{fig:time_freq_plane}](wavelets_TF.png)

# Statement of need

There are many different methods for doing time series data analysis in the 
time-frequency domain.
Similarly, there are also many different families of wavelets.
This can easily lead to confusion when comparing results between authors.
There is a need in GW data analysis community for a single, stable convention
with a publically available implementation that is well documented and has the 
mathematical details and conventions clearly described alongside the code.
Additionally, to allow Bayesian data analysis to be performed in the 
time-frequency domains, there is a need for wavelet transform to be fast and to 
be executable on multiple backends, including CPUs, GPUs, and TPUs.

`WDM_GW_Wavelets` is designed to meet these needs.
`WDM_GW_Wavelets` is a JAX-compatible Python package that implements the
WDM wavelet transform and is accompanied by extensive mathematical 
documentation which describes the conventions used.

# State of the field

The WDM wavelets have been used previously in the context of GW data analysis 
(see, for example, @Necula:2012, @Klimenko:2005 and Klimenko:2016).
However, @Cornish:2020, was the first to clearly advocare for this wavelet 
family

# Software design

`WDM_GW_Wavelets`'s design philosophy is based on the principle...

An important 

# Research impact statement

`WDM_GW_Wavelets` is a new package. 
It has demonstrated research impact with.

# AI usage disclosure

Parts of the code in `WDM_GW_Wavelets`, particularly those specific to 
performance and jit-compilation with `JAX`, were created with the assistance of 
the generative AI tool `claude`. 
However, no code was generated wholesale by any AI tools, and all the code has 
been verified by human developers. 
Other than spelling and grammar checking, this paper was written without AI 
assistance.

# Acknowledgements

We gratefully acknowledge many useful discussions on the topic of GW data 
analysis in the time-frequency domain with members of the UK LISA Ground 
Segment team:
Alberto Vecchio, Hannah Middleton, Geraint Pratten, Christian Chapman-Bird,
Ian Harry, Michael Williams, 
Graham Woan, Christopher Berry, Alexander (Ollie) Burke, and 
Leor Barack.

# References