<p align="center">
<img src="./logo_images/logo.png" alt="logo" width="350"/>
</p>

# WDM_GW_wavelets

A fast, JAX-based Python implementation of the Wilson-Daubechies-Meyer (WDM) wavelet transform for the time-frequency analysis of gravitational wave data.

# Getting Started

To install clone the repository and install using pip. Run the following in the directory cotaining the pyproject.toml file.

```bash
pip install -e .
```

# Documentation

Documentation is built using `Sphinx`. You will first need to ensure you have this installed; this can be done by installing with the docs extras.

```bash
pip install .[docs]
```

Then build the docs by running the following make command.

```bash
cd docs
make html
```

The open the documentation in your browser.

``` bash
open .docs/build/html/index.html
```
