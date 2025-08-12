<p align="center">
<img src="./logo_images/logo.png" alt="logo" width="350"/>
</p>


# WDM_GW_wavelets

A fast, JAX-based Python implementation of the Wilson-Daubechies-Meyer (WDM) wavelet transform for the time-frequency analysis of gravitational wave data.


# Getting Started

To install the package, clone the repository and use pip. Run the following in the directory cotaining pyproject.toml.

```bash
pip install -e .
```

In Python you should then be able to `import WDM`. Try running the example notebook `getting_started.ipynb`.


# Testing

To check that everything is working as expected you can run the unit tests. 
You will first need to ensure you have `pytest` installed; this can be done by installing with the dev extras.

```bash
pip install .[dev]
```

You will then be able to run all the tests. (Make sure you are in the directory cotaining the pyproject.toml file.)

```bash
python -m pytest
```


# Documentation

👉 https://cjm96.github.io/WDM_GW_wavelets/

Documentation is built using `sphinx`. 
You will first need to ensure you have this installed; this can be done by installing with the docs extras.

```bash
pip install .[docs]
```

Then build the docs by running the following make command.

```bash
cd ./docs
make clean
make html
```

Then open the local documentation files using your browser.

``` bash
open ./docs/build/html/index.html
```

If you need to rebuild the `sphinx` docs, then run the following command.

``` bash
cd ./docs
sphinx-build ./source ./build
make clean
make html
```


# To Do List

 - tidy up parts of the WDM_transform class
 - add derivations of key equations to docs
 - write vectorised version of the transform for multiple DWTs at once.
 - GW170817 data processing notebook (inc. a type of periodogram)
 - waveforms?
 - likelihoods?
