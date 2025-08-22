<p align="center">
<img src="./logo_images/logo.png" alt="logo" width="350"/>
</p>


# WDM_GW_wavelets

A fast, JAX-based Python implementation of the Wilson-Daubechies-Meyer (WDM) wavelet 
transform for the time-frequency analysis of gravitational wave data.

👉 https://cjm96.github.io/WDM_GW_wavelets/


# Getting Started

To install the package, clone the repository and use pip. 
Run the following in the directory cotaining pyproject.toml.

```bash
pip install -e .
```

In Python you should then be able to `import WDM`. 
Try running the example notebook `getting_started.ipynb`.


# Documentation

👉 https://cjm96.github.io/WDM_GW_wavelets/

Documentation is built using `sphinx`. 
You will need to ensure you have this installed; this can be done by installing with the docs extras.

```bash
pip install .[docs]
```

Build the docs by running the following command.

```bash
cd ./docs
make clean
make html
```

Open the local documentation files using your browser.

``` bash
open ./docs/build/html/index.html
```

If you need to rebuild the `sphinx` docs, then run the following command.

``` bash
cd ./docs
make clean
sphinx-build ./source ./build
make html
```


# Testing

To check that everything is working as expected you can run the unit tests. 

The tests are run with `pytest`. 
You will first need to ensure you have this installed; this can be done by installing with dev extras.

```bash
pip install .[dev]
```

You will then be able to run all the tests. 
(Make sure you are in the directory cotaining the pyproject.toml file.)

```bash
python -m pytest
```


# Authors

Christopher J. Moore


# TO DO

 - diag cov matrix noise testing in wavelet basis
 - noise docs
 - relating Snm to S(f) when stationary - add this to filtering demo
 - long duration signal in ET - time dependent response function. (f_samp~1024, f_high~500)