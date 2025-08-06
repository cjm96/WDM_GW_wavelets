import numpy as np
import matplotlib.pyplot as plt
import WDM


def test_nu_d():
    r"""
    Test utility function in WDM.
    """
    d = 4

    x = np.linspace(0, 1, 1000)
    y = WDM.code.utils.Meyer.nu_d(x, d)

    assert y.shape == x.shape, \
        "nu_d should return an array of the same shape as input x"


def test_Meyer():
    r"""
    Test utility function in WDM.
    """
    d = 4
    
    omega = np.linspace(-1, 1, 1000)
    Phi = WDM.code.utils.Meyer.Meyer(omega, d, A=0.25, B=0.5)

    assert omega.shape == Phi.shape, \
        "Meyer function should return an array of the same shape as input omega"
