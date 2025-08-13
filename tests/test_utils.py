import numpy as np
import jax
import WDM


def test_x64():
    r"""
    Test that the WDM module is using float64 precision.
    """
    assert jax.config.read("jax_enable_x64"), \
        "WDM module should be using float64 precision, check the __init__ file."
    

def test_nu_d():
    r"""
    Test the normalised incomplete beta function.
    """
    d = 4

    x = np.linspace(0, 1, 1000)
    y = WDM.code.utils.Meyer.nu_d(x, d)

    assert y.shape == x.shape, \
        "nu_d should return an array of the same shape as input x"
    
    assert np.allclose(y[0], 0) and np.allclose(y[-1], 1), \
        "should have nu_d(0)=0 and nu_d(1)=1."
    
    x = np.array([-0.1, 1.1])
    y = WDM.code.utils.Meyer.nu_d(x, d)

    assert np.all(np.isnan(y)), \
        "function should return nan when x is outside [0,1]"
    

def test_Meyer():
    r"""
    Test the Meyer window function in WDM.
    """
    d = 4
    
    omega = np.linspace(-1, 1, 1000)
    Phi = WDM.code.utils.Meyer.Meyer(omega, d, A=0.25, B=0.5)

    assert omega.shape == Phi.shape, \
        "Meyer function should return an array of the same shape as input omega"
