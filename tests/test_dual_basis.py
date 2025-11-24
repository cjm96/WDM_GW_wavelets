import numpy as np
import jax
import jax.numpy as jnp
import WDM

import matplotlib.pyplot as plt


def test_gnm_dual_basis():
    r"""
    Test the frequency-domain Gnm functions in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=8, 
                                                                N=64,
                                                                q=4,
                                                                calc_m0=True)
    
    n, m = 3, 5
    ghat = wdm.gnm_dual(n, m)

    # test output array shape
    assert ghat.shape == (wdm.N,), "gnm_dual output has incorrect shape."

    # test orthogonality
    for n in range(3):
        for m in range(1, 3): # only check for m>0
            ghat_nm = wdm.gnm_dual(n, m)
            for n_ in range(3):
                for m_ in range(1, 3): # only check for m_>0
                    ghat_nm_ = wdm.gnm_dual(n_, m_)
                    inner_prod = wdm.dt * jnp.sum(ghat_nm*ghat_nm_)

                    expected = 1.0 if (n==n_ and m==m_) else 0.0

                    assert jnp.isclose(inner_prod, expected, 
                                       rtol=1.0e-3, atol=1.0e-3), \
                                        f"Wrong"
