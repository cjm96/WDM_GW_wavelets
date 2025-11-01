import numpy as np
import jax
import jax.numpy as jnp
import WDM

import matplotlib.pyplot as plt


def test_gnm_basis_allm():
    r"""
    Test the frequency-domain Gnm functions in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=0.5, 
                                                                Nf=8, 
                                                                N=64,
                                                                q=4,
                                                                calc_m0=True)
    
    gnm_basis = wdm.gnm_basis()

    gnm_basis_allm = wdm.gnm_basis_allm()

    assert gnm_basis.shape == gnm_basis_allm.shape, \
        "gnm_basis and gnm_basis_allm should return arrays with the same shape."

    assert jnp.all(jnp.equal(gnm_basis[:,:,1:], gnm_basis_allm[:,:,1:])), \
        "gnm_basis and gnm_basis_allm should match for all m>0."
    
    gnm_basis_comp_allm = wdm.gnm_basis_comp_allm()
    
    for n in range(wdm.Nt):
        for m in range(wdm.Nf):
            phase_term = jnp.pi*m*(wdm.times-n*wdm.dT)/wdm.dT
            k = jnp.arange(wdm.N)
            if (n+m)%2 == 0:
                g = jnp.sqrt(2.) * jnp.sin(phase_term) * wdm.window_TD[(k-n*wdm.Nf)%wdm.N]
            else:
                g = jnp.sqrt(2.) * (-1)**(n*m) * jnp.cos(phase_term) * wdm.window_TD[(k-n*wdm.Nf)%wdm.N]

            assert jnp.allclose(g, gnm_basis_comp_allm[:,n,m]), \
                f"gnm_basis_comp_allm does not match expected time-domain expression for {n=}, {m=}."