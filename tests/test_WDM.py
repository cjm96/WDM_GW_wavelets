import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import WDM


def test_padding():
    """
    Test the padding function in the WDM class.
    """
    wdm = WDM.code.discrete_wavelet_transform.WDM.WDM_transform(dt=1.0, Nf=2**5, N=2**6)

    x = np.array([9.,9.,9.])

    for where in ['end', 'start', 'equal']:

        x_padded, mask = wdm.pad_signal(x, where=where)

        assert len(x_padded)%wdm.Nf == 0, \
            "Length of padded signal must be a multiple of Nf"
        
        assert jnp.all(jnp.array_equal(x, x_padded[mask])), \
            "Padded signal should match original signal when mask is applied"
        

# test Gnm

# test gnm

# test orthonormality

# test forward and inverse transforms

