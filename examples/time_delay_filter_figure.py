import jax
import jax.numpy as jnp

import matplotlib.pylab as plt

import WDM
from WDM.code.discrete_wavelet_transform import WDM
from WDM.code.time_delay_filters.filters import time_delay_filter_Tl_reference
from WDM.code.time_delay_filters.filters import time_delay_filter_Tprimel_reference

dt = 1.0

Nt = 32
Nf = 16
N = Nt*Nf

wdm = WDM.WDM_transform(dt=dt, Nf=Nf, N=N, q=8, calc_m0=True)

delta_t_vals = jnp.linspace(-12*wdm.dT, 12*wdm.dT, 500)

Tl_vals = {}
Tprimel_vals = {}

ell_max = 2

for ell in range(-ell_max, ell_max+1):
    Tl_vals[ell] = jnp.array([time_delay_filter_Tl_reference(ell, 
                                                   delta_t,
                                                   wdm.freqs,
                                                   wdm.window_FD,
                                                   wdm.dT,
                                                   wdm.df) for delta_t in delta_t_vals])
    Tprimel_vals[ell] = jnp.array([time_delay_filter_Tprimel_reference(ell,
                                                             delta_t,
                                                             wdm.freqs,
                                                             wdm.window_FD,
                                                             wdm.dT,
                                                             wdm.dF,
                                                             wdm.N,
                                                             wdm.df) for delta_t in delta_t_vals])

fig, axes = plt.subplots(nrows=2, figsize=(5, 5), sharex=True)

for ell in range(-ell_max, ell_max+1):
    label = r'$\ell={}$'.format(ell) if ell==-2 else r'${}$'.format(ell)
    axes[0].plot(delta_t_vals/wdm.dT, Tl_vals[ell], label=label)
    axes[1].plot(delta_t_vals/wdm.dT, Tprimel_vals[ell],)

axes[0].fill_between([-0.5, +0.5], 
             [-10, -10], [10, 10],
             alpha=0.3, color='gray')
axes[1].fill_between([-0.5, +0.5], 
             [-10, -10], [10, 10],
             alpha=0.3, color='gray')

axes[1].set_xlim(delta_t_vals[0]/wdm.dT, delta_t_vals[-1]/wdm.dT)
axes[1].set_xlabel(r'$\delta/\Delta T$')

axes[0].set_ylim(-0.3, 1.4)
axes[0].set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1])
axes[0].set_ylabel(r'$T_\ell(\delta)$')

axes[1].set_ylim(0, 0.095)
axes[1].set_yticks([0.02, 0.04, 0.06, 0.08])
axes[1].set_ylabel(r'$T^\prime_\ell(\delta)$')

axes[0].legend(ncols=5, loc='upper center', 
               handlelength=1, columnspacing=1,
               frameon=False, framealpha=0)

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig('TimeDelayFilters.pdf')
plt.clf()