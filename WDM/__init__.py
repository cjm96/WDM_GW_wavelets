import jax
jax.config.update("jax_enable_x64", True)

from .code.utils import Meyer

from .code.plotting import plotting

from .code.discrete_wavelet_transform import WDM
