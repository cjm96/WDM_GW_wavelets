import jax
import jax.numpy as jnp
from jax.scipy.special import betainc


def nu_d(x: jnp.ndarray, d: int) -> jnp.ndarray:
    r"""
    This function evaluates the regularised (or normalised) incomplete beta 
    function.

    .. math::

        \nu_d(x) = \frac{ \int_0^x \mathrm{d}t \, t^{d-1} (1 - t)^{d-1} }
                          { \int_0^1 \mathrm{d}t \, t^{d-1} (1 - t)^{d-1} }

    Both the input `x` and output :math:`\nu_d(x)` are in the range `[0, 1]`.

    Parameters
    ----------
    x : jnp.ndarray
        Input array of values in the interval [0, 1].
    d : int
        Steepness parameter controls smoothness of the transition region.

    Returns
    -------
    nu : jnp.ndarray
        The normalized beta function evaluated at :math:`x`.

    Notes
    -----
    This function calls `jax.scipy.special.betainc` under the hood.

    This function returns nan if :math:`x` is outside the range [0, 1].
    """
    d = int(d)
    x = jnp.asarray(x, dtype=jnp.float32)
    nu = betainc(d, d, x) 
    return nu


def Meyer(omega: jnp.ndarray, 
          d: int,
          A: float, 
          B: float) -> jnp.ndarray:
    r"""
    This function evaluates the Meyer-type frequency-domain window function:

    .. math::

        \Phi(\omega) = 
            \begin{cases}
            \frac{1}{\sqrt{\Delta \Omega}}, & |\omega| < A \\
            \frac{1}{\sqrt{\Delta \Omega}} \cos\left( \frac{\pi}{2} \, 
                \nu_d\left( \frac{|\omega| - A}{B} \right) \right), 
                & A \leq |\omega| \leq A + B \\
            0, & \text{otherwise}
            \end{cases}

    where :math:`\\nu_d(x)` is a smooth transition function defined using the
    normalized incomplete beta function, and :math:`\\Delta \\Omega = 2A + B` is 
    the total frequency support of the window.

    Parameters
    ----------
    omega : jnp.ndarray
        Input angular frequency values (radians/second).
    d : int
        Steepness parameter controls smoothness of the transition region.
    A : float
        Half-width of the flat-top region of the window (radians/second).
    B : float
        Width of the transition (roll-off) region (radians/second).

    Returns
    -------
    phi_w : jnp.ndarray
        The window function :math:`\Phi(\omega)`.

    Notes
    -----
    The window is flat in the region :math:`|\omega| < A`, rolls off smoothly 
    for :math:`A \leq |\omega| \leq A + B`, and is zero outside that range.
    The function is symmetric and real-valued.
    """
    dOmega = 2*A + B

    absw = jnp.abs(jnp.asarray(omega, dtype=jnp.float32))

    term1 = (absw < A)
    term2 = (absw >= A) & (absw <= A + B)

    phi_w = jnp.where(
        term1,
        1.0 / jnp.sqrt(dOmega),
        jnp.where(
            term2,
            (1.0 / jnp.sqrt(dOmega)) *
            jnp.cos(
                (jnp.pi / 2.0) * nu_d((absw - A) / B, d)
            ),
            0.0
        )
    )

    return phi_w
