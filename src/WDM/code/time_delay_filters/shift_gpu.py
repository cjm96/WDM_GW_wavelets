r"""
GPU backend for the variable time shift.

`WDM_transform.apply_variable_time_shift` dispatches here on CUDA devices. The
sum it evaluates is the same one, written out in `time_delay_matrix_X`,

.. math::

    \tilde{\omega}_{nm} = \sum_{l,\sigma} \omega_{(n-l)(m+\sigma)}
        X_{n(n-l);m(m+\sigma)}(-\delta_n) ,

but regrouped so that a GPU can run it as one pass over the coefficients.

The regrouping. In real arithmetic every term of the sum is

.. math::

    \omega_{(n-l)(m+\sigma)} \, F^{\sigma}_l(n)
        \left[ \alpha_{l\sigma}(n) \cos\theta_{nm}
             + \beta_{l\sigma}(n) \sin\theta_{nm} \right] s_\sigma(m) ,

with :math:`\theta_{nm}=2\pi m\Delta F\delta_n`, :math:`F^{0}=T_l` and
:math:`F^{\pm1}=T'_l` the delay filters, and :math:`s_\sigma(m)=(-1)^m` for
:math:`\sigma=\pm1` and :math:`1` for :math:`\sigma=0`. Three facts make it
factorise:

* :math:`\cos\theta` and :math:`\sin\theta` do not depend on the lag;
* :math:`\alpha` and :math:`\beta` are rank-1 in :math:`n` and depend on the
  lag only through :math:`l \bmod 4` (the lag phase :math:`(\pm i)^l` and the
  parity of :math:`l+\sigma`);
* :math:`s_\sigma` depends on :math:`\sigma` alone.

So the only sums over lags that are needed are the eight banded sums

.. math::

    U^{F}_r(n, m) = \sum_{l \equiv r \ (\mathrm{mod}\ 4)} F_l(n)\,
        \omega_{(n-l)m}, \qquad F \in \{T', T\},\ r=0\ldots3 ,

which is one load of the coefficient grid and two multiply-adds per lag. That
is the Pallas kernel. Everything after it - the rank-1 weights, the
:math:`m+\sigma` column offset and the carrier - is elementwise, and is left to
XLA to fuse.

On a CPU this regrouping does not pay: XLA:CPU already fuses the direct sum
into one pass, and the extra passes of the recombination cost more than the
saved arithmetic. On a GPU the direct sum is not fused - it reloads the grid
for each of the :math:`3(2L+1)` terms - and the regrouped form is about eight
times faster than the direct form written as a Pallas kernel.

Precision. The kernel and the recombination run in float32. The delay, the
phases and the filter interpolation stay in float64 and are only cast at the
end, which keeps the result within a relative L2 of :math:`2\times10^{-7}` of
the float64 CPU path (mismatch :math:`10^{-14}`). The carrier phase reaches
:math:`2\pi N_f \Delta F |\delta|`, around 160 rad for a year-long LISA
record, so it is reduced modulo one cycle in float64 before any float32 trig.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


#: Output tile of one kernel program: rows (time) by columns (frequency).
#: Measured on an RTX A400 at Nt=4096, Nf=1024, L=25. Taller tiles spill
#: the eight accumulators out of registers and get much slower (32x128 is 3.5x,
#: 64x64 12x).
BLOCK_ROWS = 16
BLOCK_COLS = 128
NUM_WARPS = 4

#: Width of the low part when the carrier is built by angle addition.
CARRIER_SPLIT = 32


def _round_up(x, k):
    return -(-x // k) * k


def _block_cols(Nf):
    r"""
    Column tile width: `BLOCK_COLS`, or the smallest power of two covering a
    narrower grid. Triton needs power-of-two block shapes.
    """
    width = 16
    while width < min(Nf, BLOCK_COLS):
        width *= 2
    return width


def lag_sums(T, wdm_coeff, max_lag_L, interpret=False):
    r"""
    The eight banded sums :math:`U^F_r`, as one Pallas kernel.

    Each program owns a `BLOCK_ROWS` by `BLOCK_COLS` tile and keeps all eight
    accumulators in registers. Lag :math:`l` reads the tile's rows shifted by
    :math:`l`; neighbouring lags overlap, so these reads mostly hit cache.

    Parameters
    ----------
    T : jnp.array
        Delay filters :math:`T'_l` and :math:`T_l` evaluated at
        :math:`-\delta_n`. Array, shape=(2, 2L+1, Nt).
    wdm_coeff : jnp.array
        WDM coefficient grid. Array, shape=(Nt, Nf).
    max_lag_L : int
        :math:`L`.
    interpret : bool
        Run the kernel in the Pallas interpreter, on any device. For testing.

    Returns
    -------
    U : jnp.array
        :math:`U^{T'}_0\ldots U^{T'}_3, U^{T}_0\ldots U^{T}_3`, in the dtype of
        `wdm_coeff`. Array, shape=(8, Nt, Nf).
    """
    L = max_lag_L
    Nt, Nf = wdm_coeff.shape
    dtype = wdm_coeff.dtype
    BN, BM = BLOCK_ROWS, _block_cols(Nf)
    Nt_pad, Nf_pad = _round_up(Nt, BN), _round_up(Nf, BM)

    # Cyclic extension by L rows at each end, so the wrapped read is a
    # contiguous window, then zeros to fill out the last tiles. The zero rows
    # and columns only ever feed outputs that are sliced away below.
    ext = jnp.concatenate([wdm_coeff[-L:], wdm_coeff, wdm_coeff[:L]], axis=0)
    ext = jnp.pad(ext, ((0, Nt_pad - Nt), (0, Nf_pad - Nf)))
    T = jnp.pad(T.astype(dtype), ((0, 0), (0, 0), (0, Nt_pad - Nt)))

    def kernel(ext_ref, t_ref, out_ref):
        i0 = pl.program_id(0)*BN
        j0 = pl.program_id(1)*BM

        acc = [jnp.zeros((BN, BM), dtype) for _ in range(8)]
        for i in range(2*L + 1):
            l = i - L
            rows = ext_ref[pl.ds(i0 + L - l, BN), pl.ds(j0, BM)]
            r = l % 4
            acc[r] += t_ref[0, i, pl.ds(i0, BN)][:, None]*rows
            acc[4 + r] += t_ref[1, i, pl.ds(i0, BN)][:, None]*rows

        for k in range(8):
            out_ref[k, pl.ds(i0, BN), pl.ds(j0, BM)] = acc[k]

    U = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, Nt_pad, Nf_pad), dtype),
        grid=(Nt_pad//BN, Nf_pad//BM),
        compiler_params=plgpu.CompilerParams(num_warps=NUM_WARPS,
                                             num_stages=1),
        interpret=interpret,
    )(ext, T)

    return U[:, :Nt, :Nf]


def group_weights(r, sigma, sign_n, cos_h, sin_h):
    r"""
    :math:`\alpha_{r\sigma}(n)` and :math:`\beta_{r\sigma}(n)`, the weights of
    :math:`\cos\theta` and :math:`\sin\theta` shared by every lag
    :math:`l \equiv r \pmod 4`.

    This is the per-element algebra of `apply_variable_time_shift` with the
    delay filter taken out: the lag phase :math:`(\pm i)^l` and the parity of
    :math:`l+\sigma` both depend on :math:`l` only modulo four. The sidebands
    :math:`\sigma=\pm1` add the half-bin phase
    :math:`\phi_h = \pi\Delta F\delta_n` by angle addition, so all of it stays
    rank-1 in :math:`n`.

    Parameters
    ----------
    r : int
        :math:`l \bmod 4`.
    sigma : int
        Frequency lag, :math:`0` or :math:`\pm1`.
    sign_n : jnp.array
        :math:`(-1)^n`. Array, shape=(Nt,).
    cos_h, sin_h : jnp.array
        :math:`\cos\phi_h` and :math:`\sin\phi_h`. Array, shape=(Nt,).

    Returns
    -------
    alpha, beta : jnp.array
        Array, shape=(Nt,).
    """
    i_pow = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))  # (1j)**r
    if sigma == 0:
        lag_re, lag_im = 1.0, 0.0
    else:
        lag_re = i_pow[r][0]
        lag_im = i_pow[r][1] if sigma == -1 else -i_pow[r][1]

    # cos and sin of sigma*phi_h; cos(0) is 1, not cos_h
    cos_s = 1.0 if sigma == 0 else cos_h
    sin_s = 0.0 if sigma == 0 else sigma*sin_h

    if (r + sigma) % 2 == 0:
        alpha = lag_re*cos_s + lag_im*sin_s
        beta = lag_im*cos_s - lag_re*sin_s
    else:
        alpha = sign_n*(lag_re*sin_s - lag_im*cos_s)
        beta = sign_n*(lag_re*cos_s + lag_im*sin_s)

    shape = sign_n.shape
    return jnp.broadcast_to(alpha, shape), jnp.broadcast_to(beta, shape)


def carrier(dF, delta, Nf, dtype):
    r"""
    :math:`\cos\theta_{nm}` and :math:`\sin\theta_{nm}`, with
    :math:`\theta_{nm}=2\pi m\Delta F\delta_n`.

    In float64 this is direct. In float32 the phase is too large to hand to
    trig - :math:`\theta` reaches a few hundred radians, where float32 spacing
    is :math:`10^{-5}` - so :math:`m` is split as :math:`m_{hi}+m_{lo}`, each
    part's phase is reduced to one cycle in float64, and the grid is assembled
    by angle addition from two narrow tables. That is also an order of
    magnitude cheaper than a full grid of trig calls on a GPU.

    Parameters
    ----------
    dF : float
        Frequency resolution :math:`\Delta F`.
    delta : jnp.array
        Delay, float64. Array, shape=(Nt,).
    Nf : int
        Number of frequency bands.
    dtype : jnp.dtype
        Output dtype.

    Returns
    -------
    cos_t, sin_t : jnp.array
        Array, shape=(Nt, Nf).
    """
    m = jnp.arange(Nf)
    if dtype == jnp.float64:
        theta = 2*jnp.pi*dF*jnp.outer(delta, m)
        return jnp.cos(theta), jnp.sin(theta)

    k = CARRIER_SPLIT
    cycles = (dF*delta) % 1.0
    lo = jnp.arange(k)
    hi = jnp.arange(0, _round_up(Nf, k), k)
    a = (2*jnp.pi*((cycles[:, None]*lo) % 1.0)).astype(dtype)
    b = (2*jnp.pi*((cycles[:, None]*hi) % 1.0)).astype(dtype)
    cos_a, sin_a, cos_b, sin_b = jnp.cos(a), jnp.sin(a), jnp.cos(b), jnp.sin(b)

    Nt = delta.shape[0]
    cos_t = (cos_b[:, :, None]*cos_a[:, None, :]
             - sin_b[:, :, None]*sin_a[:, None, :]).reshape(Nt, -1)
    sin_t = (sin_b[:, :, None]*cos_a[:, None, :]
             + cos_b[:, :, None]*sin_a[:, None, :]).reshape(Nt, -1)

    return cos_t[:, :Nf], sin_t[:, :Nf]


def variable_time_shift_gpu(Tl, Tprimel, wdm_coeff, delta, dF,
                            dtype=jnp.float32, interpret=False):
    r"""
    The variable time shift, regrouped for a GPU. Same contract as
    `WDM_transform.apply_variable_time_shift`, including the periodic wrap at
    the edges of the grid.

    Parameters
    ----------
    Tl, Tprimel : jnp.array
        Delay filters :math:`T_l` and :math:`T'_l` evaluated at
        :math:`-\delta_n`. Array, shape=(2L+1, Nt).
    wdm_coeff : jnp.array
        WDM coefficient grid. Array, shape=(Nt, Nf).
    delta : jnp.array
        Delay per row, float64. Array, shape=(Nt,).
    dF : float
        Frequency resolution :math:`\Delta F`.
    dtype : jnp.dtype
        Working precision of the kernel and the recombination. float64 is for
        testing the algebra; it is slow on most GPUs.
    interpret : bool
        Run the kernel in the Pallas interpreter. For testing.

    Returns
    -------
    shifted_wdm_coeff : jnp.array
        In the dtype of `wdm_coeff`. Array, shape=(Nt, Nf).
    """
    max_lag_L = (Tl.shape[0] - 1)//2
    Nt, Nf = wdm_coeff.shape

    U = lag_sums(jnp.stack([Tprimel, Tl]), wdm_coeff.astype(dtype), max_lag_L,
                 interpret=interpret)

    n = jnp.arange(Nt)
    sign_n = jnp.where(n % 2 == 0, 1.0, -1.0)
    sign_m = jnp.where(jnp.arange(Nf) % 2 == 0, 1.0, -1.0).astype(dtype)
    half = jnp.pi*dF*delta
    cos_h, sin_h = jnp.cos(half), jnp.sin(half)

    # weight of cos(theta) and of sin(theta), summed over sigma
    cos_part = 0.0
    sin_part = 0.0
    for sigma in (-1, 0, 1):
        first = 4 if sigma == 0 else 0   # T for sigma=0, T' for the sidebands
        A = 0.0
        B = 0.0
        for r in range(4):
            alpha, beta = group_weights(r, sigma, sign_n, cos_h, sin_h)
            A = A + alpha.astype(dtype)[:, None]*U[first + r]
            B = B + beta.astype(dtype)[:, None]*U[first + r]
        if sigma != 0:
            # omega_{(n-l)(m+sigma)}: column m+sigma, wrapped, then (-1)^m
            A = jnp.roll(A, -sigma, axis=1)*sign_m
            B = jnp.roll(B, -sigma, axis=1)*sign_m
        cos_part = cos_part + A
        sin_part = sin_part + B

    cos_t, sin_t = carrier(dF, delta, Nf, dtype)

    return (cos_t*cos_part + sin_t*sin_part).astype(wdm_coeff.dtype)
