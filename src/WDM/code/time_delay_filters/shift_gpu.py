r"""
GPU backend for the variable time shift.

`WDM_transform.apply_variable_time_shift` and
`WDM_transform.apply_variable_time_shift_batch` dispatch here on CUDA devices.
The sum they evaluate is the same one, written out in `time_delay_matrix_X`,

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
:math:`\sigma=\pm1` and :math:`1` for :math:`\sigma=0`. Four facts make it
factorise:

* :math:`\cos\theta` and :math:`\sin\theta` do not depend on the lag;
* :math:`\alpha` and :math:`\beta` are rank-1 in :math:`n` and depend on the
  lag only through :math:`l \bmod 4` (the lag phase :math:`(\pm i)^l` and the
  parity of :math:`l+\sigma`);
* advancing the lag by two flips the sign of the lag phase and nothing else,
  so :math:`\alpha_{l+2,\sigma}=-\alpha_{l\sigma}` for the sidebands, while for
  :math:`\sigma=0` there is no lag phase and :math:`\alpha_{l+2,0}=\alpha_{l0}`
  (likewise :math:`\beta`);
* :math:`s_\sigma` depends on :math:`\sigma` alone.

So, with :math:`\epsilon_l=(-1)^{(l-p)/2}` for :math:`l\equiv p \pmod 2`, the
only sums over lags that are needed are the four banded sums

.. math::

    P_p(n, m) = \sum_{l \equiv p} \epsilon_l T'_l(n)\, \omega_{(n-l)m},
    \qquad
    Q_p(n, m) = \sum_{l \equiv p} T_l(n)\, \omega_{(n-l)m},
    \qquad p = 0, 1 ,

which is one load of the coefficient grid and two multiply-adds per lag. That
is the Pallas kernel. Everything after it - the rank-1 weights, the
:math:`m+\sigma` column offset and the carrier - is elementwise, and is left to
XLA to fuse.

On a CPU this regrouping does not pay: XLA:CPU already fuses the direct sum
into one pass, and the extra passes of the recombination cost more than the
saved arithmetic. On a GPU the direct sum is not fused - it reloads the grid
for each of the :math:`3(2L+1)` terms - and the regrouped form is about ten
times faster than the direct form written as a Pallas kernel.

Batching. A stack of grids goes through as one kernel launch with the grid
index as the outermost axis of the launch grid. That does not make each grid
cheaper - one grid of a year-long record is already a couple of thousand
tiles, enough to fill a GPU - but it removes the per-grid launches and the
per-grid glue around them, which for six links halves the time of the shift.

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
#: Measured on an RTX A400 at L=25. The kernel's cost is dominated by loading
#: the delay filters, one value per row per lag, and each value serves every
#: column of the tile - so short, wide tiles pay: at Nt=4096, Nf=1024, 8x256
#: runs in 10.2 ms for six grids against 14.3 ms at 16x128, and the gain holds
#: from 0.33 to 2.66 yr (1.27x to 1.46x). Wider still loses (4x1024 13.3 ms),
#: and taller tiles spill the accumulators (32x128 30 ms). A grid narrower
#: than `BLOCK_COLS` gets a narrower tile, and rows are added back to keep at
#: least `MIN_TILE` elements: 8 rows at Nf=128 (1.14x over 16x128), 16 at
#: Nf=64, where 8 rows ran 1.5x slower.
BLOCK_ROWS = 8
BLOCK_COLS = 256
MIN_TILE = 1024
NUM_WARPS = 4


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


def _block_rows(block_cols):
    r"""
    Row tile height: `BLOCK_ROWS`, or more for a narrow tile so that it still
    holds `MIN_TILE` elements. A power of two, since `block_cols` is.
    """
    return max(BLOCK_ROWS, MIN_TILE//block_cols)


def lag_sums(T, wdm_coeff, max_lag_L, interpret=False):
    r"""
    The four banded sums :math:`P_0, P_1, Q_0, Q_1` for a stack of grids, as
    one Pallas kernel.

    Each program owns one `BLOCK_ROWS` by `BLOCK_COLS` tile of one grid and
    keeps the four accumulators in registers. Lag :math:`l` reads the tile's
    rows shifted by :math:`l`; neighbouring lags overlap, so these reads mostly
    hit cache.

    Parameters
    ----------
    T : jnp.array
        Delay filters :math:`\epsilon_l T'_l` and :math:`T_l` evaluated at
        :math:`-\delta_n`. Array, shape=(G, 2, 2L+1, Nt).
    wdm_coeff : jnp.array
        WDM coefficient grids. Array, shape=(G, Nt, Nf).
    max_lag_L : int
        :math:`L`.
    interpret : bool
        Run the kernel in the Pallas interpreter, on any device. For testing.

    Returns
    -------
    sums : jnp.array
        :math:`P_0, P_1, Q_0, Q_1` per grid, in the dtype of `wdm_coeff`.
        Array, shape=(G, 4, Nt, Nf).
    """
    L = max_lag_L
    G, Nt, Nf = wdm_coeff.shape
    dtype = wdm_coeff.dtype
    BM = _block_cols(Nf)
    BN = _block_rows(BM)
    Nt_pad, Nf_pad = _round_up(Nt, BN), _round_up(Nf, BM)

    # Cyclic extension by L rows at each end, so the wrapped read is a
    # contiguous window, then zeros to fill out the last tiles. The zero rows
    # and columns only ever feed outputs that are sliced away below.
    ext = jnp.concatenate([wdm_coeff[:, -L:], wdm_coeff, wdm_coeff[:, :L]],
                          axis=1)
    ext = jnp.pad(ext, ((0, 0), (0, Nt_pad - Nt), (0, Nf_pad - Nf)))
    T = jnp.pad(T.astype(dtype), ((0, 0), (0, 0), (0, 0), (0, Nt_pad - Nt)))

    def kernel(ext_ref, t_ref, out_ref):
        g = pl.program_id(0)
        i0 = pl.program_id(1)*BN
        j0 = pl.program_id(2)*BM

        # P_0, P_1, Q_0, Q_1
        acc = [jnp.zeros((BN, BM), dtype) for _ in range(4)]
        for i in range(2*L + 1):
            l = i - L
            rows = ext_ref[g, pl.ds(i0 + L - l, BN), pl.ds(j0, BM)]
            p = l % 2
            acc[p] += t_ref[g, 0, i, pl.ds(i0, BN)][:, None]*rows
            acc[2 + p] += t_ref[g, 1, i, pl.ds(i0, BN)][:, None]*rows

        for k in range(4):
            out_ref[g, k, pl.ds(i0, BN), pl.ds(j0, BM)] = acc[k]

    sums = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((G, 4, Nt_pad, Nf_pad), dtype),
        grid=(G, Nt_pad//BN, Nf_pad//BM),
        compiler_params=plgpu.CompilerParams(num_warps=NUM_WARPS,
                                             num_stages=1),
        interpret=interpret,
    )(ext, T)

    return sums[:, :, :Nt, :Nf]


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
    is :math:`10^{-5}` - so it is reduced in cycles first. Per row, in float64,
    :math:`c_n=\Delta F\delta_n \bmod 1` is split as
    :math:`c_n = c^{hi}_n + c^{lo}_n` with :math:`c^{hi}_n` on a grid of
    :math:`2^{-b}`, :math:`b = 24 - \lceil\log_2 N_f\rceil`, so that
    :math:`c^{hi}_n m` is exact in float32 for every :math:`m < N_f` and its
    fractional part is exact too, while :math:`c^{lo}_n m` stays below
    :math:`2^{-b} N_f` and loses nothing that matters. What is left is
    elementwise, so XLA fuses it into its consumer and no
    :math:`(N_t, N_f)` grid of phases is ever stored.

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

    bits = 24 - (Nf - 1).bit_length()
    cycles = (dF*delta) % 1.0
    c_hi = jnp.round(cycles*2.0**bits)/2.0**bits
    c_lo = (cycles - c_hi).astype(dtype)[:, None]
    c_hi = c_hi.astype(dtype)[:, None]

    m = jnp.arange(Nf, dtype=dtype)
    whole = c_hi*m
    theta = 2*jnp.pi*((whole - jnp.floor(whole)) + c_lo*m)

    return jnp.cos(theta), jnp.sin(theta)


def recombine(sums, delta, dF, dtype):
    r"""
    The shifted grid from its four lag sums: apply the rank-1 weights, move
    the sidebands to column :math:`m+\sigma`, and multiply in the carrier.

    Parameters
    ----------
    sums : jnp.array
        :math:`P_0, P_1, Q_0, Q_1` of one grid. Array, shape=(4, Nt, Nf).
    delta : jnp.array
        Delay, float64. Array, shape=(Nt,).
    dF : float
        Frequency resolution :math:`\Delta F`.
    dtype : jnp.dtype
        Working precision.

    Returns
    -------
    shifted : jnp.array
        Array, shape=(Nt, Nf).
    """
    _, Nt, Nf = sums.shape
    sign_n = jnp.where(jnp.arange(Nt) % 2 == 0, 1.0, -1.0)
    sign_m = jnp.where(jnp.arange(Nf) % 2 == 0, 1.0, -1.0).astype(dtype)
    half = jnp.pi*dF*delta
    cos_h, sin_h = jnp.cos(half), jnp.sin(half)

    # weight of cos(theta) and of sin(theta), summed over sigma
    cos_part = 0.0
    sin_part = 0.0
    for sigma in (-1, 0, 1):
        first = 2 if sigma == 0 else 0   # Q for sigma=0, P for the sidebands
        A = 0.0
        B = 0.0
        for p in range(2):
            alpha, beta = group_weights(p, sigma, sign_n, cos_h, sin_h)
            A = A + alpha.astype(dtype)[:, None]*sums[first + p]
            B = B + beta.astype(dtype)[:, None]*sums[first + p]
        if sigma != 0:
            # omega_{(n-l)(m+sigma)}: column m+sigma, wrapped, then (-1)^m
            A = jnp.roll(A, -sigma, axis=1)*sign_m
            B = jnp.roll(B, -sigma, axis=1)*sign_m
        cos_part = cos_part + A
        sin_part = sin_part + B

    cos_t, sin_t = carrier(dF, delta, Nf, dtype)

    return cos_t*cos_part + sin_t*sin_part


def variable_time_shift_gpu(Tl, Tprimel, wdm_coeff, delta, dF,
                            dtype=jnp.float32, interpret=False):
    r"""
    The variable time shift of a stack of grids, regrouped for a GPU. Same
    contract as `WDM_transform.apply_variable_time_shift_batch`, including the
    periodic wrap at the edges of each grid.

    Parameters
    ----------
    Tl, Tprimel : jnp.array
        Delay filters :math:`T_l` and :math:`T'_l` evaluated at
        :math:`-\delta_n`. Array, shape=(G, 2L+1, Nt).
    wdm_coeff : jnp.array
        WDM coefficient grids. Array, shape=(G, Nt, Nf).
    delta : jnp.array
        Delay per row, float64. Array, shape=(G, Nt).
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
        In the dtype of `wdm_coeff`. Array, shape=(G, Nt, Nf).
    """
    max_lag_L = (Tl.shape[1] - 1)//2

    # epsilon_l = (-1)^((l - l%2)/2): +1 for l = 0, 1 mod 4, -1 for 2, 3
    lags = jnp.arange(-max_lag_L, max_lag_L + 1)
    epsilon = jnp.where(lags % 4 < 2, 1.0, -1.0)[:, None]

    sums = lag_sums(jnp.stack([epsilon*Tprimel, Tl], axis=1),
                    wdm_coeff.astype(dtype), max_lag_L, interpret=interpret)

    shifted = jax.vmap(recombine, in_axes=(0, 0, None, None))(sums, delta, dF,
                                                              dtype)

    return shifted.astype(wdm_coeff.dtype)
