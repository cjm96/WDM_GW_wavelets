"""
Affine WDM time-shift filter table.

This module implements the local-affine WDM time-shift kernel

    Delta(t) ~= D_n + epsilon_n (t - t_n),

with sparse coupling

    w_out[n, m] ~= sum_{ell, sigma} K[n, m, ell, sigma]
                                  w_in[n - ell, m + sigma].

The implementation here is accuracy-first and NumPy-based. It matches the
validation notebooks before any FFT/JAX optimisation is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pickle
import numpy as np


ArrayLike = Any
TABLE_FORMAT_VERSION = 1


# =============================================================================
# WDM branch/parity helpers
# =============================================================================


def c_nm_complex(n: int, m: int) -> complex:
    """
    WDM parity coefficient C_nm.

    C_nm = 1 if n + m is even, and i otherwise.
    """
    return 1.0 + 0.0j if ((int(n) + int(m)) % 2 == 0) else 0.0 + 1.0j


def c_branch(n: int, m: int, rho: int) -> complex:
    """
    Branch coefficient for the positive/negative frequency branch.

    Convention used in the validated affine notebooks:

        C^(+)_nm = conj(C_nm)
        C^(-)_nm = C_nm
    """
    C = c_nm_complex(n, m)

    if int(rho) == +1:
        return np.conj(C)

    if int(rho) == -1:
        return C

    raise ValueError(f"rho must be -1 or +1, got {rho}.")


# =============================================================================
# Array validation and interpolation helpers
# =============================================================================


def _as_1d_float_array(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")

    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two points.")

    if not np.all(np.diff(arr) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")

    return arr


def _as_1d_int_array(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")

    if arr.size < 1:
        raise ValueError(f"{name} must contain at least one value.")

    return arr


def _bracket(grid: np.ndarray, x: float, name: str) -> tuple[int, float]:
    """
    Locate x in a strictly increasing grid.

    Returns
    -------
    i, t:
        x lies between grid[i] and grid[i+1], with fractional coordinate t.
    """
    x = float(x)

    if x < grid[0] or x > grid[-1]:
        raise ValueError(
            f"{name}={x} outside grid range [{grid[0]}, {grid[-1]}]."
        )

    i = int(np.searchsorted(grid, x) - 1)
    i = max(0, min(i, len(grid) - 2))

    x0 = grid[i]
    x1 = grid[i + 1]
    t = (x - x0) / (x1 - x0)

    return i, float(t)


def trilinear_interp_complex(
    table: np.ndarray,
    D_grid: np.ndarray,
    eps_grid: np.ndarray,
    eta_grid: np.ndarray,
    D: float,
    epsilon: float,
    eta: float,
) -> complex:
    """
    Trilinear interpolation of a complex table with axes:

        table[D_index, epsilon_index, eta_index].
    """
    iD, tD = _bracket(D_grid, D, "D")
    iE, tE = _bracket(eps_grid, epsilon, "epsilon")
    iH, tH = _bracket(eta_grid, eta, "eta")

    out = 0.0 + 0.0j

    for aD in (0, 1):
        wD = (1.0 - tD) if aD == 0 else tD

        for aE in (0, 1):
            wE = (1.0 - tE) if aE == 0 else tE

            for aH in (0, 1):
                wH = (1.0 - tH) if aH == 0 else tH
                out += wD * wE * wH * table[iD + aD, iE + aE, iH + aH]

    return complex(out)

def _bracket_vectorized(
    grid: np.ndarray,
    x: ArrayLike,
    name: str,
    *,
    check_domain: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised version of _bracket.

    Returns
    -------
    i, t:
        Arrays such that x lies between grid[i] and grid[i+1], with
        fractional coordinate t.

    Notes
    -----
    If check_domain=False, values are clipped to the interpolation domain.
    This is useful for fast production calls after a separate domain check.
    """
    grid = np.asarray(grid, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if check_domain:
        if np.any(x < grid[0]) or np.any(x > grid[-1]):
            xmin = float(np.min(x))
            xmax = float(np.max(x))
            raise ValueError(
                f"{name} outside grid range [{grid[0]}, {grid[-1]}]. "
                f"Observed range [{xmin}, {xmax}]."
            )

    i = np.searchsorted(grid, x, side="right") - 1
    i = np.clip(i, 0, len(grid) - 2)

    x0 = grid[i]
    x1 = grid[i + 1]
    t = (x - x0) / (x1 - x0)

    return i.astype(np.int64), np.asarray(t, dtype=np.float64)


def trilinear_interp_complex_vectorized(
    table: np.ndarray,
    D_grid: np.ndarray,
    eps_grid: np.ndarray,
    eta_grid: np.ndarray,
    D: ArrayLike,
    epsilon: ArrayLike,
    eta: ArrayLike,
    *,
    check_domain: bool = False,
) -> np.ndarray:
    """
    Vectorised trilinear interpolation of a complex table with axes:

        table[D_index, epsilon_index, eta_index].

    The inputs D, epsilon, and eta may be arrays with broadcast-compatible
    shapes. For the affine sparse operator the typical shapes are

        D       : (n_chunk, 1)
        epsilon : (n_chunk, 1)
        eta     : (n_chunk, m_count)

    and the returned array has shape (n_chunk, m_count).
    """
    D = np.asarray(D, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)

    iD, tD = _bracket_vectorized(
        D_grid,
        D,
        "D",
        check_domain=check_domain,
    )
    iE, tE = _bracket_vectorized(
        eps_grid,
        epsilon,
        "epsilon",
        check_domain=check_domain,
    )
    iH, tH = _bracket_vectorized(
        eta_grid,
        eta,
        "eta",
        check_domain=check_domain,
    )

    Q000 = table[iD,     iE,     iH    ]
    Q100 = table[iD + 1, iE,     iH    ]
    Q010 = table[iD,     iE + 1, iH    ]
    Q110 = table[iD + 1, iE + 1, iH    ]

    Q001 = table[iD,     iE,     iH + 1]
    Q101 = table[iD + 1, iE,     iH + 1]
    Q011 = table[iD,     iE + 1, iH + 1]
    Q111 = table[iD + 1, iE + 1, iH + 1]

    oneD = 1.0 - tD
    oneE = 1.0 - tE
    oneH = 1.0 - tH

    return (
        Q000 * oneD * oneE * oneH
        + Q100 * tD * oneE * oneH
        + Q010 * oneD * tE * oneH
        + Q110 * tD * tE * oneH
        + Q001 * oneD * oneE * tH
        + Q101 * tD * oneE * tH
        + Q011 * oneD * tE * tH
        + Q111 * tD * tE * tH
    )


# =============================================================================
# Cached WDM window interpolation data
# =============================================================================


@dataclass(frozen=True)
class WindowInterpData:
    """
    Cached sorted WDM frequency grid and frequency-domain window.

    This avoids repeatedly sorting wdm.freqs and repeatedly extracting the
    window support during table construction.
    """

    freqs: np.ndarray
    window: np.ndarray
    support: float


def window_interp_data_from_wdm(wdm: Any) -> WindowInterpData:
    """
    Construct cached interpolation data for Phi~(f) from a WDM object.
    """
    freqs = np.asarray(wdm.freqs, dtype=np.float64)
    window = np.asarray(wdm.window_FD, dtype=np.complex128)

    order = np.argsort(freqs)
    freqs = freqs[order]
    window = window[order]

    mask = np.abs(window) > 1.0e-14

    if not np.any(mask):
        raise ValueError("wdm.window_FD appears to have empty support.")

    support = float(np.max(np.abs(freqs[mask])))

    return WindowInterpData(
        freqs=freqs,
        window=window,
        support=support,
    )


def phi_tilde_from_window_data(
    window_data: WindowInterpData,
    f_hz: ArrayLike,
) -> np.ndarray:
    """
    Evaluate Phi~(f) using cached interpolation data.

    Values outside the stored frequency grid are set to zero.
    """
    f_hz = np.asarray(f_hz, dtype=np.float64)

    real = np.interp(
        f_hz,
        window_data.freqs,
        window_data.window.real,
        left=0.0,
        right=0.0,
    )
    imag = np.interp(
        f_hz,
        window_data.freqs,
        window_data.window.imag,
        left=0.0,
        right=0.0,
    )

    return real + 1.0j * imag


def phi_tilde_from_wdm(wdm: Any, f_hz: ArrayLike) -> np.ndarray:
    """
    Evaluate the WDM frequency-domain Meyer window Phi~(f) by interpolation
    from wdm.freqs and wdm.window_FD.

    This convenience function builds WindowInterpData internally. For table
    construction, prefer phi_tilde_from_window_data with cached window data.
    """
    window_data = window_interp_data_from_wdm(wdm)
    return phi_tilde_from_window_data(window_data, f_hz)


# =============================================================================
# Direct affine Q construction
# =============================================================================


def affine_Q_integration_bounds(
    wdm: Any,
    *,
    a: float,
    delta: float,
    rho: int,
    margin_fraction: float = 0.05,
    window_data: WindowInterpData | None = None,
) -> tuple[float, float, bool]:
    """
    Conservative integration bounds for the bare affine Q integral.

    The WDM window support is inferred from the stored window_FD array unless
    precomputed WindowInterpData is supplied.
    """
    rho = int(rho)
    a = float(a)
    delta = float(delta)

    if window_data is None:
        window_data = window_interp_data_from_wdm(wdm)

    support = float(window_data.support)
    dF = float(wdm.dF)

    # Source argument:
    #   xi/a - rho delta dF/(2a)
    # so |source_arg| <= support implies
    #   xi in [rho delta dF/2 - a support,
    #          rho delta dF/2 + a support].
    source_centre = rho * delta * dF / 2.0
    source_min = source_centre - a * support
    source_max = source_centre + a * support

    # Target argument:
    #   xi + rho delta dF/2
    # so |target_arg| <= support implies
    #   xi in [-rho delta dF/2 - support,
    #          -rho delta dF/2 + support].
    target_centre = -rho * delta * dF / 2.0
    target_min = target_centre - support
    target_max = target_centre + support

    xi_min = max(source_min, target_min)
    xi_max = min(source_max, target_max)

    if xi_min >= xi_max:
        return 0.0, 0.0, True

    margin = float(margin_fraction) * dF

    return xi_min - margin, xi_max + margin, False


def direct_affine_Q_many_lambda(
    wdm: Any,
    *,
    D_values: ArrayLike,
    ell_values: ArrayLike,
    epsilon: float,
    sigma: int,
    eta: float,
    rho: int,
    n_quad: int = 2048,
    lambda_chunk_size: int = 512,
    window_data: WindowInterpData | None = None,
) -> np.ndarray:
    """
    Compute Q for all combinations of ell and D at fixed

        epsilon, sigma, eta, rho.

    Returns
    -------
    Q_ell_D:
        ndarray with shape (len(ell_values), len(D_values)).
    """
    D_values = np.asarray(D_values, dtype=np.float64)
    ell_values = np.asarray(ell_values, dtype=np.int64)

    a = 1.0 + float(epsilon)
    delta = float(sigma) + float(eta)
    rho = int(rho)

    if a <= 0.0:
        raise ValueError(f"1 + epsilon must be positive, got {a}.")

    if window_data is None:
        window_data = window_interp_data_from_wdm(wdm)

    xi_min, xi_max, support_empty = affine_Q_integration_bounds(
        wdm,
        a=a,
        delta=delta,
        rho=rho,
        margin_fraction=0.05,
        window_data=window_data,
    )

    out = np.zeros((len(ell_values), len(D_values)), dtype=np.complex128)

    if support_empty:
        return out

    xi = np.linspace(xi_min, xi_max, int(n_quad), dtype=np.float64)

    dF = float(wdm.dF)

    arg_source = xi / a - rho * delta * dF / (2.0 * a)
    arg_target = xi + rho * delta * dF / 2.0

    W = phi_tilde_from_window_data(window_data, arg_source) * np.conj(
        phi_tilde_from_window_data(window_data, arg_target)
    )

    lambdas = np.empty((len(ell_values), len(D_values)), dtype=np.float64)

    for iell, ell in enumerate(ell_values):
        lambdas[iell, :] = (D_values + int(ell) * float(wdm.dT)) / a

    lambdas_flat = lambdas.reshape(-1)
    values_flat = np.empty_like(lambdas_flat, dtype=np.complex128)

    chunk = int(lambda_chunk_size)

    if chunk < 1:
        raise ValueError("lambda_chunk_size must be >= 1.")

    for start in range(0, lambdas_flat.size, chunk):
        stop = min(start + chunk, lambdas_flat.size)

        lam = lambdas_flat[start:stop]

        phase = np.exp(2.0j * np.pi * lam[:, None] * xi[None, :])
        integrand = phase * W[None, :]

        if hasattr(np, "trapezoid"):
            values_flat[start:stop] = np.trapezoid(integrand, xi, axis=1)
        else:
            values_flat[start:stop] = np.trapz(integrand, xi, axis=1)

    return values_flat.reshape(len(ell_values), len(D_values))


# =============================================================================
# Affine filter table class
# =============================================================================


@dataclass
class AffineFilterTable:
    """
    Interpolated affine WDM filter table.

    The stored bare filters are

        Q^(rho, sigma)(ell, D, epsilon, eta).

    With branch symmetry enabled, only rho=+1 is stored and rho=-1 is
    reconstructed using

        Q^(-, sigma)(ell, D, eps, eta)
        =
        Q^(+, -sigma)(ell, D, eps, -eta).
    """

    Q_tables: dict[tuple[int, int, int], np.ndarray]
    D_grid: np.ndarray
    eps_grid: np.ndarray
    eta_grid: np.ndarray
    ell_values: np.ndarray
    sigma_values: np.ndarray
    rho_values_stored: np.ndarray
    use_branch_symmetry: bool
    n_quad_table: int
    wdm_metadata: dict[str, Any]

    @classmethod
    def build(
        cls,
        wdm: Any,
        *,
        D_grid: ArrayLike,
        eps_grid: ArrayLike,
        eta_grid: ArrayLike,
        ell_values: ArrayLike,
        sigma_values: ArrayLike = (-1, 0, +1),
        use_branch_symmetry: bool = True,
        n_quad_table: int = 2048,
        lambda_chunk_size: int = 512,
        verbose: bool = True,
    ) -> "AffineFilterTable":
        """
        Build an affine filter table by direct quadrature.

        This is the validated accuracy-first builder. It is not yet the final
        fast FFT/JAX production builder.
        """
        D_grid = _as_1d_float_array(D_grid, "D_grid")
        eps_grid = _as_1d_float_array(eps_grid, "eps_grid")
        eta_grid = _as_1d_float_array(eta_grid, "eta_grid")
        ell_values = _as_1d_int_array(ell_values, "ell_values")
        sigma_values = _as_1d_int_array(sigma_values, "sigma_values")

        if use_branch_symmetry:
            rho_values_stored = np.array([+1], dtype=np.int64)
        else:
            rho_values_stored = np.array([-1, +1], dtype=np.int64)

        window_data = window_interp_data_from_wdm(wdm)

        Q_tables: dict[tuple[int, int, int], np.ndarray] = {}

        total = (
            len(rho_values_stored)
            * len(sigma_values)
            * len(eps_grid)
            * len(eta_grid)
        )
        count = 0

        for rho in rho_values_stored:
            for sigma in sigma_values:
                tables_for_sigma = {
                    int(ell): np.empty(
                        (len(D_grid), len(eps_grid), len(eta_grid)),
                        dtype=np.complex128,
                    )
                    for ell in ell_values
                }

                for iE, epsv in enumerate(eps_grid):
                    for iH, etav in enumerate(eta_grid):
                        Q_ell_D = direct_affine_Q_many_lambda(
                            wdm,
                            D_values=D_grid,
                            ell_values=ell_values,
                            epsilon=float(epsv),
                            sigma=int(sigma),
                            eta=float(etav),
                            rho=int(rho),
                            n_quad=n_quad_table,
                            lambda_chunk_size=lambda_chunk_size,
                            window_data=window_data,
                        )

                        for iell, ell in enumerate(ell_values):
                            tables_for_sigma[int(ell)][:, iE, iH] = Q_ell_D[
                                iell, :
                            ]

                        count += 1

                        if verbose and (count % 500 == 0 or count == total):
                            print(f"{count}/{total}")

                for ell in ell_values:
                    Q_tables[(int(rho), int(sigma), int(ell))] = tables_for_sigma[
                        int(ell)
                    ]

        return cls(
            Q_tables=Q_tables,
            D_grid=D_grid,
            eps_grid=eps_grid,
            eta_grid=eta_grid,
            ell_values=ell_values,
            sigma_values=sigma_values,
            rho_values_stored=rho_values_stored,
            use_branch_symmetry=bool(use_branch_symmetry),
            n_quad_table=int(n_quad_table),
            wdm_metadata=wdm_metadata_from_wdm(wdm),
        )

    @classmethod
    def build_or_load(
        cls,
        path: str | Path,
        wdm: Any,
        *,
        D_grid: ArrayLike,
        eps_grid: ArrayLike,
        eta_grid: ArrayLike,
        ell_values: ArrayLike,
        sigma_values: ArrayLike = (-1, 0, +1),
        use_branch_symmetry: bool = True,
        n_quad_table: int = 2048,
        lambda_chunk_size: int = 512,
        rebuild: bool = False,
        check_build_parameters: bool = True,
        verbose: bool = True,
    ) -> "AffineFilterTable":
        """
        Load an existing affine filter table if available; otherwise build and save it.

        Parameters
        ----------
        path:
            Pickle path for the cached table.
        wdm:
            WDM transform object.
        rebuild:
            If True, force rebuilding even if the cache file exists.
        check_build_parameters:
            If True, verify that the loaded table matches the requested grids,
            lag values, sigma values, branch-symmetry setting, and n_quad_table.
        """
        path = Path(path)

        D_grid = _as_1d_float_array(D_grid, "D_grid")
        eps_grid = _as_1d_float_array(eps_grid, "eps_grid")
        eta_grid = _as_1d_float_array(eta_grid, "eta_grid")
        ell_values = _as_1d_int_array(ell_values, "ell_values")
        sigma_values = _as_1d_int_array(sigma_values, "sigma_values")

        if path.exists() and not rebuild:
            table = cls.load(path)

            assert_compatible_wdm(table, wdm)

            if check_build_parameters:
                assert_compatible_build_parameters(
                    table,
                    D_grid=D_grid,
                    eps_grid=eps_grid,
                    eta_grid=eta_grid,
                    ell_values=ell_values,
                    sigma_values=sigma_values,
                    use_branch_symmetry=use_branch_symmetry,
                    n_quad_table=n_quad_table,
                )

            if verbose:
                print(f"Loaded affine filter table from {path}")

            return table

        if verbose:
            print("Building affine filter table")

        table = cls.build(
            wdm,
            D_grid=D_grid,
            eps_grid=eps_grid,
            eta_grid=eta_grid,
            ell_values=ell_values,
            sigma_values=sigma_values,
            use_branch_symmetry=use_branch_symmetry,
            n_quad_table=n_quad_table,
            lambda_chunk_size=lambda_chunk_size,
            verbose=verbose,
        )

        table.save(path)

        if verbose:
            print(f"Saved affine filter table to {path}")

        return table

    def evaluate_Q(
        self,
        *,
        rho: int,
        sigma: int,
        ell: int,
        D: float,
        epsilon: float,
        eta: float,
    ) -> complex:
        """
        Evaluate the interpolated bare filter Q.
        """
        rho = int(rho)
        sigma = int(sigma)
        ell = int(ell)

        if rho == +1:
            return self._evaluate_Q_stored(
                rho=+1,
                sigma=sigma,
                ell=ell,
                D=D,
                epsilon=epsilon,
                eta=eta,
            )

        if rho == -1:
            if self.use_branch_symmetry:
                return self._evaluate_Q_stored(
                    rho=+1,
                    sigma=-sigma,
                    ell=ell,
                    D=D,
                    epsilon=epsilon,
                    eta=-float(eta),
                )

            return self._evaluate_Q_stored(
                rho=-1,
                sigma=sigma,
                ell=ell,
                D=D,
                epsilon=epsilon,
                eta=eta,
            )

        raise ValueError(f"rho must be -1 or +1, got {rho}.")

    def _evaluate_Q_stored(
        self,
        *,
        rho: int,
        sigma: int,
        ell: int,
        D: float,
        epsilon: float,
        eta: float,
    ) -> complex:
        key = (int(rho), int(sigma), int(ell))

        if key not in self.Q_tables:
            raise KeyError(f"No stored affine Q table for key={key}.")

        return trilinear_interp_complex(
            self.Q_tables[key],
            self.D_grid,
            self.eps_grid,
            self.eta_grid,
            D,
            epsilon,
            eta,
        )
    
    
    def evaluate_Q_vectorized(
        self,
        *,
        rho: int,
        sigma: int,
        ell: int,
        D: ArrayLike,
        epsilon: ArrayLike,
        eta: ArrayLike,
        check_domain: bool = False,
        ) -> np.ndarray:
        """
        Vectorised evaluation of the interpolated bare filter Q.

        This is mathematically equivalent to evaluate_Q, but supports array
        inputs for D, epsilon, and eta. Broadcasting is allowed.

        Typical affine-operator usage:

            D       shape (n_chunk, 1)
            epsilon shape (n_chunk, 1)
            eta     shape (n_chunk, m_count)

        Returns
        -------
        ndarray
            Complex array with broadcasted shape.
        """
        rho = int(rho)
        sigma = int(sigma)
        ell = int(ell)

        if rho == +1:
            return self._evaluate_Q_stored_vectorized(
                rho=+1,
                sigma=sigma,
                ell=ell,
                D=D,
                epsilon=epsilon,
                eta=eta,
                check_domain=check_domain,
            )

        if rho == -1:
            if self.use_branch_symmetry:
                return self._evaluate_Q_stored_vectorized(
                    rho=+1,
                    sigma=-sigma,
                    ell=ell,
                    D=D,
                    epsilon=epsilon,
                    eta=-np.asarray(eta, dtype=np.float64),
                    check_domain=check_domain,
                )

            return self._evaluate_Q_stored_vectorized(
                rho=-1,
                sigma=sigma,
                ell=ell,
                D=D,
                epsilon=epsilon,
                eta=eta,
                check_domain=check_domain,
            )

        raise ValueError(f"rho must be -1 or +1, got {rho}.")


    def _evaluate_Q_stored_vectorized(
        self,
        *,
        rho: int,
        sigma: int,
        ell: int,
        D: ArrayLike,
        epsilon: ArrayLike,
        eta: ArrayLike,
        check_domain: bool = False,
    ) -> np.ndarray:
        """
        Vectorised evaluation of one stored Q table.

        This is the array-valued analogue of _evaluate_Q_stored.
        """
        key = (int(rho), int(sigma), int(ell))

        if key not in self.Q_tables:
            raise KeyError(f"No stored affine Q table for key={key}.")

        return trilinear_interp_complex_vectorized(
            self.Q_tables[key],
            self.D_grid,
            self.eps_grid,
            self.eta_grid,
            D,
            epsilon,
            eta,
            check_domain=check_domain,
        )




    def evaluate_K(
        self,
        wdm: Any,
        *,
        n: int,
        ell: int,
        m: int,
        sigma: int,
        D: float,
        epsilon: float,
    ) -> complex:
        """
        Evaluate the branch-summed affine sparse kernel entry K.

        This returns zero for invalid source/target indices.
        """
        n = int(n)
        ell = int(ell)
        m = int(m)
        sigma = int(sigma)

        n_source = n - ell
        m_source = m + sigma

        if not (0 <= n_source < int(wdm.Nt)):
            return 0.0 + 0.0j

        # m=0 is deliberately excluded because the fast WDM convention usually
        # has calc_m0=False. Handle m=0 separately later if needed.
        if not (1 <= m < int(wdm.Nf)):
            return 0.0 + 0.0j

        if not (1 <= m_source < int(wdm.Nf)):
            return 0.0 + 0.0j

        a = 1.0 + float(epsilon)

        if a <= 0.0:
            raise ValueError(f"1 + epsilon must be positive, got {a}.")

        eta = float(epsilon) * (m + sigma)

        self.validate_continuous_arguments(D=D, epsilon=epsilon, eta=eta)

        delta = sigma + eta
        mu = m + 0.5 * delta
        lambda_arg = (float(D) + ell * float(wdm.dT)) / a

        total = 0.0 + 0.0j

        for rho in (-1, +1):
            Q = self.evaluate_Q(
                rho=rho,
                sigma=sigma,
                ell=ell,
                D=D,
                epsilon=epsilon,
                eta=eta,
            )

            C_source = c_branch(n_source, m_source, rho)
            C_target = c_branch(n, m, rho)

            carrier = np.exp(
                2.0j
                * np.pi
                * rho
                * mu
                * float(wdm.dF)
                * lambda_arg
            )

            total += C_source * np.conj(C_target) * carrier * Q

        return complex(total / (2.0 * a))

    def validate_continuous_arguments(
        self,
        *,
        D: float,
        epsilon: float,
        eta: float,
    ) -> None:
        """
        Raise ValueError if D, epsilon, or eta lies outside the table domain.
        """
        _bracket(self.D_grid, D, "D")
        _bracket(self.eps_grid, epsilon, "epsilon")
        _bracket(self.eta_grid, eta, "eta")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": TABLE_FORMAT_VERSION,
            "Q_tables": self.Q_tables,
            "D_grid": self.D_grid,
            "eps_grid": self.eps_grid,
            "eta_grid": self.eta_grid,
            "ell_values": self.ell_values,
            "sigma_values": self.sigma_values,
            "rho_values_stored": self.rho_values_stored,
            "use_branch_symmetry": self.use_branch_symmetry,
            "n_quad_table": self.n_quad_table,
            "wdm_metadata": self.wdm_metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AffineFilterTable":
        # Existing cached tables without this field are treated as version 0.
        format_version = int(payload.get("format_version", 0))

        if format_version not in (0, TABLE_FORMAT_VERSION):
            raise ValueError(
                f"Unsupported affine table format_version={format_version}. "
                f"Expected {TABLE_FORMAT_VERSION}."
            )

        return cls(
            Q_tables=payload["Q_tables"],
            D_grid=np.asarray(payload["D_grid"], dtype=np.float64),
            eps_grid=np.asarray(payload["eps_grid"], dtype=np.float64),
            eta_grid=np.asarray(payload["eta_grid"], dtype=np.float64),
            ell_values=np.asarray(payload["ell_values"], dtype=np.int64),
            sigma_values=np.asarray(payload["sigma_values"], dtype=np.int64),
            rho_values_stored=np.asarray(
                payload["rho_values_stored"], dtype=np.int64
            ),
            use_branch_symmetry=bool(payload["use_branch_symmetry"]),
            n_quad_table=int(payload["n_quad_table"]),
            wdm_metadata=dict(payload["wdm_metadata"]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)

        with path.open("wb") as f:
            pickle.dump(self.to_payload(), f)

    @classmethod
    def load(cls, path: str | Path) -> "AffineFilterTable":
        path = Path(path)

        with path.open("rb") as f:
            payload = pickle.load(f)

        return cls.from_payload(payload)


# =============================================================================
# Metadata and compatibility checks
# =============================================================================


def wdm_metadata_from_wdm(wdm: Any) -> dict[str, Any]:
    """
    Store enough WDM metadata to check table compatibility later.
    """
    metadata: dict[str, Any] = {}

    int_keys = [
        "Nf",
        "N",
        "Nt",
        "q",
        "d",
    ]

    float_keys = [
        "dt",
        "A_frac",
        "dF",
        "dT",
    ]

    bool_keys = [
        "calc_m0",
    ]

    for key in int_keys:
        if hasattr(wdm, key):
            metadata[key] = int(getattr(wdm, key))

    for key in float_keys:
        if hasattr(wdm, key):
            metadata[key] = float(getattr(wdm, key))

    for key in bool_keys:
        if hasattr(wdm, key):
            metadata[key] = bool(getattr(wdm, key))

    if hasattr(wdm, "window_FD"):
        metadata["window_FD_shape"] = tuple(np.asarray(wdm.window_FD).shape)

    if hasattr(wdm, "freqs"):
        metadata["freqs_shape"] = tuple(np.asarray(wdm.freqs).shape)

    return metadata


def assert_compatible_wdm(
    table: AffineFilterTable,
    wdm: Any,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-15,
) -> None:
    """
    Check that a table was built for a WDM object compatible with the supplied one.
    """
    current = wdm_metadata_from_wdm(wdm)
    saved = table.wdm_metadata

    int_keys = ("Nf", "N", "Nt", "q", "d")
    float_keys = ("dt", "dF", "dT", "A_frac")
    bool_keys = ("calc_m0",)

    for key in int_keys:
        if key in saved and key in current:
            if int(saved[key]) != int(current[key]):
                raise ValueError(
                    f"Incompatible WDM metadata for {key}: "
                    f"table has {saved[key]}, current WDM has {current[key]}."
                )

    for key in float_keys:
        if key in saved and key in current:
            if not np.isclose(
                float(saved[key]),
                float(current[key]),
                rtol=rtol,
                atol=atol,
            ):
                raise ValueError(
                    f"Incompatible WDM metadata for {key}: "
                    f"table has {saved[key]}, current WDM has {current[key]}."
                )

    for key in bool_keys:
        if key in saved and key in current:
            if bool(saved[key]) != bool(current[key]):
                raise ValueError(
                    f"Incompatible WDM metadata for {key}: "
                    f"table has {saved[key]}, current WDM has {current[key]}."
                )

    for key in ("window_FD_shape", "freqs_shape"):
        if key in saved and key in current:
            if tuple(saved[key]) != tuple(current[key]):
                raise ValueError(
                    f"Incompatible WDM metadata for {key}: "
                    f"table has {saved[key]}, current WDM has {current[key]}."
                )


def _assert_same_float_array(
    saved: np.ndarray,
    requested: np.ndarray,
    *,
    name: str,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-15,
) -> None:
    saved = np.asarray(saved, dtype=np.float64)
    requested = np.asarray(requested, dtype=np.float64)

    if saved.shape != requested.shape:
        raise ValueError(
            f"Incompatible {name}: saved shape {saved.shape}, "
            f"requested shape {requested.shape}."
        )

    if not np.allclose(saved, requested, rtol=rtol, atol=atol):
        max_diff = float(np.max(np.abs(saved - requested)))
        raise ValueError(
            f"Incompatible {name}: saved/requested arrays differ. "
            f"max abs diff={max_diff}."
        )


def _assert_same_int_array(
    saved: np.ndarray,
    requested: np.ndarray,
    *,
    name: str,
) -> None:
    saved = np.asarray(saved, dtype=np.int64)
    requested = np.asarray(requested, dtype=np.int64)

    if saved.shape != requested.shape or not np.array_equal(saved, requested):
        raise ValueError(
            f"Incompatible {name}: saved={saved}, requested={requested}."
        )


def assert_compatible_build_parameters(
    table: AffineFilterTable,
    *,
    D_grid: ArrayLike,
    eps_grid: ArrayLike,
    eta_grid: ArrayLike,
    ell_values: ArrayLike,
    sigma_values: ArrayLike,
    use_branch_symmetry: bool,
    n_quad_table: int,
) -> None:
    """
    Check that a loaded table matches the requested build parameters.
    """
    _assert_same_float_array(
        table.D_grid,
        _as_1d_float_array(D_grid, "D_grid"),
        name="D_grid",
    )
    _assert_same_float_array(
        table.eps_grid,
        _as_1d_float_array(eps_grid, "eps_grid"),
        name="eps_grid",
    )
    _assert_same_float_array(
        table.eta_grid,
        _as_1d_float_array(eta_grid, "eta_grid"),
        name="eta_grid",
    )
    _assert_same_int_array(
        table.ell_values,
        _as_1d_int_array(ell_values, "ell_values"),
        name="ell_values",
    )
    _assert_same_int_array(
        table.sigma_values,
        _as_1d_int_array(sigma_values, "sigma_values"),
        name="sigma_values",
    )

    if bool(table.use_branch_symmetry) != bool(use_branch_symmetry):
        raise ValueError(
            "Incompatible use_branch_symmetry: "
            f"saved={table.use_branch_symmetry}, requested={use_branch_symmetry}."
        )

    if int(table.n_quad_table) != int(n_quad_table):
        raise ValueError(
            "Incompatible n_quad_table: "
            f"saved={table.n_quad_table}, requested={n_quad_table}."
        )