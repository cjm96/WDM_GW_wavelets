"""
Sparse affine WDM time-shift operator.

This module applies an AffineFilterTable to WDM coefficient arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .affine_filter_table import AffineFilterTable, assert_compatible_wdm, c_branch


def build_c_branch_cache(
    *,
    Nt: int,
    Nf: int,
    dtype: Any = np.complex128,
) -> dict[int, np.ndarray]:
    """
    Precompute WDM branch/parity factors C^(rho)_{n,m}.

    This avoids repeated scalar c_branch calls inside the vectorised affine
    sparse-operator backend.

    Returns
    -------
    dict
        Keys are rho=-1 and rho=+1. Values have shape (Nt, Nf).
    """
    Nt = int(Nt)
    Nf = int(Nf)

    if Nt < 1:
        raise ValueError(f"Nt must be positive, got {Nt}.")

    if Nf < 1:
        raise ValueError(f"Nf must be positive, got {Nf}.")

    C_cache: dict[int, np.ndarray] = {}

    for rho in (-1, +1):
        arr = np.empty((Nt, Nf), dtype=dtype)

        for n in range(Nt):
            for m in range(Nf):
                arr[n, m] = c_branch(n, m, rho)

        C_cache[rho] = arr

    return C_cache

@dataclass
class AffineSparseOperator:
    """
    Sparse affine WDM time-shift operator.

    Parameters
    ----------
    wdm:
        WDM transform object.
    table:
        Precomputed AffineFilterTable.
    """

    wdm: Any
    table: AffineFilterTable
    check_wdm_compatibility: bool = True

    def __post_init__(self) -> None:
        if self.check_wdm_compatibility:
            assert_compatible_wdm(self.table, self.wdm)

    def apply_selected(
        self,
        w_in: np.ndarray,
        *,
        D_of_n: np.ndarray,
        epsilon_of_n: np.ndarray,
        n_targets: np.ndarray,
        m_targets: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the affine sparse operator only to selected target rows/frequencies.

        Returns
        -------
        w_out_selected : ndarray
            Shape=(len(n_targets), len(m_targets)).
        """
        w_in = np.asarray(w_in, dtype=np.complex128)
        D_of_n = np.asarray(D_of_n, dtype=np.float64)
        epsilon_of_n = np.asarray(epsilon_of_n, dtype=np.float64)
        n_targets = np.asarray(n_targets, dtype=np.int64)
        m_targets = np.asarray(m_targets, dtype=np.int64)

        self._validate_inputs(w_in, D_of_n, epsilon_of_n)

        w_out = np.zeros((len(n_targets), len(m_targets)), dtype=np.complex128)

        for in_idx, n in enumerate(n_targets):
            n = int(n)

            if not (0 <= n < int(self.wdm.Nt)):
                raise ValueError(f"Target n={n} outside [0, {self.wdm.Nt}).")

            D = float(D_of_n[n])
            epsilon = float(epsilon_of_n[n])

            for im_idx, m in enumerate(m_targets):
                m = int(m)

                if not (1 <= m < int(self.wdm.Nf)):
                    raise ValueError(
                        f"Target m={m} outside valid m range [1, {self.wdm.Nf})."
                    )

                w_out[in_idx, im_idx] = self.entry(
                    w_in,
                    n=n,
                    m=m,
                    D=D,
                    epsilon=epsilon,
                )

        return w_out

    def apply_full(
        self,
        w_in: np.ndarray,
        *,
        D_of_n: np.ndarray,
        epsilon_of_n: np.ndarray,
        include_m0: bool = False,
    ) -> np.ndarray:
        """
        Apply the affine sparse operator to the full WDM coefficient array.

        This is a correctness-first NumPy implementation. It is not yet the
        final optimised production path.

        By default m=0 is left as zero because the usual fast WDM convention has
        calc_m0=False.
        """
        w_in = np.asarray(w_in, dtype=np.complex128)
        D_of_n = np.asarray(D_of_n, dtype=np.float64)
        epsilon_of_n = np.asarray(epsilon_of_n, dtype=np.float64)

        self._validate_inputs(w_in, D_of_n, epsilon_of_n)

        w_out = np.zeros_like(w_in, dtype=np.complex128)

        m_start = 0 if include_m0 else 1

        for n in range(int(self.wdm.Nt)):
            D = float(D_of_n[n])
            epsilon = float(epsilon_of_n[n])

            for m in range(m_start, int(self.wdm.Nf)):
                w_out[n, m] = self.entry(
                    w_in,
                    n=n,
                    m=m,
                    D=D,
                    epsilon=epsilon,
                )

        return w_out
    
    def apply_full_vectorized(
        self,
        w_in: np.ndarray,
        *,
        D_of_n: np.ndarray,
        epsilon_of_n: np.ndarray,
        include_m0: bool = False,
        n_chunk: int = 32,
        out_dtype: Any = np.complex128,
        check_domain: bool = False,
        ell_values: np.ndarray | None = None,
        sigma_values: np.ndarray | None = None,
        C_cache: dict[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Apply the affine sparse operator to the full WDM coefficient array using
        vectorised Q-table interpolation.

        This is mathematically equivalent to apply_full, but avoids scalar
        calls to entry/evaluate_K/evaluate_Q. The main loops are over ell,
        sigma, and chunks of n, while the m direction is vectorised.

        Parameters
        ----------
        w_in:
            Input WDM coefficient array, shape (Nt, Nf).
        D_of_n, epsilon_of_n:
            Local-affine delay arrays, shape (Nt,).
        include_m0:
            Kept for API compatibility with apply_full. The current affine
            kernel excludes target m=0, so the output m=0 row remains zero.
        n_chunk:
            Number of WDM time rows to process in one vectorised block.
        out_dtype:
            Output dtype. Use complex128 for validation.
        check_domain:
            If True, check D, epsilon, and eta against the interpolation table
            domain inside each vectorised Q evaluation. For production, this
            can be False after a separate domain check.
        ell_values:
            Optional lag subset. This allows using a table built with a large
            L_max while applying only ell in [-L, L].
        sigma_values:
            Optional sigma subset. Defaults to the table sigma values.
        C_cache:
            Optional precomputed branch/parity cache from build_c_branch_cache.

        Returns
        -------
        ndarray
            Shifted WDM coefficient array, shape (Nt, Nf).
        """
        w_in = np.asarray(w_in, dtype=np.complex128)
        D_of_n = np.asarray(D_of_n, dtype=np.float64)
        epsilon_of_n = np.asarray(epsilon_of_n, dtype=np.float64)

        self._validate_inputs(w_in, D_of_n, epsilon_of_n)

        n_chunk = int(n_chunk)

        if n_chunk < 1:
            raise ValueError(f"n_chunk must be >= 1, got {n_chunk}.")

        Nt, Nf = w_in.shape

        w_out = np.zeros((Nt, Nf), dtype=out_dtype)

        if ell_values is None:
            ell_values_use = np.asarray(self.table.ell_values, dtype=np.int64)
        else:
            ell_values_use = np.asarray(ell_values, dtype=np.int64)

        if sigma_values is None:
            sigma_values_use = np.asarray(self.table.sigma_values, dtype=np.int64)
        else:
            sigma_values_use = np.asarray(sigma_values, dtype=np.int64)

        table_ells = set(int(x) for x in np.asarray(self.table.ell_values, dtype=int))
        table_sigmas = set(int(x) for x in np.asarray(self.table.sigma_values, dtype=int))

        missing_ells = [int(x) for x in ell_values_use if int(x) not in table_ells]
        missing_sigmas = [int(x) for x in sigma_values_use if int(x) not in table_sigmas]

        if missing_ells:
            raise ValueError(
                f"Requested ell_values not present in table: {missing_ells}"
            )

        if missing_sigmas:
            raise ValueError(
                f"Requested sigma_values not present in table: {missing_sigmas}"
            )

        if C_cache is None:
            C_cache = build_c_branch_cache(
                Nt=Nt,
                Nf=Nf,
                dtype=np.complex128,
            )
        else:
            for rho in (-1, +1):
                if rho not in C_cache:
                    raise ValueError(f"C_cache is missing rho={rho}.")

                if np.asarray(C_cache[rho]).shape != (Nt, Nf):
                    raise ValueError(
                        f"C_cache[{rho}] must have shape {(Nt, Nf)}, "
                        f"got {np.asarray(C_cache[rho]).shape}."
                    )

        dT = float(self.wdm.dT)
        dF = float(self.wdm.dF)

        # The scalar evaluate_K excludes target m=0. Therefore m=0 remains zero
        # even if include_m0=True. We preserve that behaviour here.
        target_m_start = 1

        for ell in ell_values_use:
            ell = int(ell)

            # Source time index is n_source = n - ell.
            n_start = max(0, ell)
            n_stop = min(Nt, Nt + ell)

            if n_stop <= n_start:
                continue

            for sigma in sigma_values_use:
                sigma = int(sigma)

                # Need target m valid and source m + sigma valid:
                #
                #   1 <= m < Nf
                #   1 <= m + sigma < Nf
                #
                m_start = max(target_m_start, 1 - sigma)
                m_stop = min(Nf, Nf - sigma)

                if m_stop <= m_start:
                    continue

                m_values = np.arange(m_start, m_stop, dtype=np.int64)

                m_target_slice = slice(m_start, m_stop)
                m_source_slice = slice(m_start + sigma, m_stop + sigma)

                for n0 in range(n_start, n_stop, n_chunk):
                    n1 = min(n0 + n_chunk, n_stop)

                    n_target_slice = slice(n0, n1)
                    n_source_slice = slice(n0 - ell, n1 - ell)

                    D_block = D_of_n[n0:n1, None]
                    epsilon_block = epsilon_of_n[n0:n1, None]

                    a_block = 1.0 + epsilon_block

                    if np.any(a_block <= 0.0):
                        raise ValueError("Encountered 1 + epsilon <= 0.")

                    eta_block = epsilon_block * (m_values[None, :] + sigma)

                    delta_block = sigma + eta_block
                    mu_block = m_values[None, :] + 0.5 * delta_block
                    lambda_block = (D_block + ell * dT) / a_block

                    total_kernel = np.zeros(
                        (n1 - n0, m_stop - m_start),
                        dtype=np.complex128,
                    )

                    for rho in (-1, +1):
                        Q_block = self.table.evaluate_Q_vectorized(
                            rho=rho,
                            sigma=sigma,
                            ell=ell,
                            D=D_block,
                            epsilon=epsilon_block,
                            eta=eta_block,
                            check_domain=check_domain,
                        )

                        C_source = C_cache[rho][n_source_slice, m_source_slice]
                        C_target = C_cache[rho][n_target_slice, m_target_slice]

                        carrier = np.exp(
                            2.0j
                            * np.pi
                            * rho
                            * mu_block
                            * dF
                            * lambda_block
                        )

                        total_kernel += (
                            C_source
                            * np.conj(C_target)
                            * carrier
                            * Q_block
                        )

                    K_block = total_kernel / (2.0 * a_block)

                    w_out[n_target_slice, m_target_slice] += (
                        K_block * w_in[n_source_slice, m_source_slice]
                    )

        return w_out

    def entry(
        self,
        w_in: np.ndarray,
        *,
        n: int,
        m: int,
        D: float,
        epsilon: float,
    ) -> complex:
        """
        Compute one target output coefficient:

            w_out[n, m] = sum_{ell, sigma}
                          K[n,m,ell,sigma] w_in[n-ell, m+sigma].
        """
        total = 0.0 + 0.0j

        n = int(n)
        m = int(m)

        for ell in self.table.ell_values:
            ell = int(ell)
            n_source = n - ell

            if not (0 <= n_source < int(self.wdm.Nt)):
                continue

            for sigma in self.table.sigma_values:
                sigma = int(sigma)
                m_source = m + sigma

                if not (1 <= m_source < int(self.wdm.Nf)):
                    continue

                K = self.table.evaluate_K(
                    self.wdm,
                    n=n,
                    ell=ell,
                    m=m,
                    sigma=sigma,
                    D=D,
                    epsilon=epsilon,
                )

                total += K * w_in[n_source, m_source]

        return complex(total)

    def _validate_inputs(
        self,
        w_in: np.ndarray,
        D_of_n: np.ndarray,
        epsilon_of_n: np.ndarray,
    ) -> None:
        expected_shape = (int(self.wdm.Nt), int(self.wdm.Nf))

        if w_in.shape != expected_shape:
            raise ValueError(
                f"w_in must have shape {expected_shape}, got {w_in.shape}."
            )

        if D_of_n.shape != (int(self.wdm.Nt),):
            raise ValueError(
                f"D_of_n must have shape ({self.wdm.Nt},), got {D_of_n.shape}."
            )

        if epsilon_of_n.shape != (int(self.wdm.Nt),):
            raise ValueError(
                f"epsilon_of_n must have shape ({self.wdm.Nt},), "
                f"got {epsilon_of_n.shape}."
            )