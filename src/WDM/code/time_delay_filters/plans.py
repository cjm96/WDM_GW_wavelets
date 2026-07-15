"""Reusable operator plans for WDM variable time shifts.

The existing high-level shift function performs two different tasks:

1. build all delay-dependent quantities (lag range, Tl/Tp, parity matrix), and
2. apply those quantities to waveform-dependent WDM coefficients.

``VariableShiftBatchPlan`` separates those stages.  Building a plan can be
expensive, but applying it to a new coefficient batch does not reconstruct
Tl/Tp.

This first implementation intentionally reuses the existing validated helper
functions and JAX assembly kernels.  It is therefore a low-risk architectural
change rather than a numerical rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .config import VariableShiftPlanConfig
from ._time_shift_assembly import _assemble_shift_target_batch_dispatch
from .time_shift_fast import (
    _build_signed_lag_idx,
    _build_TlTp_from_shift_matrix,
    _build_TlTp_from_shift_matrix_interp,
    _build_TlTp_from_shift_matrix_interp_jax,
    _get_Cnm_parity,
    _get_kernel_precomputes,
    _infer_Nf,
    _normalize_assembly_precision,
    _resolve_assembly_backend,
    _resolve_ell_range,
    _resolve_interp_backend,
    _resolve_use_jax,
    _validate_lag_block_size,
    _validate_row_chunk_size,
    choose_Nker,
)


def _as_delay_matrix(delays: np.ndarray) -> np.ndarray:
    delays = np.asarray(delays, dtype=float)
    if delays.ndim == 1:
        delays = delays[None, :]
    if delays.ndim != 2:
        raise ValueError(
            "delays must have shape (num_jobs, Nt), or shape (Nt,) for one job."
        )
    if delays.shape[0] < 1 or delays.shape[1] < 1:
        raise ValueError("delays must contain at least one job and one time row.")
    if not np.all(np.isfinite(delays)):
        raise ValueError("delays contains NaN or infinite values.")
    return delays


def _as_coefficient_batch(
    coefficients: np.ndarray,
    *,
    num_jobs: int,
    Nt: int,
    Nm: int,
) -> np.ndarray:
    coefficients = np.asarray(coefficients)
    if coefficients.ndim == 2:
        coefficients = coefficients[None, :, :]
    expected = (num_jobs, Nt, Nm)
    if coefficients.shape != expected:
        raise ValueError(
            f"Expected coefficients with shape {expected}, got {coefficients.shape}."
        )
    return coefficients


@dataclass(frozen=True, slots=True)
class VariableShiftBatchPlan:
    """Prepared variable-delay WDM operators for one or more jobs.

    Parameters stored on the plan depend on the delay fields and WDM grid, but
    not on the waveform coefficients.  The same plan can consequently be
    applied repeatedly to different coefficient batches of identical shape.
    """

    wdm: Any
    config: VariableShiftPlanConfig

    delays: np.ndarray
    ell_all: np.ndarray
    offset: int
    Tl_all: np.ndarray
    Tp_all: np.ndarray
    Cnm: np.ndarray

    Nf: int
    Nker: int
    Nt: int
    Nm: int
    num_jobs: int

    resolved_use_jax: bool
    resolved_assembly_backend: str
    resolved_assembly_precision: str

    build_seconds: float

    # Lazily populated only by the indexed/grouped execution path.  Keeping
    # these arrays on the JAX device avoids retransferring the persistent
    # delay kernels for every waveform evaluation.
    _device_cache: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @classmethod
    def build(
        cls,
        wdm: Any,
        delays: np.ndarray,
        *,
        config: VariableShiftPlanConfig | None = None,
        Nf: int | None = None,
        Nker: int | None = None,
        safety: float = 1.02,
        kernel_kwargs: dict[str, Any] | None = None,
    ) -> "VariableShiftBatchPlan":
        """Prepare all delay-dependent data needed by the batch shifter.

        Preparation follows the same chunking convention as the existing
        ``wdm_time_shift_variable_batch`` function.  This is important for
        interpolated Tl/Tp because the legacy implementation constructs a
        separate interpolation range for every batch chunk.
        """

        started = perf_counter()
        config = VariableShiftPlanConfig() if config is None else config
        delays = _as_delay_matrix(delays)

        num_jobs, Nt = delays.shape
        Nm = int(getattr(wdm, "Nf"))
        Nf = _infer_Nf(wdm, Nt, Nf)

        resolved_use_jax = _resolve_use_jax(use_jax=config.use_jax)
        ell_all, offset = _resolve_ell_range(Nt, config.lag_truncation)

        if Nker is None:
            Nker = choose_Nker(
                offset,
                Nf,
                safety=safety,
                require_even_Ntker=True,
                require_even_Nker=True,
            )

        wdm_kernel, freqs_u, W0_u, W1_u, scale = _get_kernel_precomputes(
            wdm=wdm,
            Nker=Nker,
            Nf=Nf,
            kernel_kwargs=kernel_kwargs,
        )
        Nker = int(wdm_kernel.N)
        signed_lag_idx = _build_signed_lag_idx(ell_all, Nf, Nker)

        interp_backend = _resolve_interp_backend(
            config.tl_tp_interp_backend,
            use_jax=resolved_use_jax,
        )

        chunk_size = (
            num_jobs
            if config.batch_chunk is None
            else min(num_jobs, int(config.batch_chunk))
        )

        Tl_parts: list[np.ndarray] = []
        Tp_parts: list[np.ndarray] = []

        for start in range(0, num_jobs, chunk_size):
            stop = min(start + chunk_size, num_jobs)
            delay_chunk = delays[start:stop]

            if config.tl_tp_mode == "exact":
                Tl_chunk, Tp_chunk = _build_TlTp_from_shift_matrix(
                    delay_chunk,
                    freqs_u,
                    W0_u,
                    W1_u,
                    scale,
                    signed_lag_idx,
                )
            elif interp_backend == "jax":
                Tl_chunk, Tp_chunk = _build_TlTp_from_shift_matrix_interp_jax(
                    delay_chunk,
                    freqs_u,
                    W0_u,
                    W1_u,
                    scale,
                    signed_lag_idx,
                    interp_points=config.tl_tp_interp_points,
                    interp_pad=config.tl_tp_interp_pad,
                    interp_kind=config.tl_tp_interp_kind,
                )
            else:
                Tl_chunk, Tp_chunk = _build_TlTp_from_shift_matrix_interp(
                    delay_chunk,
                    freqs_u,
                    W0_u,
                    W1_u,
                    scale,
                    signed_lag_idx,
                    interp_points=config.tl_tp_interp_points,
                    interp_pad=config.tl_tp_interp_pad,
                    interp_kind=config.tl_tp_interp_kind,
                )

            # Plan construction is a setup operation, so holding its persistent
            # state as NumPy arrays is deliberate.  The existing assembly
            # wrappers transfer/cast these arrays when applying the plan.
            Tl_parts.append(np.asarray(Tl_chunk))
            Tp_parts.append(np.asarray(Tp_chunk))

        Tl_all = np.concatenate(Tl_parts, axis=0)
        Tp_all = np.concatenate(Tp_parts, axis=0)

        expected_kernel_shape = (num_jobs, Nt, ell_all.size)
        if Tl_all.shape != expected_kernel_shape or Tp_all.shape != expected_kernel_shape:
            raise RuntimeError(
                "Unexpected Tl/Tp shape after plan construction: "
                f"expected {expected_kernel_shape}, got Tl={Tl_all.shape}, "
                f"Tp={Tp_all.shape}."
            )

        resolved_backend = _resolve_assembly_backend(
            config.assembly_backend,
            config.assembly_vmap,
        )
        resolved_precision = _normalize_assembly_precision(
            config.assembly_precision
        )
        _validate_row_chunk_size(config.row_chunk_size)
        _validate_lag_block_size(config.lag_block_size)

        cnm_dtype = (
            np.complex64
            if resolved_precision == "complex64"
            else np.complex128
        )
        Cnm = _get_Cnm_parity(Nt, Nm, dtype=cnm_dtype)

        return cls(
            wdm=wdm,
            config=config,
            delays=delays,
            ell_all=ell_all,
            offset=int(offset),
            Tl_all=Tl_all,
            Tp_all=Tp_all,
            Cnm=Cnm,
            Nf=int(Nf),
            Nker=Nker,
            Nt=Nt,
            Nm=Nm,
            num_jobs=num_jobs,
            resolved_use_jax=resolved_use_jax,
            resolved_assembly_backend=resolved_backend,
            resolved_assembly_precision=resolved_precision,
            build_seconds=float(perf_counter() - started),
        )

    def apply(
        self,
        coefficients: np.ndarray,
        *,
        return_profile: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
        """Apply the prepared shifts to a new coefficient batch.

        No kernel FFTs or Tl/Tp interpolation are performed here.
        """

        coefficients = _as_coefficient_batch(
            coefficients,
            num_jobs=self.num_jobs,
            Nt=self.Nt,
            Nm=self.Nm,
        )

        started = perf_counter()
        chunk_size = (
            self.num_jobs
            if self.config.batch_chunk is None
            else min(self.num_jobs, int(self.config.batch_chunk))
        )

        outputs: list[np.ndarray] = []
        assembly_seconds = 0.0

        for start in range(0, self.num_jobs, chunk_size):
            stop = min(start + chunk_size, self.num_jobs)
            true_batch = stop - start

            coefficient_work = coefficients[start:stop]
            delay_work = self.delays[start:stop]
            Tl_work = self.Tl_all[start:stop]
            Tp_work = self.Tp_all[start:stop]

            should_pad = (
                self.resolved_use_jax
                and bool(self.config.assembly_vmap)
                and self.config.jax_pad_last_chunk
                and self.num_jobs > chunk_size
                and true_batch < chunk_size
            )

            if should_pad:
                pad_rows = chunk_size - true_batch
                coefficient_work = np.concatenate(
                    (
                        coefficient_work,
                        np.repeat(coefficient_work[-1:], pad_rows, axis=0),
                    ),
                    axis=0,
                )
                delay_work = np.concatenate(
                    (delay_work, np.repeat(delay_work[-1:], pad_rows, axis=0)),
                    axis=0,
                )
                Tl_work = np.concatenate(
                    (Tl_work, np.repeat(Tl_work[-1:], pad_rows, axis=0)),
                    axis=0,
                )
                Tp_work = np.concatenate(
                    (Tp_work, np.repeat(Tp_work[-1:], pad_rows, axis=0)),
                    axis=0,
                )

            assembly_started = perf_counter()
            shifted = _assemble_shift_target_batch_dispatch(
                self.wdm,
                coefficient_work,
                delay_work,
                self.ell_all,
                self.offset,
                Tl_work,
                Tp_work,
                Cnm=self.Cnm,
                use_jax=self.resolved_use_jax,
                assembly_backend=self.resolved_assembly_backend,
                assembly_precision=self.resolved_assembly_precision,
                row_chunk_size=self.config.row_chunk_size,
                lag_block_size=self.config.lag_block_size,
                job_block_size=self.config.job_block_size,
                assembly_vmap=self.config.assembly_vmap,
            )
            assembly_seconds += perf_counter() - assembly_started
            outputs.append(np.asarray(shifted)[:true_batch])

        result = np.concatenate(outputs, axis=0)
        total_seconds = perf_counter() - started

        if not return_profile:
            return result

        profile: dict[str, float | int | str] = {
            "n_jobs": self.num_jobs,
            "batch_chunk": chunk_size,
            "assembly_backend": self.resolved_assembly_backend,
            "assembly_precision": self.resolved_assembly_precision,
            "total_s": float(total_seconds),
            "assembly_s": float(assembly_seconds),
            "other_s": float(total_seconds - assembly_seconds),
            "plan_build_s": float(self.build_seconds),
        }
        return result, profile
    
    def apply_device(
        self,
        coefficients,
        *,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Apply the prepared shifts and return a JAX array.

        Unlike :meth:`apply`, this method does not copy shifted outputs back to
        NumPy. Persistent delay-dependent plan arrays may be cached on-device
        between repeated waveform evaluations.

        Parameters
        ----------
        coefficients : array-like
            WDM coefficient batch with shape ``(num_jobs, Nt, Nm)``. A
            two-dimensional array is accepted for a one-job plan.
        return_profile : bool, optional
            When true, synchronise execution and return timing metadata.
        cache_device_plan : bool, optional
            Reuse JAX copies of delays, Tl/Tp, lag indices and parity arrays.

        Returns
        -------
        jax.Array or tuple
            Shifted coefficients on the active JAX device. When
            ``return_profile=True``, returns ``(shifted, profile)``.
        """

        import jax.numpy as jnp

        started = perf_counter()

        complex_dtype = (
            jnp.complex64
            if self.resolved_assembly_precision == "complex64"
            else jnp.complex128
        )
        real_dtype = (
            jnp.float32
            if self.resolved_assembly_precision == "complex64"
            else jnp.float64
        )

        coefficients_device = jnp.asarray(
            coefficients,
            dtype=complex_dtype,
        )

        if coefficients_device.ndim == 2:
            coefficients_device = coefficients_device[None, :, :]

        expected = (self.num_jobs, self.Nt, self.Nm)
        if tuple(coefficients_device.shape) != expected:
            raise ValueError(
                f"Expected coefficients with shape {expected}, "
                f"got {tuple(coefficients_device.shape)}."
            )

        if cache_device_plan:
            device_plan = self._device_plan_arrays()
            delays_device = device_plan["delays"]
            ell_device = device_plan["ell_all"]
            Tl_device = device_plan["Tl_all"]
            Tp_device = device_plan["Tp_all"]
            Cnm_device = device_plan["Cnm"]
        else:
            delays_device = jnp.asarray(
                self.delays,
                dtype=real_dtype,
            )
            ell_device = jnp.asarray(
                self.ell_all,
                dtype=jnp.int64,
            )
            Tl_device = jnp.asarray(
                self.Tl_all,
                dtype=complex_dtype,
            )
            Tp_device = jnp.asarray(
                self.Tp_all,
                dtype=complex_dtype,
            )
            Cnm_device = jnp.asarray(
                self.Cnm,
                dtype=complex_dtype,
            )

        chunk_size = (
            self.num_jobs
            if self.config.batch_chunk is None
            else min(self.num_jobs, int(self.config.batch_chunk))
        )

        output_chunks = []
        assembly_started = perf_counter()

        for start in range(0, self.num_jobs, chunk_size):
            stop = min(start + chunk_size, self.num_jobs)
            true_batch = stop - start

            coefficient_work = coefficients_device[start:stop]
            delay_work = delays_device[start:stop]
            Tl_work = Tl_device[start:stop]
            Tp_work = Tp_device[start:stop]

            should_pad = (
                self.resolved_use_jax
                and bool(self.config.assembly_vmap)
                and self.config.jax_pad_last_chunk
                and self.num_jobs > chunk_size
                and true_batch < chunk_size
            )

            if should_pad:
                pad_rows = chunk_size - true_batch

                coefficient_work = jnp.concatenate(
                    (
                        coefficient_work,
                        jnp.repeat(
                            coefficient_work[-1:],
                            pad_rows,
                            axis=0,
                        ),
                    ),
                    axis=0,
                )
                delay_work = jnp.concatenate(
                    (
                        delay_work,
                        jnp.repeat(
                            delay_work[-1:],
                            pad_rows,
                            axis=0,
                        ),
                    ),
                    axis=0,
                )
                Tl_work = jnp.concatenate(
                    (
                        Tl_work,
                        jnp.repeat(
                            Tl_work[-1:],
                            pad_rows,
                            axis=0,
                        ),
                    ),
                    axis=0,
                )
                Tp_work = jnp.concatenate(
                    (
                        Tp_work,
                        jnp.repeat(
                            Tp_work[-1:],
                            pad_rows,
                            axis=0,
                        ),
                    ),
                    axis=0,
                )

            shifted_device = _assemble_shift_target_batch_dispatch(
                self.wdm,
                coefficient_work,
                delay_work,
                ell_device,
                self.offset,
                Tl_work,
                Tp_work,
                Cnm=Cnm_device,
                use_jax=self.resolved_use_jax,
                assembly_backend=self.resolved_assembly_backend,
                assembly_precision=self.resolved_assembly_precision,
                row_chunk_size=self.config.row_chunk_size,
                lag_block_size=self.config.lag_block_size,
                job_block_size=self.config.job_block_size,
                assembly_vmap=self.config.assembly_vmap,
                return_device=True,
            )

            output_chunks.append(shifted_device[:true_batch])

        if len(output_chunks) == 1:
            result_device = output_chunks[0]
        else:
            result_device = jnp.concatenate(
                output_chunks,
                axis=0,
            )

        # Profiling must explicitly synchronise asynchronous JAX execution.
        if return_profile:
            result_device.block_until_ready()

        assembly_seconds = perf_counter() - assembly_started
        total_seconds = perf_counter() - started

        if not return_profile:
            return result_device

        profile: dict[str, float | int | str | bool] = {
            "n_jobs": int(self.num_jobs),
            "batch_chunk": int(chunk_size),
            "assembly_backend": self.resolved_assembly_backend,
            "assembly_precision": self.resolved_assembly_precision,
            "total_s": float(total_seconds),
            "assembly_s": float(assembly_seconds),
            "other_s": float(total_seconds - assembly_seconds),
            "plan_build_s": float(self.build_seconds),
            "device_plan_cached": bool(cache_device_plan),
            "device_plan_memory_bytes": int(
                self.device_plan_memory_bytes
            ),
            "returned_on_device": True,
        }
        return result_device, profile

    def _device_plan_arrays(self) -> dict[str, Any]:
        """Return lazily cached JAX copies of the persistent plan arrays."""

        if self._device_cache:
            return self._device_cache

        import jax.numpy as jnp

        complex_dtype = (
            jnp.complex64
            if self.resolved_assembly_precision == "complex64"
            else jnp.complex128
        )
        real_dtype = (
            jnp.float32
            if self.resolved_assembly_precision == "complex64"
            else jnp.float64
        )

        self._device_cache.update(
            {
                "delays": jnp.asarray(self.delays, dtype=real_dtype),
                "ell_all": jnp.asarray(self.ell_all, dtype=jnp.int64),
                "Tl_all": jnp.asarray(self.Tl_all, dtype=complex_dtype),
                "Tp_all": jnp.asarray(self.Tp_all, dtype=complex_dtype),
                "Cnm": jnp.asarray(self.Cnm, dtype=complex_dtype),
            }
        )
        return self._device_cache

    def clear_device_cache(self) -> None:
        """Release lazily cached JAX plan arrays.

        This is mainly useful for long-running processes that construct many
        distinct plans and want explicit control over device memory.
        """

        self._device_cache.clear()

    def apply_indexed_and_accumulate(
        self,
        source_coefficients: np.ndarray,
        source_indices: np.ndarray,
        output_indices: np.ndarray,
        weights: np.ndarray,
        *,
        num_outputs: int,
        base: np.ndarray | None = None,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Shift indexed sources and accumulate directly into grouped outputs.

        Unlike :meth:`apply`, this method never materialises the complete
        ``(num_jobs, Nt, Nm)`` shifted batch on the host.  Each job chunk is
        gathered from a smaller source batch on-device, shifted with the
        existing validated kernels, and immediately scatter-added into the
        requested output groups.  Only the final grouped outputs are copied
        back to NumPy.

        Parameters
        ----------
        source_coefficients : ndarray, shape (num_sources, Nt, Nm)
            Distinct source coefficient arrays.  Several jobs may refer to the
            same source through ``source_indices``.
        source_indices : ndarray, shape (num_jobs,)
            Source row used by each prepared shift job.
        output_indices : ndarray, shape (num_jobs,)
            Output group receiving each shifted job.
        weights : ndarray, shape (num_jobs,)
            Scalar multiplier applied before accumulation.
        num_outputs : int
            Number of grouped output arrays.
        base : ndarray, optional, shape (num_outputs, Nt, Nm)
            Initial undelayed contribution to each output.
        cache_device_plan : bool
            Keep persistent delay/kernel arrays on the JAX device between
            calls.  This increases persistent device memory but avoids moving
            the plan kernels for every waveform evaluation.
        """

        import jax.numpy as jnp

        started = perf_counter()

        sources = np.asarray(source_coefficients)
        if sources.ndim != 3:
            raise ValueError(
                "source_coefficients must have shape (num_sources, Nt, Nm), "
                f"got {sources.shape}."
            )
        if sources.shape[1:] != (self.Nt, self.Nm):
            raise ValueError(
                "source_coefficients has incompatible WDM shape: "
                f"expected (*, {self.Nt}, {self.Nm}), got {sources.shape}."
            )
        if sources.shape[0] < 1:
            raise ValueError("source_coefficients must contain at least one source.")

        source_indices = np.asarray(source_indices, dtype=np.int64)
        output_indices = np.asarray(output_indices, dtype=np.int64)
        weights = np.asarray(weights)

        expected_vector_shape = (self.num_jobs,)
        for name, values in (
            ("source_indices", source_indices),
            ("output_indices", output_indices),
            ("weights", weights),
        ):
            if values.shape != expected_vector_shape:
                raise ValueError(
                    f"{name} must have shape {expected_vector_shape}, "
                    f"got {values.shape}."
                )

        num_outputs = int(num_outputs)
        if num_outputs < 1:
            raise ValueError("num_outputs must be >= 1.")
        if np.any(source_indices < 0) or np.any(source_indices >= sources.shape[0]):
            raise ValueError("source_indices contains an out-of-range source index.")
        if np.any(output_indices < 0) or np.any(output_indices >= num_outputs):
            raise ValueError("output_indices contains an out-of-range output index.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contains NaN or infinite values.")

        expected_base_shape = (num_outputs, self.Nt, self.Nm)
        if base is None:
            base_array = np.zeros(expected_base_shape, dtype=sources.dtype)
        else:
            base_array = np.asarray(base)
            if base_array.shape != expected_base_shape:
                raise ValueError(
                    f"base must have shape {expected_base_shape}, "
                    f"got {base_array.shape}."
                )

        complex_dtype = (
            jnp.complex64
            if self.resolved_assembly_precision == "complex64"
            else jnp.complex128
        )
        real_dtype = (
            jnp.float32
            if self.resolved_assembly_precision == "complex64"
            else jnp.float64
        )

        source_device = jnp.asarray(sources, dtype=complex_dtype)
        outputs_device = jnp.asarray(base_array, dtype=complex_dtype)
        source_indices_device = jnp.asarray(source_indices, dtype=jnp.int32)
        output_indices_device = jnp.asarray(output_indices, dtype=jnp.int32)
        weights_device = jnp.asarray(weights, dtype=complex_dtype)

        if cache_device_plan:
            device_plan = self._device_plan_arrays()
            delays_device = device_plan["delays"]
            ell_device = device_plan["ell_all"]
            Tl_device = device_plan["Tl_all"]
            Tp_device = device_plan["Tp_all"]
            Cnm_device = device_plan["Cnm"]
        else:
            delays_device = jnp.asarray(self.delays, dtype=real_dtype)
            ell_device = jnp.asarray(self.ell_all, dtype=jnp.int64)
            Tl_device = jnp.asarray(self.Tl_all, dtype=complex_dtype)
            Tp_device = jnp.asarray(self.Tp_all, dtype=complex_dtype)
            Cnm_device = jnp.asarray(self.Cnm, dtype=complex_dtype)

        chunk_size = (
            self.num_jobs
            if self.config.batch_chunk is None
            else min(self.num_jobs, int(self.config.batch_chunk))
        )

        for start in range(0, self.num_jobs, chunk_size):
            stop = min(start + chunk_size, self.num_jobs)
            true_batch = stop - start

            coefficient_work = source_device[
                source_indices_device[start:stop]
            ]
            delay_work = delays_device[start:stop]
            Tl_work = Tl_device[start:stop]
            Tp_work = Tp_device[start:stop]

            should_pad = (
                self.resolved_use_jax
                and bool(self.config.assembly_vmap)
                and self.config.jax_pad_last_chunk
                and self.num_jobs > chunk_size
                and true_batch < chunk_size
            )

            if should_pad:
                pad_rows = chunk_size - true_batch
                coefficient_work = jnp.concatenate(
                    (
                        coefficient_work,
                        jnp.repeat(coefficient_work[-1:], pad_rows, axis=0),
                    ),
                    axis=0,
                )
                delay_work = jnp.concatenate(
                    (delay_work, jnp.repeat(delay_work[-1:], pad_rows, axis=0)),
                    axis=0,
                )
                Tl_work = jnp.concatenate(
                    (Tl_work, jnp.repeat(Tl_work[-1:], pad_rows, axis=0)),
                    axis=0,
                )
                Tp_work = jnp.concatenate(
                    (Tp_work, jnp.repeat(Tp_work[-1:], pad_rows, axis=0)),
                    axis=0,
                )

            shifted_device = _assemble_shift_target_batch_dispatch(
                self.wdm,
                coefficient_work,
                delay_work,
                ell_device,
                self.offset,
                Tl_work,
                Tp_work,
                Cnm=Cnm_device,
                use_jax=self.resolved_use_jax,
                assembly_backend=self.resolved_assembly_backend,
                assembly_precision=self.resolved_assembly_precision,
                row_chunk_size=self.config.row_chunk_size,
                lag_block_size=self.config.lag_block_size,
                job_block_size=self.config.job_block_size,
                assembly_vmap=self.config.assembly_vmap,
                return_device=True,
            )

            shifted_device = shifted_device[:true_batch]
            weighted = (
                weights_device[start:stop, None, None] * shifted_device
            )
            outputs_device = outputs_device.at[
                output_indices_device[start:stop]
            ].add(weighted)

        # This is the only device-to-host transfer of shifted results.
        result = np.asarray(outputs_device)
        total_seconds = perf_counter() - started

        if not return_profile:
            return result

        itemsize = np.dtype(
            np.complex64
            if self.resolved_assembly_precision == "complex64"
            else np.complex128
        ).itemsize
        materialized_bytes = self.num_jobs * self.Nt * self.Nm * itemsize
        returned_bytes = num_outputs * self.Nt * self.Nm * itemsize

        host_transfer_bytes_saved = max(0, materialized_bytes - returned_bytes)

        profile: dict[str, float | int | str | bool] = {
            "n_sources": int(sources.shape[0]),
            "n_jobs": int(self.num_jobs),
            "n_outputs": int(num_outputs),
            "batch_chunk": int(chunk_size),
            "assembly_backend": self.resolved_assembly_backend,
            "assembly_precision": self.resolved_assembly_precision,
            "total_s": float(total_seconds),
            "plan_build_s": float(self.build_seconds),
            "device_plan_cached": bool(cache_device_plan),
            "device_plan_memory_bytes": int(self.device_plan_memory_bytes),
            "full_materialized_output_bytes": int(materialized_bytes),
            "returned_output_bytes": int(returned_bytes),
            "host_transfer_bytes_saved": int(host_transfer_bytes_saved),
        }
        return result, profile

    def apply_one(self, coefficients: np.ndarray) -> np.ndarray:
        """Convenience method for a plan containing exactly one delay field."""

        if self.num_jobs != 1:
            raise ValueError(
                f"apply_one requires a one-job plan, but this plan has {self.num_jobs} jobs."
            )
        return self.apply(coefficients)[0]

    @property
    def device_plan_memory_bytes(self) -> int:
        """Device memory used when the persistent plan cache is populated."""

        return int(
            self.delays.nbytes
            + self.ell_all.nbytes
            + self.Tl_all.nbytes
            + self.Tp_all.nbytes
            + self.Cnm.nbytes
        )

    @property
    def kernel_memory_bytes(self) -> int:
        """Host memory occupied by persistent Tl/Tp and parity arrays."""

        return int(self.Tl_all.nbytes + self.Tp_all.nbytes + self.Cnm.nbytes)