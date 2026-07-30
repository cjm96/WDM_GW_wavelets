"""Reusable plans for target-mode WDM variable time shifts.

A plan separates expensive delay-dependent preparation from repeated waveform
application.  The maintained plan supports one production target kernel and
one high-precision reference kernel.  Experimental prephased, weighted-source
and job-block variants were removed from the public object.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from ._time_shift_assembly import _assemble_shift_target_batch_dispatch
from .config import VariableShiftPlanConfig
from .time_shift_fast import (
    _build_signed_lag_idx,
    _build_TlTp_from_shift_matrix,
    _build_TlTp_from_shift_matrix_interp,
    TlTpInterpolationTable,
    build_tl_tp_interpolation_table,
    _get_Cnm_parity,
    _get_kernel_precomputes,
    _infer_Nf,
    _normalize_assembly_backend,
    _normalize_assembly_precision,
    _resolve_ell_range,
    _validate_lag_block_size,
    _validate_row_chunk_size,
    choose_Nker,
)


def _as_delay_matrix(delays: np.ndarray) -> np.ndarray:
    """Validate and return a float64 delay matrix for plan construction."""

    delays = np.asarray(delays, dtype=np.float64)
    if delays.ndim == 1:
        delays = delays[None, :]
    if delays.ndim != 2:
        raise ValueError(
            "delays must have shape (num_jobs, Nt), or (Nt,) for one job."
        )
    if delays.shape[0] < 1 or delays.shape[1] < 1:
        raise ValueError("delays must contain at least one job and one row.")
    if not np.all(np.isfinite(delays)):
        raise ValueError("delays contains NaN or infinite values.")
    return delays


def _coefficient_batch_shape(values, *, num_jobs, Nt, Nm):
    """Normalize a host coefficient batch without changing its dtype."""

    values = np.asarray(values)
    if values.ndim == 2:
        values = values[None, :, :]
    expected = (num_jobs, Nt, Nm)
    if values.shape != expected:
        raise ValueError(
            f"Expected coefficients with shape {expected}, got {values.shape}."
        )
    return values


def _jax_coefficient_dtype(values, precision):
    """Choose a real or complex JAX coefficient dtype."""

    import jax.numpy as jnp

    is_complex = bool(
        jnp.issubdtype(jnp.asarray(values).dtype, jnp.complexfloating)
    )
    if precision == "complex64":
        return jnp.complex64 if is_complex else jnp.float32
    return jnp.complex128 if is_complex else jnp.float64



@dataclass(frozen=True, slots=True)
class VariableShiftKernelContext:
    """Reusable variable-shift kernel state for runtime delay fields.

    Unlike :class:`VariableShiftBatchPlan`, this object does not bind a fixed
    delay matrix. It stores the WDM/lag configuration and one global ``Tl/Tp``
    interpolation table, allowing sky-dependent delays to be supplied during
    repeated response evaluations.
    """

    wdm: Any
    config: VariableShiftPlanConfig
    ell_all: np.ndarray
    offset: int
    Nf: int
    Nker: int
    Nt: int
    Nm: int
    interpolation_table: TlTpInterpolationTable
    resolved_assembly_backend: str
    resolved_assembly_precision: str
    build_seconds: float

    _device_cache: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @classmethod
    def build(
        cls,
        wdm: Any,
        *,
        delay_min: float,
        delay_max: float,
        config: VariableShiftPlanConfig | None = None,
        Nf: int | None = None,
        Nker: int | None = None,
        safety: float = 1.02,
        kernel_kwargs: dict[str, Any] | None = None,
        interpolation_points: int = 1024,
        interpolation_pad: float = 0.02,
        interpolation_kind: str | None = None,
    ) -> "VariableShiftKernelContext":
        """Build reusable production kernels for delays supplied at runtime.

        Parameters
        ----------
        wdm : object
            WDM transform defining the coefficient grid and basis windows.
        delay_min, delay_max : float
            Inclusive delay interval, in seconds, that must contain every runtime
            delay value.
        config : VariableShiftPlanConfig or None, optional
            Production configuration. The reference backend is rejected because this
            context is designed for interpolated runtime delays.
        Nf, Nker, safety, kernel_kwargs : optional
            Kernel-WDM construction controls.
        interpolation_points : int, optional
            Number of samples in the reusable delay-only ``Tl/Tp`` table.
        interpolation_pad : float, optional
            Fractional extension of the requested delay interval.
        interpolation_kind : {'linear', 'cubic'} or None, optional
            Interpolant; ``None`` uses ``config.tl_tp_interp_kind``.

        Returns
        -------
        VariableShiftKernelContext
            Lag metadata, interpolation table and assembly configuration reusable for
            arbitrary delay matrices of shape ``(num_jobs, Nt)``.

        Raises
        ------
        ValueError
            If the reference backend is requested or interpolation settings are
            invalid.
        """
        started = perf_counter()
        config = VariableShiftPlanConfig.production() if config is None else config
        backend = _normalize_assembly_backend(config.assembly_backend)
        precision = _normalize_assembly_precision(config.assembly_precision)
        if backend == "reference":
            raise ValueError(
                "VariableShiftKernelContext is intended for interpolated "
                "production runtime delays; use VariableShiftBatchPlan for "
                "the exact reference backend."
            )
        _validate_row_chunk_size(config.row_chunk_size)
        _validate_lag_block_size(config.lag_block_size)

        kind = (
            config.tl_tp_interp_kind
            if interpolation_kind is None
            else str(interpolation_kind)
        )
        table, ell_all, offset, resolved_Nker = build_tl_tp_interpolation_table(
            wdm,
            delay_min=delay_min,
            delay_max=delay_max,
            L_trunc=config.lag_truncation,
            Nf=Nf,
            Nker=Nker,
            safety=safety,
            kernel_kwargs=kernel_kwargs,
            interpolation_points=interpolation_points,
            interpolation_pad=interpolation_pad,
            interpolation_kind=kind,
        )
        return cls(
            wdm=wdm,
            config=config,
            ell_all=np.asarray(ell_all, dtype=np.int32),
            offset=int(offset),
            Nf=int(wdm.Nf if Nf is None else Nf),
            Nker=int(resolved_Nker),
            Nt=int(wdm.Nt),
            Nm=int(wdm.Nf),
            interpolation_table=table,
            resolved_assembly_backend=backend,
            resolved_assembly_precision=precision,
            build_seconds=float(perf_counter() - started),
        )

    def _validate_delays(self, delays: np.ndarray) -> np.ndarray:
        values = _as_delay_matrix(delays)
        if values.shape[1] != self.Nt:
            raise ValueError(
                f"Expected runtime delays with Nt={self.Nt}, got {values.shape}."
            )
        # Reuse the table's explicit host-side range validation.
        self.interpolation_table._validate_delays(values)
        return values

    def _validate_coefficients(self, coefficients, num_jobs):
        return _coefficient_batch_shape(
            coefficients,
            num_jobs=num_jobs,
            Nt=self.Nt,
            Nm=self.Nm,
        )

    def apply(
        self,
        coefficients: np.ndarray,
        delays: np.ndarray,
        *,
        return_profile: bool = False,
    ):
        """Interpolate runtime delay kernels and return a NumPy batch."""

        delays = self._validate_delays(delays)
        coefficients = self._validate_coefficients(coefficients, delays.shape[0])
        started = perf_counter()
        interp_started = perf_counter()
        Tl_all, Tp_all = self.interpolation_table.evaluate(delays)
        interpolation_seconds = perf_counter() - interp_started

        assembly_started = perf_counter()
        shifted = _assemble_shift_target_batch_dispatch(
            self.wdm,
            coefficients,
            delays,
            self.ell_all,
            self.offset,
            Tl_all,
            Tp_all,
            Cnm=None,
            assembly_backend=self.resolved_assembly_backend,
            assembly_precision=self.resolved_assembly_precision,
            row_chunk_size=self.config.row_chunk_size,
            lag_block_size=self.config.lag_block_size,
        )
        assembly_seconds = perf_counter() - assembly_started
        result = np.asarray(shifted)
        if not return_profile:
            return result
        return result, {
            "n_jobs": int(delays.shape[0]),
            "runtime_interpolation_s": float(interpolation_seconds),
            "assembly_s": float(assembly_seconds),
            "total_s": float(perf_counter() - started),
            "kernel_context_build_s": float(self.build_seconds),
            "interpolation_table_memory_bytes": int(
                self.interpolation_table.memory_bytes
            ),
            "assembly_backend": self.resolved_assembly_backend,
            "assembly_precision": self.resolved_assembly_precision,
            "returned_on_device": False,
        }

    def _device_static_arrays(self):
        import jax.numpy as jnp
        if not self._device_cache:
            if self.resolved_assembly_precision == "complex64":
                complex_dtype = jnp.complex64
            else:
                complex_dtype = jnp.complex128
            self._device_cache.update(
                {
                    "ell_all": jnp.asarray(self.ell_all, dtype=jnp.int32),
                    "Tl_grid": jnp.asarray(
                        self.interpolation_table.Tl_grid,
                        dtype=complex_dtype,
                    ),
                    "Tp_grid": jnp.asarray(
                        self.interpolation_table.Tp_grid,
                        dtype=complex_dtype,
                    ),
                }
            )
        return self._device_cache

    def apply_device(
        self,
        coefficients,
        delays,
        *,
        return_profile: bool = False,
        cache_device_context: bool = True,
    ):
        """Interpolate runtime delays and assemble entirely on the JAX device."""

        import jax.numpy as jnp
        started = perf_counter()
        host_delays = self._validate_delays(np.asarray(delays))
        coefficient_dtype = _jax_coefficient_dtype(
            coefficients,
            self.resolved_assembly_precision,
        )
        coefficients_device = jnp.asarray(coefficients, dtype=coefficient_dtype)
        if coefficients_device.ndim == 2:
            coefficients_device = coefficients_device[None, :, :]
        expected = (host_delays.shape[0], self.Nt, self.Nm)
        if tuple(coefficients_device.shape) != expected:
            raise ValueError(
                f"Expected coefficients with shape {expected}, "
                f"got {tuple(coefficients_device.shape)}."
            )

        if self.resolved_assembly_precision == "complex64":
            real_dtype = jnp.float32
            complex_dtype = jnp.complex64
        else:
            real_dtype = jnp.float64
            complex_dtype = jnp.complex128
        delay_device = jnp.asarray(host_delays, dtype=real_dtype)

        interp_started = perf_counter()
        Tl_all, Tp_all = self.interpolation_table.evaluate_device(
            delay_device,
            complex_dtype=complex_dtype,
        )
        interpolation_seconds = perf_counter() - interp_started
        ell_all = (
            self._device_static_arrays()["ell_all"]
            if cache_device_context
            else jnp.asarray(self.ell_all, dtype=jnp.int32)
        )

        assembly_started = perf_counter()
        result = _assemble_shift_target_batch_dispatch(
            self.wdm,
            coefficients_device,
            delay_device,
            ell_all,
            self.offset,
            Tl_all,
            Tp_all,
            Cnm=None,
            assembly_backend=self.resolved_assembly_backend,
            assembly_precision=self.resolved_assembly_precision,
            row_chunk_size=self.config.row_chunk_size,
            lag_block_size=self.config.lag_block_size,
            return_device=True,
        )
        if return_profile:
            result.block_until_ready()
        assembly_seconds = perf_counter() - assembly_started
        if not return_profile:
            return result
        return result, {
            "n_jobs": int(host_delays.shape[0]),
            "runtime_interpolation_s": float(interpolation_seconds),
            "assembly_s": float(assembly_seconds),
            "total_s": float(perf_counter() - started),
            "kernel_context_build_s": float(self.build_seconds),
            "interpolation_table_memory_bytes": int(
                self.interpolation_table.memory_bytes
            ),
            "assembly_backend": self.resolved_assembly_backend,
            "assembly_precision": self.resolved_assembly_precision,
            "returned_on_device": True,
            "device_context_cached": bool(cache_device_context),
        }

    def build_fixed_plan(self, delays: np.ndarray) -> "VariableShiftBatchPlan":
        """Materialise a conventional fixed-delay plan from this context."""

        return VariableShiftBatchPlan.build_from_kernel_context(self, delays)

    def clear_device_cache(self) -> None:
        """Release cached device copies of lag and interpolation-table arrays."""
        self._device_cache.clear()

    @property
    def kernel_memory_bytes(self) -> int:
        """Bytes occupied by the persistent delay interpolation table."""
        return int(self.interpolation_table.memory_bytes)


@dataclass(frozen=True, slots=True)
class VariableShiftBatchPlan:
    """Prepared target-mode WDM operators for one or more delay fields."""

    wdm: Any
    config: VariableShiftPlanConfig

    delays: np.ndarray
    ell_all: np.ndarray
    offset: int
    Tl_all: np.ndarray
    Tp_all: np.ndarray
    Cnm: np.ndarray | None

    Nf: int
    Nker: int
    Nt: int
    Nm: int
    num_jobs: int

    resolved_assembly_backend: str
    resolved_assembly_precision: str
    build_seconds: float

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
        """Prepare all delay-dependent data for repeated batch application.

        Interpolated kernels retain the historical per-``batch_chunk`` delay
        range so this refactor does not silently change interpolation values.
        Persistent arrays are stored directly in the configured production or
        reference precision.
        """

        started = perf_counter()
        config = VariableShiftPlanConfig() if config is None else config
        construction_delays = _as_delay_matrix(delays)

        num_jobs, Nt = construction_delays.shape
        Nm = int(getattr(wdm, "Nf"))
        Nf = _infer_Nf(wdm, Nt, Nf)
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
        signed_lag_idx = _build_signed_lag_idx(
            ell_all,
            Nf,
            Nker,
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
            delay_chunk = construction_delays[start:stop]

            if config.tl_tp_mode == "exact":
                Tl_chunk, Tp_chunk = _build_TlTp_from_shift_matrix(
                    delay_chunk,
                    freqs_u,
                    W0_u,
                    W1_u,
                    scale,
                    signed_lag_idx,
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

            Tl_parts.append(np.asarray(Tl_chunk))
            Tp_parts.append(np.asarray(Tp_chunk))

        Tl_all = np.concatenate(Tl_parts, axis=0)
        Tp_all = np.concatenate(Tp_parts, axis=0)
        expected_kernel_shape = (num_jobs, Nt, ell_all.size)
        if (
            Tl_all.shape != expected_kernel_shape
            or Tp_all.shape != expected_kernel_shape
        ):
            raise RuntimeError(
                "Unexpected Tl/Tp shape after plan construction: "
                f"expected {expected_kernel_shape}, got "
                f"Tl={Tl_all.shape}, Tp={Tp_all.shape}."
            )

        backend = _normalize_assembly_backend(config.assembly_backend)
        precision = _normalize_assembly_precision(config.assembly_precision)
        _validate_row_chunk_size(config.row_chunk_size)
        _validate_lag_block_size(config.lag_block_size)

        if precision == "complex64":
            real_dtype = np.float32
            complex_dtype = np.complex64
        else:
            real_dtype = np.float64
            complex_dtype = np.complex128

        # The production analytic-parity kernel does not read Cnm.  Building
        # and caching the full checkerboard is therefore restricted to the
        # explicit reference plan.
        Cnm = (
            _get_Cnm_parity(Nt, Nm, dtype=np.complex128)
            if backend == "reference"
            else None
        )

        return cls(
            wdm=wdm,
            config=config,
            delays=np.asarray(construction_delays, dtype=real_dtype),
            ell_all=np.asarray(ell_all, dtype=np.int32),
            offset=int(offset),
            Tl_all=np.asarray(Tl_all, dtype=complex_dtype),
            Tp_all=np.asarray(Tp_all, dtype=complex_dtype),
            Cnm=Cnm,
            Nf=int(Nf),
            Nker=Nker,
            Nt=Nt,
            Nm=Nm,
            num_jobs=num_jobs,
            resolved_assembly_backend=backend,
            resolved_assembly_precision=precision,
            build_seconds=float(perf_counter() - started),
        )

    @classmethod
    def build_from_kernel_context(
        cls,
        kernel_context: VariableShiftKernelContext,
        delays: np.ndarray,
    ) -> "VariableShiftBatchPlan":
        """Materialise fixed ``Tl/Tp`` arrays using a reusable kernel context."""

        started = perf_counter()
        construction_delays = kernel_context._validate_delays(delays)
        Tl_all, Tp_all = kernel_context.interpolation_table.evaluate(
            construction_delays
        )
        precision = kernel_context.resolved_assembly_precision
        if precision == "complex64":
            real_dtype = np.float32
            complex_dtype = np.complex64
        else:
            real_dtype = np.float64
            complex_dtype = np.complex128
        return cls(
            wdm=kernel_context.wdm,
            config=kernel_context.config,
            delays=np.asarray(construction_delays, dtype=real_dtype),
            ell_all=np.asarray(kernel_context.ell_all, dtype=np.int32),
            offset=int(kernel_context.offset),
            Tl_all=np.asarray(Tl_all, dtype=complex_dtype),
            Tp_all=np.asarray(Tp_all, dtype=complex_dtype),
            Cnm=None,
            Nf=int(kernel_context.Nf),
            Nker=int(kernel_context.Nker),
            Nt=int(kernel_context.Nt),
            Nm=int(kernel_context.Nm),
            num_jobs=int(construction_delays.shape[0]),
            resolved_assembly_backend=kernel_context.resolved_assembly_backend,
            resolved_assembly_precision=precision,
            build_seconds=float(perf_counter() - started),
        )

    def apply(
        self,
        coefficients: np.ndarray,
        *,
        return_profile: bool = False,
    ):
        """Apply the plan and return a NumPy batch."""

        coefficients = _coefficient_batch_shape(
            coefficients,
            num_jobs=self.num_jobs,
            Nt=self.Nt,
            Nm=self.Nm,
        )
        started = perf_counter()
        chunk_size = self._chunk_size
        outputs = []
        assembly_seconds = 0.0

        for start in range(0, self.num_jobs, chunk_size):
            stop = min(start + chunk_size, self.num_jobs)
            assembly_started = perf_counter()
            shifted = _assemble_shift_target_batch_dispatch(
                self.wdm,
                coefficients[start:stop],
                self.delays[start:stop],
                self.ell_all,
                self.offset,
                self.Tl_all[start:stop],
                self.Tp_all[start:stop],
                Cnm=self.Cnm,
                assembly_backend=self.resolved_assembly_backend,
                assembly_precision=self.resolved_assembly_precision,
                row_chunk_size=self.config.row_chunk_size,
                lag_block_size=self.config.lag_block_size,
            )
            assembly_seconds += perf_counter() - assembly_started
            outputs.append(np.asarray(shifted))

        result = outputs[0] if len(outputs) == 1 else np.concatenate(outputs)
        if not return_profile:
            return result

        total_seconds = perf_counter() - started
        return result, self._profile(
            total_seconds=total_seconds,
            assembly_seconds=assembly_seconds,
            returned_on_device=False,
            device_plan_cached=False,
        )

    def apply_device(
        self,
        coefficients,
        *,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Apply the plan and keep the result on the active JAX device."""

        import jax.numpy as jnp

        started = perf_counter()
        coefficient_dtype = _jax_coefficient_dtype(
            coefficients,
            self.resolved_assembly_precision,
        )
        coefficients_device = jnp.asarray(
            coefficients,
            dtype=coefficient_dtype,
        )
        if coefficients_device.ndim == 2:
            coefficients_device = coefficients_device[None, :, :]
        self._validate_device_coefficient_shape(coefficients_device)

        plan = self._device_plan_arrays() if cache_device_plan else self._new_device_plan()
        outputs = []
        assembly_started = perf_counter()

        for start in range(0, self.num_jobs, self._chunk_size):
            stop = min(start + self._chunk_size, self.num_jobs)
            shifted = _assemble_shift_target_batch_dispatch(
                self.wdm,
                coefficients_device[start:stop],
                plan["delays"][start:stop],
                plan["ell_all"],
                self.offset,
                plan["Tl_all"][start:stop],
                plan["Tp_all"][start:stop],
                Cnm=plan.get("Cnm"),
                assembly_backend=self.resolved_assembly_backend,
                assembly_precision=self.resolved_assembly_precision,
                row_chunk_size=self.config.row_chunk_size,
                lag_block_size=self.config.lag_block_size,
                return_device=True,
            )
            outputs.append(shifted)

        result = outputs[0] if len(outputs) == 1 else jnp.concatenate(outputs)
        if return_profile:
            result.block_until_ready()

        assembly_seconds = perf_counter() - assembly_started
        if not return_profile:
            return result

        total_seconds = perf_counter() - started
        return result, self._profile(
            total_seconds=total_seconds,
            assembly_seconds=assembly_seconds,
            returned_on_device=True,
            device_plan_cached=cache_device_plan,
        )

    def apply_device_parallel_groups(
        self,
        coefficients,
        *,
        group_size: int,
        max_workers: int | None = None,
        return_profile: bool = False,
        cache_device_plan: bool = True,
    ):
        """Apply contiguous job groups concurrently on CPU-backed JAX.

        This is the retained coarse-grained one-arm execution strategy.  Every
        worker synchronizes before returning so timings represent completed
        execution rather than asynchronous dispatch.
        """

        import jax.numpy as jnp

        group_size = int(group_size)
        if group_size < 1 or group_size > self.num_jobs:
            raise ValueError(
                f"group_size must lie in [1, {self.num_jobs}], got {group_size}."
            )
        bounds = tuple(
            (start, min(start + group_size, self.num_jobs))
            for start in range(0, self.num_jobs, group_size)
        )
        workers = len(bounds) if max_workers is None else int(max_workers)
        if workers < 1:
            raise ValueError("max_workers must be >= 1.")
        workers = min(workers, len(bounds))

        started = perf_counter()
        coefficient_dtype = _jax_coefficient_dtype(
            coefficients,
            self.resolved_assembly_precision,
        )
        coefficients_device = jnp.asarray(coefficients, dtype=coefficient_dtype)
        if coefficients_device.ndim == 2:
            coefficients_device = coefficients_device[None, :, :]
        self._validate_device_coefficient_shape(coefficients_device)

        plan = self._device_plan_arrays() if cache_device_plan else self._new_device_plan()

        def run_group(group_bounds):
            group_start, group_stop = group_bounds
            group_started = perf_counter()
            shifted = _assemble_shift_target_batch_dispatch(
                self.wdm,
                coefficients_device[group_start:group_stop],
                plan["delays"][group_start:group_stop],
                plan["ell_all"],
                self.offset,
                plan["Tl_all"][group_start:group_stop],
                plan["Tp_all"][group_start:group_stop],
                Cnm=plan.get("Cnm"),
                assembly_backend=self.resolved_assembly_backend,
                assembly_precision=self.resolved_assembly_precision,
                row_chunk_size=self.config.row_chunk_size,
                lag_block_size=self.config.lag_block_size,
                return_device=True,
            )
            shifted.block_until_ready()
            return shifted, float(perf_counter() - group_started)

        assembly_started = perf_counter()
        if len(bounds) == 1:
            completed = [run_group(bounds[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="wdm-shift-group",
            ) as executor:
                completed = list(executor.map(run_group, bounds))

        outputs = [item[0] for item in completed]
        group_seconds = tuple(item[1] for item in completed)
        result = outputs[0] if len(outputs) == 1 else jnp.concatenate(outputs)
        result.block_until_ready()

        assembly_seconds = perf_counter() - assembly_started
        if not return_profile:
            return result

        total_seconds = perf_counter() - started
        profile = self._profile(
            total_seconds=total_seconds,
            assembly_seconds=assembly_seconds,
            returned_on_device=True,
            device_plan_cached=cache_device_plan,
        )
        profile.update(
            {
                "parallel_group_size": group_size,
                "parallel_num_groups": len(bounds),
                "parallel_max_workers": workers,
                "parallel_group_bounds": bounds,
                "parallel_group_seconds": group_seconds,
                "execution_synchronous": True,
            }
        )
        return result, profile

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
        """Gather repeated sources, shift by job, and accumulate by output.

        Only the grouped outputs are copied to the host.  The method is retained
        for the later TDI layer; it is not part of one-arm source construction.
        """

        import jax.numpy as jnp

        started = perf_counter()
        sources = np.asarray(source_coefficients)
        if sources.ndim != 3 or sources.shape[1:] != (self.Nt, self.Nm):
            raise ValueError(
                "source_coefficients must have shape "
                f"(num_sources, {self.Nt}, {self.Nm})."
            )
        if sources.shape[0] < 1:
            raise ValueError("At least one source is required.")

        source_indices = np.asarray(source_indices, dtype=np.int32)
        output_indices = np.asarray(output_indices, dtype=np.int32)
        weights = np.asarray(weights)
        expected_vector = (self.num_jobs,)
        for name, values in (
            ("source_indices", source_indices),
            ("output_indices", output_indices),
            ("weights", weights),
        ):
            if values.shape != expected_vector:
                raise ValueError(
                    f"{name} must have shape {expected_vector}, got {values.shape}."
                )

        num_outputs = int(num_outputs)
        if num_outputs < 1:
            raise ValueError("num_outputs must be >= 1.")
        if np.any(source_indices < 0) or np.any(source_indices >= sources.shape[0]):
            raise ValueError("source_indices contains an out-of-range index.")
        if np.any(output_indices < 0) or np.any(output_indices >= num_outputs):
            raise ValueError("output_indices contains an out-of-range index.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contains NaN or infinite values.")

        expected_base = (num_outputs, self.Nt, self.Nm)
        if base is None:
            base_array = np.zeros(expected_base, dtype=sources.dtype)
        else:
            base_array = np.asarray(base)
            if base_array.shape != expected_base:
                raise ValueError(
                    f"base must have shape {expected_base}, got {base_array.shape}."
                )

        requires_complex = any(
            np.issubdtype(np.asarray(values).dtype, np.complexfloating)
            for values in (sources, weights, base_array)
        )
        if self.resolved_assembly_precision == "complex64":
            coefficient_dtype = jnp.complex64 if requires_complex else jnp.float32
        else:
            coefficient_dtype = jnp.complex128 if requires_complex else jnp.float64

        source_device = jnp.asarray(sources, dtype=coefficient_dtype)
        outputs_device = jnp.asarray(base_array, dtype=coefficient_dtype)
        source_index_device = jnp.asarray(source_indices, dtype=jnp.int32)
        output_index_device = jnp.asarray(output_indices, dtype=jnp.int32)
        weights_device = jnp.asarray(weights, dtype=coefficient_dtype)
        plan = self._device_plan_arrays() if cache_device_plan else self._new_device_plan()

        for start in range(0, self.num_jobs, self._chunk_size):
            stop = min(start + self._chunk_size, self.num_jobs)
            shifted = _assemble_shift_target_batch_dispatch(
                self.wdm,
                source_device[source_index_device[start:stop]],
                plan["delays"][start:stop],
                plan["ell_all"],
                self.offset,
                plan["Tl_all"][start:stop],
                plan["Tp_all"][start:stop],
                Cnm=plan.get("Cnm"),
                assembly_backend=self.resolved_assembly_backend,
                assembly_precision=self.resolved_assembly_precision,
                row_chunk_size=self.config.row_chunk_size,
                lag_block_size=self.config.lag_block_size,
                return_device=True,
            )
            weighted = weights_device[start:stop, None, None] * shifted
            outputs_device = outputs_device.at[
                output_index_device[start:stop]
            ].add(weighted)

        result = np.asarray(outputs_device)
        if not return_profile:
            return result

        itemsize = np.dtype(
            np.complex64
            if self.resolved_assembly_precision == "complex64" and requires_complex
            else np.float32
            if self.resolved_assembly_precision == "complex64"
            else np.complex128
            if requires_complex
            else np.float64
        ).itemsize
        materialized_bytes = self.num_jobs * self.Nt * self.Nm * itemsize
        returned_bytes = num_outputs * self.Nt * self.Nm * itemsize
        return result, {
            "n_sources": sources.shape[0],
            "n_jobs": self.num_jobs,
            "n_outputs": num_outputs,
            "batch_chunk": self._chunk_size,
            "assembly_backend": self.resolved_assembly_backend,
            "assembly_precision": self.resolved_assembly_precision,
            "total_s": float(perf_counter() - started),
            "plan_build_s": self.build_seconds,
            "device_plan_cached": cache_device_plan,
            "device_plan_memory_bytes": self.device_plan_memory_bytes,
            "full_materialized_output_bytes": materialized_bytes,
            "returned_output_bytes": returned_bytes,
            "host_transfer_bytes_saved": max(
                0,
                materialized_bytes - returned_bytes,
            ),
        }

    def apply_one(self, coefficients: np.ndarray) -> np.ndarray:
        """Apply a plan containing exactly one delay field."""

        if self.num_jobs != 1:
            raise ValueError(
                "apply_one requires a one-job plan, but this plan has "
                f"{self.num_jobs} jobs."
            )
        return self.apply(coefficients)[0]

    def _validate_device_coefficient_shape(self, coefficients) -> None:
        expected = (self.num_jobs, self.Nt, self.Nm)
        if tuple(coefficients.shape) != expected:
            raise ValueError(
                f"Expected coefficients with shape {expected}, "
                f"got {tuple(coefficients.shape)}."
            )

    @property
    def _chunk_size(self) -> int:
        return (
            self.num_jobs
            if self.config.batch_chunk is None
            else min(self.num_jobs, int(self.config.batch_chunk))
        )

    def _new_device_plan(self) -> dict[str, Any]:
        import jax.numpy as jnp

        if self.resolved_assembly_precision == "complex64":
            real_dtype = jnp.float32
            complex_dtype = jnp.complex64
        else:
            real_dtype = jnp.float64
            complex_dtype = jnp.complex128

        plan = {
            "delays": jnp.asarray(self.delays, dtype=real_dtype),
            "ell_all": jnp.asarray(self.ell_all, dtype=jnp.int32),
            "Tl_all": jnp.asarray(self.Tl_all, dtype=complex_dtype),
            "Tp_all": jnp.asarray(self.Tp_all, dtype=complex_dtype),
        }
        if self.Cnm is not None:
            plan["Cnm"] = jnp.asarray(self.Cnm, dtype=jnp.complex128)
        return plan

    def _device_plan_arrays(self) -> dict[str, Any]:
        """Return lazily cached JAX copies of persistent plan arrays."""

        if not self._device_cache:
            self._device_cache.update(self._new_device_plan())
        return self._device_cache

    def clear_device_cache(self) -> None:
        """Release lazily cached JAX plan arrays."""

        self._device_cache.clear()

    def _profile(
        self,
        *,
        total_seconds,
        assembly_seconds,
        returned_on_device,
        device_plan_cached,
    ) -> dict[str, Any]:
        return {
            "n_jobs": self.num_jobs,
            "batch_chunk": self._chunk_size,
            "assembly_backend": self.resolved_assembly_backend,
            "assembly_precision": self.resolved_assembly_precision,
            "total_s": float(total_seconds),
            "assembly_s": float(assembly_seconds),
            "other_s": float(total_seconds - assembly_seconds),
            "plan_build_s": self.build_seconds,
            "device_plan_cached": bool(device_plan_cached),
            "device_plan_memory_bytes": self.device_plan_memory_bytes,
            "returned_on_device": bool(returned_on_device),
        }

    @property
    def device_plan_memory_bytes(self) -> int:
        """Persistent host/device payload used by the prepared plan."""

        arrays = [self.delays, self.ell_all, self.Tl_all, self.Tp_all]
        if self.Cnm is not None:
            arrays.append(self.Cnm)
        return int(sum(array.nbytes for array in arrays))

    @property
    def kernel_memory_bytes(self) -> int:
        """Host memory occupied by persistent shift-kernel arrays."""

        total = self.Tl_all.nbytes + self.Tp_all.nbytes
        if self.Cnm is not None:
            total += self.Cnm.nbytes
        return int(total)
