"""Configuration objects for reusable WDM time-shift plans.

This module deliberately contains no LISA-specific concepts.  It belongs in
``WDM/code/time_delay_filters`` and describes only how a generic variable WDM
shift is prepared and assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TlTpMode = Literal["exact", "interp"]
InterpolationKind = Literal["linear", "cubic"]
InterpolationBackend = Literal["numpy", "jax", "auto"]
AssemblyPrecision = Literal["complex64", "complex128", "float32", "float64"]


@dataclass(frozen=True, slots=True)
class VariableShiftPlanConfig:
    """Numerical configuration for a reusable variable-delay shift plan.

    The defaults mirror the existing ``wdm_time_shift_variable_batch`` API so
    introducing the plan object does not silently change numerical behaviour.
    Use :meth:`production` for the faster configuration currently used by the
    LISA response pipeline.
    """

    lag_truncation: int | None = None

    tl_tp_mode: TlTpMode = "exact"
    tl_tp_interp_points: int = 64
    tl_tp_interp_pad: float = 0.0
    tl_tp_interp_kind: InterpolationKind = "linear"
    tl_tp_interp_backend: InterpolationBackend = "numpy"

    assembly_backend: str | None = None
    assembly_precision: AssemblyPrecision = "complex64"
    row_chunk_size: int = 128
    lag_block_size: int = 1
    job_block_size: int = 1
    assembly_vmap: bool | None = None

    batch_chunk: int | None = 32
    jax_pad_last_chunk: bool = False
    use_jax: bool | None = None

    def __post_init__(self) -> None:
        if self.lag_truncation is not None and self.lag_truncation < 0:
            raise ValueError("lag_truncation must be non-negative or None.")

        if self.tl_tp_mode not in ("exact", "interp"):
            raise ValueError("tl_tp_mode must be 'exact' or 'interp'.")

        if self.tl_tp_interp_points < 2:
            raise ValueError("tl_tp_interp_points must be at least 2.")

        if self.tl_tp_interp_kind == "cubic" and self.tl_tp_interp_points < 4:
            raise ValueError(
                "Cubic Tl/Tp interpolation requires at least 4 grid points."
            )

        if self.tl_tp_interp_pad < 0.0:
            raise ValueError("tl_tp_interp_pad must be non-negative.")

        if self.tl_tp_interp_backend not in ("numpy", "jax", "auto"):
            raise ValueError(
                "tl_tp_interp_backend must be 'numpy', 'jax', or 'auto'."
            )

        if self.row_chunk_size < 1:
            raise ValueError("row_chunk_size must be at least 1.")

        if self.lag_block_size < 1:
            raise ValueError("lag_block_size must be at least 1.")

        if self.job_block_size < 1:
            raise ValueError("job_block_size must be at least 1.")

        if self.batch_chunk is not None and self.batch_chunk < 1:
            raise ValueError("batch_chunk must be at least 1 or None.")

    @classmethod
    def production(
        cls,
        *,
        lag_truncation: int = 25,
        interpolation_points: int = 16,
    ) -> "VariableShiftPlanConfig":
        """Return the current fast interpolated configuration.

        This is intentionally an explicit alternative to the legacy-matching
        defaults above.
        """

        return cls(
            lag_truncation=lag_truncation,
            tl_tp_mode="interp",
            tl_tp_interp_points=interpolation_points,
            tl_tp_interp_kind="linear",
            tl_tp_interp_backend="numpy",
            assembly_backend="lagfirst_chunked_lagblock",
            assembly_precision="complex64",
            row_chunk_size=2048,
            lag_block_size=32,
            job_block_size=1,
            batch_chunk=32,
        )

    @classmethod
    def reference(
        cls,
        *,
        lag_truncation: int | None = None,
    ) -> "VariableShiftPlanConfig":
        """Return a high-accuracy configuration for regression testing."""

        return cls(
            lag_truncation=lag_truncation,
            tl_tp_mode="exact",
            assembly_backend="lagfirst_chunked",
            assembly_precision="complex128",
            row_chunk_size=128,
            lag_block_size=1,
            job_block_size=1,
            batch_chunk=1,
        )