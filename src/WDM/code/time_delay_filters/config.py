"""Configuration for reusable WDM variable-delay plans.

Only numerical choices that remain active in the maintained implementation are
exposed.  The small ``assembly_variant`` switch supports isolated performance
experiments without replacing the validated production backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TlTpMode = Literal["exact", "interp"]
InterpolationKind = Literal["linear", "cubic"]
AssemblyBackend = Literal["production", "reference"]
AssemblyVariant = Literal["baseline", "reordered"]
AssemblyPrecision = Literal["complex64", "complex128", "float32", "float64"]


@dataclass(frozen=True, slots=True)
class VariableShiftPlanConfig:
    """Numerical configuration for a reusable target-mode shift plan.

    ``production`` uses interpolated delay kernels and the analytic-parity JAX
    assembly.  ``reference`` uses exact delay kernels and the explicit
    checkerboard implementation for regression tests.
    """

    lag_truncation: int | None = None

    tl_tp_mode: TlTpMode = "exact"
    tl_tp_interp_points: int = 64
    tl_tp_interp_pad: float = 0.0
    tl_tp_interp_kind: InterpolationKind = "linear"

    assembly_backend: AssemblyBackend = "reference"
    assembly_variant: AssemblyVariant = "baseline"
    assembly_precision: AssemblyPrecision = "complex128"
    row_chunk_size: int = 128
    lag_block_size: int = 1
    batch_chunk: int | None = 1

    def __post_init__(self) -> None:
        if self.lag_truncation is not None and self.lag_truncation < 0:
            raise ValueError("lag_truncation must be non-negative or None.")

        if self.tl_tp_mode not in ("exact", "interp"):
            raise ValueError("tl_tp_mode must be 'exact' or 'interp'.")

        if self.tl_tp_interp_points < 2:
            raise ValueError("tl_tp_interp_points must be at least 2.")
        if self.tl_tp_interp_kind not in ("linear", "cubic"):
            raise ValueError("tl_tp_interp_kind must be 'linear' or 'cubic'.")
        if self.tl_tp_interp_kind == "cubic" and self.tl_tp_interp_points < 4:
            raise ValueError(
                "Cubic Tl/Tp interpolation requires at least 4 grid points."
            )
        if self.tl_tp_interp_pad < 0.0:
            raise ValueError("tl_tp_interp_pad must be non-negative.")

        if self.assembly_backend not in ("production", "reference"):
            raise ValueError(
                "assembly_backend must be 'production' or 'reference'."
            )
        if self.assembly_variant not in ("baseline", "reordered"):
            raise ValueError(
                "assembly_variant must be 'baseline' or 'reordered'."
            )
        if (
            self.assembly_backend == "reference"
            and self.assembly_variant != "baseline"
        ):
            raise ValueError(
                "The reference backend supports only "
                "assembly_variant='baseline'."
            )
        if str(self.assembly_precision).lower() not in (
            "complex64",
            "complex128",
            "float32",
            "float64",
        ):
            raise ValueError(
                "assembly_precision must be complex64/float32 or "
                "complex128/float64."
            )

        if self.row_chunk_size < 1:
            raise ValueError("row_chunk_size must be at least 1.")
        if self.lag_block_size < 1:
            raise ValueError("lag_block_size must be at least 1.")
        if self.batch_chunk is not None and self.batch_chunk < 1:
            raise ValueError("batch_chunk must be at least 1 or None.")

    @classmethod
    def production(
        cls,
        *,
        lag_truncation: int = 25,
        interpolation_points: int = 16,
        row_chunk_size: int = 2048,
        lag_block_size: int = 17,
        batch_chunk: int | None = 32,
        assembly_variant: AssemblyVariant = "baseline",
    ) -> "VariableShiftPlanConfig":
        """Return the maintained fast configuration.

        The row and lag blocks remain explicit because their optimum is
        hardware- and problem-size dependent.
        """

        return cls(
            lag_truncation=lag_truncation,
            tl_tp_mode="interp",
            tl_tp_interp_points=interpolation_points,
            tl_tp_interp_kind="linear",
            assembly_backend="production",
            assembly_variant=assembly_variant,
            assembly_precision="complex64",
            row_chunk_size=row_chunk_size,
            lag_block_size=lag_block_size,
            batch_chunk=batch_chunk,
        )

    @classmethod
    def reference(
        cls,
        *,
        lag_truncation: int | None = None,
        batch_chunk: int | None = 1,
    ) -> "VariableShiftPlanConfig":
        """Return the high-accuracy regression configuration."""

        return cls(
            lag_truncation=lag_truncation,
            tl_tp_mode="exact",
            assembly_backend="reference",
            assembly_precision="complex128",
            row_chunk_size=128,
            lag_block_size=1,
            batch_chunk=batch_chunk,
        )
