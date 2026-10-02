from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Role = Literal["bias", "dark", "flat", "science"]
ROLES: tuple[Role, ...] = ("bias", "dark", "flat", "science")

MASK_FINITE_INVALID = 1
MASK_SATURATED = 2
MASK_FLAT_NONPOSITIVE = 4
MASK_CALIBRATION_INVALID = 8

MASK_DESCRIPTIONS = {
    MASK_FINITE_INVALID: "input pixel is not finite",
    MASK_SATURATED: "raw pixel exceeds SATURATE",
    MASK_FLAT_NONPOSITIVE: "master-flat pixel is zero, negative, or invalid",
    MASK_CALIBRATION_INVALID: "bias or dark calibration pixel has no valid sample",
}


class CalibrationError(Exception):
    """Batch-level validation failure."""

    def __init__(self, message: str, file_errors: dict[str, str] | None = None):
        super().__init__(message)
        self.message = message
        self.file_errors = file_errors or {}


@dataclass(frozen=True)
class CalibrationParameters:
    sigma: float = 3.0
    maxiters: int = 5

    @classmethod
    def from_value(cls, value: Any) -> "CalibrationParameters":
        if value is None or value == "":
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise CalibrationError("calibration parameters must be valid JSON") from exc
        if not isinstance(value, dict):
            raise CalibrationError("calibration parameters must be a JSON object")
        try:
            result = cls(
                sigma=float(value.get("sigma", 3.0)),
                maxiters=int(value.get("maxiters", 5)),
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid calibration parameters: {exc}") from exc
        if not np.isfinite(result.sigma) or result.sigma <= 0:
            raise CalibrationError("sigma must be a finite positive number")
        if result.maxiters < 0:
            raise CalibrationError("maxiters must be zero or a positive integer")
        return result


@dataclass(frozen=True)
class ObservationContext:
    instrument: str
    temperature: float
    gain: float
    binning: tuple[int, int]
    shape: tuple[int, int]
    filter_name: str | None = None


@dataclass
class Exposure:
    role: Role
    filename: str
    data: NDArray[np.float64]
    exptime: float
    saturate: float
    file_hash: str
    context: ObservationContext
    header: Any

    def bad_input_mask(self) -> NDArray[np.bool_]:
        return (~np.isfinite(self.data)) | (self.data > self.saturate)


@dataclass
class CalibratedScience:
    source: Exposure
    data: NDArray[np.float64]
    mask: NDArray[np.uint8]


@dataclass
class CalibrationResult:
    master_bias: NDArray[np.float64]
    master_dark_rate: NDArray[np.float64]
    master_flat: NDArray[np.float64]
    sciences: list[CalibratedScience] = field(default_factory=list)
