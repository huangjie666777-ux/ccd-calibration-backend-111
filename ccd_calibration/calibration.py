from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clip
from numpy.typing import NDArray

from .models import (
    CalibratedScience,
    CalibrationError,
    CalibrationParameters,
    CalibrationResult,
    Exposure,
    MASK_CALIBRATION_INVALID,
    MASK_FINITE_INVALID,
    MASK_FLAT_NONPOSITIVE,
    MASK_SATURATED,
)


def _stack(items: list[Exposure]) -> NDArray[np.float64]:
    return np.stack([item.data for item in items], axis=0).astype(np.float64)


def _input_mask(items: list[Exposure]) -> NDArray[np.bool_]:
    mask = np.zeros((len(items), *items[0].data.shape), dtype=bool)
    for index, item in enumerate(items):
        mask[index] = (~np.isfinite(item.data)) | (item.data > item.saturate)
    return mask


def sigma_clipped_mean(
    values: NDArray[np.float64],
    invalid: NDArray[np.bool_],
    params: CalibrationParameters,
) -> NDArray[np.float64]:
    clean = values.astype(np.float64, copy=True)
    clean[invalid] = np.nan
    clipped = sigma_clip(
        clean,
        sigma=params.sigma,
        maxiters=params.maxiters,
        cenfunc=np.ma.median,
        axis=0,
        masked=True,
        copy=True,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        averages = np.ma.mean(clipped, axis=0)
    result = np.ma.filled(averages, np.nan).astype(np.float64)
    result[np.all(invalid, axis=0)] = np.nan
    return result


def _positive_median(values: NDArray[np.float64], what: str) -> float:
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        raise CalibrationError(f"{what}: no positive effective pixels available for normalization")
    median = float(np.median(valid))
    if not np.isfinite(median) or median <= 0:
        raise CalibrationError(f"{what}: invalid median normalization factor")
    return median



def calibrate_batch(
    exposures: list[Exposure],
    params: CalibrationParameters | None = None,
) -> CalibrationResult:
    params = params or CalibrationParameters()
    grouped = {
        role: [item for item in exposures if item.role == role]
        for role in ("bias", "dark", "flat", "science")
    }
    if any(not items for items in grouped.values()):
        raise CalibrationError("each of bias, dark, flat and science must contain a file")

    master_bias = sigma_clipped_mean(
        _stack(grouped["bias"]), _input_mask(grouped["bias"]), params
    )

    dark_values = np.zeros((len(grouped["dark"]), *master_bias.shape), dtype=np.float64)
    dark_invalid = np.zeros_like(dark_values, dtype=bool)
    for index, dark in enumerate(grouped["dark"]):
        source_invalid = _input_mask([dark])[0] | ~np.isfinite(master_bias)
        corrected = dark.data - master_bias
        dark_values[index] = corrected / dark.exptime
        dark_invalid[index] = source_invalid | ~np.isfinite(dark_values[index])
    master_dark_rate = sigma_clipped_mean(dark_values, dark_invalid, params)

    flat_values = np.zeros((len(grouped["flat"]), *master_bias.shape), dtype=np.float64)
    flat_invalid = np.zeros_like(flat_values, dtype=bool)
    for index, flat in enumerate(grouped["flat"]):
        source_invalid = (
            _input_mask([flat])[0]
            | ~np.isfinite(master_bias)
            | ~np.isfinite(master_dark_rate)
        )
        corresponding_dark = master_dark_rate * flat.exptime
        corrected = flat.data - master_bias - corresponding_dark
        flat_values[index] = corrected
        flat_invalid[index] = source_invalid | ~np.isfinite(corrected) | (corrected <= 0)

        per_image = corrected.copy()
        per_image[flat_invalid[index]] = np.nan
        scale = _positive_median(per_image, f"{flat.filename} flat")
        flat_values[index] = corrected / scale

    combined_flat = sigma_clipped_mean(flat_values, flat_invalid, params)
    renorm = _positive_median(combined_flat, "master flat")
    master_flat = combined_flat / renorm
    master_flat[~np.isfinite(master_flat) | (master_flat <= 0)] = np.nan

    sciences: list[CalibratedScience] = []
    for science in grouped["science"]:
        nonfinite = ~np.isfinite(science.data)
        saturated = science.data > science.saturate
        calibration_invalid = ~(np.isfinite(master_bias) & np.isfinite(master_dark_rate))
        flat_invalid_pixel = ~(np.isfinite(master_flat) & (master_flat > 0))

        corrected = (
            science.data - master_bias - master_dark_rate * science.exptime
        ) / master_flat
        invalid = nonfinite | saturated | calibration_invalid | flat_invalid_pixel
        corrected[invalid] = np.nan

        mask = np.zeros(science.data.shape, dtype=np.uint8)
        mask[nonfinite] |= MASK_FINITE_INVALID
        mask[saturated] |= MASK_SATURATED
        mask[calibration_invalid] |= MASK_CALIBRATION_INVALID
        mask[flat_invalid_pixel] |= MASK_FLAT_NONPOSITIVE
        sciences.append(CalibratedScience(science, corrected, mask))

    return CalibrationResult(
        master_bias=master_bias,
        master_dark_rate=master_dark_rate,
        master_flat=master_flat,
        sciences=sciences,
    )
