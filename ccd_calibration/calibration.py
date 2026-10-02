"""Calibration math: sigma-clipped combination and frame reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MASK_NONFINITE = 1
MASK_SATURATED = 2
MASK_BADFLAT = 4


@dataclass
class CalibConfig:
    sigma: float = 3.0
    maxiters: int = 5


def sigma_clip_mean(stack, sigma, maxiters):
    """Per-pixel sigma-clipped mean along axis 0.

    Non-finite samples are always excluded. Pixels with no surviving
    samples after clipping yield NaN.
    """
    data = np.asarray(stack, dtype=np.float64)
    work = np.where(np.isfinite(data), data, np.nan)

    for _ in range(max(0, maxiters)):
        count = np.sum(np.isfinite(work), axis=0)
        if not np.any(count > 1):
            break
        center = np.nanmedian(work, axis=0)
        mad = np.nanmedian(np.abs(work - center), axis=0)
        spread = 1.4826 * mad
        with np.errstate(invalid="ignore"):
            clip = np.abs(work - center) > sigma * spread
        clip &= np.isfinite(work)
        if not clip.any():
            break
        work = np.where(clip, np.nan, work)

    count = np.sum(np.isfinite(work), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(work, axis=0)
    return np.where(count > 0, mean, np.nan)


def combine(frames, cfg):
    return sigma_clip_mean(np.stack(frames, axis=0), cfg.sigma, cfg.maxiters)


def build_master_bias(bias_frames, cfg):
    """Master bias: sigma-clipped mean of raw bias frames."""
    return combine(bias_frames, cfg)


def build_master_dark(dark_frames, master_bias, cfg):
    """Master dark current per second (counts/s).

    Each dark is bias-subtracted first, then divided by its own exposure
    time, then combined.
    """
    rates = [(d - master_bias) / t for d, t in dark_frames]
    return combine(rates, cfg)


def build_master_flat(flat_frames, master_bias, dark_rate, cfg):
    """Master flat, normalized to a median of 1 over valid pixels.

    Each flat is corrected for bias and for dark current scaled by its own
    exposure time (the bias level itself is never exposure-scaled), then
    normalized by the median of its finite pixels before combination.
    The combined flat is renormalized to median 1.
    """
    normed = []
    for raw, exptime in flat_frames:
        corrected = raw - master_bias - dark_rate * exptime
        finite = corrected[np.isfinite(corrected)]
        if finite.size == 0:
            normed.append(np.full_like(corrected, np.nan))
            continue
        med = np.median(finite)
        if not np.isfinite(med) or med == 0:
            normed.append(np.full_like(corrected, np.nan))
            continue
        normed.append(corrected / med)
    master = combine(normed, cfg)
    finite = master[np.isfinite(master)]
    if finite.size:
        med = np.median(finite)
        if np.isfinite(med) and med != 0:
            master = master / med
    return master


def calibrate_science(raw, exptime, master_bias, dark_rate, master_flat,
                      saturate=np.inf):
    """Calibrate one science frame.

    Returns (calibrated float array, uint8 reason mask). Bad pixels are
    NaN in the output and flagged in the mask; negative values from
    legitimate subtraction are preserved.
    """
    mask = np.zeros(raw.shape, dtype=np.uint8)
    mask[~np.isfinite(raw)] |= MASK_NONFINITE
    mask[raw > saturate] |= MASK_SATURATED
    badflat = ~np.isfinite(master_flat) | (master_flat <= 0)
    mask[badflat] |= MASK_BADFLAT

    corrected = raw - master_bias - dark_rate * exptime
    with np.errstate(invalid="ignore", divide="ignore"):
        out = corrected / master_flat
    out[mask != 0] = np.nan
    return out, mask
