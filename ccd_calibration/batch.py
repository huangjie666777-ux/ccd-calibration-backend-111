"""Batch orchestration: run a full calibration batch and package results."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass

import numpy as np
from astropy.io import fits

from .calibration import (
    CalibConfig,
    build_master_bias,
    build_master_dark,
    build_master_flat,
    calibrate_science,
)
from .fits_io import Frame, FrameError, read_frame, split_roles, validate_group


class BatchError(ValueError):
    """Whole-batch failure; carries per-file reasons. Nothing is delivered."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass
class BatchResult:
    master_bias: np.ndarray
    master_dark: np.ndarray
    master_flat: np.ndarray
    calibrated: list[tuple[Frame, np.ndarray, np.ndarray]]
    manifest: dict


def run_batch(uploads: list[tuple[str, str, bytes]],
              cfg: CalibConfig | None = None) -> BatchResult:
    """uploads: list of (filename, role, payload). All-or-nothing."""
    cfg = cfg or CalibConfig()
    reasons: list[str] = []
    frames: list[Frame] = []
    for filename, role, payload in uploads:
        try:
            frames.append(read_frame(payload, filename, role))
        except FrameError as exc:
            reasons.append(str(exc))
    if not frames:
        reasons.append("no usable frames submitted")
    if frames:
        reasons.extend(validate_group(frames))
    if reasons:
        raise BatchError(reasons)

    by_role = split_roles(frames)
    missing = [r for r in ("bias", "dark", "flat", "science") if not by_role[r]]
    if missing:
        raise BatchError([f"missing frames for role(s): {', '.join(missing)}"])

    master_bias = build_master_bias([f.data for f in by_role["bias"]], cfg)
    master_dark = build_master_dark(
        [(f.data, f.exptime) for f in by_role["dark"]], master_bias, cfg)
    master_flat = build_master_flat(
        [(f.data, f.exptime) for f in by_role["flat"]], master_bias, master_dark, cfg)

    calibrated = []
    for f in by_role["science"]:
        data, mask = calibrate_science(
            f.data, f.exptime, master_bias, master_dark, master_flat, f.saturate)
        calibrated.append((f, data, mask))

    manifest = {
        "parameters": {"sigma": cfg.sigma, "maxiters": cfg.maxiters},
        "steps": [
            "master_bias = sigma-clipped mean of raw bias frames",
            "master_dark = sigma-clipped mean of (dark - master_bias) / EXPTIME",
            "master_flat = sigma-clipped mean of (flat - master_bias - master_dark*EXPTIME)"
            " normalized per-frame by median, renormalized to median 1",
            "science = (raw - master_bias - master_dark*EXPTIME) / master_flat",
        ],
        "mask_bits": {"1": "non-finite input", "2": "raw > SATURATE", "4": "flat non-positive/invalid"},
        "sources": {
            role: [f.filename for f in by_role[role]] for role in ("bias", "dark", "flat", "science")
        },
    }
    return BatchResult(master_bias, master_dark, master_flat, calibrated, manifest)


def _stamp_header(header: fits.Header, cfg: CalibConfig, product: str) -> fits.Header:
    h = header.copy()
    h["CALPROD"] = (product, "calibration product identifier")
    h["CLSIGMA"] = (cfg.sigma, "sigma-clipping threshold")
    h["CLNITER"] = (cfg.maxiters, "sigma-clipping max iterations")
    h.add_history("calibrated: (raw - bias - dark_rate*EXPTIME) / flat")
    h.add_history("masters combined with per-pixel sigma-clipped mean")
    return h


def _fits_bytes(hdul: fits.HDUList) -> bytes:
    buf = io.BytesIO()
    hdul.writeto(buf, overwrite=True)
    return buf.getvalue()


def build_zip(result: BatchResult, cfg: CalibConfig) -> bytes:
    """Package masters, calibrated science (with MASK extension) and manifest."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data, unit in (
            ("master_bias.fits", result.master_bias, "adu"),
            ("master_dark.fits", result.master_dark, "adu/s"),
            ("master_flat.fits", result.master_flat, ""),
        ):
            hdr = _stamp_header(fits.Header(), cfg, name)
            hdr["BUNIT"] = unit
            hdul = fits.HDUList([fits.PrimaryHDU(data=data.astype(np.float64), header=hdr)])
            zf.writestr(name, _fits_bytes(hdul))

        for frame, data, mask in result.calibrated:
            hdr = _stamp_header(frame.header, cfg, "calibrated science")
            hdul = fits.HDUList([
                fits.PrimaryHDU(data=data.astype(np.float64), header=hdr),
                fits.ImageHDU(data=mask.astype(np.uint8), name="MASK"),
            ])
            out_name = f"calibrated/{frame.filename.rsplit('.', 1)[0]}_cal.fits"
            zf.writestr(out_name, _fits_bytes(hdul))

        zf.writestr("manifest.json", json.dumps(result.manifest, indent=2))
    return buf.getvalue()
