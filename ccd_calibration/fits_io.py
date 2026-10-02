"""FITS reading, header conventions and per-file / per-batch validation."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

ROLES = ("bias", "dark", "flat", "science")

# Header keywords that must be present and identical across the whole batch.
GROUP_KEYS = ("INSTRUME", "CCD-TEMP", "GAIN", "XBINNING", "YBINNING")
# FILTER must additionally match between flat and science frames.
FILTER_KEY = "FILTER"
SATURATE_KEY = "SATURATE"


class FrameError(ValueError):
    """File-level rejection with a human readable reason."""


@dataclass
class Frame:
    filename: str
    role: str
    data: np.ndarray
    header: fits.Header
    exptime: float
    saturate: float = np.inf

    @property
    def shape(self):
        return self.data.shape


def _require(header: fits.Header, key: str, filename: str):
    if key not in header:
        raise FrameError(f"{filename}: missing required header keyword {key}")
    return header[key]


def read_frame(payload: bytes, filename: str, role: str) -> Frame:
    """Parse one uploaded FITS file into a Frame, raising FrameError on any
    file-level problem (bad FITS, non-2D, missing/illegal headers)."""
    if role not in ROLES:
        raise FrameError(f"{filename}: unknown role '{role}' (expected one of {ROLES})")
    try:
        hdul = fits.open(io.BytesIO(payload), memmap=False)
    except Exception as exc:
        raise FrameError(f"{filename}: not a readable FITS file ({exc})") from exc
    with hdul:
        if len(hdul) == 0 or hdul[0].data is None:
            raise FrameError(f"{filename}: primary HDU has no data")
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()

    if data.ndim != 2:
        raise FrameError(f"{filename}: expected a 2D monochrome image, got {data.ndim}D")

    for key in GROUP_KEYS:
        _require(header, key, filename)
    if role in ("flat", "science"):
        _require(header, FILTER_KEY, filename)

    raw_exptime = _require(header, "EXPTIME", filename)
    try:
        exptime = float(raw_exptime)
    except (TypeError, ValueError):
        raise FrameError(f"{filename}: EXPTIME={raw_exptime!r} is not numeric")
    if not np.isfinite(exptime) or exptime < 0:
        raise FrameError(f"{filename}: illegal EXPTIME={raw_exptime!r}")
    if role != "bias" and exptime <= 0:
        raise FrameError(f"{filename}: {role} frame requires EXPTIME > 0, got {exptime}")

    saturate = float(header.get(SATURATE_KEY, np.inf))

    return Frame(
        filename=filename,
        role=role,
        data=data,
        header=header,
        exptime=exptime,
        saturate=saturate,
    )


def validate_group(frames: list[Frame]) -> list[str]:
    """Return a list of file-level incompatibility reasons (empty if OK)."""
    errors: list[str] = []
    if not frames:
        return ["no frames submitted"]

    ref = frames[0]
    for f in frames[1:]:
        if f.shape != ref.shape:
            errors.append(
                f"{f.filename}: image size {f.shape} differs from {ref.filename} {ref.shape}"
            )
        for key in GROUP_KEYS:
            if f.header.get(key) != ref.header.get(key):
                errors.append(
                    f"{f.filename}: {key}={f.header.get(key)!r} differs from "
                    f"{ref.filename} {key}={ref.header.get(key)!r}"
                )

    filters = {
        f.header.get(FILTER_KEY)
        for f in frames
        if f.role in ("flat", "science")
    }
    if len(filters) > 1:
        bad = [f.filename for f in frames if f.role in ("flat", "science")]
        errors.append(
            f"FILTER mismatch among flat/science frames {bad}: {sorted(map(str, filters))}"
        )
    return errors


def split_roles(frames: list[Frame]) -> dict[str, list[Frame]]:
    out: dict[str, list[Frame]] = {r: [] for r in ROLES}
    for f in frames:
        out[f.role].append(f)
    return out
