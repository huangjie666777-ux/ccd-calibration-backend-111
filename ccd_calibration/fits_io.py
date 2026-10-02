from __future__ import annotations

import io
from collections import Counter
import hashlib

import numpy as np
from astropy.io import fits

from .models import CalibrationError, Exposure, ObservationContext, Role, ROLES


def _header_value(header: fits.Header, key: str) -> object:
    value = header.get(key)
    if isinstance(value, str):
        value = value.strip()
    return value


def _required_float(header: fits.Header, key: str, filename: str) -> float:
    value = _header_value(header, key)
    if value is None or value == "":
        raise CalibrationError(f"{filename}: missing required header {key}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{filename}: header {key} must be numeric") from exc
    if not np.isfinite(number):
        raise CalibrationError(f"{filename}: header {key} must be finite")
    return number


def _required_positive_float(header: fits.Header, key: str, filename: str) -> float:
    value = _required_float(header, key, filename)
    if value <= 0:
        raise CalibrationError(f"{filename}: header {key} must be positive")
    return value


def _required_int(header: fits.Header, key: str, filename: str) -> int:
    value = _required_float(header, key, filename)
    if not float(value).is_integer():
        raise CalibrationError(f"{filename}: header {key} must be an integer")
    number = int(value)
    if number <= 0:
        raise CalibrationError(f"{filename}: header {key} must be positive")
    return number


def read_exposure(filename: str, payload: bytes, role: Role) -> Exposure:
    try:
        with fits.open(io.BytesIO(payload), memmap=False, mode="readonly") as hdul:
            hdu = hdul[0]
            if hdu.data is None:
                raise CalibrationError(f"{filename}: primary HDU must contain image data")
            data = np.asarray(hdu.data)
            header = hdu.header.copy()
    except CalibrationError:
        raise
    except Exception as exc:
        raise CalibrationError(f"{filename}: unreadable FITS file: {exc}") from exc

    if data.ndim != 2:
        raise CalibrationError(f"{filename}: primary image must be two-dimensional")
    if not np.issubdtype(data.dtype, np.number):
        raise CalibrationError(f"{filename}: primary image must be numeric")

    instrument = _header_value(header, "INSTRUME")
    if not isinstance(instrument, str) or not instrument:
        raise CalibrationError(f"{filename}: missing required header INSTRUME")
    temperature = _required_float(header, "CCDTEMP", filename)
    gain = _required_positive_float(header, "GAIN", filename)
    xbin = _required_int(header, "XBINNING", filename)
    ybin = _required_int(header, "YBINNING", filename)
    exptime = _required_float(header, "EXPTIME", filename)
    saturate = _required_positive_float(header, "SATURATE", filename)

    if role == "bias":
        if exptime < 0:
            raise CalibrationError(f"{filename}: EXPTIME must be non-negative")
    elif exptime <= 0:
        raise CalibrationError(f"{filename}: EXPTIME must be positive")

    filter_name = None
    if role in {"flat", "science"}:
        filter_name = _header_value(header, "FILTER")
        if not isinstance(filter_name, str) or not filter_name:
            raise CalibrationError(f"{filename}: missing required header FILTER")

    return Exposure(
        role=role,
        filename=filename,
        data=data.astype(np.float64, copy=True),
        exptime=exptime,
        saturate=saturate,
        file_hash=hashlib.sha256(payload).hexdigest(),
        context=ObservationContext(
            instrument=instrument,
            temperature=temperature,
            gain=gain,
            binning=(xbin, ybin),
            shape=data.shape,
            filter_name=filter_name,
        ),
        header=header,
    )


def validate_batch(exposures: list[Exposure]) -> None:
    errors: dict[str, str] = {}
    counts = Counter(item.role for item in exposures)
    for role in ROLES:
        if counts[role] == 0:
            errors[f"__batch__:{role}"] = f"batch must contain at least one {role} file"

    names = [item.filename for item in exposures]
    duplicated = {name for name, count in Counter(names).items() if count > 1}
    for name in duplicated:
        errors[name] = "duplicate filename in submitted batch"

    if exposures:
        reference = exposures[0]
        ref = reference.context
        for exposure in exposures[1:]:
            current = exposure.context
            checks = (
                ("INSTRUME", ref.instrument, current.instrument),
                ("CCDTEMP", ref.temperature, current.temperature),
                ("GAIN", ref.gain, current.gain),
                ("XBINNING/YBINNING", ref.binning, current.binning),
                ("image dimensions", ref.shape, current.shape),
            )
            mismatch = next((label for label, left, right in checks if left != right), None)
            if mismatch:
                errors[exposure.filename] = (
                    f"does not match {reference.filename} for {mismatch}"
                )

        flats = [item for item in exposures if item.role == "flat"]
        sciences = [item for item in exposures if item.role == "science"]
        filters = {item.context.filter_name for item in flats + sciences}
        if len(filters) > 1:
            for item in flats[1:] + sciences:
                errors.setdefault(
                    item.filename,
                    f"flat/science FILTER must match {flats[0].filename if flats else 'the batch'}",
                )

    if errors:
        raise CalibrationError("batch validation failed", dict(sorted(errors.items())))


def read_batch(items: list[tuple[str, bytes, Role]]) -> list[Exposure]:
    exposures: list[Exposure] = []
    errors: dict[str, str] = {}
    for filename, payload, role in items:
        try:
            exposures.append(read_exposure(filename, payload, role))
        except CalibrationError as exc:
            errors[filename] = exc.message.replace(f"{filename}: ", "", 1)
    if errors:
        raise CalibrationError("one or more files could not be read", errors)
    validate_batch(exposures)
    return exposures
