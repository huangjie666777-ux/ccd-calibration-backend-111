from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import numpy as np
from astropy.io import fits

from .models import (
    MASK_DESCRIPTIONS,
    ROLES,
    CalibratedScience,
    CalibrationParameters,
    CalibrationResult,
    Exposure,
)


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "unnamed.fits"


def _stem(filename: str) -> str:
    return Path(_safe_name(filename)).stem


def _processing_header(
    source: Exposure | None,
    params: CalibrationParameters,
    product: str,
    step: str,
) -> fits.Header:
    if source is None:
        header = fits.Header()
    else:
        header = source.header.copy()
    header["CALPROD"] = (product, "Calibration product type")
    header["CALSIGMA"] = (params.sigma, "Per-pixel sigma-clipping threshold")
    header["CALITER"] = (params.maxiters, "Sigma-clipping maximum iterations")
    header.add_history("Physical pixel values read from primary HDU as float64.")
    header.add_history(step)
    header.add_history(
        f"Sigma-clipped mean: sigma={params.sigma}, maxiters={params.maxiters}."
    )
    return header


def _reason_hdu() -> fits.BinTableHDU:
    columns = fits.ColDefs(
        [
            fits.Column(name="BIT", format="J", array=list(MASK_DESCRIPTIONS.keys())),
            fits.Column(
                name="REASON",
                format="A64",
                array=list(MASK_DESCRIPTIONS.values()),
            ),
        ]
    )
    return fits.BinTableHDU.from_columns(columns, name="MASK_REASONS")


def _master_bytes(
    result: CalibrationResult,
    exposures: list[Exposure],
    params: CalibrationParameters,
) -> list[tuple[str, bytes]]:
    specs = (
        (
            "master_bias.fits",
            result.master_bias,
            "bias",
            "master bias",
            "Combined bias; no exposure-time scaling applied.",
        ),
        (
            "master_dark_current_per_second.fits",
            result.master_dark_rate,
            "dark",
            "master dark current per second",
            "Bias-subtracted dark values divided by EXPTIME before combining.",
        ),
        (
            "master_flat.fits",
            result.master_flat,
            "flat",
            "renormalized master flat",
            "Bias and matching-exposure dark current subtracted; flats normalized before and after combining.",
        ),
    )
    outputs: list[tuple[str, bytes]] = []
    for filename, data, role, product, step in specs:
        source = next(item for item in exposures if item.role == role)
        header = _processing_header(source, params, product, step)
        hdu = fits.PrimaryHDU(data=data.astype(np.float64, copy=False), header=header)
        stream = io.BytesIO()
        fits.HDUList([hdu]).writeto(stream, overwrite=True)
        outputs.append((filename, stream.getvalue()))
    return outputs


def _science_bytes(
    science: CalibratedScience,
    params: CalibrationParameters,
) -> tuple[str, bytes]:
    source = science.source
    header = _processing_header(
        source,
        params,
        "calibrated science",
        "Science corrected as (data - bias - dark_rate*EXPTIME)/flat; negative values retained.",
    )
    header["SRCFILE"] = (_safe_name(source.filename), "Source exposure")
    primary = fits.PrimaryHDU(data=science.data.astype(np.float64, copy=False), header=header)
    mask_hdu = fits.ImageHDU(data=science.mask.astype(np.uint8, copy=False), name="BADMASK")
    reason_hdu = _reason_hdu()
    stream = io.BytesIO()
    fits.HDUList([primary, mask_hdu, reason_hdu]).writeto(stream, overwrite=True)
    return f"science_{_stem(source.filename)}_calibrated.fits", stream.getvalue()


def build_result_zip(
    result: CalibrationResult,
    exposures: list[Exposure],
    params: CalibrationParameters,
) -> bytes:
    import zipfile

    files = _master_bytes(result, exposures, params)
    files.extend(_science_bytes(item, params) for item in result.sciences)

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files:
            archive.writestr(f"calibration/{name}", payload)

        manifest = io.StringIO()
        writer = csv.DictWriter(
            manifest,
            fieldnames=("filename", "role", "exptime", "filter", "sha256", "output"),
        )
        writer.writeheader()
        master_outputs = {
            "bias": "master_bias.fits",
            "dark": "master_dark_current_per_second.fits",
            "flat": "master_flat.fits",
            "science": None,
        }
        for source in exposures:
            output = master_outputs[source.role] or (
                f"science_{_stem(source.filename)}_calibrated.fits"
            )
            writer.writerow(
                {
                    "filename": source.filename,
                    "role": source.role,
                    "exptime": source.exptime,
                    "filter": source.context.filter_name or "",
                    "sha256": source.file_hash,
                    "output": output,
                }
            )
        archive.writestr("calibration/source_manifest.csv", manifest.getvalue())

        metadata = {
            "parameters": {"sigma": params.sigma, "maxiters": params.maxiters},
            "roles": {
                role: sum(item.role == role for item in exposures) for role in ROLES
            },
            "products": [name for name, _ in files] + ["source_manifest.csv"],
        }
        archive.writestr(
            "calibration/processing_metadata.json", json.dumps(metadata, indent=2)
        )
    return stream.getvalue()
