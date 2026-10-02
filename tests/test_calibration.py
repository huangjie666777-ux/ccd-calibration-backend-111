from __future__ import annotations

import io
import zipfile

import numpy as np
from astropy.io import fits
from fastapi.testclient import TestClient

from ccd_calibration.api import app
from ccd_calibration.calibration import calibrate_batch
from ccd_calibration.delivery import build_result_zip
from ccd_calibration.fits_io import read_batch
from ccd_calibration.models import MASK_FINITE_INVALID, MASK_SATURATED, CalibrationParameters


SHAPE = (6, 8)
SATURATE = 60_000.0


def make_header(role: str, exptime: float, filter_name: str = "R") -> fits.Header:
    header = fits.Header()
    header["INSTRUME"] = "TEST-CCD"
    header["CCDTEMP"] = -15.0
    header["GAIN"] = 2.0
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["EXPTIME"] = exptime
    header["SATURATE"] = SATURATE
    if role in {"flat", "science"}:
        header["FILTER"] = filter_name
    return header


def fits_bytes(data: np.ndarray, role: str, exptime: float, filter_name: str = "R") -> bytes:
    stream = io.BytesIO()
    fits.PrimaryHDU(data=data.astype(np.float64), header=make_header(role, exptime, filter_name)).writeto(stream)
    return stream.getvalue()


def make_batch():
    rows, columns = np.indices(SHAPE, dtype=np.float64)
    bias_level = 100.0 + rows * 0.02
    dark_rate = 2.0 + columns * 0.01
    response = 0.95 + rows * 0.002
    rng = np.random.default_rng(7)
    items = []

    for index in range(10):
        data = bias_level + rng.normal(0, 0.001, SHAPE)
        if index == 9:
            data[0, 0] += 20.0
        items.append((f"bias{index}.fits", fits_bytes(data, "bias", 0.0), "bias"))

    for index, exptime in enumerate([4.0, 8.0, 16.0]):
        data = bias_level + dark_rate * exptime
        items.append((f"dark{index}.fits", fits_bytes(data, "dark", exptime), "dark"))

    for index, exptime in enumerate([1.0, 2.0]):
        level = 400.0 + index * 20.0
        data = bias_level + dark_rate * exptime + response * level
        items.append((f"flat{index}.fits", fits_bytes(data, "flat", exptime), "flat"))

    science = bias_level + dark_rate * 10.0 + 300.0 * response
    science[0, 0] = np.nan
    science[5, 7] = SATURATE + 1
    items.append(("science0.fits", fits_bytes(science, "science", 10.0), "science"))
    return items


def test_calibration_math_bad_pixels_and_package():
    items = make_batch()
    exposures = read_batch(items)
    params = CalibrationParameters(sigma=3.0, maxiters=5)
    result = calibrate_batch(exposures, params)

    expected_bias = 100.0 + np.indices(SHAPE, dtype=np.float64)[0] * 0.02
    expected_dark = 2.0 + np.indices(SHAPE, dtype=np.float64)[1] * 0.01
    np.testing.assert_allclose(result.master_bias, expected_bias, atol=2e-3)
    np.testing.assert_allclose(result.master_dark_rate, expected_dark, atol=1e-3)
    np.testing.assert_allclose(np.nanmedian(result.master_flat), 1.0, atol=1e-8)
    assert np.all(np.isfinite(result.master_flat))

    calibrated = result.sciences[0]
    good_signal = calibrated.data[1:5, 1:6]
    rows, _ = np.indices(SHAPE, dtype=np.float64)
    response = 0.95 + rows * 0.002
    expected_signal = 300.0 * np.median(response)
    np.testing.assert_allclose(good_signal, expected_signal, atol=0.02)
    assert np.isnan(calibrated.data[0, 0])
    assert np.isnan(calibrated.data[5, 7])
    assert calibrated.mask[0, 0] & MASK_FINITE_INVALID
    assert calibrated.mask[5, 7] & MASK_SATURATED

    package = build_result_zip(result, exposures, params)
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names = set(archive.namelist())
        assert "calibration/master_bias.fits" in names
        assert "calibration/master_dark_current_per_second.fits" in names
        assert "calibration/master_flat.fits" in names
        assert "calibration/science_science0_calibrated.fits" in names
        assert "calibration/source_manifest.csv" in names
        with archive.open("calibration/science_science0_calibrated.fits") as member:
            with fits.open(member) as hdul:
                assert hdul[1].name == "BADMASK"
                assert hdul[2].name == "MASK_REASONS"
                assert hdul[0].header["INSTRUME"] == "TEST-CCD"
                assert hdul[0].header["CALSIGMA"] == 3.0


def test_file_level_errors_and_linear_dark_requirement():
    data = np.full(SHAPE, 100.0)
    payload = fits_bytes(data, "science", 0.0)
    items = make_batch() + [("bad.fits", payload, "science")]
    response = TestClient(app).post(
        "/calibrate/batch",
        files=[("files", (name, content, "application/fits")) for name, content, _ in items],
        data={
            "roles": [role for _, _, role in items],
            "parameters": '{"sigma": 3, "maxiters": 2}',
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "bad.fits" in detail["file_errors"]


def test_http_batch_returns_zip():
    items = make_batch()
    response = TestClient(app).post(
        "/calibrate/batch",
        files=[("files", (name, content, "application/fits")) for name, content, _ in items],
        data={"roles": [role for _, _, role in items]},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert zipfile.is_zipfile(io.BytesIO(response.content))
