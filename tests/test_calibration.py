import io

import numpy as np
import pytest
from astropy.io import fits

from ccd_calibration.calibration import (
    CalibConfig,
    MASK_BADFLAT,
    MASK_NONFINITE,
    MASK_SATURATED,
    build_master_bias,
    build_master_dark,
    build_master_flat,
    calibrate_science,
    sigma_clip_mean,
)
from ccd_calibration.fits_io import FrameError, read_frame, validate_group

CFG = CalibConfig(sigma=3.0, maxiters=5)


def base_header(role, exptime, filter_="r"):
    h = fits.Header()
    h["INSTRUME"] = "CAM"
    h["CCD-TEMP"] = -100.0
    h["GAIN"] = 1.0
    h["XBINNING"] = 1
    h["YBINNING"] = 1
    h["EXPTIME"] = exptime
    h["SATURATE"] = 60000.0
    if role in ("flat", "science"):
        h["FILTER"] = filter_
    return h


def to_bytes(data, header):
    buf = io.BytesIO()
    fits.writeto(buf, np.asarray(data, dtype=np.float32), header)
    return buf.getvalue()


def test_sigma_clip_excludes_outlier_and_nonfinite():
    good = np.full((4, 5, 5), 10.0)
    stack = np.concatenate([good, np.full((1, 5, 5), 1000.0)], axis=0)
    stack[0, 0, 0] = np.nan
    out = sigma_clip_mean(stack, 3.0, 5)
    assert np.allclose(out, 10.0)


def test_sigma_clip_no_samples_yields_nan():
    stack = np.full((3, 2, 2), np.nan)
    out = sigma_clip_mean(stack, 3.0, 5)
    assert np.isnan(out).all()


def test_master_dark_is_per_second_and_bias_not_scaled():
    bias = np.full((4, 4), 100.0)
    mb = build_master_bias([bias, bias], CFG)
    darks = [(bias + 2.0 * t, t) for t in (10.0, 30.0)]
    rate = build_master_dark(darks, mb, CFG)
    assert np.allclose(rate, 2.0)


def test_master_flat_normalized_to_one():
    mb = np.zeros((4, 4))
    rate = np.zeros((4, 4))
    flats = [(np.full((4, 4), 20000.0), 5.0), (np.full((4, 4), 40000.0), 10.0)]
    mf = build_master_flat(flats, mb, rate, CFG)
    assert np.allclose(mf, 1.0)


def test_calibrate_science_masks_and_negatives():
    mb = np.full((2, 2), 100.0)
    rate = np.ones((2, 2))
    flat = np.array([[1.0, 0.0], [1.0, 1.0]])
    raw = np.array([[150.0, 200.0], [90.0, 70000.0]])
    raw[0, 0] = np.nan
    out, mask = calibrate_science(raw, 10.0, mb, rate, flat, saturate=60000.0)
    assert np.isnan(out[0, 0]) and mask[0, 0] & MASK_NONFINITE
    assert np.isnan(out[0, 1]) and mask[0, 1] & MASK_BADFLAT
    assert np.isnan(out[1, 1]) and mask[1, 1] & MASK_SATURATED
    assert out[1, 0] == pytest.approx(-20.0)
    assert mask[1, 0] == 0


def test_read_frame_missing_exptime():
    h = base_header("bias", 0.0)
    del h["EXPTIME"]
    with pytest.raises(FrameError, match="EXPTIME"):
        read_frame(to_bytes(np.zeros((4, 4)), h), "b0.fits", "bias")


def test_read_frame_rejects_3d_and_bad_exptime():
    with pytest.raises(FrameError, match="2D"):
        read_frame(to_bytes(np.zeros((2, 2, 2)), base_header("bias", 0.0)), "b.fits", "bias")
    with pytest.raises(FrameError, match="EXPTIME > 0"):
        read_frame(to_bytes(np.zeros((4, 4)), base_header("science", 0.0)), "s.fits", "science")


def test_validate_group_mismatch():
    f1 = read_frame(to_bytes(np.zeros((4, 4)), base_header("bias", 0.0)), "b0.fits", "bias")
    h = base_header("bias", 0.0)
    h["CCD-TEMP"] = -90.0
    f2 = read_frame(to_bytes(np.zeros((4, 4)), h), "b1.fits", "bias")
    errors = validate_group([f1, f2])
    assert any("CCD-TEMP" in e for e in errors)


def test_validate_group_filter_mismatch():
    f1 = read_frame(to_bytes(np.zeros((4, 4)), base_header("flat", 5.0, "r")), "f.fits", "flat")
    f2 = read_frame(to_bytes(np.zeros((4, 4)), base_header("science", 5.0, "g")), "s.fits", "science")
    errors = validate_group([f1, f2])
    assert any("FILTER" in e for e in errors)
