import io
import json
import zipfile

import numpy as np
from astropy.io import fits
from fastapi.testclient import TestClient

from ccd_calibration.app import app

client = TestClient(app)


def make_fits(role, exptime, value, filter_="r", shape=(8, 8)):
    h = fits.Header()
    h["INSTRUME"] = "CAM"
    h["CCD-TEMP"] = -100.0
    h["GAIN"] = 1.0
    h["XBINNING"] = 1
    h["YBINNING"] = 1
    h["EXPTIME"] = exptime
    h["SATURATE"] = 60000.0
    h["OBJECT"] = "test-target"
    if role in ("flat", "science"):
        h["FILTER"] = filter_
    buf = io.BytesIO()
    fits.writeto(buf, np.full(shape, value, dtype=np.float32), h)
    return buf.getvalue()


def batch_files(**overrides):
    frames = [
        ("b0.fits", "bias", make_fits("bias", 0.0, 100.0)),
        ("b1.fits", "bias", make_fits("bias", 0.0, 100.0)),
        ("d0.fits", "dark", make_fits("dark", 10.0, 120.0)),
        ("f0.fits", "flat", make_fits("flat", 5.0, 20100.0)),
        ("s0.fits", "science", make_fits("science", 10.0, 1120.0)),
    ]
    files = [("files", (name, payload, "application/fits")) for name, _, payload in frames]
    roles = [role for _, role, _ in frames]
    return files, roles


def test_calibrate_success_zip_contents():
    files, roles = batch_files()
    resp = client.post(
        "/calibrate",
        files=files,
        data={"roles": roles, "sigma": "3.0", "maxiters": "5"},
    )
    assert resp.status_code == 200, resp.text
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert {"master_bias.fits", "master_dark.fits", "master_flat.fits",
            "calibrated/s0_cal.fits", "manifest.json"} <= names

    with fits.open(io.BytesIO(zf.read("master_dark.fits"))) as hdul:
        assert np.allclose(hdul[0].data, 2.0)  # (120-100)/10
        assert hdul[0].header["BUNIT"] == "adu/s"

    with fits.open(io.BytesIO(zf.read("calibrated/s0_cal.fits"))) as hdul:
        assert hdul[0].header["OBJECT"] == "test-target"  # header preserved
        assert hdul[0].header["CLSIGMA"] == 3.0
        assert any("MASK" == h.name for h in hdul[1:])
        # science: (1120 - 100 - 2*10)/1.0 = 1000
        assert np.allclose(hdul[0].data, 1000.0)
        assert (hdul["MASK"].data == 0).all()

    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["sources"]["science"] == ["s0.fits"]
    assert manifest["parameters"]["sigma"] == 3.0


def test_calibrate_batch_failure_returns_file_reasons():
    files, roles = batch_files()
    files.append(("files", ("bad.fits", make_fits("science", 10.0, 1.0, filter_="g"), "application/fits")))
    roles.append("science")
    resp = client.post("/calibrate", files=files, data={"roles": roles})
    assert resp.status_code == 422
    body = resp.json()
    assert any("FILTER" in e for e in body["errors"])


def test_calibrate_missing_role_fails_whole_batch():
    files, roles = batch_files()
    files = files[:3]  # drop flat and science
    roles = roles[:3]
    resp = client.post("/calibrate", files=files, data={"roles": roles})
    assert resp.status_code == 422
    assert any("missing frames" in e for e in resp.json()["errors"])
