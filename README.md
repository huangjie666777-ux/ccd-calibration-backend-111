# CCD Calibration Backend

Pure-backend CCD calibration service: combines bias / dark / flat frames into
master calibration frames and calibrates science exposures for photometry.

## Layout

- `ccd_calibration/fits_io.py` — FITS reading, header conventions, per-file and per-batch validation
- `ccd_calibration/calibration.py` — calibration math (sigma-clipped combination, frame reduction, bad-pixel masks)
- `ccd_calibration/batch.py` — batch orchestration and result packaging (zip)
- `ccd_calibration/app.py` — FastAPI HTTP entry point
- `examples/make_synthetic.py` — small synthetic exposure set generator
- `tests/` — pytest suite

## Header conventions

All frames are 2D monochrome FITS with data in the primary HDU. Required
keywords (missing or illegal values reject the file with a per-file reason):

| Keyword    | Meaning                                             |
|------------|-----------------------------------------------------|
| EXPTIME    | Exposure seconds; 0 allowed for bias, >0 otherwise  |
| INSTRUME   | Instrument name, must match across the batch        |
| CCD-TEMP   | Detector temperature, must match across the batch   |
| GAIN       | Gain, must match across the batch                   |
| XBINNING / YBINNING | Binning factors, must match across the batch |
| FILTER     | Required for flat and science; must match between them |
| SATURATE   | Optional saturation level (ADU); raw pixels above it are masked |

Image dimensions must be identical for all frames in a batch.

## Processing

1. **Master bias**: per-pixel sigma-clipped mean of the raw bias frames.
2. **Master dark**: each dark is bias-subtracted, divided by its own EXPTIME,
   then combined — the result is dark current per second (adu/s).
3. **Master flat**: each flat is corrected for bias and for dark current
   scaled by its own EXPTIME (the bias level itself is never exposure-scaled),
   normalized by the median of its finite pixels, combined, and the result
   renormalized to median 1.
4. **Science**: `(raw - master_bias - master_dark * EXPTIME) / master_flat`.
   Negative values from legitimate subtraction are preserved.

Combination uses a per-pixel sigma-clipped mean (median/MAD based), with
configurable `sigma` (default 3.0) and `maxiters` (default 5). Non-finite
samples are always excluded; pixels with no surviving samples become NaN.

Bad pixels in calibrated science frames are NaN and flagged in the `MASK`
extension (uint8 bit mask): 1 = non-finite input, 2 = raw > SATURATE,
4 = flat non-positive/invalid.

## Dark current linearity scope

The master dark is a *per-second* dark-current rate derived from the submitted
dark exposures, and is linearly rescaled to the EXPTIME of flats and science
frames. This is only valid while the detector's dark current is linear in
exposure time (no significant hot-pixel non-linearity, glow saturation, or
thermal drift). For best results submit darks whose exposure times bracket
the flat/science exposure times, taken at the same CCD-TEMP; extrapolating
far beyond the dark exposure range is not recommended.

## HTTP API

`POST /calibrate` (multipart/form-data):

- `files`: repeated file parts, one per FITS frame
- `roles`: repeated form field, one role per file (`bias`/`dark`/`flat`/`science`), same order
- `sigma`, `maxiters`: optional combination parameters

Success: `200` with a zip containing `master_bias.fits`, `master_dark.fits`,
`master_flat.fits`, `calibrated/<name>_cal.fits` (float data + `MASK`
extension, original observation header preserved with added HISTORY and
`CLSIGMA`/`CLNITER`/`CALPROD` keywords) and `manifest.json` (sources,
steps, parameters, mask bit meanings).

Failure: `422` with `{"errors": [...]}` listing per-file reasons. The batch
is all-or-nothing — no partial results are delivered, and uploaded files are
never modified (all processing is in memory).

## Run

```bash
.venv/bin/python -m uvicorn ccd_calibration.app:app --port 8000
.venv/bin/python examples/make_synthetic.py          # writes examples/data/
cd examples/data
curl -s -o /tmp/result.zip   -F "files=@bias_0.fits"  -F "roles=bias"   -F "files=@bias_1.fits"  -F "roles=bias"   -F "files=@bias_2.fits"  -F "roles=bias"   -F "files=@dark_0.fits"  -F "roles=dark"   -F "files=@dark_1.fits"  -F "roles=dark"   -F "files=@flat_0.fits"  -F "roles=flat"   -F "files=@flat_1.fits"  -F "roles=flat"   -F "files=@flat_2.fits"  -F "roles=flat"   -F "files=@science_0.fits" -F "roles=science"   -F "sigma=3.0" -F "maxiters=5"   http://127.0.0.1:8000/calibrate
```

## Tests

```bash
.venv/bin/python -m pytest tests -q
```
