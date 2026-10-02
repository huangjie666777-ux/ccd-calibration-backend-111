"""Generate a small synthetic exposure set (bias/dark/flat/science) for demo.

Usage: .venv/bin/python examples/make_synthetic.py [outdir]
"""

import os
import sys

import numpy as np
from astropy.io import fits

SHAPE = (64, 64)
BIAS_LEVEL = 1000.0
DARK_RATE = 0.5
SATURATE = 60000.0


def header(role, exptime, filter_="r"):
    h = fits.Header()
    h["INSTRUME"] = "SYNTHCAM"
    h["CCD-TEMP"] = -100.0
    h["GAIN"] = 1.5
    h["XBINNING"] = 1
    h["YBINNING"] = 1
    h["EXPTIME"] = exptime
    h["SATURATE"] = SATURATE
    h["IMAGETYP"] = role.upper()
    if role in ("flat", "science"):
        h["FILTER"] = filter_
    h["OBJECT"] = "synthetic"
    return h


def make(outdir):
    rng = np.random.default_rng(42)
    os.makedirs(outdir, exist_ok=True)

    def write(name, role, exptime, data, filter_="r"):
        fits.writeto(
            os.path.join(outdir, name),
            data.astype(np.float32),
            header(role, exptime, filter_),
            overwrite=True,
        )

    for i in range(3):
        bias = BIAS_LEVEL + rng.normal(0, 3, SHAPE)
        write(f"bias_{i}.fits", "bias", 0.0, bias)

    for i, t in enumerate((30.0, 60.0)):
        dark = BIAS_LEVEL + DARK_RATE * t + rng.normal(0, 3, SHAPE)
        write(f"dark_{i}.fits", "dark", t, dark)

    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    illumination = 1.0 - 0.2 * (((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 32.0**2))
    for i, t in enumerate((5.0, 8.0, 10.0)):
        flat = BIAS_LEVEL + DARK_RATE * t + 20000.0 * illumination + rng.normal(0, 20, SHAPE)
        write(f"flat_{i}.fits", "flat", t, flat)

    sci = BIAS_LEVEL + DARK_RATE * 120.0 + 500.0 * illumination + rng.normal(0, 5, SHAPE)
    sci[10, 10] = 70000.0
    sci[20, 20] = np.nan
    write("science_0.fits", "science", 120.0, sci)

    print(f"wrote synthetic set to {outdir}")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data"))
