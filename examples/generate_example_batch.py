from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits


SHAPE = (8, 10)
SATURATE = 55_000.0


def header(role: str, exptime: float, filter_name: str | None = None) -> fits.Header:
    header = fits.Header()
    header["INSTRUME"] = "SYNTH-CCD"
    header["CCDTEMP"] = -20.0
    header["GAIN"] = 1.5
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["EXPTIME"] = exptime
    header["SATURATE"] = SATURATE
    header["OBJECT"] = "synthetic calibration example"
    if filter_name is not None:
        header["FILTER"] = filter_name
    header["ROLE"] = role
    return header


def write_fits(path: Path, data: np.ndarray, hdr: fits.Header) -> None:
    fits.PrimaryHDU(data=data.astype(np.float64), header=hdr).writeto(
        path, overwrite=True
    )


def main() -> None:
    rng = np.random.default_rng(20261002)
    output = Path(__file__).resolve().parent / "batch"
    output.mkdir(parents=True, exist_ok=True)

    rows, columns = np.indices(SHAPE, dtype=np.float64)
    bias_level = 100.0 + 0.02 * rows + 0.01 * columns
    dark_rate = 1.5 + 0.002 * rows + 0.001 * columns
    response = 0.90 + 0.003 * rows + 0.002 * columns

    for index in range(3):
        data = bias_level + rng.normal(0.0, 0.03, SHAPE)
        if index == 2:
            data[0, 0] += 0.7
        write_fits(output / f"bias_{index + 1:02d}.fits", data, header("bias", 0.0))

    dark_exptimes = [5.0, 10.0, 20.0]
    for index, exptime in enumerate(dark_exptimes):
        data = bias_level + dark_rate * exptime + rng.normal(0.0, 0.04, SHAPE)
        write_fits(
            output / f"dark_{index + 1:02d}.fits", data, header("dark", exptime)
        )

    flat_exptimes = [1.0, 1.5, 2.0]
    illumination = [500.0, 520.0, 480.0]
    for index, (exptime, level) in enumerate(zip(flat_exptimes, illumination)):
        data = (
            bias_level
            + dark_rate * exptime
            + level * response
            + rng.normal(0.0, 0.08, SHAPE)
        )
        write_fits(
            output / f"flat_{index + 1:02d}.fits",
            data,
            header("flat", exptime, "R"),
        )

    for index, exptime in enumerate([30.0, 45.0]):
        signal = (120.0 + index * 20.0) * response
        data = bias_level + dark_rate * exptime + signal
        data += rng.normal(0.0, 0.05, SHAPE)
        if index == 0:
            data[1, 1] = np.nan
            data[2, 2] = SATURATE + 100.0
        hdr = header("science", exptime, "R")
        hdr["OBJECT"] = f"synthetic target {index + 1}"
        write_fits(output / f"science_{index + 1:02d}.fits", data, hdr)

    print(output)


if __name__ == "__main__":
    main()
