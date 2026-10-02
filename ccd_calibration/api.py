from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .calibration import calibrate_batch
from .delivery import build_result_zip
from .fits_io import read_batch
from .models import CalibrationError, CalibrationParameters, ROLES

app = FastAPI(
    title="CCD Calibration Backend",
    version="0.1.0",
    description="Create bias/dark/flat masters and calibrate monochrome science FITS files.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/calibrate/batch")
async def calibrate_batch_endpoint(
    files: list[UploadFile] = File(..., description="Monochrome 2-D FITS exposures"),
    roles: list[str] = Form(
        ...,
        description="One role per uploaded file, in the same order: bias/dark/flat/science",
    ),
    parameters: str = Form("{}", description="JSON: sigma and maxiters"),
) -> Response:
    if len(files) != len(roles):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "one role must be supplied for each uploaded file",
                "file_errors": {},
            },
        )

    invalid_roles = {
        file.filename or "unnamed": f"role must be one of {', '.join(ROLES)}"
        for file, role in zip(files, roles)
        if role not in ROLES
    }
    if invalid_roles:
        raise HTTPException(
            status_code=422,
            detail={"message": "invalid role", "file_errors": invalid_roles},
        )

    items = []
    for upload, role in zip(files, roles):
        items.append((upload.filename or "unnamed.fits", await upload.read(), role))

    try:
        params = CalibrationParameters.from_value(parameters)
        exposures = read_batch(items)
        result = calibrate_batch(exposures, params)
        package = build_result_zip(result, exposures, params)
    except CalibrationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": exc.message, "file_errors": exc.file_errors},
        ) from exc
    finally:
        for upload in files:
            await upload.close()

    return Response(
        content=package,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="ccd_calibration_package.zip"'
        },
    )
