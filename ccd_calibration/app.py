"""FastAPI entry point: receive a calibration batch, return a result zip."""

from __future__ import annotations

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from .batch import BatchError, build_zip, run_batch
from .calibration import CalibConfig

app = FastAPI(title="CCD Calibration Backend", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/calibrate")
async def calibrate(
    files: list[UploadFile] = File(...),
    roles: list[str] = Form(...),
    sigma: float = Form(3.0),
    maxiters: int = Form(5),
):
    if len(files) != len(roles):
        raise HTTPException(
            status_code=422,
            detail=f"got {len(files)} files but {len(roles)} roles; they must match",
        )
    cfg = CalibConfig(sigma=sigma, maxiters=maxiters)
    uploads = []
    for f, role in zip(files, roles):
        payload = await f.read()
        uploads.append((f.filename or "unnamed.fits", role.strip().lower(), payload))

    try:
        result = run_batch(uploads, cfg)
    except BatchError as exc:
        return JSONResponse(status_code=422, content={"errors": exc.reasons})

    archive = build_zip(result, cfg)
    return StreamingResponse(
        io.BytesIO(archive),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="calibration_result.zip"'},
    )
