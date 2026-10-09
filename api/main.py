"""REST API for the inspection system.

Run:   uvicorn api.main:app --reload
Docs:  http://127.0.0.1:8000/docs   (try every endpoint in the browser)

    curl -F "file=@part.jpeg" "http://127.0.0.1:8000/inspect?framework=keras&explain=true"
"""
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qi.inference import Inspector, png_base64  # noqa: E402

app = FastAPI(title="AI Quality Inspection API",
              description="Detects casting defects in impeller photos (Keras / PyTorch / TFLite), "
                          "explains the decision with Grad-CAM and checks the bore against CAD tolerance.",
              version="1.0.0")
inspector = Inspector()


@app.get("/health")
def health():
    return {"status": "ok", "models": inspector.available, "model_dir": str(inspector.model_dir)}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...),
                  framework: str = Query(None, description="keras | pytorch | tflite (default: first available)"),
                  explain: bool = Query(True, description="return a Grad-CAM heatmap overlay (PNG, base64)"),
                  measure: bool = Query(False, description="also measure the bore vs CAD tolerance")):
    try:
        image = Image.open(file.file)
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Upload a valid image file (jpg/png).")
    try:
        r = inspector.inspect(image, framework, explain=explain, measure=measure)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    r.pop("heatmap", None)
    if "overlay" in r:
        r["gradcam_png_base64"] = png_base64(r.pop("overlay"))
    if "measurement_overlay" in r:
        r["measurement_png_base64"] = png_base64(r.pop("measurement_overlay"))
    return r
