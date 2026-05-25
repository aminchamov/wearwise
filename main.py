import io
import os

# Keep ONNX/BLAS from allocating thread pools that exhaust Render Free memory.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps
from rembg import new_session, remove

MAX_INPUT_DIMENSION = int(os.getenv("MAX_INPUT_DIMENSION", "768"))
MODEL_NAME = os.getenv("REMBG_MODEL", "u2netp")

app = FastAPI(title="WearWise Background Removal")
session = None


def get_session():
    global session
    if session is None:
        session = new_session(MODEL_NAME)
    return session


@app.get("/")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "service": "wearwise-rembg",
        "model": MODEL_NAME,
        "modelLoaded": session is not None,
    }


@app.post("/remove-background")
async def remove_background(image: UploadFile = File(...)) -> Response:
    try:
        source = Image.open(io.BytesIO(await image.read()))
        source = ImageOps.exif_transpose(source).convert("RGB")
        source.thumbnail((MAX_INPUT_DIMENSION, MAX_INPUT_DIMENSION))

        encoded_input = io.BytesIO()
        source.save(encoded_input, format="JPEG", quality=88, optimize=True)
        foreground_bytes = remove(
            encoded_input.getvalue(),
            session=get_session(),
        )
        foreground = Image.open(io.BytesIO(foreground_bytes)).convert("RGBA")

        canvas = Image.new("RGB", foreground.size, "#FFFFFF")
        canvas.paste(foreground, mask=foreground.getchannel("A"))

        output = io.BytesIO()
        canvas.save(output, format="WEBP", quality=82, method=4)
        return Response(content=output.getvalue(), media_type="image/webp")
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Background removal failed.",
        ) from error
