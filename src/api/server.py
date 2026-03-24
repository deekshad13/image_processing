import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, FileResponse
from src.api.routes import router
from fastapi.staticfiles import StaticFiles
from src.config import RAW_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO2_PATH = os.path.join(BASE_DIR, "..", "..", "demo2.html")

app = FastAPI(title="Rootstalk API")

app.include_router(router)

if os.path.exists(RAW_DIR):
    app.mount("/images", StaticFiles(directory=RAW_DIR), name="images")

@app.get("/")
def root():
    return RedirectResponse(url="/constrained")

@app.get("/constrained")
def constrained():
    return FileResponse(DEMO2_PATH)