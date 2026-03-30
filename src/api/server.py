import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, FileResponse
from src.api.routes import router
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO2_PATH = os.path.join(BASE_DIR, "..", "..", "demo2.html")

app = FastAPI(title="Rootstalk API")

app.include_router(router)

app.mount("/images", StaticFiles(directory="data/raw"), name="images")

@app.get("/")
def root():
    return RedirectResponse(url="/constrained")

@app.get("/constrained")
def constrained():
    return FileResponse(DEMO2_PATH)

@app.get("/debug-paths")
def debug_paths():
    return {
        "cwd": os.getcwd(),
        "data_raw_exists": os.path.exists("data/raw"),
        "contents": os.listdir("data/raw") if os.path.exists("data/raw") else []
    }