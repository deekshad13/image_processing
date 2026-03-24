from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api.routes import router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.config import RAW_DIR

app = FastAPI(title="Rootstalk API")

app.include_router(router)
app.mount("/images", StaticFiles(directory=RAW_DIR), name="images")

@app.get("/")
def root():
    return RedirectResponse(url="/demo")

@app.get("/demo")
def demo():
    return FileResponse("demo.html")


@app.get("/constrained")
def constrained():
    return FileResponse("demo2.html")

import threading

def open_browser():
    import time
    time.sleep(1.5)
    print("\n🌿 Rootstalk is running!")
    print("   Demo:        http://127.0.0.1:8000/demo")
    print("   Constrained: http://127.0.0.1:8000/constrained")
    print("   API Docs:    http://127.0.0.1:8000/docs\n")

threading.Thread(target=open_browser, daemon=True).start()