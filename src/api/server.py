from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api.routes import router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Rootstalk API")

app.include_router(router)
app.mount("/images", StaticFiles(directory="data/raw"), name="images")

@app.get("/")
def root():
    return RedirectResponse(url="/demo")

@app.get("/demo")
def demo():
    return FileResponse("demo.html")
