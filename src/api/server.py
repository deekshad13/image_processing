from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from src.api.routes import router

app = FastAPI(title="Rootstalk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/images", StaticFiles(directory="data/raw"), name="images")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")