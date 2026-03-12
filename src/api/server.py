from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api.routes import router

app = FastAPI(title="Rootstalk API")

app.include_router(router)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")