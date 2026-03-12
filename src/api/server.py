from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title = "Rootstalk API")

app.include_router(router)