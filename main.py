from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
import logging

# logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(api_router)

# CORS
origins = [
    "https://handledangerrequest-aoviljd5hq-du.a.run.app",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],  # Authorization, Content-Type etc.
)

# test
@app.get("/")
def root():
    return {"message": "SIREN API 서버 작동 중"}

