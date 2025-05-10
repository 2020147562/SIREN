from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(api_router)

# CORS 설정
origins = [
    "https://your-firebase-app.web.app",  # firebase 백엔드 주소소
    "http://localhost:8000",              # 개발용 로컬 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],  # Authorization, Content-Type 등
)

# 테스트용 기본 경로
@app.get("/")
def root():
    return {"message": "SIREN API 서버 작동 중"}

