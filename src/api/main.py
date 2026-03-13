from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.routes import upload, jobs,projects
from src.db.init_jobs_table import init_jobs_table
from src.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="DEX API",
    version="1.0.0",
    description="AI-powered structural estimation engine"
)

@app.on_event("startup")
def startup():
    init_jobs_table()
    logger.info("DEX backend starting...")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/api/v1/assets", StaticFiles(directory="assets"), name="assets")

app.include_router(upload.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(projects.router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "service": "DEX API"}