from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import upload, jobs,projects

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(projects.router)


@app.get("/")
def health():
    return {"status": "DEX backend running"}