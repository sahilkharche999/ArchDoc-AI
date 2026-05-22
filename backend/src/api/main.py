from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from src.api.routes import upload, jobs, projects
from src.db.init_jobs_table import init_jobs_table
from src.db.init_job_progress_table import init_job_progress_table
from src.db.connection import get_conn,release_conn
from src.logger import setup_logger
from dotenv import load_dotenv
from src.redis_conn import connect_redis
from src.api.routes import pdf_fixes
load_dotenv()

logger = setup_logger(__name__)
app = FastAPI(
    title="DAX API",
    version="1.0.0",
    description="AI-powered structural estimation engine"
)

@app.on_event("startup")
def startup():
    logger.info("Starting DAX backend...")

    try:
        init_jobs_table()
        init_job_progress_table()
        connect_redis(os.getenv('REDIS_HOST', 'localhost'), int(os.getenv('REDIS_PORT', 6379)))
        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Database initialization failed | error={str(e)}")
        raise

    logger.info("Startup complete")



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    assets_dir = os.getenv("ASSETS_DIR", "/data/assets")
    app.mount("/api/v1/assets", StaticFiles(directory=assets_dir), name="assets")
    logger.info("Static files mounted | path=/api/v1/assets")

except Exception as e:
    logger.error(f"Failed to mount static files | error={str(e)}")

app.include_router(upload.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(projects.router, prefix="/api/v1")
app.include_router(pdf_fixes.router, prefix="/api/v1")

@app.get("/health")
def health():
    checks={}
    overall="ok"

    #Check Redis 
    redis_client=connect_redis()
    try:
        redis_client.ping()
        checks["radis"]="ok"
    except Exception as e:
        checks['radis']=f'faild with exception :{str(e)}'
        overall="degraded"

    # Check Postgres
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        release_conn(conn)
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"failed: {str(e)}"
        overall = "degraded"

    # Check memegraph 

    try:
        from src.infrastructure.graph_db import graph_db
        with graph_db.driver.session() as s:
            s.run("RETURN 1")
        checks["memgraph"] = "ok"
    except Exception as e:
        checks["memgraph"] = f"failed: {str(e)}"
        overall = "degraded"

    status_code=200 if overall == "ok" else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall,
            "service": "DAX API",
            "checks": checks
        }
    )