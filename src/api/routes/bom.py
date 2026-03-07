from fastapi import APIRouter

router = APIRouter()

memory_store = {}

@router.get("/bom/{job_id}")
def get_bom(job_id: str):
    return memory_store.get(job_id, {})