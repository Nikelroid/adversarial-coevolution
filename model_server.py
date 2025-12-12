import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List
import uuid
import asyncio
from datetime import datetime
from utils.config import get_config

# --- Master Node Configuration ---
CONFIG = get_config()
DIST_CONFIG = CONFIG.get("distributed", {}).get("master", {})
PORT = DIST_CONFIG.get("port", 8000)

# Queues for different job types
# fast: action, evaluation (high priority, short latency)
# slow: prompt enhancing (low priority, long duration)
QUEUE_NAMES = DIST_CONFIG.get("queues", {"fast": "fast", "slow": "slow"})
QUEUES = {
    "fast": asyncio.Queue(),
    "slow": asyncio.Queue()
}

# Results storage: job_id -> result
RESULTS: Dict[str, Dict[str, Any]] = {}

# Active Workers
WORKERS: Dict[str, Dict[str, Any]] = {}


app = FastAPI(title="LLM Master Node")

# --- Data Models ---

class JobSubmission(BaseModel):
    type: str = Field(..., description="Job type: 'action', 'eval', 'enhance'")
    prompt: str
    options: Optional[Dict[str, Any]] = {}
    priority: str = "fast" # 'fast' or 'slow'

class JobResult(BaseModel):
    job_id: str
    result: str
    worker_id: str
    duration: float

class WorkerRegistration(BaseModel):
    worker_id: str
    worker_type: str # 'fast', 'slow'
    model_id: str

class SyncResponse(BaseModel):
    response: str
    
# --- Endpoints ---

@app.post("/register_worker")
async def register_worker(worker: WorkerRegistration):
    WORKERS[worker.worker_id] = {
        "type": worker.worker_type,
        "model": worker.model_id,
        "last_seen": datetime.now()
    }
    print(f"Worker registered: {worker.worker_id} ({worker.worker_type})")
    return {"status": "registered"}

@app.post("/api/generate", response_model=SyncResponse)
async def submit_sync_job(payload: JobSubmission):
    """
    Simulates the synchronous Ollama API but routes via the distributed queue.
    Waits for the result (long polling).
    """
    # Determine queue based on job type or priority override
    queue_name = payload.priority
    if payload.type == "enhance":
        queue_name = "slow"
    
    # Create Job
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "type": payload.type,
        "payload": {
            "prompt": payload.prompt,
            "options": payload.options
        },
        "submitted_at": datetime.now(),
        "status": "pending"
    }
    
    # Enqueue
    if queue_name not in QUEUES:
        raise HTTPException(status_code=400, detail="Invalid priority queue")
        
    await QUEUES[queue_name].put(job)
    print(f"Job {job_id} ({payload.type}) queued in {queue_name}")

    # Wait for result (Polling loop)
    # Timeout 60s for fast, 300s for slow
    timeout = 60 if queue_name == "fast" else 300
    start_time = asyncio.get_event_loop().time()
    
    while True:
        if job_id in RESULTS:
            result = RESULTS.pop(job_id)
            return {"response": result["result"]}
        
        if asyncio.get_event_loop().time() - start_time > timeout:
             raise HTTPException(status_code=504, detail="Job timed out")
             
        await asyncio.sleep(0.1)


@app.get("/get_job")
async def get_job(worker_type: str = "fast"):
    """
    Workers poll this endpoint to get jobs.
    """
    queue = QUEUES.get(worker_type)
    if not queue:
        return None
        
    if queue.empty():
        return None
        
    # Get job without blocking HTTP request too long
    try:
        job = queue.get_nowait()
        return job
    except asyncio.QueueEmpty:
        return None

@app.post("/submit_result")
async def submit_result(result: JobResult):
    """
    Workers submit completed results here.
    """
    RESULTS[result.job_id] = result.dict()
    print(f"Result received for {result.job_id} from {result.worker_id}")
    
    # Handle "Enhance" Result side-effects (overwrite prompt)
    # In a real system, we might separate this logic, but here the Master manages config.
    # We need to know the job type logic, but we stored just the result.
    # Ideally, we inspect the result content or metadata.
    # For now, we assume the Client handles the prompt update via the response.
    # OR, if the Master is responsible for updating "config/prompt.txt", we'd do it here.
    # But the Prompt Enhancer Agent likely receives the new prompt and saves it itself.
    # Let's keep Master simple: it just passes messages.
    
    return {"status": "received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)