
import os
import sys
import subprocess
import signal
import asyncio
import uuid
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

# --- Configuration & Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add parent dir to path if needed for local imports, though we run scripts as subprocesses
sys.path.append(BASE_DIR)

app = FastAPI(title="Adversarial Co-Evolution Dashboard")

# Serve Static Files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Process Storage
class ProcessInfo(BaseModel):
    id: str
    name: str
    command: str
    status: str  # 'running', 'stopped', 'error'
    pid: Optional[int] = None

class ActiveProcess:
    def __init__(self, id: str, name: str, process: subprocess.Popen, command: str):
        self.id = id
        self.name = name
        self.process = process
        self.command = command
        self.log_queue = asyncio.Queue()  # Use queue for websocket broadcasting

processes: Dict[str, ActiveProcess] = {}

# --- Utility: Process Manager ---

async def stream_output(process: subprocess.Popen, process_id: str):
    """
    Reads stdout/stderr from the subprocess and pushes lines to the log queue.
    """
    if process.stdout:
        while True:
            line = await asyncio.to_thread(process.stdout.readline)
            if not line:
                break
            line_decoded = line.decode('utf-8').rstrip()
            if process_id in processes:
                 await processes[process_id].log_queue.put(line_decoded)
    
    # Handle exit
    if process_id in processes:
        await processes[process_id].log_queue.put(f"[SYSTEM] Process exited with code {process.poll()}")
        # Check if we should mark as stopped, but usually we keep the history until cleared?
        # For simplicity, we just log exit.

# --- Endpoints ---

@app.get("/")
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class RunRequest(BaseModel):
    script_name: str # e.g., "ppo_train.py"
    args: Optional[Dict[str, str]] = {} # e.g., {"--timesteps": "1000", "--reward-system": "short"}

@app.post("/api/run")
async def run_script(req: RunRequest):
    """
    Execute a python script as a subprocess.
    """
    script_path = os.path.join(BASE_DIR, req.script_name)
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"Script {req.script_name} not found")

    # Construct Command
    cmd = [sys.executable, "-u", script_path] # -u for unbuffered output
    for key, value in req.args.items():
        cmd.append(key)
        if value: # Handle flags vs key-value pairs
            cmd.append(str(value))
    
    # Special Handling for play_ui.py (Environment variables maybe?)
    env = os.environ.copy()
    
    try:
        # Start Process
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout
            env=env
        )
        
        proc_id = str(uuid.uuid4())[:8]
        active_proc = ActiveProcess(proc_id, req.script_name, proc, " ".join(cmd))
        processes[proc_id] = active_proc
        
        # Start background task to stream logs
        asyncio.create_task(stream_output(proc, proc_id))
        
        return {"status": "success", "process_id": proc_id, "message": f"Started {req.script_name}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop/{process_id}")
async def stop_process(process_id: str):
    if process_id not in processes:
        raise HTTPException(status_code=404, detail="Process not found")
        
    proc = processes[process_id]
    if proc.process.poll() is None: # Still running
        proc.process.terminate()
        try:
            proc.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.process.kill()
            
    # del processes[process_id] # Keep it in list to show 'Stopped' status? 
    # For now, let's keep it but update internal tracking if we were doing more complex state management.
    return {"status": "success", "message": "Process terminated"}

@app.get("/api/status")
async def get_status():
    """
    Get list of all managed processes.
    """
    status_list = []
    ids_to_remove = []
    
    for pid, p in processes.items():
        code = p.process.poll()
        alive = code is None
        state = "running" if alive else f"stopped (code {code})"
        
        status_list.append({
            "id": pid,
            "name": p.name,
            "command": p.command,
            "status": state,
            "pid": p.process.pid
        })
        
    return status_list

@app.delete("/api/process/{process_id}")
async def clear_process(process_id: str):
    """Remove a stopped process from the dashboard list"""
    if process_id in processes:
        if processes[process_id].process.poll() is None:
             raise HTTPException(status_code=400, detail="Cannot clear running process")
        del processes[process_id]
    return {"status": "success"}

# --- Prompt Management ---

class PromptRequest(BaseModel):
    content: str
    
PROMPT_FILE = os.path.join(BASE_DIR, "config", "prompt.txt")

@app.get("/api/prompt")
async def get_prompt():
    """Read current default prompt"""
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            return {"content": f.read()}
    return {"content": ""}

@app.post("/api/prompt")
async def save_prompt(req: PromptRequest):
    """Save new default prompt"""
    try:
        os.makedirs(os.path.dirname(PROMPT_FILE), exist_ok=True)
        with open(PROMPT_FILE, "w") as f:
            f.write(req.content)
        return {"status": "success", "message": "Prompt saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- WebSockets for Logging ---

@app.websocket("/ws/logs/{process_id}")
async def websocket_endpoint(websocket: WebSocket, process_id: str):
    await websocket.accept()
    
    if process_id not in processes:
        await websocket.send_text("[ERROR] Process not found or session expired.")
        await websocket.close()
        return

    proc = processes[process_id]
    
    try:
        # Send initial "Connected" message
        await websocket.send_text(f"[SYSTEM] Connected to logs for {proc.name}...")
        
        while True:
            # Wait for new log line
            line = await proc.log_queue.get()
            await websocket.send_text(line)
            
            # If process is dead and queue is empty, we might want to close?
            # actually we keep it open so user can see final logs.
            
    except WebSocketDisconnect:
        print(f"Client disconnected from {process_id} logs")

if __name__ == "__main__":
    import uvicorn
    # Allow running directly: python dashboard/app.py
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8001, reload=True)
