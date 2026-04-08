from fastapi import FastAPI, HTTPException, APIRouter, Body, Request
from pydantic import BaseModel
import uuid
import json
from typing import Any, Dict, Optional
import gradio as gr

from models import Observation, Action, Reward
from tasks import get_task
from gradio_app import demo

app = FastAPI(title="LifeLoop")

# In-memory storage for active environments/sessions
sessions = {}

# --- Environment API ---
api_router = APIRouter()

class ResetRequest(BaseModel):
    task_id: int = 0

class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]

@api_router.post("/reset")
async def reset_env(raw_request: Request):
    """
    Reset the environment. Accepts:
      - empty body / no Content-Type
      - {} (empty JSON object)
      - {"task_id": 0}  (full request)
    Meta's automated checker may send any of these.
    """
    task_id = 0
    try:
        body_bytes = await raw_request.body()
        if body_bytes and body_bytes.strip():
            body_json = json.loads(body_bytes)
            if isinstance(body_json, dict) and "task_id" in body_json:
                task_id = int(body_json["task_id"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass  # Fall back to default task_id=0
    session_id = str(uuid.uuid4())
    try:
        task = get_task(task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    sessions[session_id] = task
    obs = task.reset()
    
    return {
        "session_id": session_id,
        "observation": obs.dict()
    }

@api_router.post("/step")
def step_env(request: StepRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    task = sessions[request.session_id]
    
    # Parse action
    action_data = request.action
    action_type = action_data.get("action_type")
    
    try:
        from models import ReadTicket, RouteTicket, ReplyTicket, CloseTicket
        if action_type == "read":
            action = ReadTicket(**action_data)
        elif action_type == "route":
            action = RouteTicket(**action_data)
        elif action_type == "reply":
            action = ReplyTicket(**action_data)
        elif action_type == "close":
            action = CloseTicket(**action_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid action type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Action validation failed: {str(e)}")
        
    obs, reward, done, info = task.step(action)
    
    # Clean up session if done
    if done:
        del sessions[request.session_id]
        
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@api_router.get("/models")
def list_models():
    """Optional but often checked by OpenAI-compatible evaluators."""
    return {
        "object": "list",
        "data": [
            {"id": "lifeloop-v1", "object": "model", "created": 1712500000, "owned_by": "lifeloop"}
        ]
    }

@api_router.get("/health")
def health_check():
    return {"status": "ok"}

app.include_router(api_router)
app.include_router(api_router, prefix="/v1")

# --- Gradio UI ---
# Mount Gradio at /ui to keep API routes accessible at root
# (mounting at / would hijack /reset, /step, etc.)
app = gr.mount_gradio_app(app, demo, path="/ui")

# Health check is now in api_router included above
