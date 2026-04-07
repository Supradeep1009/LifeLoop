from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import uuid
from typing import Any, Dict
import gradio as gr

from models import Observation, Action, Reward
from tasks import get_task
from gradio_app import demo

app = FastAPI(title="LifeLoop")

# In-memory storage for active environments/sessions
sessions = {}

# --- Environment API (v1) ---
v1_router = APIRouter(prefix="/v1")

class ResetRequest(BaseModel):
    task_id: int

class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]

@v1_router.get("/")
def v1_root():
    return {"message": "LifeLoop Environment API v1"}

@v1_router.post("/reset")
def reset_env(request: ResetRequest):
    session_id = str(uuid.uuid4())
    try:
        task = get_task(request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    sessions[session_id] = task
    obs = task.reset()
    
    return {
        "session_id": session_id,
        "observation": obs.dict()
    }

@v1_router.post("/step")
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

@v1_router.get("/models")
def list_models():
    """Optional but often checked by OpenAI-compatible evaluators."""
    return {
        "object": "list",
        "data": [
            {"id": "lifeloop-v1", "object": "model", "created": 1712500000, "owned_by": "lifeloop"}
        ]
    }

app.include_router(v1_router)

# --- Gradio UI ---
# Mount the Gradio app to the root
app = gr.mount_gradio_app(app, demo, path="/")

@app.get("/health")
def health_check():
    return {"status": "ok"}
