---
title: LifeLoop
emoji: 🎫
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
---
# LifeLoop — Customer Support Triage Demo

An interactive **AI agent evaluation environment** for Customer Support Triage, built for the **Meta Hackathon** using the [OpenEnv](https://github.com/openenv) framework.

## Overview
LifeLoop simulates a customer support queue. The goal is to triage tickets by reading them and then routing, replying to, or closing each one correctly to maximise the reward score.

Three tasks of increasing difficulty are included:
- **Task 0 (Easy)** – Single password-reset ticket → route to IT
- **Task 1 (Medium)** – Angry refund request → reply with apology + "refund", then close
- **Task 2 (Hard)** – Three mixed tickets → route/close each correctly

## Running Locally

```bash
# Install requirements
pip install -r requirements.txt

# Launch the Gradio demo (self-contained, no separate server needed)
python gradio_app.py
```

The app will be available at `http://localhost:7860`.

## Running the Agent Evaluator (optional)

```bash
# Start FastAPI backend separately
uvicorn main:app --host 0.0.0.0 --port 7860

# In another terminal
export API_BASE_URL="http://localhost:7860/v1"
python inference.py
```

## Project Structure
- `gradio_app.py`: Self-contained Gradio demo (used by Hugging Face Spaces).
- `models.py`: Pydantic schemas for observations, actions, and rewards.
- `tasks.py`: Three tasks with increasing difficulty.
- `main.py`: FastAPI server (for programmatic / agent access).
- `inference.py`: Baseline agent using the OpenAI client.
