---
title: LifeLoop
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# LifeLoop

A complete, reproducible `OpenEnv` environment centered around **LifeLoop**, designed for the Meta Hackathon Task.

## Overview
This environment exposes an API mimicking a customer support queue. The agent's goal is to read tickets and appropriately route, reply to, or close them based on their content.

## Setup Local Environment

```bash
# Install requirements
pip install -r requirements.txt

# Start the environment API via FastAPI
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Running the Agent Evaluator
Run the provided inference script which will interact with the environment and output standard formatted STDOUT:

```bash
export API_BASE_URL="http://localhost:7860/v1"
python inference.py
```

## Structure
- `models.py`: Pydantic schemes for the state and tool definitions.
- `tasks.py`: Implementation of 3 tasks with increasing difficulty (Easy, Medium, Hard).
- `main.py`: FastAPI server bridging to the tasks.
- `inference.py`: Baseline agent code interacting with the `OpenAI` client.
