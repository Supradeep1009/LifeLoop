"""
LifeLoop – Inference Agent
Runs the LifeLoop environment by calling the LLM via the hackathon's LiteLLM proxy.

Required env vars (injected by the hackathon validator):
  API_BASE_URL  – LiteLLM proxy base URL
  API_KEY       – API key for the proxy
  MODEL_NAME    – (optional) model to use, defaults to "gpt-4o-mini"
"""

import os
import json
import requests
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]          # required – hackathon injects this
API_KEY      = os.environ["API_KEY"]               # required – hackathon injects this
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Base URL of the LifeLoop environment server (the FastAPI app on HF Space)
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# ── OpenAI client pointing at the proxy ──────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a customer-support triage agent.
You will be given the current observation of a ticket queue and must decide the
single best next action to take.

Available actions (respond with ONLY valid JSON, no markdown, no explanation):

1. Read a ticket:
   {"action_type": "read", "ticket_id": "<id>"}

2. Route a ticket to a department:
   {"action_type": "route", "ticket_id": "<id>", "department": "<dept>"}
   Valid departments: IT, Billing, General, Technical

3. Reply to a ticket:
   {"action_type": "reply", "ticket_id": "<id>", "message": "<your message>"}

4. Close a ticket:
   {"action_type": "close", "ticket_id": "<id>"}

Strategy:
- First READ a ticket to learn its content.
- Then decide: route it to a department OR reply (apologize + mention refund
  where relevant) and close it.
- Aim to handle all unhandled tickets.
"""

# ── Environment helpers ───────────────────────────────────────────────────────

def env_reset(task_id: int = 0) -> dict:
    """POST /reset to start a new episode."""
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(session_id: str, action: dict) -> dict:
    """POST /step to apply an action."""
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── LLM decision ─────────────────────────────────────────────────────────────

def choose_action(obs: dict, history: list) -> dict:
    """Ask the LLM (via proxy) for the next action given the current observation."""
    user_msg = (
        f"Current observation:\n{json.dumps(obs, indent=2)}\n\n"
        "What is your next action? Reply with a single JSON object only."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps with ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    action = json.loads(raw)
    # Append to history so the model keeps context
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": raw})
    return action


# ── Main agent loop ───────────────────────────────────────────────────────────

def run_episode(task_id: int = 0, max_steps: int = 20):
    """Reset the environment and run one full episode."""
    reset_data = env_reset(task_id)
    session_id = reset_data["session_id"]
    obs        = reset_data["observation"]

    print("[START]")

    history = []
    for step in range(max_steps):
        action = choose_action(obs, history)
        print(f"[STEP] Observation: {json.dumps(obs)} -> Action: {json.dumps(action)}")

        result = env_step(session_id, action)
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]

        if done:
            print(f"[STEP] Observation: {json.dumps(obs)} -> Action: done")
            break

    print("[END]")
    return reward


def main():
    # Run all three tasks so the validator sees LLM calls for a variety of inputs
    for task_id in range(3):
        try:
            run_episode(task_id=task_id)
        except Exception as e:
            # Don't crash the whole run if one task fails
            print(f"[ERROR] Task {task_id} failed: {e}")


if __name__ == "__main__":
    main()
