import gradio as gr
import requests

API_URL = "http://localhost:7860"

def format_obs(obs):
    if not obs:
        return "No observation yet.", "", "", ""
    return (
        str(obs.get("unhandled_tickets_count", 0)),
        obs.get("current_ticket_id", "") or "None",
        obs.get("ticket_details", "") or "None",
        ", ".join(obs.get("departments", []))
    )

def init_env(task_id):
    try:
        resp = requests.post(f"{API_URL}/init", json={"task_id": int(task_id)})
        resp.raise_for_status()
        data = resp.json()
        session_id = data["session_id"]
        obs = data["observation"]
        unhandled, current_id, details, depts = format_obs(obs)
        return (
            session_id,         # session_id_state
            session_id,         # session_text
            unhandled,          # unhandled_tickets
            current_id,         # current_ticket_id
            details,            # ticket_details
            depts,              # valid_departments
            "Environment Initialized.", # status_msg
            "0.0",              # reward_score
            "False"             # done_status
        )
    except Exception as e:
        return "", "", "", "", "", "", f"Error initializing: {str(e)}", "0.0", "False"

def step_action(session_id, action_payload):
    if not session_id:
        return gr.update(), gr.update(), gr.update(), gr.update(), "Error: No active session. Please initialize first.", gr.update(), gr.update()
    try:
        req = {
            "session_id": session_id,
            "action": action_payload
        }
        resp = requests.post(f"{API_URL}/step", json=req)
        resp.raise_for_status()
        data = resp.json()
        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]
        unhandled, current_id, details, depts = format_obs(obs)
        return (
            unhandled,
            current_id,
            details,
            depts,
            "Action submitted successfully.",
            str(reward.get("score")),
            str(done)
        )
    except Exception as e:
        # Instead of clearing everything, we return gr.update() for fields we aren't changing
        return gr.update(), gr.update(), gr.update(), gr.update(), f"Error submitting action: {str(e)}", gr.update(), gr.update()

def read_action(session_id, ticket_id):
    payload = {"action_type": "read", "ticket_id": ticket_id}
    return step_action(session_id, payload)

def route_action(session_id, ticket_id, department):
    payload = {"action_type": "route", "ticket_id": ticket_id, "department": department}
    return step_action(session_id, payload)

def reply_action(session_id, ticket_id, message):
    payload = {"action_type": "reply", "ticket_id": ticket_id, "message": message}
    return step_action(session_id, payload)

def close_action(session_id, ticket_id):
    payload = {"action_type": "close", "ticket_id": ticket_id}
    return step_action(session_id, payload)

with gr.Blocks(title="LifeLoop UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LifeLoop Interactive Environment")
    gr.Markdown("Select a task, initialize the session, and act to resolve support tickets manually.")
    
    session_id_state = gr.State("")
    
    with gr.Row():
        task_dropdown = gr.Dropdown(choices=[0, 1, 2], label="Select Task ID", value=0)
        init_btn = gr.Button("Initialize Session", variant="primary")
        
    with gr.Row():
        session_text = gr.Textbox(label="Session ID", interactive=False)
        status_msg = gr.Textbox(label="Status Message", interactive=False, value="Awaiting initialization...")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Observation")
            with gr.Group():
                unhandled_tickets = gr.Textbox(label="Unhandled Tickets Count", interactive=False)
                current_ticket_id = gr.Textbox(label="Current Ticket ID", interactive=False)
                ticket_details = gr.TextArea(label="Ticket Details", interactive=False, lines=4)
                valid_departments = gr.Textbox(label="Valid Departments", interactive=False)
            
            gr.Markdown("### Environment Status")
            with gr.Row():
                reward_score = gr.Textbox(label="Reward Score", interactive=False)
                done_status = gr.Textbox(label="Task Done", interactive=False)
                
        with gr.Column(scale=1):
            gr.Markdown("### Action Control Panel")
            with gr.Tabs():
                with gr.Tab("Read Ticket"):
                    read_tid = gr.Textbox(label="Ticket ID (e.g. T001, T002)")
                    read_btn = gr.Button("Read", variant="secondary")
                with gr.Tab("Route Ticket"):
                    route_tid = gr.Textbox(label="Ticket ID")
                    route_dept = gr.Dropdown(choices=["IT", "Billing", "General", "Technical"], label="Department")
                    route_btn = gr.Button("Route", variant="secondary")
                with gr.Tab("Reply Ticket"):
                    reply_tid = gr.Textbox(label="Ticket ID")
                    reply_msg = gr.TextArea(label="Message Box", lines=3)
                    reply_btn = gr.Button("Send Reply", variant="secondary")
                with gr.Tab("Close Ticket"):
                    close_tid = gr.Textbox(label="Ticket ID")
                    close_btn = gr.Button("Close Ticket", variant="secondary")

    # Collect outputs for updating after actions
    obs_outputs = [unhandled_tickets, current_ticket_id, ticket_details, valid_departments, status_msg, reward_score, done_status]
    
    # Wire events
    init_btn.click(
        fn=init_env,
        inputs=[task_dropdown],
        outputs=[session_id_state, session_text] + obs_outputs
    )
    
    read_btn.click(fn=read_action, inputs=[session_id_state, read_tid], outputs=obs_outputs)
    route_btn.click(fn=route_action, inputs=[session_id_state, route_tid, route_dept], outputs=obs_outputs)
    reply_btn.click(fn=reply_action, inputs=[session_id_state, reply_tid, reply_msg], outputs=obs_outputs)
    close_btn.click(fn=close_action, inputs=[session_id_state, close_tid], outputs=obs_outputs)

if __name__ == "__main__":
    # Runs on port 7861 to not conflict with FastAPI on 7860
    demo.launch(server_port=7861)
