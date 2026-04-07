"""
LifeLoop – Customer Support Triage Demo
A self-contained Gradio app that runs the LifeLoop OpenEnv environment
directly without needing a separate FastAPI server.
"""

import gradio as gr
from tasks import Task0, Task1, Task2
from models import ReadTicket, RouteTicket, ReplyTicket, CloseTicket

# ── In-process session store ──────────────────────────────────────────────────
_sessions: dict = {}          # session_key -> task instance
_current_task = [None]        # holds the active task object in this demo


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_obs(obs):
    """Convert an Observation object into display strings."""
    if obs is None:
        return "—", "—", "Awaiting initialization…", "IT, Billing, General, Technical"
    return (
        str(obs.unhandled_tickets_count),
        obs.current_ticket_id or "—",
        obs.ticket_details or "No ticket loaded yet.",
        ", ".join(obs.departments),
    )


def _status(msg: str, ok: bool = True) -> str:
    prefix = "✅" if ok else "❌"
    return f"{prefix}  {msg}"


# ── Button callbacks ──────────────────────────────────────────────────────────

def init_env(task_id: int):
    task_map = {0: Task0, 1: Task1, 2: Task2}
    task = task_map[int(task_id)]()
    obs = task.reset()
    _current_task[0] = task

    unhandled, cur_id, details, depts = _fmt_obs(obs)
    return (
        unhandled,                          # unhandled_out
        cur_id,                             # cur_id_out
        details,                            # details_out
        depts,                              # depts_out
        _status(f"Task {task_id} initialised — ready to triage!"),  # status_out
        "0.00",                             # score_out
        "No",                               # done_out
    )


def _run_action(action):
    task = _current_task[0]
    if task is None:
        return "—", "—", "—", "—", _status("No active session. Please initialise first.", ok=False), "0.00", "No"
    obs, reward, done, _ = task.step(action)
    unhandled, cur_id, details, depts = _fmt_obs(obs)
    done_str = "Yes 🎉" if done else "No"
    status = _status("Action applied successfully." if not done else "Task complete! All tickets resolved.")
    return unhandled, cur_id, details, depts, status, f"{reward.score:.2f}", done_str


def read_ticket(ticket_id: str):
    tid = ticket_id.strip()
    if not tid:
        return "—", "—", "—", "—", _status("Please enter a Ticket ID.", ok=False), gr.update(), gr.update()
    return _run_action(ReadTicket(ticket_id=tid))


def route_ticket(ticket_id: str, department: str):
    tid = ticket_id.strip()
    if not tid or not department:
        return "—", "—", "—", "—", _status("Please enter both Ticket ID and Department.", ok=False), gr.update(), gr.update()
    return _run_action(RouteTicket(ticket_id=tid, department=department))


def reply_ticket(ticket_id: str, message: str):
    tid, msg = ticket_id.strip(), message.strip()
    if not tid or not msg:
        return "—", "—", "—", "—", _status("Please enter Ticket ID and a message.", ok=False), gr.update(), gr.update()
    return _run_action(ReplyTicket(ticket_id=tid, message=msg))


def close_ticket(ticket_id: str):
    tid = ticket_id.strip()
    if not tid:
        return "—", "—", "—", "—", _status("Please enter a Ticket ID.", ok=False), gr.update(), gr.update()
    return _run_action(CloseTicket(ticket_id=tid))


# ── Shared output list helper ─────────────────────────────────────────────────

def _obs_outputs(unhandled, cur_id, details, depts, status, score, done):
    return [unhandled, cur_id, details, depts, status, score, done]


# ── UI ────────────────────────────────────────────────────────────────────────

css = """
/* ── Base ── */
body, .gradio-container { background: #0f111a !important; color: #e2e8f0 !important; }

/* ── Banner ── */
#banner-title { font-size: 2rem !important; font-weight: 800 !important; color: #a78bfa !important; }
#banner-sub   { color: #94a3b8 !important; font-size: 0.97rem !important; }

/* ── Section headers ── */
.section-label {
    background: linear-gradient(90deg, #6d28d9, #4f46e5);
    color: #fff !important;
    padding: 4px 14px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    display: inline-block;
    margin-bottom: 6px;
}

/* ── Panels ── */
.panel-box {
    background: #1a1d2e !important;
    border: 1px solid #2d3152 !important;
    border-radius: 12px !important;
    padding: 18px !important;
}

/* ── Inputs / textboxes ── */
textarea, input[type=text] {
    background: #141625 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3152 !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.primary-btn {
    background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    transition: opacity .2s;
}
.primary-btn:hover { opacity: 0.85 !important; }

.action-btn {
    background: #1e2136 !important;
    color: #a78bfa !important;
    border: 1px solid #4f46e5 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background .2s;
}
.action-btn:hover { background: #2d3152 !important; }

/* ── Status box ── */
#status-box textarea {
    color: #86efac !important;
    font-family: monospace !important;
    font-size: 0.88rem !important;
}

/* ── Score / done highlights ── */
#score-box textarea { color: #fbbf24 !important; font-weight: 700 !important; }
#done-box  textarea { color: #38bdf8 !important; font-weight: 700 !important; }

/* ── Tabs ── */
.tab-nav button { color: #94a3b8 !important; border-color: transparent !important; }
.tab-nav button.selected { color: #a78bfa !important; border-color: #6d28d9 !important; }
"""

with gr.Blocks(css=css, title="LifeLoop Demo") as demo:

    # ── Banner ──────────────────────────────────────────────────────────────
    gr.Markdown("# LifeLoop Demo", elem_id="banner-title")
    gr.Markdown(
        "An **AI agent evaluation environment** for Customer Support Triage — load a task, "
        "then use the action controls to read, route, reply to, or close tickets. "
        "Built for the **Meta Hackathon** using the OpenEnv framework.",
        elem_id="banner-sub",
    )

    # ── Initialise row ───────────────────────────────────────────────────────
    with gr.Row():
        task_drop = gr.Dropdown(
            choices=[(f"Task {i} – {d}", i) for i, d in enumerate(
                ["Easy (password reset → IT)", "Medium (refund → reply + close)", "Hard (3 mixed tickets)"]
            )],
            value=0,
            label="Select Task",
            interactive=True,
        )
        init_btn = gr.Button("▶  Initialise Session", elem_classes="primary-btn", scale=0, min_width=200)

    # ── Main layout ──────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left column: Observation ─────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="panel-box"):
            gr.HTML('<span class="section-label">📋  Observation</span>')

            unhandled_out = gr.Textbox(label="Unhandled Tickets", interactive=False, value="—")
            cur_id_out    = gr.Textbox(label="Current Ticket ID",  interactive=False, value="—")
            details_out   = gr.Textbox(label="Ticket Content", interactive=False, lines=5, value="Awaiting initialization…")
            depts_out     = gr.Textbox(label="Valid Departments", interactive=False, value="IT, Billing, General, Technical")

            gr.HTML('<span class="section-label">📊  Environment Status</span>')
            with gr.Row():
                score_out = gr.Textbox(label="Reward Score", interactive=False, value="0.00", elem_id="score-box")
                done_out  = gr.Textbox(label="Task Complete?", interactive=False, value="No",  elem_id="done-box")

            gr.HTML('<span class="section-label">📡  Status</span>')
            status_out = gr.Textbox(
                label="",
                interactive=False,
                value="Awaiting initialization…",
                elem_id="status-box",
                lines=2,
            )

        # ── Right column: Actions ────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes="panel-box"):
            gr.HTML('<span class="section-label">🎮  Action Control Panel</span>')

            with gr.Tabs():
                # ── Read ────────────────────────────────────────────────────
                with gr.Tab("📖  Read Ticket"):
                    gr.Markdown(
                        "*Read a ticket to load its content into the Observation panel.*\n\n"
                        "Available IDs depend on the task: **T001**, **T002**, **T003**, **T004**, **T005**."
                    )
                    read_tid = gr.Textbox(label="Ticket ID", placeholder="e.g. T001")
                    read_btn = gr.Button("Read Ticket", elem_classes="action-btn")

                # ── Route ───────────────────────────────────────────────────
                with gr.Tab("📤  Route Ticket"):
                    gr.Markdown("*Route the ticket to the correct department.*")
                    route_tid  = gr.Textbox(label="Ticket ID", placeholder="e.g. T001")
                    route_dept = gr.Dropdown(
                        choices=["IT", "Billing", "General", "Technical"],
                        label="Department",
                        value="IT",
                    )
                    route_btn = gr.Button("Route Ticket", elem_classes="action-btn")

                # ── Reply ───────────────────────────────────────────────────
                with gr.Tab("💬  Reply to Ticket"):
                    gr.Markdown(
                        "*Reply to the customer. For a refund task, include 'refund' and 'sorry' / 'apologize'.*"
                    )
                    reply_tid = gr.Textbox(label="Ticket ID", placeholder="e.g. T002")
                    reply_msg = gr.Textbox(label="Your Message", lines=4, placeholder="Type your reply here…")
                    reply_btn = gr.Button("Send Reply", elem_classes="action-btn")

                # ── Close ───────────────────────────────────────────────────
                with gr.Tab("✅  Close Ticket"):
                    gr.Markdown("*Close a resolved ticket to mark it as done.*")
                    close_tid = gr.Textbox(label="Ticket ID", placeholder="e.g. T002")
                    close_btn = gr.Button("Close Ticket", elem_classes="action-btn")

    # ── Shared output list ───────────────────────────────────────────────────
    _outs = [unhandled_out, cur_id_out, details_out, depts_out, status_out, score_out, done_out]

    # ── Wire events ─────────────────────────────────────────────────────────
    init_btn.click(fn=init_env,    inputs=[task_drop],                      outputs=_outs)
    read_btn.click(fn=read_ticket, inputs=[read_tid],                       outputs=_outs)
    route_btn.click(fn=route_ticket, inputs=[route_tid, route_dept],        outputs=_outs)
    reply_btn.click(fn=reply_ticket, inputs=[reply_tid, reply_msg],         outputs=_outs)
    close_btn.click(fn=close_ticket, inputs=[close_tid],                    outputs=_outs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
