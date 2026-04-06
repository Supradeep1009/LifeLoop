from models import Observation, Reward

class BaseTask:
    def __init__(self):
        self.tickets = {}
        self.ticket_states = {}
        self.current_ticket_id = None
        self.departments = ["IT", "Billing", "General", "Technical"]

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_observation(self) -> Observation:
        unhandled = len([tid for tid, state in self.ticket_states.items() if state == "open"])
        details = self.tickets.get(self.current_ticket_id) if self.current_ticket_id else None
        return Observation(
            current_ticket_id=self.current_ticket_id,
            ticket_details=details,
            unhandled_tickets_count=unhandled,
            departments=self.departments
        )

class Task0(BaseTask):
    # Easy: A single open ticket about a password reset. Route to IT.
    def reset(self):
        self.tickets = {"T001": "I forgot my password, can you please reset it?"}
        self.ticket_states = {"T001": "open"}
        self.current_ticket_id = None
        return self.get_observation()

    def step(self, action):
        if action.action_type == "read":
            if action.ticket_id in self.tickets:
                self.current_ticket_id = action.ticket_id
        elif action.action_type == "route":
            if action.ticket_id == "T001" and action.department == "IT":
                self.ticket_states["T001"] = "routed"
        
        # Calculate Reward
        score = 1.0 if self.ticket_states.get("T001") == "routed" else 0.0
        done = score == 1.0
        return self.get_observation(), Reward(score=score), done, {}

class Task1(BaseTask):
    # Medium: Requesting a refund. Reply with an apology and 'refund', then close.
    def reset(self):
        self.tickets = {"T002": "I hate this product! I demand a refund right now!"}
        self.ticket_states = {"T002": "open"}
        self.current_ticket_id = None
        self.replied_properly = False
        return self.get_observation()

    def step(self, action):
        if action.action_type == "read":
            if action.ticket_id in self.tickets:
                self.current_ticket_id = action.ticket_id
        elif action.action_type == "reply":
            if action.ticket_id == "T002" and "refund" in action.message.lower() and ("sorry" in action.message.lower() or "apologize" in action.message.lower()):
                self.replied_properly = True
        elif action.action_type == "close":
            if action.ticket_id == "T002":
                self.ticket_states["T002"] = "closed"
        elif action.action_type == "route":
            self.replied_properly = False # Penalty for routing instead
                
        # Calculate Reward
        score = 0.0
        if self.replied_properly:
            score += 0.5
        if self.ticket_states.get("T002") == "closed":
            score += 0.5
            
        done = self.ticket_states.get("T002") == "closed"
        return self.get_observation(), Reward(score=score), done, {}

class Task2(BaseTask):
    # Hard: Mix of 3 tickets
    def reset(self):
        self.tickets = {
            "T003": "My server is throwing a 500 internal server error.", # Technical
            "T004": "Please update my credit card info.", # Billing
            "T005": "What are your business hours?" # General
        }
        self.ticket_states = {k: "open" for k in self.tickets.keys()}
        self.current_ticket_id = None
        self.correct_steps = {"T003": False, "T004": False, "T005": False}
        return self.get_observation()

    def step(self, action):
        if action.action_type == "read":
            if action.ticket_id in self.tickets:
                self.current_ticket_id = action.ticket_id
        elif action.action_type == "route":
            if action.ticket_id == "T003" and action.department == "Technical":
                self.correct_steps["T003"] = True
                self.ticket_states["T003"] = "routed"
        elif action.action_type == "reply":
            pass # Simplified: any reply gets closer to close
        elif action.action_type == "close":
            if action.ticket_id in ["T004", "T005"]:
                self.correct_steps[action.ticket_id] = True
                self.ticket_states[action.ticket_id] = "closed"
                
        # Calculate Reward
        handled = sum(1 for v in self.correct_steps.values() if v)
        score = handled / 3.0
        done = handled == 3
        return self.get_observation(), Reward(score=score), done, {}

def get_task(task_id: int):
    if task_id == 0:
        return Task0()
    elif task_id == 1:
        return Task1()
    elif task_id == 2:
        return Task2()
    else:
        raise ValueError("Invalid task ID")
