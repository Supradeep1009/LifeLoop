from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal

class Ticket(BaseModel):
    ticket_id: str
    body: str

class Observation(BaseModel):
    current_ticket_id: Optional[str] = Field(default=None, description="The ID of the ticket currently being viewed, if any.")
    ticket_details: Optional[str] = Field(default=None, description="The body of the current ticket.")
    unhandled_tickets_count: int = Field(description="Number of tickets remaining in the queue.")
    departments: List[str] = Field(description="List of valid departments to route tickets to.")

class ReadTicket(BaseModel):
    action_type: Literal["read"] = "read"
    ticket_id: str

class RouteTicket(BaseModel):
    action_type: Literal["route"] = "route"
    ticket_id: str
    department: str

class ReplyTicket(BaseModel):
    action_type: Literal["reply"] = "reply"
    ticket_id: str
    message: str

class CloseTicket(BaseModel):
    action_type: Literal["close"] = "close"
    ticket_id: str

Action = Union[ReadTicket, RouteTicket, ReplyTicket, CloseTicket]

class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
