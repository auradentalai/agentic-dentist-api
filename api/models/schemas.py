"""
Pydantic schemas for the Agentic Dentist API.
"""

from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime


# ============================================
# Agent Context — what agents receive
# ============================================
class AgentContext(BaseModel):
    workspace_id: str
    patient_ref: Optional[str] = None
    provider_ref: Optional[str] = None
    intent: Optional[str] = None
    prior_outputs: dict = {}
    allowed_tools: list[str] = []
    phi_access: bool = False
    max_steps: int = 10
    timeout_ms: int = 30000


# ============================================
# Trigger Events
# ============================================
class TriggerEvent(BaseModel):
    event_type: Literal[
        "inbound_call",
        "inbound_sms",
        "web_chat",
        "manual_trigger",
        "scheduled_job",
        "system_event",
    ]
    workspace_id: str
    patient_ref: Optional[str] = None
    provider_ref: Optional[str] = None
    payload: dict = {}


# ============================================
# Agent Run
# ============================================
class AgentRunRequest(BaseModel):
    """Manual trigger to run a specific agent."""
    workspace_id: str
    agent: Literal["concierge", "diagnostician", "liaison", "auditor"]
    patient_ref: Optional[str] = None
    intent: Optional[str] = None
    payload: dict = {}


class AgentRunResponse(BaseModel):
    run_id: str
    agent: str
    status: Literal["completed", "escalated", "error"]
    output: dict = {}
    steps: int = 0
    duration_ms: int = 0


# ============================================
# Orchestrator
# ============================================
class InteractionState(BaseModel):
    """Full state tracked by the Orchestrator across an interaction."""
    interaction_id: str
    workspace_id: str
    patient_ref: Optional[str] = None
    provider_ref: Optional[str] = None
    trigger_event: TriggerEvent
    current_agent: Optional[str] = None
    pattern: Literal["sequential", "parallel", "conditional"] = "sequential"
    agent_outputs: dict = {}
    escalated: bool = False
    escalation_reason: Optional[str] = None
    completed: bool = False
    steps: int = 0
    started_at: Optional[datetime] = None


# ============================================
# Health
# ============================================
class HealthResponse(BaseModel):
    status: str
    environment: str
    llm_provider: str
    llm_model_primary: str
    llm_model_fast: str
    supabase_connected: bool
    agents: list[str]
