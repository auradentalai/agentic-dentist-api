"""
Orchestrator — the central nervous system of the agent swarm.

Uses LangGraph to coordinate agents through interaction patterns:
- Sequential: Concierge → Diagnostician → Liaison (standard patient flow)
- Parallel: Diagnostician + Auditor run simultaneously (briefing + compliance)
- Conditional: Route based on intent classification
"""

import uuid
from datetime import datetime
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from api.models.schemas import InteractionState, TriggerEvent
from api.agents.concierge.agent import run_concierge
from api.agents.diagnostician.agent import run_diagnostician
from api.agents.liaison.agent import run_liaison
from api.agents.auditor.agent import run_auditor
from api.services.supabase_client import log_audit_event


class OrchestratorState(TypedDict):
    interaction_id: str
    workspace_id: str
    patient_ref: str | None
    provider_ref: str | None
    trigger_type: str
    intent: str | None
    payload: dict
    agent_outputs: dict
    current_agent: str | None
    escalated: bool
    escalation_reason: str | None
    completed: bool
    steps: int


def classify_intent(state: OrchestratorState) -> OrchestratorState:
    """Step 1: Classify the inbound event intent."""
    payload = state["payload"]
    trigger = state["trigger_type"]

    # Simple intent classification — will be replaced by LLM classification
    if trigger == "inbound_call":
        state["intent"] = payload.get("intent", "appointment_request")
    elif trigger == "inbound_sms":
        text = payload.get("text", "").lower()
        if any(w in text for w in ["cancel", "reschedule", "move"]):
            state["intent"] = "schedule_change"
        elif any(w in text for w in ["confirm", "yes"]):
            state["intent"] = "appointment_confirm"
        elif any(w in text for w in ["bill", "pay", "charge", "insurance"]):
            state["intent"] = "billing_inquiry"
        else:
            state["intent"] = "general_inquiry"
    elif trigger == "manual_trigger":
        state["intent"] = payload.get("intent", "manual")
    elif trigger == "scheduled_job":
        state["intent"] = payload.get("job_type", "recall_campaign")
    else:
        state["intent"] = "unknown"

    state["steps"] += 1
    return state


def route_to_agent(state: OrchestratorState) -> str:
    """Routing function — decides which agent runs next based on intent."""
    intent = state["intent"]
    outputs = state["agent_outputs"]

    # If already escalated, end
    if state["escalated"]:
        return "finalize"

    # First pass — always start with Concierge for inbound
    if "concierge" not in outputs and state["trigger_type"] in ("inbound_call", "inbound_sms", "web_chat"):
        return "concierge"

    # After Concierge, route by intent
    if "concierge" in outputs:
        refined_intent = outputs["concierge"].get("refined_intent", intent)

        if refined_intent in ("clinical_question", "treatment_plan", "chart_review"):
            if "diagnostician" not in outputs:
                return "diagnostician"

        if refined_intent in ("billing_inquiry", "insurance_question"):
            if "auditor" not in outputs:
                return "auditor"

        # After specialist, Liaison sends communications
        if "diagnostician" in outputs or "auditor" in outputs:
            if "liaison" not in outputs:
                return "liaison"

    # Manual triggers go directly to the specified agent
    if state["trigger_type"] == "manual_trigger":
        target = state["payload"].get("agent", "concierge")
        if target not in outputs:
            return target

    # Scheduled jobs go to Liaison
    if state["trigger_type"] == "scheduled_job" and "liaison" not in outputs:
        return "liaison"

    return "finalize"


async def run_concierge_node(state: OrchestratorState) -> OrchestratorState:
    """Execute Concierge agent."""
    result = await run_concierge(
        workspace_id=state["workspace_id"],
        patient_ref=state["patient_ref"],
        intent=state["intent"],
        payload=state["payload"],
    )
    state["agent_outputs"]["concierge"] = result
    state["current_agent"] = "concierge"
    state["steps"] += 1

    if result.get("escalate"):
        state["escalated"] = True
        state["escalation_reason"] = result.get("escalation_reason", "Concierge escalated")

    return state


async def run_diagnostician_node(state: OrchestratorState) -> OrchestratorState:
    """Execute Diagnostician agent."""
    result = await run_diagnostician(
        workspace_id=state["workspace_id"],
        patient_ref=state["patient_ref"],
        prior_outputs=state["agent_outputs"],
    )
    state["agent_outputs"]["diagnostician"] = result
    state["current_agent"] = "diagnostician"
    state["steps"] += 1
    return state


async def run_liaison_node(state: OrchestratorState) -> OrchestratorState:
    """Execute Liaison agent."""
    result = await run_liaison(
        workspace_id=state["workspace_id"],
        patient_ref=state["patient_ref"],
        prior_outputs=state["agent_outputs"],
    )
    state["agent_outputs"]["liaison"] = result
    state["current_agent"] = "liaison"
    state["steps"] += 1
    return state


async def run_auditor_node(state: OrchestratorState) -> OrchestratorState:
    """Execute Auditor agent."""
    result = await run_auditor(
        workspace_id=state["workspace_id"],
        patient_ref=state["patient_ref"],
        prior_outputs=state["agent_outputs"],
    )
    state["agent_outputs"]["auditor"] = result
    state["current_agent"] = "auditor"
    state["steps"] += 1
    return state


async def finalize_node(state: OrchestratorState) -> OrchestratorState:
    """Finalize the interaction — audit log and cleanup."""
    state["completed"] = True

    await log_audit_event(
        workspace_id=state["workspace_id"],
        actor_type="agent",
        actor_id="orchestrator",
        action="interaction_completed",
        resource_type="interaction",
        resource_id=state["interaction_id"],
        metadata={
            "trigger_type": state["trigger_type"],
            "intent": state["intent"],
            "agents_used": list(state["agent_outputs"].keys()),
            "escalated": state["escalated"],
            "steps": state["steps"],
        },
    )

    return state


# Build the graph
def build_orchestrator_graph():
    """Build the LangGraph state machine for the orchestrator."""
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("concierge", run_concierge_node)
    graph.add_node("diagnostician", run_diagnostician_node)
    graph.add_node("liaison", run_liaison_node)
    graph.add_node("auditor", run_auditor_node)
    graph.add_node("finalize", finalize_node)

    # Entry point
    graph.set_entry_point("classify")

    # After classification, route to appropriate agent
    graph.add_conditional_edges("classify", route_to_agent)

    # After each agent, route again (enables chaining)
    graph.add_conditional_edges("concierge", route_to_agent)
    graph.add_conditional_edges("diagnostician", route_to_agent)
    graph.add_conditional_edges("liaison", route_to_agent)
    graph.add_conditional_edges("auditor", route_to_agent)

    # Finalize ends the graph
    graph.add_edge("finalize", END)

    return graph.compile()


# Singleton compiled graph
orchestrator = build_orchestrator_graph()


async def run_interaction(event: TriggerEvent) -> dict:
    """
    Main entry point — run a complete interaction through the orchestrator.
    """
    initial_state: OrchestratorState = {
        "interaction_id": str(uuid.uuid4()),
        "workspace_id": event.workspace_id,
        "patient_ref": event.patient_ref,
        "provider_ref": event.provider_ref,
        "trigger_type": event.event_type,
        "intent": None,
        "payload": event.payload,
        "agent_outputs": {},
        "current_agent": None,
        "escalated": False,
        "escalation_reason": None,
        "completed": False,
        "steps": 0,
    }

    start = datetime.utcnow()
    result = await orchestrator.ainvoke(initial_state)
    duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

    return {
        "interaction_id": result["interaction_id"],
        "intent": result["intent"],
        "agents_used": list(result["agent_outputs"].keys()),
        "agent_outputs": result["agent_outputs"],
        "escalated": result["escalated"],
        "escalation_reason": result.get("escalation_reason"),
        "steps": result["steps"],
        "duration_ms": duration_ms,
    }
