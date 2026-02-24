"""
Agent API routes.
Endpoints for triggering agents and the orchestrator.
"""

from fastapi import APIRouter, Depends
from api.models.schemas import TriggerEvent, AgentRunRequest, AgentRunResponse
from api.core.security import verify_auth, verify_membership
from api.agents.orchestrator import run_interaction
from api.agents.concierge.agent import run_concierge
from api.agents.diagnostician.agent import run_diagnostician
from api.agents.liaison.agent import run_liaison
from api.agents.auditor.agent import run_auditor
import uuid
import time

router = APIRouter()


@router.post("/trigger", summary="Trigger the orchestrator with an event")
async def trigger_interaction(
    event: TriggerEvent,
    user: dict = Depends(verify_auth),
):
    """
    Main entry point — sends an event through the orchestrator.
    The orchestrator classifies intent and routes to appropriate agents.
    """
    await verify_membership(user["id"], event.workspace_id)
    result = await run_interaction(event)
    return result


@router.post("/run", response_model=AgentRunResponse, summary="Run a specific agent directly")
async def run_agent_directly(
    request: AgentRunRequest,
    user: dict = Depends(verify_auth),
):
    """
    Manually run a specific agent — bypasses orchestrator routing.
    Useful for testing and specific workflows.
    """
    await verify_membership(user["id"], request.workspace_id)

    start = time.time()
    run_id = str(uuid.uuid4())

    agent_map = {
        "concierge": lambda: run_concierge(
            workspace_id=request.workspace_id,
            patient_ref=request.patient_ref,
            intent=request.intent,
            payload=request.payload,
        ),
        "diagnostician": lambda: run_diagnostician(
            workspace_id=request.workspace_id,
            patient_ref=request.patient_ref,
        ),
        "liaison": lambda: run_liaison(
            workspace_id=request.workspace_id,
            patient_ref=request.patient_ref,
        ),
        "auditor": lambda: run_auditor(
            workspace_id=request.workspace_id,
            patient_ref=request.patient_ref,
        ),
    }

    agent_fn = agent_map.get(request.agent)
    if not agent_fn:
        return AgentRunResponse(
            run_id=run_id,
            agent=request.agent,
            status="error",
            output={"error": f"Unknown agent: {request.agent}"},
        )

    try:
        output = await agent_fn()
        duration_ms = int((time.time() - start) * 1000)
        status = "error" if output.get("error") else "escalated" if output.get("escalate") else "completed"

        return AgentRunResponse(
            run_id=run_id,
            agent=request.agent,
            status=status,
            output=output,
            steps=1,
            duration_ms=duration_ms,
        )
    except Exception as e:
        return AgentRunResponse(
            run_id=run_id,
            agent=request.agent,
            status="error",
            output={"error": str(e)},
        )


@router.get("/status", summary="Get agent swarm status")
async def agent_status():
    """Returns the current status of all agents."""
    return {
        "agents": [
            {
                "name": "concierge",
                "status": "ready",
                "model": "gpt-4o-mini",
                "channels": ["phone", "sms", "web_chat"],
            },
            {
                "name": "diagnostician",
                "status": "ready",
                "model": "gpt-4o",
                "channels": ["internal"],
            },
            {
                "name": "liaison",
                "status": "ready",
                "model": "gpt-4o-mini",
                "channels": ["sms", "email", "phone"],
            },
            {
                "name": "auditor",
                "status": "ready",
                "model": "gpt-4o",
                "channels": ["event_stream"],
            },
        ],
        "orchestrator": {
            "status": "ready",
            "engine": "langgraph",
            "patterns": ["sequential", "parallel", "conditional"],
        },
    }
