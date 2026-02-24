"""
Vapi Webhook Handler.

Receives real-time call events from Vapi and routes them
through the orchestrator for processing.

Webhook events:
- assistant-request: Vapi asks for assistant config (dynamic)
- function-call: Vapi asks us to execute a function
- status-update: Call status changes (ringing, in-progress, ended)
- end-of-call-report: Full call summary after hangup
- transcript: Real-time transcript updates
"""

from fastapi import APIRouter, Request
from api.agents.orchestrator import run_interaction
from api.models.schemas import TriggerEvent
from api.services.supabase_client import log_audit_event
import json

router = APIRouter()

CONCIERGE_SYSTEM_PROMPT = """You are the Concierge AI assistant for a dental practice. You are the first point of contact for patients calling in.

Your responsibilities:
- Greet the patient warmly and professionally
- Identify who they are (ask for name or appointment reference)
- Determine the reason for their call
- Handle simple requests: appointment confirmations, cancellations, rescheduling, office hours, directions
- For clinical questions or emergencies, let them know you'll connect them with the clinical team
- For billing questions, let them know you'll transfer to the billing department

Key behaviors:
- Be warm, empathetic, and efficient
- Keep responses concise — this is a phone call, not an essay
- If someone is in pain or mentions an emergency, prioritize getting them help immediately
- Always confirm actions before taking them
- Use natural conversational language
- NEVER mention that you are an AI or that you cannot do something — always frame it as connecting them with the right person"""


@router.post("/webhook")
async def vapi_webhook(request: Request):
    """
    Main Vapi webhook endpoint.
    Vapi sends events here during and after calls.
    """
    body = await request.json()
    message = body.get("message", {})
    event_type = message.get("type", "")

    # assistant-request: Vapi wants dynamic assistant config
    if event_type == "assistant-request":
        call = message.get("call", {})
        metadata = call.get("metadata", {})
        workspace_id = metadata.get("workspace_id", "")

        return {
            "assistant": {
                "model": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "systemPrompt": CONCIERGE_SYSTEM_PROMPT,
                    "temperature": 0.3,
                },
                "voice": {
                    "provider": "11labs",
                    "voiceId": "21m00Tcm4TlvDq8ikWAM",
                },
                "firstMessage": "Hello! Thank you for calling. This is the dental practice AI assistant. How can I help you today?",
                "transcriber": {
                    "provider": "deepgram",
                    "model": "nova-2",
                    "language": "en",
                },
                "metadata": {
                    "workspace_id": workspace_id,
                },
            }
        }

    # function-call: Vapi wants us to execute a tool
    if event_type == "function-call":
        function_call = message.get("functionCall", {})
        fn_name = function_call.get("name", "")
        fn_params = function_call.get("parameters", {})
        call = message.get("call", {})
        metadata = call.get("metadata", {})
        workspace_id = metadata.get("workspace_id", "")

        result = await handle_function_call(fn_name, fn_params, workspace_id)
        return {"result": result}

    # status-update: Call status changed
    if event_type == "status-update":
        call = message.get("call", {})
        status = message.get("status", "")
        metadata = call.get("metadata", {})
        workspace_id = metadata.get("workspace_id", "")

        if workspace_id:
            await log_audit_event(
                workspace_id=workspace_id,
                actor_type="system",
                actor_id="vapi",
                action=f"call_{status}",
                resource_type="call",
                resource_id=call.get("id"),
                metadata={
                    "status": status,
                    "phone_number": call.get("customer", {}).get("number", "unknown"),
                    "source": "vapi_webhook",
                },
            )

        return {"ok": True}

    # end-of-call-report: Call completed — run through orchestrator
    if event_type == "end-of-call-report":
        call = message.get("call", {})
        metadata = call.get("metadata", {})
        workspace_id = metadata.get("workspace_id", "")
        transcript = message.get("transcript", "")
        summary = message.get("summary", "")
        duration_seconds = message.get("durationSeconds", 0)
        ended_reason = message.get("endedReason", "")

        if workspace_id:
            # Log the call
            await log_audit_event(
                workspace_id=workspace_id,
                actor_type="system",
                actor_id="vapi",
                action="call_completed",
                resource_type="call",
                resource_id=call.get("id"),
                metadata={
                    "duration_seconds": duration_seconds,
                    "ended_reason": ended_reason,
                    "summary": summary[:500] if summary else None,
                    "source": "vapi_webhook",
                },
            )

            # Route through orchestrator for post-call processing
            try:
                event = TriggerEvent(
                    event_type="inbound_call",
                    workspace_id=workspace_id,
                    payload={
                        "text": summary or transcript[:1000],
                        "channel": "phone",
                        "call_id": call.get("id"),
                        "duration_seconds": duration_seconds,
                        "transcript": transcript[:2000] if transcript else "",
                        "post_call": True,
                    },
                )
                await run_interaction(event)
            except Exception as e:
                print(f"Post-call orchestrator error: {e}")

        return {"ok": True}

    # transcript: Real-time transcript (logged but not processed)
    if event_type == "transcript":
        return {"ok": True}

    return {"ok": True}


async def handle_function_call(fn_name: str, params: dict, workspace_id: str) -> str:
    """
    Handle tool calls from Vapi during a live call.
    These are functions the Concierge can invoke mid-conversation.
    """
    if fn_name == "check_appointment":
        # TODO: Look up actual appointment
        return json.dumps({
            "found": True,
            "message": "I found an appointment. Let me check the details for you.",
        })

    if fn_name == "schedule_appointment":
        # TODO: Actually create appointment
        date = params.get("date", "")
        time = params.get("time", "")
        return json.dumps({
            "scheduled": True,
            "message": f"I've noted your preferred time of {date} at {time}. A team member will confirm this shortly.",
        })

    if fn_name == "transfer_to_human":
        reason = params.get("reason", "patient request")
        await log_audit_event(
            workspace_id=workspace_id,
            actor_type="agent",
            actor_id="concierge_voice",
            action="transfer_to_human",
            metadata={"reason": reason},
        )
        return json.dumps({
            "transferred": True,
            "message": "Transferring you to a team member now.",
        })

    return json.dumps({"error": f"Unknown function: {fn_name}"})
