"""
Vapi Webhook Handler — now with real scheduling tools.

The Concierge can check availability, book, cancel, and reschedule
during live phone calls via Vapi function calling.
"""

from fastapi import APIRouter, Request
from api.agents.orchestrator import run_interaction
from api.models.schemas import TriggerEvent
from api.services.supabase_client import log_audit_event
from api.services.appointments import (
    check_availability,
    find_next_available,
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
    get_patient_appointments,
)
import json
import os

DEFAULT_WORKSPACE_ID = os.environ.get("DEFAULT_WORKSPACE_ID", "")

router = APIRouter()

CONCIERGE_SYSTEM_PROMPT = """You are the Concierge AI assistant for a dental practice. You are the first point of contact for patients calling in.

Your responsibilities:
- Greet the patient warmly and professionally
- Identify who they are (ask for name or appointment reference)
- Determine the reason for their call
- Use your tools to take REAL action on appointments

You have these tools:
- check_availability: Check open slots for a specific date
- find_next_available: Find the next available appointment slots
- book_appointment: Book a new appointment
- cancel_appointment: Cancel an existing appointment (will also suggest reschedule dates)
- reschedule_appointment: Move an appointment to a new date/time
- get_patient_appointments: Look up a patient's upcoming appointments

Key behaviors:
- Be warm, empathetic, and efficient
- Keep responses concise — this is a phone call
- When cancelling, ALWAYS offer available reschedule dates from the tool result
- When booking, confirm date/time before finalizing
- If someone is in pain or mentions an emergency, prioritize getting them help
- NEVER mention that you are an AI unless directly asked"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check available appointment slots for a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to check in YYYY-MM-DD format",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration of appointment in minutes (default 30)",
                        "default": 30,
                    },
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_next_available",
            "description": "Find the next available appointment slots across the next 14 days",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration needed in minutes (default 30)",
                        "default": 30,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a new appointment at a specific date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Appointment date in YYYY-MM-DD format",
                    },
                    "time": {
                        "type": "string",
                        "description": "Appointment time in HH:MM format (24-hour)",
                    },
                    "appointment_type": {
                        "type": "string",
                        "description": "Type: cleaning, exam, filling, crown, root_canal, extraction, consultation, emergency, follow_up, general",
                        "default": "general",
                    },
                },
                "required": ["date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel a patient's next upcoming appointment. Will return suggested reschedule dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for cancellation",
                        "default": "Patient requested cancellation",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Reschedule an appointment to a new date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_date": {
                        "type": "string",
                        "description": "New date in YYYY-MM-DD format",
                    },
                    "new_time": {
                        "type": "string",
                        "description": "New time in HH:MM format (24-hour)",
                    },
                },
                "required": ["new_date", "new_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_appointments",
            "description": "Look up a patient's upcoming appointments",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


@router.post("/webhook")
async def vapi_webhook(request: Request):
    """Main Vapi webhook endpoint."""
    body = await request.json()
    message = body.get("message", {})
    event_type = message.get("type", "")

    # assistant-request: Dynamic assistant config with tools
    if event_type == "assistant-request":
        call = message.get("call", {})
        metadata = call.get("metadata") or {}
        if not metadata.get("workspace_id"):
            metadata["workspace_id"] = DEFAULT_WORKSPACE_ID

        return {
            "assistant": {
                "model": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "systemPrompt": CONCIERGE_SYSTEM_PROMPT,
                    "temperature": 0.3,
                    
                },
                "serverUrl": "https://agentic-dentist-api-production.up.railway.app/api/vapi/webhook",
                "tools": TOOLS,
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
                "metadata": metadata,
            }
        }

    # function-call: Execute real scheduling tools
    if event_type == "function-call":
        function_call = message.get("functionCall", {})
        fn_name = function_call.get("name", "")
        fn_params = function_call.get("parameters", {})
        call = message.get("call", {})
        metadata = call.get("metadata") or {}
        workspace_id = metadata.get("workspace_id", "") or DEFAULT_WORKSPACE_ID
        print(f"[VAPI DEBUG] function={fn_name}, params={fn_params}, workspace_id={workspace_id}, metadata={metadata}")
        patient_ref = metadata.get("patient_ref", None)

        result = await handle_function_call(fn_name, fn_params, workspace_id, patient_ref)
        return {"result": result}

    # status-update
    if event_type == "status-update":
        call = message.get("call", {})
        status = message.get("status", "")
        metadata = call.get("metadata") or {}
        workspace_id = metadata.get("workspace_id", "") or DEFAULT_WORKSPACE_ID

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
                },
            )
        return {"ok": True}

    # end-of-call-report
    if event_type == "end-of-call-report":
        call = message.get("call", {})
        metadata = call.get("metadata") or {}
        workspace_id = metadata.get("workspace_id", "") or DEFAULT_WORKSPACE_ID
        summary = message.get("summary", "")
        transcript = message.get("transcript", "")
        duration_seconds = message.get("durationSeconds", 0)

        if workspace_id:
            await log_audit_event(
                workspace_id=workspace_id,
                actor_type="system",
                actor_id="vapi",
                action="call_completed",
                resource_type="call",
                resource_id=call.get("id"),
                metadata={
                    "duration_seconds": duration_seconds,
                    "summary": summary[:500] if summary else None,
                },
            )

            # Post-call processing
            try:
                event = TriggerEvent(
                    event_type="inbound_call",
                    workspace_id=workspace_id,
                    payload={
                        "text": summary or transcript[:1000],
                        "channel": "phone",
                        "call_id": call.get("id"),
                        "post_call": True,
                    },
                )
                await run_interaction(event)
            except Exception as e:
                print(f"Post-call orchestrator error: {e}")

        return {"ok": True}

    return {"ok": True}


async def handle_function_call(
    fn_name: str, params: dict, workspace_id: str, patient_ref: str | None
) -> str:
    """Execute real scheduling tools during a live voice call."""

    try:
        if fn_name == "check_availability":
            date = params.get("date", "")
            duration = params.get("duration_minutes", 30)
            slots = await check_availability(workspace_id, date, duration)
            if slots:
                slot_list = ", ".join([s["start"] for s in slots[:6]])
                return json.dumps({
                    "available": True,
                    "date": date,
                    "slots": slots[:6],
                    "message": f"On {date}, I have these openings: {slot_list}",
                })
            return json.dumps({
                "available": False,
                "message": f"I'm sorry, there are no openings on {date}. Would you like me to check another date?",
            })

        if fn_name == "find_next_available":
            duration = params.get("duration_minutes", 30)
            results = await find_next_available(workspace_id, duration)
            if results:
                summary_parts = []
                for day in results[:3]:
                    slots_str = ", ".join([s["start"] for s in day["slots"][:2]])
                    summary_parts.append(f"{day['day_name']} {day['date']}: {slots_str}")
                summary = "; ".join(summary_parts)
                return json.dumps({
                    "found": True,
                    "options": results[:3],
                    "message": f"Here are the next available slots: {summary}. Which works best for you?",
                })
            return json.dumps({
                "found": False,
                "message": "I'm sorry, I couldn't find any available slots in the next two weeks. Let me connect you with the front desk.",
            })

        if fn_name == "book_appointment":
            result = await book_appointment(
                workspace_id=workspace_id,
                date=params.get("date", ""),
                time=params.get("time", ""),
                appointment_type=params.get("appointment_type", "general"),
                patient_id=patient_ref,
                source="phone",
            )
            if result.get("success"):
                appt = result["appointment"]
                return json.dumps({
                    "booked": True,
                    "message": f"I've booked your {appt['type']} appointment for {appt['date']} at {appt['time']}. You'll receive a confirmation shortly.",
                })
            return json.dumps({
                "booked": False,
                "message": result.get("error", "Sorry, I couldn't book that slot."),
                "available_slots": result.get("available_slots", []),
            })

        if fn_name == "cancel_appointment":
            result = await cancel_appointment(
                workspace_id=workspace_id,
                patient_id=patient_ref,
                reason=params.get("reason", "Patient requested cancellation"),
            )
            if result.get("success"):
                cancelled = result["cancelled_appointment"]
                msg = f"I've cancelled your {cancelled['type']} appointment on {cancelled['date']} at {cancelled['time']}."
                if result.get("suggested_reschedule"):
                    rebook_parts = []
                    for day in result["suggested_reschedule"][:3]:
                        slots_str = ", ".join([s["start"] for s in day["slots"][:2]])
                        rebook_parts.append(f"{day['day_name']} {day['date']}: {slots_str}")
                    msg += f" Would you like to reschedule? I have openings on: {'; '.join(rebook_parts)}"
                return json.dumps({"cancelled": True, "message": msg})
            return json.dumps({
                "cancelled": False,
                "message": result.get("error", "I couldn't find an appointment to cancel."),
            })

        if fn_name == "reschedule_appointment":
            # Get patient's next appointment to reschedule
            if patient_ref:
                appts = await get_patient_appointments(workspace_id, patient_ref)
                if appts:
                    result = await reschedule_appointment(
                        workspace_id=workspace_id,
                        appointment_id=appts[0]["id"],
                        new_date=params.get("new_date", ""),
                        new_time=params.get("new_time", ""),
                    )
                    if result.get("success"):
                        r = result["rescheduled"]
                        return json.dumps({
                            "rescheduled": True,
                            "message": f"Done! I've moved your appointment from {r['old_date']} at {r['old_time']} to {r['new_date']} at {r['new_time']}.",
                        })
                    return json.dumps({
                        "rescheduled": False,
                        "message": result.get("error", "That time isn't available."),
                    })
            return json.dumps({
                "rescheduled": False,
                "message": "I need to verify your identity first. Can you provide your patient reference number?",
            })

        if fn_name == "get_patient_appointments":
            if patient_ref:
                appts = await get_patient_appointments(workspace_id, patient_ref)
                if appts:
                    appt_list = []
                    for a in appts:
                        appt_list.append(f"{a['appointment_type']} on {a['start_time'][:10]} at {a['start_time'][11:16]}")
                    return json.dumps({
                        "found": True,
                        "appointments": appts,
                        "message": f"I found {len(appts)} upcoming appointment(s): {', '.join(appt_list)}",
                    })
                return json.dumps({
                    "found": False,
                    "message": "I don't see any upcoming appointments on file.",
                })
            return json.dumps({
                "found": False,
                "message": "I need your patient reference number to look up your appointments.",
            })

        if fn_name == "transfer_to_human":
            reason = params.get("reason", "patient request")
            if workspace_id:
                await log_audit_event(
                    workspace_id=workspace_id,
                    actor_type="agent",
                    actor_id="concierge_voice",
                    action="transfer_to_human",
                    metadata={"reason": reason},
                )
            return json.dumps({"transferred": True, "message": "Transferring you now."})

        return json.dumps({"error": f"Unknown function: {fn_name}"})

    except Exception as e:
        return json.dumps({
            "error": True,
            "message": f"I'm sorry, I encountered an issue. Let me connect you with the front desk. Error: {str(e)}",
        })
