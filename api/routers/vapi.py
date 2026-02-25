"""
Vapi Webhook Handler — with real scheduling tools.

The Concierge can check availability, book, cancel, and reschedule
during live phone calls via Vapi function calling.
"""

from fastapi import APIRouter, Request
from api.agents.orchestrator import run_interaction
from api.models.schemas import TriggerEvent
from api.services.supabase_client import log_audit_event
from api.services.appointments import (
    lookup_patient_by_name,
    check_availability,
    find_next_available,
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
    get_patient_appointments,
)
import json
import os
import traceback

router = APIRouter()

DEFAULT_WORKSPACE_ID = os.environ.get("DEFAULT_WORKSPACE_ID", "")

CONCIERGE_SYSTEM_PROMPT = """You are the Concierge AI assistant for a dental practice. You are the first point of contact for patients calling in.

IMPORTANT — ALWAYS collect the patient's full name before taking any action:
1. Greet the patient and ask for their full name
2. Ask what they need help with
3. When calling book_appointment, cancel_appointment, reschedule_appointment, or get_patient_appointments, ALWAYS include the patient_name parameter — the system will verify them automatically

You have these tools:
- check_availability: Check open slots for a specific date
- find_next_available: Find the next available appointment slots
- book_appointment: Book a new appointment — ALWAYS include patient_name
- cancel_appointment: Cancel an existing appointment — ALWAYS include patient_name
- reschedule_appointment: Move an appointment — ALWAYS include patient_name
- get_patient_appointments: Look up upcoming appointments — ALWAYS include patient_name

Key behaviors:
- ALWAYS ask for the patient's full name early in the conversation
- ALWAYS pass patient_name when calling any scheduling tool
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
            "description": "Check available appointment slots for a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_next_available",
            "description": "Find next available appointment slots across the next 14 days.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a new appointment. You MUST include the patient's full name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "The patient's full name as they told you (e.g. 'John Smith'). REQUIRED."},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM 24-hour format"},
                    "appointment_type": {"type": "string", "description": "Type: cleaning, exam, filling, crown, consultation, general"},
                },
                "required": ["patient_name", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel the patient's next upcoming appointment. You MUST include the patient's full name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "The patient's full name. REQUIRED."},
                    "reason": {"type": "string", "description": "Reason for cancellation"},
                },
                "required": ["patient_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Reschedule the patient's next upcoming appointment to a new date/time. You MUST include the patient's full name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "The patient's full name. REQUIRED."},
                    "new_date": {"type": "string", "description": "New date in YYYY-MM-DD format"},
                    "new_time": {"type": "string", "description": "New time in HH:MM 24-hour format"},
                },
                "required": ["patient_name", "new_date", "new_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_appointments",
            "description": "Look up the patient's upcoming appointments. You MUST include the patient's full name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string", "description": "The patient's full name. REQUIRED."},
                },
                "required": ["patient_name"],
            },
        },
    },
]
@router.post("/webhook")
async def vapi_webhook(request: Request):
    """Main Vapi webhook endpoint."""
    try:
        body = await request.json()
        print(f"[VAPI RAW] {json.dumps(body)[:3000]}")
        message = body.get("message") or {}
        event_type = message.get("type", "")

        # Safe extraction — Vapi sends null for call/metadata in some events
        call = message.get("call") or {}
        metadata = call.get("metadata") or {}
        if not metadata.get("workspace_id"):
            metadata["workspace_id"] = DEFAULT_WORKSPACE_ID
        workspace_id = metadata.get("workspace_id", "") or DEFAULT_WORKSPACE_ID
        patient_ref = metadata.get("patient_ref")

        print(f"[VAPI] event={event_type}, workspace={workspace_id}")

        # assistant-request: Dynamic assistant config with tools
        if event_type == "assistant-request":
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
                    "firstMessage": "Hello! Thank you for calling the dental practice. May I have your full name so I can pull up your file?",
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
            function_call = message.get("functionCall") or {}
            fn_name = function_call.get("name", "")
            fn_params = function_call.get("parameters") or {}
            print(f"[VAPI DEBUG] function={fn_name}, params={fn_params}, workspace_id={workspace_id}")
            print(f"[VAPI FUNCTION-CALL] message keys: {list(message.keys())}")

            # Try to extract patient name from conversation for this format too
            artifact = message.get("artifact") or {}
            messages_list = artifact.get("messages") or message.get("messages") or []
            if messages_list and not fn_params.get("patient_name"):
                for m in messages_list:
                    role = m.get("role", "")
                    content = (m.get("content") or m.get("text") or "").strip()
                    if role == "user" and content and len(content) < 60:
                        lower = content.lower()
                        if not any(w in lower for w in ["book", "cancel", "reschedule", "appointment", "schedule", "yes", "no", "okay", "sure", "please", "thank"]):
                            if fn_name in ("book_appointment", "cancel_appointment", "reschedule_appointment", "get_patient_appointments"):
                                fn_params["patient_name"] = content
                                print(f"[VAPI INJECT] Added patient_name='{content}' to {fn_name}")
                            break

            result = await handle_function_call(fn_name, fn_params, workspace_id, patient_ref)
            return {"result": result}
        # tool-calls: Vapi server-side tool execution
        if event_type == "tool-calls":
            tool_calls = message.get("toolCallList") or message.get("toolCalls") or []

            # Log all available keys in the message to find conversation context
            print(f"[VAPI TOOL-CALLS] message keys: {list(message.keys())}")

            # Vapi may send conversation messages — try to extract patient name
            conversation_name = None
            artifact = message.get("artifact") or {}
            messages_list = artifact.get("messages") or message.get("messages") or []
            if messages_list:
                print(f"[VAPI MESSAGES] Found {len(messages_list)} messages in conversation")
                for m in messages_list:
                    print(f"[VAPI MSG] role={m.get('role')}, content={str(m.get('content',''))[:200]}")

            # Extract name: look for the user's first substantive response
            # (their reply to "May I have your full name?")
            for m in messages_list:
                role = m.get("role", "")
                content = (m.get("content") or m.get("text") or "").strip()
                if role == "user" and content and len(content) < 60:
                    # First short user message is likely the name
                    # Skip messages that are clearly intents, not names
                    lower = content.lower()
                    if not any(w in lower for w in ["book", "cancel", "reschedule", "appointment", "schedule", "yes", "no", "okay", "sure", "please", "thank"]):
                        conversation_name = content
                        print(f"[VAPI NAME EXTRACTED] '{conversation_name}' from conversation")
                        break

            results = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                fn_name = fn.get("name", "")
                fn_params = fn.get("arguments") or {}
                if isinstance(fn_params, str):
                    try:
                        fn_params = json.loads(fn_params)
                    except:
                        fn_params = {}
                tc_id = tc.get("id", "")

                # Inject extracted patient name if the tool needs it and AI didn't provide it
                if conversation_name and not fn_params.get("patient_name"):
                    if fn_name in ("book_appointment", "cancel_appointment", "reschedule_appointment", "get_patient_appointments"):
                        fn_params["patient_name"] = conversation_name
                        print(f"[VAPI INJECT] Added patient_name='{conversation_name}' to {fn_name}")

                print(f"[VAPI TOOL] function={fn_name}, params={fn_params}, workspace={workspace_id}")

                result_str = await handle_function_call(fn_name, fn_params, workspace_id, patient_ref)
                results.append({
                    "toolCallId": tc_id,
                    "result": result_str,
                })

            return {"results": results}
        # status-update
        if event_type == "status-update":
            status = message.get("status", "")
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
                        "phone_number": (call.get("customer") or {}).get("number", "unknown"),
                    },
                )
            return {"ok": True}

        # end-of-call-report
        if event_type == "end-of-call-report":
            summary = message.get("summary", "")
            transcript_text = message.get("transcript", "")
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
                            "text": summary or transcript_text[:1000],
                            "channel": "phone",
                            "call_id": call.get("id"),
                            "post_call": True,
                        },
                    )
                    await run_interaction(event)
                except Exception as e:
                    print(f"[VAPI] Post-call orchestrator error: {e}")

            return {"ok": True}

        # transcript or unknown events
        return {"ok": True}

    except Exception as e:
        print(f"[VAPI ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"ok": True}


async def resolve_patient(workspace_id: str, params: dict, patient_ref: str | None) -> tuple[str | None, str | None]:
    """
    Resolve patient from params. Tries patient_name lookup first, falls back to patient_ref.
    Returns (patient_id, error_message). If error_message is set, patient_id is None.
    """
    patient_name = params.get("patient_name", "")

    if patient_name:
        print(f"[VAPI LOOKUP] Resolving patient_name='{patient_name}', workspace={workspace_id}")
        lookup = await lookup_patient_by_name(workspace_id, patient_name)
        print(f"[VAPI LOOKUP RESULT] {lookup}")

        if lookup["found"] and lookup["patient"]:
            return lookup["patient"]["id"], None
        elif lookup["candidates"]:
            names = [c["full_name"] for c in lookup["candidates"]]
            return None, f"I found multiple patients matching that name: {', '.join(names)}. Could you confirm the full name or date of birth?"
        else:
            return None, f"I don't have a patient named {patient_name} on file. They may need to register as a new patient first."

    if patient_ref:
        return patient_ref, None

    return None, "I need the patient's full name before I can help with that. Could you tell me your full name?"


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
            patient_id, error = await resolve_patient(workspace_id, params, patient_ref)
            if error:
                return json.dumps({"booked": False, "message": error})
            print(f"[VAPI BOOK] date={params.get('date')}, time={params.get('time')}, type={params.get('appointment_type')}, patient={patient_id}, workspace={workspace_id}")
            try:
                result = await book_appointment(
                    workspace_id=workspace_id,
                    date=params.get("date", ""),
                    time=params.get("time", ""),
                    appointment_type=params.get("appointment_type", "general"),
                    patient_id=patient_id,
                    source="phone",
                )
                print(f"[VAPI BOOK RESULT] {result}")
                if result.get("success"):
                    appt = result["appointment"]
                    return json.dumps({
                        "booked": True,
                        "message": f"I've booked your {appt['type']} appointment for {appt['date']} at {appt['time']}. You'll receive a confirmation shortly.",
                    })
                return json.dumps({
                    "booked": False,
                    "message": result.get("error", "Sorry, I couldn't book that slot."),
                })
            except Exception as e:
                print(f"[VAPI BOOK ERROR] {e}")
                traceback.print_exc()
                return json.dumps({"error": True, "message": f"Booking failed: {str(e)}"})

        if fn_name == "cancel_appointment":
            patient_id, error = await resolve_patient(workspace_id, params, patient_ref)
            if error:
                return json.dumps({"cancelled": False, "message": error})
            result = await cancel_appointment(
                workspace_id=workspace_id,
                patient_id=patient_id,
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
            patient_id, error = await resolve_patient(workspace_id, params, patient_ref)
            if error:
                return json.dumps({"rescheduled": False, "message": error})
            appts = await get_patient_appointments(workspace_id, patient_id)
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
                "message": "I don't see any upcoming appointments to reschedule.",
            })

        if fn_name == "get_patient_appointments":
            patient_id, error = await resolve_patient(workspace_id, params, patient_ref)
            if error:
                return json.dumps({"found": False, "message": error})
            appts = await get_patient_appointments(workspace_id, patient_id)
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
        print(f"[VAPI TOOL ERROR] {fn_name}: {e}")
        traceback.print_exc()
        return json.dumps({
            "error": True,
            "message": f"I'm sorry, I encountered an issue. Let me connect you with the front desk.",
        })