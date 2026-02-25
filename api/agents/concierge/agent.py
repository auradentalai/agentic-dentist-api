"""
Concierge Agent — First point of contact.

Now equipped with real scheduling tools:
- lookup_patient_by_name: Verify patient exists before booking
- check_availability: See open slots
- book_appointment: Create appointments
- cancel_appointment: Cancel + suggest reschedule
- reschedule_appointment: Move to new date/time
- get_patient_appointments: View patient's upcoming visits
"""

from langchain_core.messages import SystemMessage, HumanMessage
from api.core.llm import get_fast_llm
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

CONCIERGE_SYSTEM_PROMPT = """You are the Concierge agent for a dental practice. You are the first point of contact.

You have ACCESS to real scheduling tools. When a patient wants to book, cancel, or reschedule, you MUST use the tools to take action — don't just say you'll do it.

Your job:
1. Identify the patient — ALWAYS look them up by name before booking/cancelling/rescheduling
2. Classify the intent of the interaction
3. Use your tools to take real action when possible
4. Respond with structured JSON

TOOLS AVAILABLE:
- lookup_patient(name): Look up a patient by name. ALWAYS call this first when a patient gives their name.
- check_availability(date): See open slots for a date
- find_next_available(duration): Find next open slots across 14 days
- book_appointment(date, time, type, patient_id): Book an appointment — requires a valid patient_id
- cancel_appointment(appointment_id or patient_id): Cancel + get reschedule options
- reschedule_appointment(appointment_id, new_date, new_time): Move appointment
- get_patient_appointments(patient_id): View upcoming appointments

PATIENT IDENTIFICATION RULES:
- When a caller provides their name, ALWAYS verify them via lookup_patient before taking action
- If lookup returns found=true, use the patient's ID for all subsequent operations
- If lookup returns multiple candidates, ask the caller to clarify (e.g. "I see a few patients with that name — can you confirm your date of birth?")
- If lookup returns not found, inform the caller they may need to register as a new patient
- NEVER book an appointment without a verified patient_id

Intent categories:
- appointment_request: wants to book → lookup patient, then find_next_available, then book_appointment
- appointment_confirm: confirming existing → lookup patient, then get_patient_appointments
- schedule_change: cancel or reschedule → lookup patient, then cancel/reschedule
- clinical_question: symptom, treatment, pain → route to diagnostician
- billing_inquiry: balance, insurance → route to auditor
- emergency: severe pain, trauma → escalate immediately
- general_inquiry: hours, directions, policies → handle directly

Respond ONLY with a JSON object:
{
    "patient_identified": true/false,
    "patient_ref": "patient ID if found or null",
    "refined_intent": "one of the intents above",
    "confidence": 0.0-1.0,
    "can_handle": true/false,
    "response": "what to say to the patient",
    "action_taken": "what tool was used and result",
    "tool_results": {},
    "escalate": false,
    "escalation_reason": null,
    "notes": "context for the next agent"
}

CRITICAL RULES:
- NEVER access or display patient PII (DOB, phone, email) — name is OK for verification
- Only reference patients by their external_ref token in downstream systems
- Only set escalate=true for real emergencies (severe pain, trauma, bleeding)
- When you can't handle a request (clinical, billing), set can_handle=false but escalate=false
- When cancelling, ALWAYS include suggested reschedule dates from the tool result
- When booking, confirm the date/time with the patient before finalizing
- Be warm, professional, efficient
"""


async def run_concierge(
    workspace_id: str,
    patient_ref: str | None = None,
    intent: str | None = None,
    payload: dict = {},
) -> dict:
    """Run the Concierge agent with real scheduling tools."""
    llm = get_fast_llm()

    # Build context
    context_parts = [f"Workspace: {workspace_id}"]
    if patient_ref:
        context_parts.append(f"Patient ref: {patient_ref}")
    if intent:
        context_parts.append(f"Initial intent classification: {intent}")
    if payload.get("text"):
        context_parts.append(f"Patient message: {payload['text']}")
    if payload.get("channel"):
        context_parts.append(f"Channel: {payload['channel']}")

    # Pre-fetch data based on likely intent
    tool_context = []
    tool_results = {}

    text = (payload.get("text") or "").lower()

    # ── Step 0: Patient name lookup ──────────────────────────────────
    # If a patient_name is provided in payload (from Vapi or frontend)
    # OR no patient_ref yet, try to identify the patient by name.
    patient_name = payload.get("patient_name")
    if patient_name and not patient_ref:
        try:
            lookup = await lookup_patient_by_name(workspace_id, patient_name)
            tool_results["patient_lookup"] = lookup

            if lookup["found"] and lookup["patient"]:
                patient_ref = lookup["patient"]["id"]
                context_parts.append(f"Patient ref: {patient_ref}")
                tool_context.append(
                    f"\n✅ Patient verified: {lookup['patient']['full_name']} "
                    f"(ID: {patient_ref})"
                )
            elif lookup["candidates"]:
                names = ", ".join(c["full_name"] for c in lookup["candidates"])
                tool_context.append(
                    f"\n⚠️ Multiple patients match '{patient_name}': {names}. "
                    f"Ask the caller to clarify."
                )
            else:
                tool_context.append(
                    f"\n❌ No patient named '{patient_name}' found. "
                    f"They may need to register as a new patient first."
                )
        except Exception as e:
            tool_context.append(f"\nPatient lookup failed: {e}")

    # If patient wants to cancel or reschedule, get their appointments
    if patient_ref and any(w in text for w in ["cancel", "reschedule", "move", "change", "appointment"]):
        try:
            patient_appts = await get_patient_appointments(workspace_id, patient_ref)
            if patient_appts:
                tool_context.append(f"\nPatient's upcoming appointments:")
                for appt in patient_appts:
                    tool_context.append(
                        f"  - ID: {appt['id']} | {appt['appointment_type']} | "
                        f"{appt['start_time'][:10]} at {appt['start_time'][11:16]} | "
                        f"Status: {appt['status']}"
                    )
                tool_results["patient_appointments"] = patient_appts
            else:
                tool_context.append("\nPatient has no upcoming appointments.")
                tool_results["patient_appointments"] = []
        except Exception as e:
            tool_context.append(f"\nFailed to fetch appointments: {e}")

    # If patient wants to book or reschedule, get availability
    if any(w in text for w in ["book", "schedule", "appointment", "available", "opening", "reschedule", "next"]):
        try:
            next_slots = await find_next_available(workspace_id, duration_minutes=30, max_results=3)
            if next_slots:
                tool_context.append(f"\nNext available slots (30 min):")
                for day in next_slots:
                    slots_str = ", ".join([s["start"] for s in day["slots"]])
                    tool_context.append(f"  - {day['day_name']} {day['date']}: {slots_str}")
                tool_results["availability"] = next_slots
            else:
                tool_context.append("\nNo available slots found in the next 14 days.")
        except Exception as e:
            tool_context.append(f"\nFailed to check availability: {e}")

    # If patient wants to cancel
    if patient_ref and any(w in text for w in ["cancel"]):
        try:
            cancel_result = await cancel_appointment(
                workspace_id=workspace_id,
                patient_id=patient_ref,
                reason="Patient requested cancellation via Concierge",
            )
            tool_results["cancellation"] = cancel_result
            if cancel_result.get("success"):
                cancelled = cancel_result["cancelled_appointment"]
                tool_context.append(
                    f"\n✅ CANCELLED appointment: {cancelled['type']} on "
                    f"{cancelled['date']} at {cancelled['time']}"
                )
                if cancel_result.get("suggested_reschedule"):
                    tool_context.append("Suggested reschedule options:")
                    for day in cancel_result["suggested_reschedule"]:
                        slots_str = ", ".join([s["start"] for s in day["slots"]])
                        tool_context.append(f"  - {day['day_name']} {day['date']}: {slots_str}")
            else:
                tool_context.append(f"\n❌ Could not cancel: {cancel_result.get('error')}")
        except Exception as e:
            tool_context.append(f"\nCancellation failed: {e}")

    context = "\n".join(context_parts + tool_context)

    messages = [
        SystemMessage(content=CONCIERGE_SYSTEM_PROMPT),
        HumanMessage(content=f"Process this interaction:\n\n{context}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        # Merge tool results into the response
        result["tool_results"] = tool_results

        await log_audit_event(
            workspace_id=workspace_id,
            actor_type="agent",
            actor_id="concierge",
            action="intent_classified",
            resource_type="patient" if patient_ref else None,
            resource_id=patient_ref,
            metadata={
                "intent": result.get("refined_intent"),
                "confidence": result.get("confidence"),
                "can_handle": result.get("can_handle"),
                "action_taken": result.get("action_taken"),
                "tools_used": list(tool_results.keys()),
            },
        )

        return result

    except json.JSONDecodeError:
        return {
            "patient_identified": False,
            "patient_ref": patient_ref,
            "refined_intent": intent or "unknown",
            "confidence": 0.0,
            "can_handle": False,
            "response": None,
            "action_taken": None,
            "tool_results": tool_results,
            "escalate": False,
            "escalation_reason": None,
            "notes": f"Failed to parse LLM response: {response.content[:200]}",
            "error": True,
        }
    except Exception as e:
        return {
            "patient_identified": False,
            "patient_ref": patient_ref,
            "refined_intent": "error",
            "confidence": 0.0,
            "can_handle": False,
            "response": None,
            "action_taken": None,
            "tool_results": tool_results,
            "escalate": False,
            "escalation_reason": f"Concierge error: {str(e)}",
            "notes": None,
            "error": True,
        }
