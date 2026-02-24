"""
Concierge Agent — First point of contact.

Responsibilities:
- Identify patient (by phone, name, ref)
- Classify intent (appointment, billing, clinical, emergency)
- Route to appropriate specialist agent
- Handle simple requests directly (confirm, cancel)

Uses GPT-4o-mini for speed.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from api.core.llm import get_fast_llm
from api.services.supabase_client import log_audit_event
import json

CONCIERGE_SYSTEM_PROMPT = """You are the Concierge agent for a dental practice. You are the first point of contact.

Your job:
1. Identify the patient (if possible from context)
2. Classify the intent of the interaction
3. Determine if you can handle it yourself or if it needs a specialist agent
4. Respond with structured JSON

Intent categories:
- appointment_request: wants to book, reschedule, or check appointment
- appointment_confirm: confirming an existing appointment
- schedule_change: cancel or reschedule
- clinical_question: symptom, treatment, pain, post-op concern
- treatment_plan: wants to discuss or review treatment
- chart_review: provider needs chart summary
- billing_inquiry: balance, insurance, payment question
- insurance_question: coverage, verification
- recall_response: responding to a recall/reminder
- emergency: dental emergency, severe pain, trauma
- general_inquiry: hours, directions, policies
- unknown: cannot determine

Respond ONLY with a JSON object:
{
    "patient_identified": true/false,
    "patient_ref": "ref if found or null",
    "refined_intent": "one of the intents above",
    "confidence": 0.0-1.0,
    "can_handle": true/false,
    "response": "what to say to the patient (if can_handle)",
    "escalate": false,
    "escalation_reason": null,
    "notes": "any context for the next agent"
}

CRITICAL RULES:
- NEVER access or display patient PII (name, DOB, phone, email)
- Only reference patients by their external_ref token
- Only set escalate=true for real emergencies (severe pain, trauma, uncontrolled bleeding)
- When you can't handle a request yourself (clinical, billing), set can_handle=false but escalate=false — the orchestrator will route to the right specialist
- Be warm, professional, efficient
"""


async def run_concierge(
    workspace_id: str,
    patient_ref: str | None = None,
    intent: str | None = None,
    payload: dict = {},
) -> dict:
    """Run the Concierge agent."""
    llm = get_fast_llm()

    # Build context message
    context_parts = [f"Workspace: {workspace_id}"]
    if patient_ref:
        context_parts.append(f"Patient ref: {patient_ref}")
    if intent:
        context_parts.append(f"Initial intent classification: {intent}")
    if payload.get("text"):
        context_parts.append(f"Patient message: {payload['text']}")
    if payload.get("channel"):
        context_parts.append(f"Channel: {payload['channel']}")

    context = "\n".join(context_parts)

    messages = [
        SystemMessage(content=CONCIERGE_SYSTEM_PROMPT),
        HumanMessage(content=f"Process this interaction:\n\n{context}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        # Audit log
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
            "escalate": True,
            "escalation_reason": f"Concierge error: {str(e)}",
            "notes": None,
            "error": True,
        }
