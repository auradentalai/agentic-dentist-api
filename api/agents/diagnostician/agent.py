"""
Diagnostician Agent — Clinical intelligence.

Responsibilities:
- Generate pre-appointment briefing cards
- Identify treatment gaps and overdue procedures
- Flag risk factors from chart history
- Summarize patient clinical history

Uses GPT-4o for complex medical reasoning.
NEVER outputs PII — only references external_ref tokens.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from api.core.llm import get_primary_llm
from api.services.supabase_client import log_audit_event
import json

DIAGNOSTICIAN_SYSTEM_PROMPT = """You are the Diagnostician agent for a dental practice. You provide clinical intelligence.

Your capabilities:
1. Pre-appointment briefing cards (summary of patient history, pending treatments, alerts)
2. Treatment gap analysis (overdue cleanings, incomplete treatment plans)
3. Risk flagging (medical history concerns, drug interactions, allergy alerts)
4. Chart summarization for providers

You will receive:
- Patient reference ID (external_ref) — NEVER their real name
- Any prior agent outputs (from Concierge routing)
- Clinical context as available

Respond ONLY with a JSON object:
{
    "briefing_card": {
        "patient_ref": "the external ref",
        "summary": "2-3 sentence clinical summary",
        "alerts": ["list of clinical alerts"],
        "pending_treatments": ["list of pending items"],
        "treatment_gaps": ["overdue procedures"],
        "risk_flags": ["medical/drug/allergy concerns"],
        "last_visit": "date or unknown",
        "next_recommended": "what should happen next"
    },
    "confidence": 0.0-1.0,
    "data_quality": "good/partial/insufficient",
    "notes": "any notes for the provider or other agents"
}

CRITICAL RULES:
- NEVER include patient PII (name, DOB, SSN, phone, email) in any output
- Only reference patients by external_ref
- Flag insufficient data clearly — never fabricate clinical information
- If data is insufficient, say so — don't guess about medical history
- Use proper dental terminology (CDT codes, ADA classifications)
"""


async def run_diagnostician(
    workspace_id: str,
    patient_ref: str | None = None,
    prior_outputs: dict = {},
) -> dict:
    """Run the Diagnostician agent."""
    llm = get_primary_llm()

    context_parts = [f"Workspace: {workspace_id}"]
    if patient_ref:
        context_parts.append(f"Patient ref: {patient_ref}")
    else:
        context_parts.append("No specific patient — general analysis requested")

    if prior_outputs.get("concierge"):
        concierge = prior_outputs["concierge"]
        context_parts.append(f"Concierge intent: {concierge.get('refined_intent', 'unknown')}")
        if concierge.get("notes"):
            context_parts.append(f"Concierge notes: {concierge['notes']}")

    # TODO: Fetch actual clinical data from Supabase
    context_parts.append("\n[Note: Clinical data integration pending — generating template briefing]")

    context = "\n".join(context_parts)

    messages = [
        SystemMessage(content=DIAGNOSTICIAN_SYSTEM_PROMPT),
        HumanMessage(content=f"Generate clinical intelligence:\n\n{context}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        await log_audit_event(
            workspace_id=workspace_id,
            actor_type="agent",
            actor_id="diagnostician",
            action="briefing_generated",
            resource_type="patient" if patient_ref else None,
            resource_id=patient_ref,
            metadata={
                "data_quality": result.get("data_quality"),
                "alerts_count": len(result.get("briefing_card", {}).get("alerts", [])),
                "gaps_count": len(result.get("briefing_card", {}).get("treatment_gaps", [])),
            },
        )

        return result

    except Exception as e:
        return {
            "briefing_card": None,
            "confidence": 0.0,
            "data_quality": "error",
            "notes": f"Diagnostician error: {str(e)}",
            "error": True,
        }
