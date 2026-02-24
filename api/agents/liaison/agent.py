"""
Liaison Agent — Communication execution.

Responsibilities:
- Draft and send appointment reminders
- Recall campaigns (6-month cleanings, overdue patients)
- Post-op follow-up messages
- Balance notifications
- Multi-channel: SMS (Twilio), Email, Phone (Vapi)

Uses GPT-4o-mini for template generation.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from api.core.llm import get_fast_llm
from api.services.supabase_client import log_audit_event
import json

LIAISON_SYSTEM_PROMPT = """You are the Liaison agent for a dental practice. You handle all outbound communications.

Your capabilities:
1. Appointment reminders (24h, 48h, 1-week)
2. Recall campaigns (6-month cleaning, overdue treatment)
3. Post-op follow-up messages
4. Balance/billing notifications
5. General practice communications

You will receive:
- Patient reference (external_ref only)
- Communication type/context
- Prior agent outputs (from Concierge/Diagnostician)
- Preferred language for the patient

Respond ONLY with a JSON object:
{
    "messages": [
        {
            "channel": "sms|email|phone",
            "recipient_ref": "patient external_ref",
            "template_type": "reminder|recall|post_op|balance|general",
            "subject": "email subject if email",
            "body": "message content",
            "language": "en|es|fr|etc",
            "urgency": "low|medium|high",
            "send_at": "now|scheduled datetime",
            "requires_approval": true/false
        }
    ],
    "campaign_id": "if part of a campaign",
    "notes": "any context"
}

CRITICAL RULES:
- NEVER include patient PII in message templates — use placeholders like {patient_name}
- The actual PII substitution happens at send-time by the secure delivery layer
- Always include opt-out language for SMS
- Post-op messages must include emergency contact info
- Balance messages must include dispute instructions
- Multi-language: draft in patient's preferred language
"""


async def run_liaison(
    workspace_id: str,
    patient_ref: str | None = None,
    prior_outputs: dict = {},
) -> dict:
    """Run the Liaison agent."""
    llm = get_fast_llm()

    context_parts = [f"Workspace: {workspace_id}"]
    if patient_ref:
        context_parts.append(f"Patient ref: {patient_ref}")

    # Determine communication context from prior outputs
    if prior_outputs.get("concierge"):
        concierge = prior_outputs["concierge"]
        intent = concierge.get("refined_intent", "general_inquiry")
        context_parts.append(f"Intent: {intent}")
        if concierge.get("response"):
            context_parts.append(f"Concierge response to patient: {concierge['response']}")

    if prior_outputs.get("diagnostician"):
        diag = prior_outputs["diagnostician"]
        card = diag.get("briefing_card", {})
        if card.get("treatment_gaps"):
            context_parts.append(f"Treatment gaps found: {card['treatment_gaps']}")
        if card.get("next_recommended"):
            context_parts.append(f"Recommended next step: {card['next_recommended']}")

    if prior_outputs.get("auditor"):
        auditor = prior_outputs["auditor"]
        if auditor.get("balance_info"):
            context_parts.append(f"Balance info: {auditor['balance_info']}")

    context = "\n".join(context_parts)

    messages = [
        SystemMessage(content=LIAISON_SYSTEM_PROMPT),
        HumanMessage(content=f"Draft communications based on this context:\n\n{context}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        msg_count = len(result.get("messages", []))

        await log_audit_event(
            workspace_id=workspace_id,
            actor_type="agent",
            actor_id="liaison",
            action="communications_drafted",
            resource_type="patient" if patient_ref else None,
            resource_id=patient_ref,
            metadata={
                "message_count": msg_count,
                "channels": list({m.get("channel") for m in result.get("messages", [])}),
            },
        )

        return result

    except Exception as e:
        return {
            "messages": [],
            "notes": f"Liaison error: {str(e)}",
            "error": True,
        }
