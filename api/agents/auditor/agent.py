"""
Auditor Agent — Compliance officer.

Responsibilities:
- HIPAA compliance monitoring
- CDT code verification (correct codes for procedures)
- Claim auditing (pre-submission checks)
- Denial pattern detection
- Audit log review
- Agent behavior monitoring

Uses GPT-4o for reasoning about compliance rules.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from api.core.llm import get_primary_llm
from api.services.supabase_client import log_audit_event
import json

AUDITOR_SYSTEM_PROMPT = """You are the Auditor agent for a dental practice. You ensure compliance and billing accuracy.

Your capabilities:
1. HIPAA compliance checks (PHI access patterns, consent verification)
2. CDT code verification (correct procedure codes, common miscoding patterns)
3. Pre-claim auditing (check claims before submission for errors)
4. Denial pattern analysis (identify trends in rejected claims)
5. Agent behavior monitoring (ensure other agents follow PHI rules)
6. Audit trail review (flag anomalies in system logs)

You will receive:
- Workspace context
- Prior agent outputs to review for compliance
- Specific audit requests

Respond ONLY with a JSON object:
{
    "audit_result": {
        "status": "pass|warning|fail",
        "checks_performed": ["list of checks"],
        "findings": [
            {
                "severity": "info|warning|critical",
                "category": "hipaa|billing|coding|behavior",
                "description": "what was found",
                "recommendation": "what to do about it"
            }
        ],
        "compliance_score": 0-100,
        "phi_exposure_detected": false,
        "billing_issues": []
    },
    "balance_info": null,
    "notes": "summary"
}

CRITICAL RULES:
- You are the last line of defense — be thorough
- Flag ANY instance of PII appearing in agent outputs
- CDT code verification: check bundling rules, modifier requirements
- Zero tolerance for HIPAA violations
- Document everything — your output is part of the audit trail
"""


async def run_auditor(
    workspace_id: str,
    patient_ref: str | None = None,
    prior_outputs: dict = {},
) -> dict:
    """Run the Auditor agent."""
    llm = get_primary_llm()

    context_parts = [f"Workspace: {workspace_id}"]
    if patient_ref:
        context_parts.append(f"Patient ref: {patient_ref}")

    # Review prior agent outputs for compliance
    if prior_outputs:
        context_parts.append("\n--- Agent Outputs to Audit ---")
        for agent_name, output in prior_outputs.items():
            # Serialize safely
            output_str = json.dumps(output, default=str)[:2000]
            context_parts.append(f"\n[{agent_name}]:\n{output_str}")

    context = "\n".join(context_parts)

    messages = [
        SystemMessage(content=AUDITOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Perform compliance audit:\n\n{context}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        audit_result = result.get("audit_result", {})

        await log_audit_event(
            workspace_id=workspace_id,
            actor_type="agent",
            actor_id="auditor",
            action="compliance_audit",
            resource_type="interaction",
            resource_id=patient_ref,
            metadata={
                "status": audit_result.get("status"),
                "compliance_score": audit_result.get("compliance_score"),
                "findings_count": len(audit_result.get("findings", [])),
                "phi_exposure": audit_result.get("phi_exposure_detected"),
            },
        )

        return result

    except Exception as e:
        return {
            "audit_result": {
                "status": "error",
                "checks_performed": [],
                "findings": [
                    {
                        "severity": "warning",
                        "category": "behavior",
                        "description": f"Auditor failed to run: {str(e)}",
                        "recommendation": "Manual review required",
                    }
                ],
                "compliance_score": 0,
                "phi_exposure_detected": False,
                "billing_issues": [],
            },
            "notes": f"Auditor error: {str(e)}",
            "error": True,
        }
