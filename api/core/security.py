"""
Security helpers for HIPAA-compliant operations.
- JWT verification against Supabase
- PHI masking for agent context
"""

from fastapi import HTTPException, Header
from api.services.supabase_client import get_supabase_admin


async def verify_auth(authorization: str = Header(None)) -> dict:
    """
    Verify Supabase JWT token from Authorization header.
    Returns the user object if valid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization.replace("Bearer ", "")
    supabase = get_supabase_admin()

    try:
        user = supabase.auth.get_user(token)
        return {"id": user.user.id, "email": user.user.email}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def verify_membership(user_id: str, workspace_id: str) -> dict:
    """
    Verify user is an active member of the workspace.
    Returns the membership record.
    """
    supabase = get_supabase_admin()

    result = (
        supabase.table("clinic_memberships")
        .select("id, role, status")
        .eq("profile_id", user_id)
        .eq("clinic_id", workspace_id)
        .eq("status", "active")
        .single()
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=403, detail="Not a member of this workspace")

    return result.data


def mask_phi(text: str) -> str:
    """
    Strip any accidentally included PHI from text before sending to agents.
    This is a safety net — agents should never receive PHI in the first place.
    """
    # TODO: implement NER-based PHI detection
    # For now, this is a placeholder
    return text
