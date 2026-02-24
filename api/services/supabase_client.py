"""
Supabase admin client — service role, bypasses RLS.
Used for agent operations and encrypted PHI access.
"""

from supabase import create_client, Client
from api.core.config import settings

_client: Client | None = None


def get_supabase_admin() -> Client:
    """Get or create the Supabase admin client (singleton)."""
    global _client
    if _client is None:
        _client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
    return _client


async def log_audit_event(
    workspace_id: str,
    actor_type: str,  # "user" | "agent" | "system"
    actor_id: str,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict | None = None,
):
    """Write to the immutable audit log."""
    supabase = get_supabase_admin()
    supabase.table("audit_log").insert(
        {
            "workspace_id": workspace_id,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "metadata": metadata or {},
        }
    ).execute()
