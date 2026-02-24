from fastapi import APIRouter
from api.core.config import settings
from api.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — verifies config and connectivity."""
    supabase_ok = False
    try:
        from api.services.supabase_client import get_supabase_admin
        client = get_supabase_admin()
        client.table("profiles").select("id").limit(1).execute()
        supabase_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if supabase_ok else "degraded",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        llm_model_primary=settings.llm_model_primary,
        llm_model_fast=settings.llm_model_fast,
        supabase_connected=supabase_ok,
        agents=["concierge", "diagnostician", "liaison", "auditor"],
    )
