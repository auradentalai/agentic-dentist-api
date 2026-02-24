from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Environment
    environment: str = "dev"

    # Supabase
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # LLM — provider-agnostic
    llm_provider: str = "openai"
    openai_api_key: str = ""
    llm_model_primary: str = "gpt-4o"  # Orchestrator, Diagnostician, Auditor
    llm_model_fast: str = "gpt-4o-mini"  # Concierge, Liaison

    # PHI Encryption
    phi_encryption_key: str = "CHANGE_ME_IN_PRODUCTION"

    # API Security
    api_secret_key: str = ""

    # CORS
    frontend_url: str = "https://agentic-dentist.vercel.app"

    @property
    def cors_origins(self) -> List[str]:
        origins = [self.frontend_url]
        if self.environment == "dev":
            origins.append("http://localhost:3000")
        return origins

    # Future
    vapi_api_key: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    langchain_api_key: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
