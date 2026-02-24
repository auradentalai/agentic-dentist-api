"""
Provider-agnostic LLM wrapper.
All agent code calls this module — never OpenAI directly.
Swapping providers is a config change, not a rewrite.
"""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from api.core.config import settings


def get_llm(tier: str = "primary") -> BaseChatModel:
    """
    Get an LLM instance by tier.
    
    Tiers:
        - "primary": GPT-4o — complex reasoning (Orchestrator, Diagnostician, Auditor)
        - "fast": GPT-4o-mini — routing, templates (Concierge, Liaison)
    """
    model = settings.llm_model_primary if tier == "primary" else settings.llm_model_fast

    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=2000,
        )
    else:
        # Future: add anthropic, cohere, etc.
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


# Pre-configured instances for convenience
def get_primary_llm() -> BaseChatModel:
    """GPT-4o for complex reasoning."""
    return get_llm("primary")


def get_fast_llm() -> BaseChatModel:
    """GPT-4o-mini for fast routing."""
    return get_llm("fast")
