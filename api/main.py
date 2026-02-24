from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.core.config import settings
from api.routers import health, agents

app = FastAPI(
    title="The Agentic Dentist — API",
    description="AI agent swarm backend for dental practice management",
    version="0.1.0",
)

# CORS — allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(agents.router, prefix="/api/agents", tags=["Agents"])


@app.get("/")
async def root():
    return {
        "name": "The Agentic Dentist API",
        "version": "0.1.0",
        "status": "running",
        "agents": ["concierge", "diagnostician", "liaison", "auditor"],
    }
