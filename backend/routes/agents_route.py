"""
GET /api/agents & POST /api/agents — inspect and configure agent models.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.config import AGENT_NAMES, get_agent_model
from backend.models import AgentConfig, AgentConfigUpdate

router = APIRouter()

# In-memory agent config overrides
_overrides: dict[str, AgentConfig] = {}


@router.get("/api/agents")
async def list_agents():
    """Return current agent configuration."""
    agents = []
    for name in AGENT_NAMES:
        if name in _overrides:
            agents.append(_overrides[name].model_dump())
        else:
            agents.append(
                AgentConfig(name=name, model=get_agent_model(name)).model_dump()
            )
    return {"agents": agents}


@router.post("/api/agents")
async def update_agents(req: AgentConfigUpdate):
    """Dynamically update agent models and routing."""
    updated = []
    for cfg in req.agents:
        _overrides[cfg.name] = cfg
        updated.append(cfg.name)
    return {"updated": updated, "agents": (await list_agents())["agents"]}
