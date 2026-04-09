"""
SRE Incident Response — OpenEnv Environment
============================================
FastAPI application implementing the full OpenEnv spec.

Endpoints:
  POST /reset          — Start a new episode
  POST /step           — Take one action
  GET  /state          — Get current session state
  GET  /tasks          — List tasks and action schema
  POST /grader         — Get grader score for a session
  POST /baseline       — Run baseline agent on all tasks
  GET  /health         — Health check
"""

import os
import json
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.models import (
    Observation, Action, Reward, StepResponse, ResetRequest,
    StateResponse, TaskInfo, GraderResponse, BaselineResult,
)
from app.environment import EnvironmentManager
from app.tasks import TASK_REGISTRY
from app.tasks.base import ACTION_SCHEMA, AVAILABLE_ACTIONS


# ─── App Setup ───────────────────────────────────────────────────────────────

env_manager = EnvironmentManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("SRE Incident Response Environment starting up...")
    yield
    print("Shutting down...")


app = FastAPI(
    title="SRE Incident Response — OpenEnv Environment",
    description=(
        "An OpenEnv-compliant reinforcement learning environment where an AI agent "
        "acts as an on-call Site Reliability Engineer. The agent receives production "
        "incident alerts, investigates root causes using logs/metrics/configs, and "
        "takes remediation actions to restore service health.\n\n"
        "Three tasks of increasing difficulty: CPU spike investigation (easy), "
        "database connection pool exhaustion (medium), and cascading service failure (hard)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ─────────────────────────────────────────────────

class StepRequest(BaseModel):
    session_id: str
    action: Action


class GraderRequest(BaseModel):
    session_id: str


class BaselineRequest(BaseModel):
    model: str = "gpt-4o-mini"
    max_steps: int = 12
    tasks: List[str] = ["task1", "task2", "task3"]


# ─── OpenEnv Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": "sre-incident-response",
        "version": "1.0.0",
        "active_sessions": env_manager.active_sessions(),
    }


@app.post("/reset", response_model=Observation, tags=["OpenEnv"])
async def reset(request: Optional[ResetRequest] = Body(None)):
    """
    Start a new episode.

    Returns the initial observation for the specified task.
    The session_id in the response must be passed to subsequent /step calls.

    - **task_id**: One of `task1` (easy), `task2` (medium), `task3` (hard)
    - **seed**: Optional random seed for reproducibility
    """
    if request is None:
        request = ResetRequest()
    try:
        obs, session_id = env_manager.reset(
            task_id=request.task_id,
            seed=request.seed or 42,
        )
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse, tags=["OpenEnv"])
async def step(request: StepRequest = Body(...)):
    """
    Take one action in the environment.

    Returns observation, reward, done flag, and info dict.

    Action types: `query_logs`, `check_metrics`, `restart_service`,
    `rollback_deployment`, `scale_service`, `kill_query`, `acknowledge_alert`,
    `examine_trace`, `check_config`, `resolve_incident`
    """
    try:
        return env_manager.step(request.session_id, request.action)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step error: {str(e)}")


@app.get("/state", response_model=StateResponse, tags=["OpenEnv"])
async def state(session_id: str = Query(..., description="Session ID from /reset")):
    """
    Get the full current state of a session (for debugging/grading).

    Returns internal world state including hidden state variables and
    action history. The grader_score field shows current progress.
    """
    try:
        return env_manager.get_state(session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── Additional Required Endpoints ───────────────────────────────────────────

@app.get("/tasks", tags=["Environment Info"])
async def list_tasks():
    """
    List all available tasks with descriptions, difficulty levels,
    and the full action schema.
    """
    tasks_info = []
    for task_id, task in TASK_REGISTRY.items():
        tasks_info.append({
            "task_id": task.task_id,
            "name": task.name,
            "description": task.description,
            "difficulty": task.difficulty,
            "max_steps": task.max_steps,
            "passing_score": task.passing_score,
        })

    return {
        "tasks": tasks_info,
        "action_schema": ACTION_SCHEMA,
        "available_actions": AVAILABLE_ACTIONS,
        "observation_fields": [
            "session_id", "task_id", "step", "timestamp",
            "alerts", "services", "logs", "metrics",
            "available_actions", "incident_resolved", "message",
            "recent_deployments", "runbook_hints",
        ],
        "reward_range": [-999.0, 1.0],
    }


@app.post("/grader", response_model=GraderResponse, tags=["Environment Info"])
async def grader(request: GraderRequest = Body(...)):
    """
    Get the current grader score for a session.

    Graders are deterministic and can be called mid-episode or after completion.
    Score range: 0.0 (no progress) to 1.0 (perfect solution).
    """
    try:
        return env_manager.grade(request.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/baseline", tags=["Evaluation"])
async def baseline(request: BaselineRequest = Body(default=BaselineRequest())):
    """
    Run the baseline inference agent against all tasks.

    Requires OPENAI_API_KEY environment variable.
    Uses a ReAct-style prompting strategy with the specified model.
    Returns per-task scores and episode logs.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY environment variable not set. "
                   "Set it to run the baseline agent.",
        )

    try:
        results = await run_baseline_agent(
            api_key=api_key,
            model=request.model,
            max_steps=request.max_steps,
            task_ids=request.tasks,
        )

        # Ensure validator-safe scores even if baseline errored or returned edge values.
        safe_scores = [env_manager._clamp_score_strict(r.get("score", 0.0)) for r in results]
        for r, s in zip(results, safe_scores):
            r["score"] = s

        return {
            "model": request.model,
            "results": results,
            "summary": {
                "mean_score": round(
                    sum(safe_scores) / len(safe_scores), 4
                ) if safe_scores else env_manager._clamp_score_strict(0.0),
                "tasks_passed": sum(
                    1 for r in results
                    if r["score"] >= TASK_REGISTRY[r["task_id"]].passing_score
                ),
                "total_tasks": len(results),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline error: {str(e)}")


# ─── Baseline Agent Logic ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident.
You will receive alerts, service statuses, and investigation results.
Your goal is to identify the root cause and resolve the incident efficiently.

At each step, respond with ONLY a valid JSON object in this exact format:
{"action_type": "<action>", "parameters": {<params>}}

Available actions:
- query_logs: {"service": "<name>"} — fetch recent logs for a service
- check_metrics: {"service": "<name>"} — get current metrics for a service
- check_config: {"service": "<name>"} — inspect live runtime configuration
- restart_service: {"service": "<name>"} — restart a service (use carefully)
- rollback_deployment: {"service": "<name>"} — roll back to previous version
- kill_query: {"source": "<service>"} — terminate long-running DB queries from a source
- scale_service: {"service": "<name>", "replicas": <int>} — change replica count
- examine_trace: {"trace_id": "<id>"} — examine distributed trace
- acknowledge_alert: {"alert_id": "<id>"} — acknowledge an alert
- resolve_incident: {} — mark incident as resolved (only when services are healthy)

SRE Investigation Strategy:
1. Read ALL alerts and service statuses carefully
2. Look at recent deployments — they are often correlated with incidents
3. Use query_logs and check_metrics to gather evidence before acting
4. Form a clear hypothesis about the root cause
5. Apply the most targeted fix (prefer rollback over restart when deployment changed)
6. Verify all affected services are healthy
7. Call resolve_incident to complete the episode

Respond ONLY with JSON. No markdown. No explanation."""


def format_observation(obs: dict) -> str:
    """Format observation dict into a concise prompt string."""
    lines = [f"=== INCIDENT — Step {obs.get('step', 0)} ===\n"]

    lines.append("ACTIVE ALERTS:")
    for alert in obs.get("alerts", []):
        ack = " [ACK]" if alert.get("acknowledged") else ""
        sev = alert.get("severity", "?").upper()
        lines.append(f"  [{sev}]{ack} {alert.get('service')}: {alert.get('message')}")

    lines.append("\nSERVICE STATUS:")
    for name, svc in obs.get("services", {}).items():
        conn = ""
        if svc.get("connections") is not None:
            conn = f" | conns: {svc['connections']}/{svc.get('max_connections', '?')}"
        lines.append(
            f"  {name}: {svc.get('status', '?').upper()} | "
            f"cpu: {svc.get('cpu_percent', 0):.1f}% | "
            f"mem: {svc.get('memory_percent', 0):.1f}% | "
            f"errors: {svc.get('error_rate', 0):.1f}/s | "
            f"v{svc.get('version', '?')}{conn}"
        )

    if obs.get("recent_deployments"):
        lines.append("\nRECENT DEPLOYMENTS:")
        for dep in obs["recent_deployments"]:
            lines.append(
                f"  {dep.get('service')}: v{dep.get('previous', '?')} → "
                f"v{dep.get('version')} deployed at {dep.get('deployed_at')}"
            )

    if obs.get("message"):
        lines.append(f"\nLAST ACTION RESULT:\n{obs['message']}")

    if obs.get("runbook_hints"):
        lines.append("\nRUNBOOK HINTS:")
        for h in obs["runbook_hints"]:
            lines.append(f"  • {h}")

    return "\n".join(lines)


async def run_baseline_agent(api_key: str, model: str, max_steps: int, task_ids: List[str]):
    """Run baseline agent asynchronously."""
    import httpx

    results = []
    base_url = "http://localhost:7860"  # Assume local

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        for task_id in task_ids:
            result = await run_task_async(client, api_key, task_id, model, max_steps)
            results.append(result)

    return results


async def run_task_async(client: httpx.AsyncClient, api_key: str, task_id: str, model: str, max_steps: int):
    """Run one task asynchronously."""
    import httpx

    # Reset
    reset_resp = await client.post("/reset", json={"task_id": task_id, "seed": 42})
    reset_resp.raise_for_status()
    obs = reset_resp.json()
    session_id = obs["session_id"]

    conversation = []
    score = 0.0
    steps_taken = 0

    for step_num in range(max_steps):
        obs_text = format_observation(obs)
        conversation.append({"role": "user", "content": obs_text})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += conversation[-8:]

        # Call LLM (synchronously for now, but could be async)
        llm_resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=30.0,
        )
        llm_resp.raise_for_status()
        action_text = llm_resp.json()["choices"][0]["message"]["content"].strip()

        conversation.append({"role": "assistant", "content": action_text})

        # Parse action
        action_dict = parse_action(action_text)
        action_type = action_dict.get("action_type", "unknown")
        parameters = action_dict.get("parameters", {})

        # Step
        step_resp = await client.post("/step", json={
            "session_id": session_id,
            "action": {"action_type": action_type, "parameters": parameters},
        })
        step_resp.raise_for_status()
        step_data = step_resp.json()

        obs = step_data["observation"]
        done = step_data["done"]
        steps_taken = step_num + 1

        if done:
            break

    # Grade
    grader_resp = await client.post("/grader", json={"session_id": session_id})
    grader_resp.raise_for_status()
    grader_data = grader_resp.json()
    score = grader_data["score"]

    # Task info
    tasks_resp = await client.get("/tasks")
    task_info = {}
    if tasks_resp.status_code == 200:
        for t in tasks_resp.json().get("tasks", []):
            if t["task_id"] == task_id:
                task_info = t
                break

    return {
        "task_id": task_id,
        "task_name": task_info.get("name", task_id),
        "difficulty": task_info.get("difficulty", "?"),
        "score": score,
        "steps_taken": steps_taken,
        "success": score >= task_info.get("passing_score", 0.6),
    }


def parse_action(text: str) -> dict:
    """Parse JSON action from LLM output."""
    import re
    import json

    text = text.strip()
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"action_type": "acknowledge_alert", "parameters": {"alert_id": "ALT-001"}}


def main():
    """Entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()