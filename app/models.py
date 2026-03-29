"""
OpenEnv typed models for SRE Incident Response environment.
Complies with OpenEnv spec: Observation, Action, Reward as Pydantic models.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime


# ─── Core Domain Models ──────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    name: str
    status: Literal["healthy", "degraded", "down", "unknown"]
    cpu_percent: float = Field(..., ge=0.0, le=100.0)
    memory_percent: float = Field(..., ge=0.0, le=100.0)
    error_rate: float = Field(..., ge=0.0, description="Errors per second")
    connections: Optional[int] = None
    max_connections: Optional[int] = None
    replicas: int = 1
    version: str = "1.0.0"
    tags: Dict[str, str] = {}


class Alert(BaseModel):
    alert_id: str
    severity: Literal["critical", "warning", "info"]
    service: str
    message: str
    triggered_at: str
    acknowledged: bool = False


class LogEntry(BaseModel):
    timestamp: str
    level: Literal["ERROR", "WARN", "INFO", "DEBUG"]
    service: str
    message: str
    trace_id: Optional[str] = None


class MetricPoint(BaseModel):
    name: str
    value: float
    unit: str
    service: str
    timestamp: str


# ─── OpenEnv Core Types ───────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    The agent's view of the environment at each step.
    Implements OpenEnv Observation spec.
    """
    session_id: str
    task_id: str
    step: int
    timestamp: str

    # Incident data (always visible)
    alerts: List[Alert]
    services: Dict[str, ServiceStatus]

    # Queried data (only populated after agent investigates)
    logs: List[LogEntry] = []
    metrics: List[MetricPoint] = []

    # Episode state
    available_actions: List[str]
    incident_resolved: bool = False
    message: str = ""

    # Contextual hints
    recent_deployments: List[Dict[str, Any]] = []
    runbook_hints: List[str] = []


class Action(BaseModel):
    """
    An action the agent can take in the environment.
    Implements OpenEnv Action spec.

    action_type options:
      - query_logs: Fetch recent logs for a service
      - check_metrics: Retrieve metrics for a service
      - restart_service: Restart a named service
      - rollback_deployment: Roll back a service to its previous version
      - scale_service: Change replica count
      - kill_query: Terminate a running database query from a named source
      - acknowledge_alert: Acknowledge an alert by ID
      - examine_trace: Examine a distributed trace by trace_id
      - check_config: Inspect the live configuration of a service
      - resolve_incident: Mark the incident as resolved (terminal action)
    """
    action_type: str = Field(
        ...,
        description="The type of action to perform",
        examples=["query_logs", "restart_service", "resolve_incident"],
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters. E.g., {'service': 'web-api'}",
        examples=[{"service": "web-api"}, {"service": "db-primary", "source": "analytics-worker"}],
    )


class Reward(BaseModel):
    """
    Per-step reward with breakdown for interpretability.
    Implements OpenEnv Reward spec.
    """
    value: float = Field(..., description="Reward for this step")
    cumulative: float = Field(..., description="Total reward so far this episode")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Named reward components for debugging",
    )
    message: str = Field("", description="Human-readable explanation of reward")


class StepResponse(BaseModel):
    """Full response from a step() call."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


class ResetRequest(BaseModel):
    """Request body for reset()."""
    task_id: str = Field("task1", description="One of: task1, task2, task3")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class StateResponse(BaseModel):
    """Full internal state (for grading/debugging)."""
    session_id: str
    task_id: str
    step: int
    done: bool
    total_reward: float
    world_state: Dict[str, Any]
    action_history: List[Dict[str, Any]]
    grader_score: Optional[float] = None


class TaskInfo(BaseModel):
    """Metadata about a task."""
    task_id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    passing_score: float
    action_schema: Dict[str, Any]
    observation_schema: Dict[str, Any]


class GraderResponse(BaseModel):
    """Response from /grader endpoint."""
    session_id: str
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    episode_complete: bool
    steps_taken: int
    message: str


class BaselineResult(BaseModel):
    """Result from /baseline endpoint."""
    task_id: str
    task_name: str
    difficulty: str
    score: float
    steps_taken: int
    episode_log: List[Dict[str, Any]]
    success: bool
