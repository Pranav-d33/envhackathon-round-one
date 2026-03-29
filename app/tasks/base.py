"""Base class for all SRE incident tasks."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from app.models import Observation, Alert, ServiceStatus, LogEntry, MetricPoint
from datetime import datetime, timezone


BASE_INCIDENT_TIME = "2024-11-15T09:47:00Z"

AVAILABLE_ACTIONS = [
    "query_logs",
    "check_metrics",
    "restart_service",
    "rollback_deployment",
    "scale_service",
    "kill_query",
    "acknowledge_alert",
    "examine_trace",
    "check_config",
    "resolve_incident",
]

ACTION_SCHEMA = {
    "query_logs": {
        "description": "Fetch recent log entries for a service",
        "parameters": {
            "service": {"type": "string", "required": True, "description": "Service name"},
            "lines": {"type": "integer", "required": False, "default": 50},
        },
    },
    "check_metrics": {
        "description": "Retrieve current metrics for a service",
        "parameters": {
            "service": {"type": "string", "required": True},
        },
    },
    "restart_service": {
        "description": "Restart a service (rolling restart, brief downtime)",
        "parameters": {
            "service": {"type": "string", "required": True},
        },
    },
    "rollback_deployment": {
        "description": "Roll back a service to its previous deployment version",
        "parameters": {
            "service": {"type": "string", "required": True},
        },
    },
    "scale_service": {
        "description": "Change the number of replicas for a service",
        "parameters": {
            "service": {"type": "string", "required": True},
            "replicas": {"type": "integer", "required": True, "min": 1, "max": 20},
        },
    },
    "kill_query": {
        "description": "Kill long-running database queries from a specific source/application",
        "parameters": {
            "source": {"type": "string", "required": True, "description": "Application or service holding queries"},
        },
    },
    "acknowledge_alert": {
        "description": "Acknowledge an alert to stop paging",
        "parameters": {
            "alert_id": {"type": "string", "required": True},
        },
    },
    "examine_trace": {
        "description": "Examine a distributed trace to identify slow spans",
        "parameters": {
            "trace_id": {"type": "string", "required": True},
        },
    },
    "check_config": {
        "description": "Inspect the live runtime configuration of a service",
        "parameters": {
            "service": {"type": "string", "required": True},
        },
    },
    "resolve_incident": {
        "description": "Mark the incident as resolved. Terminal action — ends the episode.",
        "parameters": {},
    },
}


class BaseTask(ABC):
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    passing_score: float = 0.6

    @abstractmethod
    def initial_state(self, seed: int = 42) -> Dict[str, Any]:
        """Return initial world state dict."""
        ...

    @abstractmethod
    def process_action(
        self, action_type: str, params: Dict[str, Any], state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, str]:
        """
        Apply action to state.
        Returns: (new_state, step_reward, done, message)
        """
        ...

    @abstractmethod
    def get_observation(self, state: Dict[str, Any], session_id: str, step: int) -> Observation:
        """Build Observation from state."""
        ...

    @abstractmethod
    def grade(self, state: Dict[str, Any], history: List[Dict]) -> Tuple[float, Dict[str, float]]:
        """
        Grade the episode.
        Returns: (score 0.0–1.0, breakdown dict)
        """
        ...

    def _make_service(self, name: str, status: str, cpu: float, mem: float,
                       err: float, **kwargs) -> ServiceStatus:
        return ServiceStatus(
            name=name, status=status,
            cpu_percent=cpu, memory_percent=mem, error_rate=err,
            **kwargs,
        )

    def _make_alert(self, alert_id: str, severity: str, service: str,
                     message: str, ack: bool = False) -> Alert:
        return Alert(
            alert_id=alert_id, severity=severity, service=service,
            message=message, triggered_at=BASE_INCIDENT_TIME, acknowledged=ack,
        )
