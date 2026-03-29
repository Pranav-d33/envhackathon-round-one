"""
Task 1: CPU Spike Investigation (Easy)
=======================================
Scenario: The web-api service has been pegged at 95% CPU for 12 minutes,
causing elevated latency. A memory leak in the request handler is causing
a GC spin loop. The agent must investigate and restart the service.

Optimal solution: query_logs(web-api) → restart_service(web-api) → resolve_incident()
Max steps: 15  |  Passing score: 0.6
"""

from typing import Dict, Any, Tuple, List
from app.models import Observation, Alert, ServiceStatus, LogEntry, MetricPoint
from app.tasks.base import BaseTask, AVAILABLE_ACTIONS, BASE_INCIDENT_TIME


class CPUSpikeTask(BaseTask):
    task_id = "task1"
    name = "CPU Spike Investigation"
    description = (
        "The web-api service is consuming 95% CPU. Investigate the root cause "
        "using logs and metrics, then remediate the incident."
    )
    difficulty = "easy"
    max_steps = 15
    passing_score = 0.6

    def initial_state(self, seed: int = 42) -> Dict[str, Any]:
        return {
            "services": {
                "web-api": {
                    "status": "degraded", "cpu": 95.2, "memory": 78.4,
                    "error_rate": 3.1, "version": "2.3.1", "replicas": 2,
                },
                "db-primary": {
                    "status": "healthy", "cpu": 14.0, "memory": 48.0,
                    "error_rate": 0.0, "connections": 28, "max_connections": 100,
                    "version": "14.5",
                },
                "cache": {
                    "status": "healthy", "cpu": 6.0, "memory": 32.0,
                    "error_rate": 0.0, "version": "7.2.0",
                },
                "load-balancer": {
                    "status": "healthy", "cpu": 3.0, "memory": 15.0,
                    "error_rate": 0.0, "version": "1.28.0",
                },
            },
            "alerts": [
                {"id": "ALT-001", "sev": "critical", "svc": "web-api",
                 "msg": "CPU utilization at 95.2% — sustained for 12 minutes (threshold: 80%)",
                 "ack": False},
                {"id": "ALT-002", "sev": "warning", "svc": "web-api",
                 "msg": "Request latency P99 = 8.4s (SLA threshold: 2s)",
                 "ack": False},
                {"id": "ALT-003", "sev": "info", "svc": "load-balancer",
                 "msg": "Increased error routing to healthy upstream",
                 "ack": False},
            ],
            "recent_deployments": [
                {"service": "web-api", "version": "2.3.1", "previous": "2.3.0",
                 "deployed_at": "2024-11-15T06:00:00Z", "deployer": "ci-pipeline"},
                {"service": "cache", "version": "7.2.0", "previous": "7.1.9",
                 "deployed_at": "2024-11-14T22:00:00Z", "deployer": "ci-pipeline"},
            ],
            # Tracking agent progress
            "logs_queried": [],
            "metrics_checked": [],
            "configs_checked": [],
            "traces_examined": [],
            "web_api_restarted": False,
            "wrong_actions": 0,
            "incident_resolved": False,
        }

    def process_action(
        self, action_type: str, params: Dict[str, Any], state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, str]:
        reward = 0.0
        done = False
        message = ""
        service = params.get("service", "").strip()

        if action_type == "query_logs":
            if not service:
                return state, -0.02, False, "Parameter 'service' is required for query_logs."
            if service in state["logs_queried"]:
                reward = 0.0
                message = f"[Cached] Logs for {service} already retrieved."
            elif service == "web-api":
                state["logs_queried"].append(service)
                reward = 0.12
                message = (
                    "2024-11-15T09:34:11Z [ERROR] web-api RequestHandler: OutOfMemoryError "
                    "caught, forcing GC — heap 98% full\n"
                    "2024-11-15T09:35:02Z [ERROR] web-api RequestHandler: GC pause 1240ms, "
                    "thread stalled — possible memory leak in connection pool\n"
                    "2024-11-15T09:36:45Z [WARN]  web-api ConnectionPool: 512 leaked "
                    "connections detected, pool not releasing objects\n"
                    "2024-11-15T09:38:00Z [ERROR] web-api RequestHandler: CPU spin on GC "
                    "collect() — recommend service restart\n"
                    "2024-11-15T09:44:20Z [ERROR] web-api: request timeout after 8000ms "
                    "(3 occurrences in last 60s)\n"
                    "ROOT CAUSE HINT: Connection pool is leaking — GC is spinning trying "
                    "to reclaim memory, causing CPU spike."
                )
            elif service == "db-primary":
                state["logs_queried"].append(service)
                reward = 0.02
                message = (
                    "2024-11-15T09:40:00Z [INFO] db-primary: checkpoint completed\n"
                    "2024-11-15T09:45:00Z [INFO] db-primary: autovacuum finished on table users\n"
                    "No anomalies in database logs."
                )
            elif service == "cache":
                state["logs_queried"].append(service)
                reward = 0.02
                message = (
                    "2024-11-15T09:30:00Z [INFO] cache: eviction rate 0.2%\n"
                    "No anomalies in cache logs."
                )
            else:
                state["logs_queried"].append(service)
                reward = 0.01
                message = f"No logs found for service '{service}'."

        elif action_type == "check_metrics":
            if not service:
                return state, -0.02, False, "Parameter 'service' is required for check_metrics."
            if service in state["metrics_checked"]:
                message = f"[Cached] Metrics for {service} already retrieved."
                reward = 0.0
            elif service == "web-api":
                state["metrics_checked"].append(service)
                reward = 0.06
                message = (
                    "web-api metrics (last 15 min):\n"
                    "  cpu_percent:       95.2% ↑ (was 12% before 09:30)\n"
                    "  memory_percent:    78.4% ↑ (growing 1.2%/min)\n"
                    "  heap_used_mb:      3840 / 4096\n"
                    "  gc_pause_ms_avg:   950ms (normal: <50ms)\n"
                    "  request_latency_p99: 8.4s\n"
                    "  connections_leaked: 512 (normal: 0)\n"
                    "INSIGHT: Memory growth is linear — classic leak pattern."
                )
            else:
                state["metrics_checked"].append(service)
                reward = 0.02
                message = f"Metrics for {service}: All values within normal operating ranges."

        elif action_type == "check_config":
            if not service:
                return state, -0.02, False, "Parameter 'service' is required for check_config."
            state["configs_checked"].append(service)
            if service == "web-api":
                reward = 0.04
                message = (
                    "web-api live config:\n"
                    "  connection_pool_max: 512\n"
                    "  connection_pool_timeout: 30s\n"
                    "  connection_pool_recycle: DISABLED  ← note: recycle was enabled in v2.3.0\n"
                    "  heap_size: 4096m\n"
                    "  gc_policy: G1GC\n"
                    "Config change in v2.3.1: connection_pool_recycle was accidentally disabled."
                )
            else:
                reward = 0.01
                message = f"Config for {service}: No unusual settings detected."

        elif action_type == "restart_service":
            if not service:
                return state, -0.02, False, "Parameter 'service' is required for restart_service."
            if service == "web-api":
                state["web_api_restarted"] = True
                state["services"]["web-api"]["status"] = "healthy"
                state["services"]["web-api"]["cpu"] = 11.0
                state["services"]["web-api"]["memory"] = 34.0
                state["services"]["web-api"]["error_rate"] = 0.0
                reward = 0.40
                message = (
                    "✓ web-api restarted successfully (rolling restart, ~15s downtime).\n"
                    "  CPU: 95.2% → 11.0%\n"
                    "  Memory: 78.4% → 34.0%\n"
                    "  Error rate: 3.1/s → 0.0/s\n"
                    "  Status: healthy\n"
                    "NOTE: Root cause (disabled connection pool recycle) is still present. "
                    "CPU spike may recur without a permanent fix."
                )
            else:
                state["wrong_actions"] += 1
                reward = -0.08
                message = (
                    f"Restarted {service}, but this service is healthy and unrelated to "
                    f"the incident. No improvement observed. Avoid unnecessary restarts."
                )

        elif action_type == "rollback_deployment":
            if not service:
                return state, -0.02, False, "Parameter 'service' is required for rollback_deployment."
            if service == "web-api":
                # Rollback also fixes it (alternative valid solution)
                state["web_api_restarted"] = True  # treat as resolved
                state["services"]["web-api"]["status"] = "healthy"
                state["services"]["web-api"]["cpu"] = 10.0
                state["services"]["web-api"]["memory"] = 32.0
                state["services"]["web-api"]["error_rate"] = 0.0
                reward = 0.45  # slightly better because it fixes root cause
                message = (
                    "✓ web-api rolled back to v2.3.0.\n"
                    "  connection_pool_recycle re-enabled.\n"
                    "  CPU: 95.2% → 10.0%\n"
                    "  Memory: 78.4% → 32.0%\n"
                    "  Status: healthy\n"
                    "EXCELLENT: This addresses the root cause, not just the symptom."
                )
            else:
                state["wrong_actions"] += 1
                reward = -0.05
                message = f"Rolled back {service}, but this is unrelated to the incident."

        elif action_type == "scale_service":
            if service == "web-api":
                replicas = params.get("replicas", 2)
                reward = 0.05
                message = (
                    f"Scaled web-api to {replicas} replicas. This distributes load but "
                    f"does NOT fix the underlying memory leak. CPU per instance still high."
                )
            else:
                reward = -0.03
                message = f"Scaling {service} has no effect on the current incident."

        elif action_type == "acknowledge_alert":
            alert_id = params.get("alert_id", "")
            for a in state["alerts"]:
                if a["id"] == alert_id:
                    a["ack"] = True
            reward = 0.01
            message = f"Alert {alert_id} acknowledged."

        elif action_type == "examine_trace":
            state["traces_examined"].append(params.get("trace_id", "unknown"))
            reward = 0.03
            message = (
                "Trace analysis: Request span shows 7.8s blocked in GC pause within "
                "web-api RequestHandler. All downstream services (db, cache) responding "
                "normally. Bottleneck is exclusively in web-api."
            )

        elif action_type == "resolve_incident":
            if state["web_api_restarted"]:
                state["incident_resolved"] = True
                done = True
                reward = 0.30
                message = (
                    "✓ Incident resolved.\n"
                    "Summary: web-api experienced a CPU spike due to a memory leak "
                    "(connection pool not recycling in v2.3.1). Service was remediated."
                )
            else:
                reward = -0.05
                message = (
                    "Cannot resolve: web-api is still degraded (CPU 95.2%). "
                    "Investigate and fix the root cause before resolving."
                )

        else:
            reward = -0.03
            message = f"Unknown or inapplicable action: {action_type}."

        return state, reward, done, message

    def get_observation(self, state: Dict[str, Any], session_id: str, step: int) -> Observation:
        services = {}
        for name, s in state["services"].items():
            services[name] = ServiceStatus(
                name=name,
                status=s["status"],
                cpu_percent=s["cpu"],
                memory_percent=s["memory"],
                error_rate=s["error_rate"],
                connections=s.get("connections"),
                max_connections=s.get("max_connections"),
                version=s.get("version", "1.0.0"),
                replicas=s.get("replicas", 1),
            )

        alerts = [
            Alert(
                alert_id=a["id"], severity=a["sev"], service=a["svc"],
                message=a["msg"], triggered_at=BASE_INCIDENT_TIME,
                acknowledged=a["ack"],
            )
            for a in state["alerts"]
        ]

        return Observation(
            session_id=session_id,
            task_id=self.task_id,
            step=step,
            timestamp=BASE_INCIDENT_TIME,
            alerts=alerts,
            services=services,
            logs=[],
            metrics=[],
            available_actions=AVAILABLE_ACTIONS,
            incident_resolved=state["incident_resolved"],
            message="",
            recent_deployments=state["recent_deployments"],
            runbook_hints=[
                "High CPU often caused by: memory leaks, infinite loops, or hot code paths.",
                "Check recent deployments for configuration changes.",
                "query_logs and check_metrics before taking disruptive actions.",
            ],
        )

    def grade(self, state: Dict[str, Any], history: List[Dict]) -> Tuple[float, Dict[str, float]]:
        breakdown = {}
        score = 0.0

        # Root cause investigated?
        if "web-api" in state.get("logs_queried", []) or \
           "web-api" in state.get("metrics_checked", []) or \
           "web-api" in state.get("configs_checked", []):
            breakdown["investigated_root_service"] = 0.15
            score += 0.15

        # Service remediated?
        if state.get("web_api_restarted", False):
            breakdown["service_remediated"] = 0.45
            score += 0.45

        # Incident formally closed?
        if state.get("incident_resolved", False):
            breakdown["incident_resolved"] = 0.25
            score += 0.25

        # Efficiency bonus
        steps = len(history)
        if steps <= 4:
            breakdown["efficiency_bonus"] = 0.15
            score += 0.15
        elif steps <= 7:
            breakdown["efficiency_bonus"] = 0.10
            score += 0.10
        elif steps <= 10:
            breakdown["efficiency_bonus"] = 0.05
            score += 0.05
        else:
            breakdown["efficiency_bonus"] = 0.0

        # Penalty for wrong actions
        wrong = state.get("wrong_actions", 0)
        if wrong > 0:
            penalty = min(wrong * 0.08, 0.20)
            breakdown["wrong_action_penalty"] = -penalty
            score -= penalty

        return round(min(max(score, 0.0), 1.0), 4), breakdown
