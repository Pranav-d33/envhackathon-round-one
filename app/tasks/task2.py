"""
Task 2: Database Connection Pool Exhaustion (Medium)
=====================================================
Scenario: The db-primary connection pool is fully exhausted (100/100 connections).
Multiple dependent services (payment-api, user-service) are timing out.
Root cause: analytics-worker is running unbounded full-table scans, holding 78
long-running connections that never release.

Trap: Restarting the DB makes things temporarily worse (all services lose their
remaining connections during restart and have to reconnect).

Optimal: check_metrics(db-primary) → query_logs(db-primary) →
         kill_query(analytics-worker) → resolve_incident()
Max steps: 18  |  Passing score: 0.6
"""

from typing import Dict, Any, Tuple, List
from app.models import Observation, Alert, ServiceStatus, LogEntry, MetricPoint
from app.tasks.base import BaseTask, AVAILABLE_ACTIONS, BASE_INCIDENT_TIME


class DBConnectionPoolTask(BaseTask):
    task_id = "task2"
    name = "Database Connection Pool Exhaustion"
    description = (
        "The db-primary connection pool is exhausted (100/100). Multiple services "
        "are timing out on DB calls. Identify which application is holding excess "
        "connections and remediate without restarting the database."
    )
    difficulty = "medium"
    max_steps = 18
    passing_score = 0.6

    def initial_state(self, seed: int = 42) -> Dict[str, Any]:
        return {
            "services": {
                "db-primary": {
                    "status": "degraded", "cpu": 31.0, "memory": 62.0,
                    "error_rate": 8.2, "connections": 100, "max_connections": 100,
                    "version": "14.8",
                },
                "payment-api": {
                    "status": "degraded", "cpu": 45.0, "memory": 55.0,
                    "error_rate": 12.4, "connections": 8, "max_connections": 20,
                    "version": "3.1.2",
                },
                "user-service": {
                    "status": "degraded", "cpu": 38.0, "memory": 50.0,
                    "error_rate": 9.7, "connections": 6, "max_connections": 20,
                    "version": "2.4.0",
                },
                "analytics-worker": {
                    "status": "healthy", "cpu": 42.0, "memory": 70.0,
                    "error_rate": 0.0, "connections": 78, "max_connections": 80,
                    "version": "1.0.9",
                },
                "cache": {
                    "status": "healthy", "cpu": 5.0, "memory": 28.0,
                    "error_rate": 0.0, "version": "7.2.0",
                },
            },
            "alerts": [
                {"id": "ALT-010", "sev": "critical", "svc": "db-primary",
                 "msg": "Connection pool exhausted: 100/100 connections in use",
                 "ack": False},
                {"id": "ALT-011", "sev": "critical", "svc": "payment-api",
                 "msg": "High error rate 12.4/s — DB connection timeout after 30s",
                 "ack": False},
                {"id": "ALT-012", "sev": "critical", "svc": "user-service",
                 "msg": "High error rate 9.7/s — DB connection timeout after 30s",
                 "ack": False},
                {"id": "ALT-013", "sev": "warning", "svc": "analytics-worker",
                 "msg": "High memory usage 70% on analytics-worker",
                 "ack": False},
            ],
            "recent_deployments": [
                {"service": "analytics-worker", "version": "1.0.9", "previous": "1.0.8",
                 "deployed_at": "2024-11-15T08:00:00Z", "deployer": "data-team"},
                {"service": "payment-api", "version": "3.1.2", "previous": "3.1.1",
                 "deployed_at": "2024-11-14T16:00:00Z", "deployer": "ci-pipeline"},
            ],
            # Tracking agent progress
            "logs_queried": [],
            "metrics_checked": [],
            "configs_checked": [],
            "queries_killed": [],
            "db_restarted": False,
            "analytics_worker_killed": False,
            "wrong_actions": 0,
            "incident_resolved": False,
            # Hidden state — what's actually happening
            "_analytics_holding_connections": True,
            "_db_connections_freed": False,
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
                return state, -0.02, False, "Parameter 'service' is required."
            if service in state["logs_queried"]:
                return state, 0.0, False, f"[Cached] Logs for {service} already retrieved."
            state["logs_queried"].append(service)

            if service == "db-primary":
                reward = 0.14
                message = (
                    "2024-11-15T09:32:00Z [WARN]  db-primary: connection count 95/100\n"
                    "2024-11-15T09:35:00Z [ERROR] db-primary: connection pool full, "
                    "new connections queued\n"
                    "2024-11-15T09:38:00Z [ERROR] db-primary: query from analytics-worker "
                    "pid=28441 running 18min on table 'events' (full table scan, no index)\n"
                    "2024-11-15T09:40:00Z [ERROR] db-primary: 78 long-running queries from "
                    "analytics-worker — these are consuming all connections\n"
                    "2024-11-15T09:44:00Z [ERROR] db-primary: payment-api cannot acquire "
                    "connection — pool exhausted\n"
                    "ROOT CAUSE: analytics-worker is running full table scans on 'events' "
                    "table, holding 78 connections indefinitely."
                )
            elif service == "analytics-worker":
                reward = 0.10
                message = (
                    "2024-11-15T08:01:00Z [INFO]  analytics-worker: v1.0.9 deployed\n"
                    "2024-11-15T08:05:00Z [INFO]  analytics-worker: starting daily report job\n"
                    "2024-11-15T08:05:10Z [WARN]  analytics-worker: query_timeout config "
                    "missing — defaulting to no timeout\n"
                    "2024-11-15T09:10:00Z [WARN]  analytics-worker: 78 concurrent queries "
                    "running for >60min — possible misconfiguration\n"
                    "BUG IN v1.0.9: query_timeout was removed from config, causing unbounded "
                    "full-table scans that never terminate."
                )
            elif service == "payment-api":
                reward = 0.06
                message = (
                    "2024-11-15T09:38:00Z [ERROR] payment-api: db connection timeout "
                    "after 30s — pool exhausted upstream\n"
                    "2024-11-15T09:39:00Z [ERROR] payment-api: 12 transaction failures "
                    "due to DB unavailability\n"
                    "payment-api is a victim, not the root cause."
                )
            elif service == "user-service":
                reward = 0.05
                message = (
                    "2024-11-15T09:38:30Z [ERROR] user-service: DB connection timeout — "
                    "pool exhausted upstream\n"
                    "user-service is a victim, not the root cause."
                )
            else:
                reward = 0.01
                message = f"No relevant logs found for '{service}'."

        elif action_type == "check_metrics":
            if not service:
                return state, -0.02, False, "Parameter 'service' is required."
            if service in state["metrics_checked"]:
                return state, 0.0, False, f"[Cached] Metrics for {service} already retrieved."
            state["metrics_checked"].append(service)

            if service == "db-primary":
                reward = 0.12
                message = (
                    "db-primary metrics:\n"
                    "  active_connections:    100 / 100  ← FULL\n"
                    "  connections_by_client:\n"
                    "    analytics-worker:    78  ← 78% of pool\n"
                    "    payment-api:          8\n"
                    "    user-service:         6\n"
                    "    other:                8\n"
                    "  longest_query_duration: 18m 42s (from analytics-worker)\n"
                    "  queries_waiting_for_lock: 12\n"
                    "  replication_lag:       0ms\n"
                    "CRITICAL: analytics-worker holds 78/100 connections."
                )
            elif service == "analytics-worker":
                reward = 0.06
                message = (
                    "analytics-worker metrics:\n"
                    "  active_db_connections: 78\n"
                    "  cpu_percent:           42.0%\n"
                    "  memory_percent:        70.2%\n"
                    "  rows_scanned_per_sec:  45000 (full table scan pattern)\n"
                    "  queries_timed_out:     0  ← no query timeout configured!\n"
                )
            else:
                reward = 0.02
                message = f"Metrics for {service}: Elevated error rates due to DB connection failures."

        elif action_type == "check_config":
            state["configs_checked"].append(service)
            if service == "analytics-worker":
                reward = 0.08
                message = (
                    "analytics-worker config (v1.0.9):\n"
                    "  db_connection_pool_size: 80\n"
                    "  query_timeout:           (not set)  ← MISSING in v1.0.9\n"
                    "  max_concurrent_queries:  (not set)\n"
                    "  report_schedule:         0 8 * * *\n\n"
                    "analytics-worker config (v1.0.8 — previous):\n"
                    "  db_connection_pool_size: 20\n"
                    "  query_timeout:           600  ← was 10 minutes\n"
                    "  max_concurrent_queries:  5\n"
                    "REGRESSION: v1.0.9 removed query_timeout and raised pool_size to 80."
                )
            elif service == "db-primary":
                reward = 0.04
                message = (
                    "db-primary config:\n"
                    "  max_connections: 100\n"
                    "  statement_timeout: (not set at server level)\n"
                    "  idle_in_transaction_session_timeout: 0 (disabled)\n"
                )
            else:
                reward = 0.01
                message = f"Config for {service}: No unusual settings."

        elif action_type == "kill_query":
            source = params.get("source", "").strip()
            if source == "analytics-worker":
                state["queries_killed"].append("analytics-worker")
                state["analytics_worker_killed"] = True
                state["_db_connections_freed"] = True
                # Update db-primary state
                state["services"]["db-primary"]["connections"] = 22
                state["services"]["db-primary"]["status"] = "healthy"
                state["services"]["db-primary"]["error_rate"] = 0.0
                # Update dependent services
                state["services"]["payment-api"]["status"] = "healthy"
                state["services"]["payment-api"]["error_rate"] = 0.1
                state["services"]["user-service"]["status"] = "healthy"
                state["services"]["user-service"]["error_rate"] = 0.1
                reward = 0.40
                message = (
                    "✓ Killed 78 long-running queries from analytics-worker.\n"
                    "  db-primary connections: 100 → 22\n"
                    "  db-primary status: degraded → healthy\n"
                    "  payment-api: recovering (error rate dropping)\n"
                    "  user-service: recovering (error rate dropping)\n"
                    "NOTE: analytics-worker may restart the runaway queries on next job run. "
                    "Consider also rolling back analytics-worker to v1.0.8."
                )
            else:
                state["wrong_actions"] += 1
                reward = -0.05
                message = (
                    f"No long-running queries found from '{source}'. "
                    "Check db metrics to identify the actual source of connection exhaustion."
                )

        elif action_type == "restart_service":
            if service == "db-primary":
                state["db_restarted"] = True
                state["wrong_actions"] += 1
                # Restarting DB causes brief outage for all — connections drop but analytics-worker
                # reconnects immediately and fills the pool again
                reward = -0.15
                message = (
                    "⚠ db-primary restarted — ALL services lost their connections.\n"
                    "  payment-api: connection errors spiking\n"
                    "  user-service: connection errors spiking\n"
                    "  analytics-worker: reconnected immediately, refilling pool with 78 queries\n"
                    "RESULT: Restart did not fix the root cause. analytics-worker filled the "
                    "pool again within 30 seconds. This approach is ineffective here."
                )
            elif service == "analytics-worker":
                # Partial fix — stops current queries but doesn't prevent recurrence
                state["queries_killed"].append("analytics-worker-restart")
                state["services"]["db-primary"]["connections"] = 22
                state["services"]["db-primary"]["status"] = "healthy"
                state["services"]["payment-api"]["status"] = "healthy"
                state["services"]["user-service"]["status"] = "healthy"
                reward = 0.20  # partial credit — works but is heavy-handed
                message = (
                    "analytics-worker restarted. Current runaway queries terminated.\n"
                    "  db-primary connections: 100 → 22 (analytics-worker queries cleared)\n"
                    "  payment-api, user-service: recovering\n"
                    "NOTE: This is a blunt fix. analytics-worker will restart its job and "
                    "may cause the same issue again without a config fix."
                )
            else:
                state["wrong_actions"] += 1
                reward = -0.08
                message = f"Restarting {service} does not affect the root cause."

        elif action_type == "rollback_deployment":
            if service == "analytics-worker":
                if not state["analytics_worker_killed"] and not any("analytics-worker" in k for k in state["queries_killed"]):
                    # Rollback also kills queries as part of restart
                    state["services"]["db-primary"]["connections"] = 22
                    state["services"]["db-primary"]["status"] = "healthy"
                    state["services"]["payment-api"]["status"] = "healthy"
                    state["services"]["user-service"]["status"] = "healthy"
                reward = 0.35
                message = (
                    "✓ analytics-worker rolled back to v1.0.8.\n"
                    "  query_timeout restored to 600s\n"
                    "  max_concurrent_queries restored to 5\n"
                    "  db_connection_pool_size reduced to 20\n"
                    "  db-primary connections: dropping to 22\n"
                    "This addresses the root cause AND prevents recurrence."
                )
            else:
                state["wrong_actions"] += 1
                reward = -0.05
                message = f"Rolling back {service} does not address the connection pool issue."

        elif action_type == "acknowledge_alert":
            alert_id = params.get("alert_id", "")
            for a in state["alerts"]:
                if a["id"] == alert_id:
                    a["ack"] = True
            reward = 0.01
            message = f"Alert {alert_id} acknowledged."

        elif action_type == "resolve_incident":
            db_ok = state["services"]["db-primary"]["connections"] < 80
            if db_ok:
                state["incident_resolved"] = True
                done = True
                reward = 0.25
                message = (
                    "✓ Incident resolved.\n"
                    "Summary: analytics-worker v1.0.9 introduced unbounded DB queries "
                    "(no query_timeout, large connection pool) causing pool exhaustion. "
                    "Remediated by killing runaway queries and/or rolling back analytics-worker."
                )
            else:
                reward = -0.05
                message = (
                    f"Cannot resolve: db-primary still at "
                    f"{state['services']['db-primary']['connections']}/100 connections. "
                    "Identify and fix the source of connection exhaustion first."
                )

        else:
            reward = -0.03
            message = f"Unknown or inapplicable action: {action_type}."

        return state, reward, done, message

    def get_observation(self, state: Dict[str, Any], session_id: str, step: int) -> Observation:
        services = {}
        for name, s in state["services"].items():
            services[name] = ServiceStatus(
                name=name, status=s["status"],
                cpu_percent=s["cpu"], memory_percent=s["memory"],
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
            available_actions=AVAILABLE_ACTIONS,
            incident_resolved=state["incident_resolved"],
            message="",
            recent_deployments=state["recent_deployments"],
            runbook_hints=[
                "check_metrics(db-primary) shows connections broken down by client.",
                "Restarting the database during connection exhaustion can worsen the situation.",
                "kill_query terminates long-running queries from a specific source application.",
                "Recent deployments are often correlated with sudden incidents.",
            ],
        )

    def grade(self, state: Dict[str, Any], history: List[Dict]) -> Tuple[float, Dict[str, float]]:
        breakdown = {}
        score = 0.0

        # Root cause identified? (checked db metrics OR db logs which reveal analytics-worker)
        root_cause_found = (
            "db-primary" in state.get("metrics_checked", []) or
            "db-primary" in state.get("logs_queried", []) or
            "analytics-worker" in state.get("logs_queried", []) or
            "analytics-worker" in state.get("configs_checked", [])
        )
        if root_cause_found:
            breakdown["root_cause_identified"] = 0.20
            score += 0.20

        # Correct attribution? (analytics-worker named as source)
        targeted_analytics = (
            state.get("analytics_worker_killed", False) or
            any("analytics-worker" in k for k in state.get("queries_killed", []))
        )
        if targeted_analytics:
            breakdown["correct_attribution"] = 0.30
            score += 0.30

        # DB recovered (connections < 80)?
        if state["services"]["db-primary"]["connections"] < 80:
            breakdown["db_recovered"] = 0.20
            score += 0.20

        # Incident formally resolved?
        if state.get("incident_resolved", False):
            breakdown["incident_resolved"] = 0.20
            score += 0.20

        # Efficiency bonus
        steps = len(history)
        if steps <= 5:
            breakdown["efficiency_bonus"] = 0.10
            score += 0.10
        elif steps <= 8:
            breakdown["efficiency_bonus"] = 0.07
            score += 0.07
        elif steps <= 12:
            breakdown["efficiency_bonus"] = 0.03
            score += 0.03

        # Penalty: restarted DB (bad practice)
        if state.get("db_restarted", False):
            breakdown["db_restart_penalty"] = -0.15
            score -= 0.15

        # Penalty: other wrong actions
        wrong = state.get("wrong_actions", 0) - (1 if state.get("db_restarted", False) else 0)
        if wrong > 0:
            p = min(wrong * 0.07, 0.15)
            breakdown["wrong_action_penalty"] = -p
            score -= p

        return round(min(max(score, 0.0), 1.0), 4), breakdown
