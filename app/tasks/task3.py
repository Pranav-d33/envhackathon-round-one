"""
Task 3: Cascading Service Failure (Hard)
=========================================
Scenario: Four production services are down simultaneously. A recent deployment
of config-service v1.2.0 introduced a broken service-discovery URL, causing all
services that depend on it to fail to resolve each other.

Traps:
  - user-service has an unrelated memory warning (red herring)
  - order-service v2.1.0 was deployed 2h ago (not the cause)
  - Restarting any individual service without fixing config-service makes no progress
  - Need to investigate and identify config-service as the common dependency

Optimal: check_config(config-service) OR query_logs(any-down-service) →
         rollback_deployment(config-service) → resolve_incident()
Max steps: 20  |  Passing score: 0.6
"""

from typing import Dict, Any, Tuple, List
from app.models import Observation, Alert, ServiceStatus
from app.tasks.base import BaseTask, AVAILABLE_ACTIONS, BASE_INCIDENT_TIME


class CascadingFailureTask(BaseTask):
    task_id = "task3"
    name = "Cascading Service Failure"
    description = (
        "Four services (api-gateway, user-service, order-service, payment-service) "
        "are simultaneously down. Identify the common root cause and remediate "
        "the entire incident with minimal blast radius."
    )
    difficulty = "hard"
    max_steps = 20
    passing_score = 0.6

    def initial_state(self, seed: int = 42) -> Dict[str, Any]:
        return {
            "services": {
                "api-gateway": {
                    "status": "down", "cpu": 0.5, "memory": 12.0,
                    "error_rate": 100.0, "version": "1.4.2", "replicas": 3,
                },
                "user-service": {
                    "status": "down", "cpu": 0.8, "memory": 88.0,
                    "error_rate": 100.0, "version": "4.0.1", "replicas": 2,
                },
                "order-service": {
                    "status": "down", "cpu": 0.3, "memory": 45.0,
                    "error_rate": 100.0, "version": "2.1.0", "replicas": 2,
                },
                "payment-service": {
                    "status": "down", "cpu": 0.2, "memory": 40.0,
                    "error_rate": 100.0, "version": "5.2.3", "replicas": 2,
                },
                "config-service": {
                    "status": "healthy", "cpu": 18.0, "memory": 35.0,
                    "error_rate": 0.0, "version": "1.2.0", "replicas": 1,
                },
                "db-primary": {
                    "status": "healthy", "cpu": 22.0, "memory": 55.0,
                    "error_rate": 0.0, "connections": 15, "max_connections": 100,
                    "version": "14.8",
                },
                "message-queue": {
                    "status": "healthy", "cpu": 10.0, "memory": 42.0,
                    "error_rate": 0.0, "version": "3.12.0",
                },
            },
            "alerts": [
                {"id": "ALT-020", "sev": "critical", "svc": "api-gateway",
                 "msg": "api-gateway is DOWN — all requests returning 503",
                 "ack": False},
                {"id": "ALT-021", "sev": "critical", "svc": "user-service",
                 "msg": "user-service is DOWN — health check failing for 8 minutes",
                 "ack": False},
                {"id": "ALT-022", "sev": "critical", "svc": "order-service",
                 "msg": "order-service is DOWN — all replicas unhealthy",
                 "ack": False},
                {"id": "ALT-023", "sev": "critical", "svc": "payment-service",
                 "msg": "payment-service is DOWN — cannot process transactions",
                 "ack": False},
                {"id": "ALT-024", "sev": "warning", "svc": "user-service",
                 "msg": "user-service memory at 88% (elevated but not critical)",
                 "ack": False},
            ],
            "recent_deployments": [
                {"service": "config-service", "version": "1.2.0", "previous": "1.1.9",
                 "deployed_at": "2024-11-15T09:30:00Z", "deployer": "platform-team",
                 "change": "Updated service discovery URLs for new datacenter migration"},
                {"service": "order-service", "version": "2.1.0", "previous": "2.0.8",
                 "deployed_at": "2024-11-15T07:45:00Z", "deployer": "ci-pipeline",
                 "change": "New checkout flow feature"},
                {"service": "user-service", "version": "4.0.1", "previous": "4.0.0",
                 "deployed_at": "2024-11-14T14:00:00Z", "deployer": "ci-pipeline",
                 "change": "Bug fix for profile update endpoint"},
            ],
            # Tracking agent progress
            "logs_queried": [],
            "metrics_checked": [],
            "configs_checked": [],
            "services_restarted": [],
            "rollbacks_attempted": {},
            "wrong_rollbacks": 0,
            "config_service_rolledback": False,
            "services_recovered": [],
            "incident_resolved": False,
            "_all_services_down_due_to_config": True,
        }

    def _check_recovery(self, state: Dict[str, Any]) -> None:
        """Update service statuses based on whether config-service was fixed."""
        if state["config_service_rolledback"]:
            for svc in ["api-gateway", "user-service", "order-service", "payment-service"]:
                if svc not in state["services_recovered"]:
                    state["services_recovered"].append(svc)
                state["services"][svc]["status"] = "healthy"
                state["services"][svc]["error_rate"] = 0.0
                state["services"][svc]["cpu"] = float(
                    {"api-gateway": 8.0, "user-service": 22.0,
                     "order-service": 15.0, "payment-service": 12.0}[svc]
                )

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

            down_services = ["api-gateway", "user-service", "order-service", "payment-service"]

            if service in down_services:
                reward = 0.08
                service_logs = {
                    "api-gateway": (
                        "2024-11-15T09:31:00Z [ERROR] api-gateway: failed to resolve "
                        "user-service endpoint via config-service: "
                        "GET http://config-service/discover/user-service → "
                        "returned 'http://svc-mesh-BROKEN.internal:8080' (unreachable)\n"
                        "2024-11-15T09:31:01Z [ERROR] api-gateway: failed to resolve "
                        "order-service — same issue\n"
                        "2024-11-15T09:31:05Z [FATAL] api-gateway: no healthy upstreams "
                        "available — entering 503 mode\n"
                        "PATTERN: All service discovery calls returning broken URLs from config-service."
                    ),
                    "user-service": (
                        "2024-11-15T09:31:00Z [ERROR] user-service: startup failed — cannot "
                        "resolve db endpoint via config-service: received 'db-BROKEN.internal' "
                        "(expected 'db-primary.internal')\n"
                        "2024-11-15T09:31:02Z [FATAL] user-service: health check failed — "
                        "cannot connect to database\n"
                        "PATTERN: config-service returning incorrect service discovery data."
                    ),
                    "order-service": (
                        "2024-11-15T09:31:00Z [ERROR] order-service: failed to start — "
                        "config-service returned broken payment-service URL\n"
                        "2024-11-15T09:31:03Z [FATAL] order-service: dependency check failed, "
                        "exiting\n"
                        "NOTE: order-service v2.1.0 deployed at 07:45 ran fine until 09:30 "
                        "when config-service was updated."
                    ),
                    "payment-service": (
                        "2024-11-15T09:31:00Z [ERROR] payment-service: cannot resolve fraud-check "
                        "service — config-service lookup returned null endpoint\n"
                        "2024-11-15T09:31:05Z [FATAL] payment-service: aborting startup due to "
                        "missing required service dependencies\n"
                    ),
                }
                message = service_logs.get(service, "No logs found.")

            elif service == "config-service":
                reward = 0.12
                message = (
                    "2024-11-15T09:28:00Z [INFO]  config-service: v1.2.0 deployment started\n"
                    "2024-11-15T09:29:50Z [INFO]  config-service: service discovery URLs updated "
                    "for datacenter migration\n"
                    "2024-11-15T09:30:00Z [INFO]  config-service: v1.2.0 deployment complete\n"
                    "2024-11-15T09:30:05Z [WARN]  config-service: 4 downstream services "
                    "reporting connection failures immediately after deploy\n"
                    "2024-11-15T09:30:10Z [ERROR] config-service: config validation failed "
                    "in post-deploy check — service_discovery_urls contain unreachable hosts\n"
                    "ROOT CAUSE CONFIRMED: config-service v1.2.0 deployed broken service "
                    "discovery URLs. All dependent services cannot resolve each other."
                )
            else:
                reward = 0.02
                message = f"No anomalies in logs for {service}."

        elif action_type == "check_metrics":
            if not service:
                return state, -0.02, False, "Parameter 'service' is required."
            if service in state["metrics_checked"]:
                return state, 0.0, False, f"[Cached] Metrics for {service} already retrieved."
            state["metrics_checked"].append(service)

            if service in ["api-gateway", "user-service", "order-service", "payment-service"]:
                reward = 0.06
                message = (
                    f"{service} metrics:\n"
                    f"  status: DOWN\n"
                    f"  error_rate: 100% (all requests failing)\n"
                    f"  last_healthy: 2024-11-15T09:30:02Z\n"
                    f"  restart_attempts: 3 (all failed)\n"
                    f"  failure_reason: dependency resolution failure at startup\n"
                    f"CORRELATES: All 4 services went down within 15 seconds of each other "
                    f"at 09:30 — timing matches config-service v1.2.0 deployment."
                )
            elif service == "config-service":
                reward = 0.08
                message = (
                    "config-service metrics:\n"
                    "  status: healthy\n"
                    "  cpu: 18%\n"
                    "  requests_per_sec: 240\n"
                    "  cache_hit_rate: 12%  ← very low (normal: 95%+)\n"
                    "  discovery_errors_per_sec: 180  ← HIGH\n"
                    "  version: 1.2.0  (deployed 17min ago)\n"
                    "SUSPICIOUS: High discovery_errors and low cache_hit_rate after recent deploy."
                )
            else:
                reward = 0.02
                message = f"Metrics for {service}: Normal."

        elif action_type == "check_config":
            state["configs_checked"].append(service)
            if service == "config-service":
                reward = 0.15
                message = (
                    "config-service LIVE CONFIG (v1.2.0):\n"
                    "  service_discovery:\n"
                    "    user-service:    http://svc-mesh-BROKEN.dc2.internal:8080\n"
                    "    order-service:   http://svc-mesh-BROKEN.dc2.internal:8081\n"
                    "    payment-service: http://svc-mesh-BROKEN.dc2.internal:8082\n"
                    "    db-primary:      http://db-BROKEN.dc2.internal:5432\n"
                    "    api-gateway:     http://gw-BROKEN.dc2.internal:80\n\n"
                    "config-service PREVIOUS CONFIG (v1.1.9):\n"
                    "  service_discovery:\n"
                    "    user-service:    http://user-service.svc.cluster.local:8080  ✓\n"
                    "    order-service:   http://order-service.svc.cluster.local:8081  ✓\n"
                    "    payment-service: http://payment-service.svc.cluster.local:8082 ✓\n"
                    "    db-primary:      http://db-primary.svc.cluster.local:5432  ✓\n\n"
                    "ROOT CAUSE CONFIRMED: v1.2.0 changed ALL service discovery URLs to "
                    "non-existent dc2.internal addresses. Datacenter migration was incomplete."
                )
            elif service in ["api-gateway", "user-service", "order-service", "payment-service"]:
                reward = 0.04
                message = (
                    f"{service} config appears normal. Service discovery endpoint "
                    f"points to config-service (as expected). The issue is in what "
                    f"config-service returns, not in {service}'s config itself."
                )
            else:
                reward = 0.01
                message = f"Config for {service}: Nothing unusual."

        elif action_type == "examine_trace":
            trace_id = params.get("trace_id", "unknown")
            state["logs_queried"].append(f"trace:{trace_id}")
            reward = 0.06
            message = (
                f"Trace {trace_id}:\n"
                "  api-gateway → [service discovery lookup] → config-service (2ms)\n"
                "  config-service → returned URL: http://svc-mesh-BROKEN.dc2.internal\n"
                "  api-gateway → [connection attempt to broken URL] → TIMEOUT after 5000ms\n"
                "  Root span: 100% of failures originate from bad service discovery response."
            )

        elif action_type == "restart_service":
            if service in ["api-gateway", "user-service", "order-service", "payment-service"]:
                if not state["config_service_rolledback"]:
                    state["services_restarted"].append(service)
                    # Restarting without fixing config does nothing
                    reward = -0.05
                    message = (
                        f"Restarted {service}... but it failed to start again.\n"
                        f"  Startup error: cannot resolve service dependencies via config-service\n"
                        f"  Status: still DOWN\n"
                        f"The underlying config-service issue must be fixed first."
                    )
                else:
                    # After config fix, manual restart not needed (auto-recovery)
                    reward = 0.0
                    message = f"{service} already recovering after config-service rollback."
            elif service == "config-service":
                # Restarting config-service doesn't fix the bad config
                state["services_restarted"].append(service)
                reward = -0.08
                message = (
                    "config-service restarted — but it loaded the same broken v1.2.0 config.\n"
                    "  All downstream services still failing.\n"
                    "  A restart does not fix a misconfiguration. Use rollback_deployment."
                )
            else:
                reward = 0.0
                message = f"{service} is healthy and does not need a restart."

        elif action_type == "rollback_deployment":
            if service == "config-service":
                state["config_service_rolledback"] = True
                self._check_recovery(state)
                reward = 0.45
                message = (
                    "✓ config-service rolled back from v1.2.0 → v1.1.9.\n"
                    "  Service discovery URLs restored to cluster-internal addresses.\n"
                    "  api-gateway:       DOWN → healthy (restarted automatically)\n"
                    "  user-service:      DOWN → healthy (restarted automatically)\n"
                    "  order-service:     DOWN → healthy (restarted automatically)\n"
                    "  payment-service:   DOWN → healthy (restarted automatically)\n"
                    "All 4 services recovered within 45 seconds of config-service rollback."
                )
            elif service == "order-service":
                # Red herring — order-service v2.1.0 was NOT the cause
                state["rollbacks_attempted"][service] = True
                state["wrong_rollbacks"] += 1
                reward = -0.08
                message = (
                    "Rolled back order-service to v2.0.8... but it immediately failed again.\n"
                    "  Error: still cannot resolve service dependencies via config-service.\n"
                    "  RESULT: order-service v2.1.0 was not the root cause. "
                    "The issue is upstream."
                )
            elif service in ["api-gateway", "user-service", "payment-service"]:
                state["rollbacks_attempted"][service] = True
                state["wrong_rollbacks"] += 1
                reward = -0.06
                message = (
                    f"Rolled back {service}... but it still cannot start.\n"
                    f"  Error: service discovery failing — same as before.\n"
                    f"  This service is not the root cause."
                )
            else:
                reward = -0.02
                message = f"Rolling back {service} has no effect on the current incident."

        elif action_type == "scale_service":
            reward = -0.05
            message = (
                "Scaling has no effect — services are failing due to misconfiguration, "
                "not insufficient capacity."
            )

        elif action_type == "acknowledge_alert":
            alert_id = params.get("alert_id", "")
            for a in state["alerts"]:
                if a["id"] == alert_id:
                    a["ack"] = True
            reward = 0.01
            message = f"Alert {alert_id} acknowledged."

        elif action_type == "resolve_incident":
            all_healthy = all(
                state["services"][svc]["status"] == "healthy"
                for svc in ["api-gateway", "user-service", "order-service", "payment-service"]
            )
            if all_healthy:
                state["incident_resolved"] = True
                done = True
                reward = 0.25
                message = (
                    "✓ Incident resolved.\n"
                    "Post-mortem: config-service v1.2.0 was deployed with incorrect service "
                    "discovery URLs targeting a non-existent dc2 datacenter. This caused all "
                    "dependent services to fail at startup. Rollback to v1.1.9 restored service.\n"
                    "Recommendation: Add config validation to deployment pipeline."
                )
            else:
                still_down = [
                    s for s in ["api-gateway", "user-service", "order-service", "payment-service"]
                    if state["services"][s]["status"] != "healthy"
                ]
                reward = -0.05
                message = (
                    f"Cannot resolve: {len(still_down)} services still down: "
                    f"{', '.join(still_down)}. Fix the root cause first."
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
                "When multiple services fail simultaneously, look for a common dependency.",
                "Check the timing: what changed just before the incident?",
                "check_config reveals live runtime configuration values.",
                "Restarting services without fixing the root cause will not help.",
                "rollback_deployment reverts to the previous known-good version.",
            ],
        )

    def grade(self, state: Dict[str, Any], history: List[Dict]) -> Tuple[float, Dict[str, float]]:
        breakdown = {}
        score = 0.0

        # Root cause investigated?
        root_investigated = (
            "config-service" in state.get("configs_checked", []) or
            "config-service" in state.get("logs_queried", []) or
            "config-service" in state.get("metrics_checked", []) or
            any(svc in state.get("logs_queried", [])
                for svc in ["api-gateway", "user-service", "order-service", "payment-service"])
        )
        if root_investigated:
            breakdown["investigated_root_cause"] = 0.14
            score += 0.14

        # config-service identified and rolled back?
        if state.get("config_service_rolledback", False):
            breakdown["correct_rollback"] = 0.40
            score += 0.40

        # All services recovered?
        recovered = state.get("services_recovered", [])
        if len(recovered) >= 4:
            breakdown["full_recovery"] = 0.20
            score += 0.20
        elif len(recovered) >= 2:
            breakdown["partial_recovery"] = 0.10
            score += 0.10

        # Incident formally resolved?
        if state.get("incident_resolved", False):
            breakdown["incident_resolved"] = 0.15
            score += 0.15

        # Efficiency bonus
        steps = len(history)
        if steps <= 4:
            breakdown["efficiency_bonus"] = 0.10
            score += 0.10
        elif steps <= 7:
            breakdown["efficiency_bonus"] = 0.07
            score += 0.07
        elif steps <= 12:
            breakdown["efficiency_bonus"] = 0.03
            score += 0.03

        # Penalty for wrong rollbacks
        wrong_rollbacks = state.get("wrong_rollbacks", 0)
        if wrong_rollbacks > 0:
            p = min(wrong_rollbacks * 0.08, 0.20)
            breakdown["wrong_rollback_penalty"] = -p
            score -= p

        score = round(min(max(score, 0.0), 1.0), 4)
        return self.clamp_score_strict(score), breakdown
