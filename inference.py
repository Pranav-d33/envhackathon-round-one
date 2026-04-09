#!/usr/bin/env python3
"""
Inference Script — SRE Incident Response OpenEnv
=================================================
Runs a ReAct-style OpenAI agent against all three tasks and
reports reproducible baseline scores.

Usage:
    export OPENAI_API_KEY="sk-..."
    export OPENENV_BASE_URL="http://localhost:7860"  # or your HF Space URL
    python inference.py

    # Run specific tasks:
    python inference.py --tasks task1 task2

    # Use a different model:
    python inference.py --model gpt-4o

Requirements:
    pip install openai httpx rich
"""

import os
import sys
import json
import re
import argparse
import time
import subprocess
import signal
from typing import Optional

import httpx

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    RICH = True
except ImportError:
    RICH = False
    Console = None


# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _supports_unicode_stdout() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or ""
    try:
        "✓⚠ℹ✗═─→•".encode(enc or "utf-8")
        return True
    except Exception:
        return False


UNICODE_OK = _supports_unicode_stdout()
HR_THICK = "═" if UNICODE_OK else "="
HR_THIN = "─" if UNICODE_OK else "-"
ARROW = "→" if UNICODE_OK else "->"


def safe_print(s: str = "", **kwargs):
    """
    Print without crashing on Windows codepages that can't encode Unicode.
    """
    try:
        print(s, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        s2 = s.encode(enc, errors="replace").decode(enc, errors="replace")
        print(s2, **kwargs)


def log(msg: str, level: str = "INFO"):
    if UNICODE_OK:
        prefix = {"INFO": "ℹ", "OK": "✓", "WARN": "⚠", "ERR": "✗"}.get(level, "•")
    else:
        prefix = {"INFO": "i", "OK": "+", "WARN": "!", "ERR": "x"}.get(level, "-")
    safe_print(f"  {prefix} {msg}")


def format_observation(obs: dict) -> str:
    """Format observation dict into a concise prompt string."""
    dash = "—" if UNICODE_OK else "-"
    lines = [f"=== INCIDENT {dash} Step {obs.get('step', 0)} ===\n"]

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
            arrow = "→" if UNICODE_OK else "->"
            lines.append(
                f"  {dep.get('service')}: v{dep.get('previous', '?')} {arrow} "
                f"v{dep.get('version')} deployed at {dep.get('deployed_at')}"
            )

    if obs.get("message"):
        lines.append(f"\nLAST ACTION RESULT:\n{obs['message']}")

    if obs.get("runbook_hints"):
        lines.append("\nRUNBOOK HINTS:")
        for h in obs["runbook_hints"]:
            lines.append(f"  • {h}")

    return "\n".join(lines)


def call_llm(client: httpx.Client, model: str, messages: list) -> str:
    """Call OpenAI chat completions API."""
    try:
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
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
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except httpx.RequestError as e:
        raise Exception(f"Network error calling OpenAI API: {e}")
    except httpx.HTTPStatusError as e:
        raise Exception(f"OpenAI API error (status {e.response.status_code}): {e.response.text}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected response format from OpenAI API: {e}")


def _is_localhost_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return u.startswith("http://localhost") or u.startswith("http://127.0.0.1")


def _wait_for_health(base_url: str, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with httpx.Client(base_url=base_url, timeout=2.5) as c:
                r = c.get("/health")
                r.raise_for_status()
                return True
        except Exception as e:
            last_err = e
            time.sleep(0.4)
    if last_err:
        log(f"Health check still failing: {last_err}", "WARN")
    return False


def _start_local_server() -> subprocess.Popen:
    """
    Start the environment server in a subprocess.
    Intended for runners that execute inference without already running the env.
    """
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "7860",
        "--workers",
        "1",
    ]
    kwargs = {}
    if os.name == "nt":
        # Avoid CTRL-C propagation weirdness on Windows runners.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)


def _stop_local_server(p: subprocess.Popen):
    try:
        if p.poll() is not None:
            return
        if os.name == "nt":
            p.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            try:
                p.wait(timeout=5)
                return
            except Exception:
                pass
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    except Exception:
        pass


def parse_action(text: str) -> dict:
    """Parse JSON action from LLM output, with fallback."""
    text = text.strip()
    # Remove markdown code blocks if present
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
    # Fallback: safe no-op
    return {"action_type": "acknowledge_alert", "parameters": {"alert_id": "ALT-001"}}


def _pick_target_service(obs: dict) -> Optional[str]:
    services = obs.get("services") or {}
    if not isinstance(services, dict) or not services:
        return None

    def score_service(item):
        _, svc = item
        try:
            err = float(svc.get("error_rate") or 0.0)
        except Exception:
            err = 0.0
        status = str(svc.get("status") or "").lower()
        bad = 1.0 if status not in ("healthy", "ok", "passing") else 0.0
        return (bad, err)

    return max(services.items(), key=score_service)[0]


def fallback_policy(obs: dict, step_num: int) -> dict:
    """
    Deterministic, no-network fallback agent.
    This is intentionally conservative: gather evidence, prefer rollback on recent deploys,
    and only resolve when things look healthy.
    """
    alerts = obs.get("alerts") or []
    if isinstance(alerts, list):
        for a in alerts:
            if isinstance(a, dict) and a.get("acknowledged") is False and a.get("alert_id"):
                return {"action_type": "acknowledge_alert", "parameters": {"alert_id": a["alert_id"]}}

    # If there's a recent deployment on a sick service, prefer rollback early.
    recent = obs.get("recent_deployments") or []
    if isinstance(recent, list) and recent:
        target = _pick_target_service(obs)
        for dep in recent:
            if not isinstance(dep, dict):
                continue
            svc = dep.get("service")
            if svc and (target is None or svc == target):
                return {"action_type": "rollback_deployment", "parameters": {"service": svc}}

    target = _pick_target_service(obs) or "api"

    # Alternate between logs/metrics/config early to build context.
    if step_num % 3 == 0:
        return {"action_type": "query_logs", "parameters": {"service": target}}
    if step_num % 3 == 1:
        return {"action_type": "check_metrics", "parameters": {"service": target}}
    return {"action_type": "check_config", "parameters": {"service": target}}


# ─── Core Runner ─────────────────────────────────────────────────────────────

def run_task(
    env_client: httpx.Client,
    llm_client: httpx.Client,
    task_id: str,
    model: str,
    max_steps: int,
    verbose: bool = True,
) -> dict:
    """Run one complete episode for a task. Returns result dict."""

    if verbose:
        safe_print(f"\n{HR_THIN*60}")
        safe_print(f"  Task: {task_id.upper()}")
        safe_print(f"{HR_THIN*60}")

    episode_log = []
    score = 0.0
    steps_taken = 0
    session_id = None

    try:
        # ── Reset ────────────────────────────────────────────────────
        reset_resp = env_client.post("/reset", json={"task_id": task_id, "seed": 42})
        reset_resp.raise_for_status()
        obs = reset_resp.json()
        session_id = obs["session_id"]

        if verbose:
            task_name = obs.get("message", "").split("Task:")[1].split("(")[0].strip() \
                if "Task:" in obs.get("message", "") else task_id
            log(f"Session: {session_id[:8]}...", "INFO")
            log(obs.get("message", ""), "INFO")

        conversation = []
        done = False

        # ── Episode Loop ─────────────────────────────────────────────
        for step_num in range(max_steps):
            obs_text = format_observation(obs)
            conversation.append({"role": "user", "content": obs_text})

            # Trim conversation to last 4 turns (keep it focused)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages += conversation[-8:]

            # Get action from LLM (or fallback policy if no API key)
            if OPENAI_API_KEY:
                action_text = call_llm(llm_client, model, messages)
                conversation.append({"role": "assistant", "content": action_text})
                action_dict = parse_action(action_text)
            else:
                action_dict = fallback_policy(obs, step_num)
                conversation.append({"role": "assistant", "content": json.dumps(action_dict)})
            action_type = action_dict.get("action_type", "unknown")
            parameters = action_dict.get("parameters", {})

            if verbose:
                params_str = json.dumps(parameters) if parameters else "{}"
                safe_print(f"  Step {step_num+1:2d}: {action_type}({params_str})", end="")

            # Take step
            step_resp = env_client.post("/step", json={
                "session_id": session_id,
                "action": {"action_type": action_type, "parameters": parameters},
            })
            step_resp.raise_for_status()
            step_data = step_resp.json()

            obs = step_data["observation"]
            reward_val = step_data["reward"]["value"]
            done = step_data["done"]
            steps_taken = step_num + 1

            if verbose:
                reward_str = f"{reward_val:+.3f}"
                current_score = step_data["info"].get("grader_score", 0.0)
                safe_print(f" {ARROW} reward: {reward_str} | score: {current_score:.3f}")

            episode_log.append({
                "step": step_num + 1,
                "action_type": action_type,
                "parameters": parameters,
                "reward": reward_val,
                "message_preview": obs.get("message", "")[:150],
            })

            if done:
                break

        # ── Get Final Grade ───────────────────────────────────────────
        grader_resp = env_client.post("/grader", json={"session_id": session_id})
        grader_resp.raise_for_status()
        grader_data = grader_resp.json()
        score = grader_data["score"]
        breakdown = grader_data.get("breakdown", {})

        if verbose:
            safe_print(f"\n  {HR_THIN*30}")
            log(f"Final score: {score:.4f}", "OK" if score >= 0.6 else "WARN")
            log(f"Steps taken: {steps_taken}", "INFO")
            if breakdown:
                log("Breakdown:", "INFO")
                for k, v in breakdown.items():
                    safe_print(f"      {k}: {v:+.4f}")

    except httpx.RequestError as e:
        error_msg = f"Network error communicating with environment: {e}"
        if verbose:
            log(error_msg, "ERR")
        episode_log.append({"error": error_msg})
    except httpx.HTTPStatusError as e:
        error_msg = f"Environment API error (status {e.response.status_code}): {e.response.text}"
        if verbose:
            log(error_msg, "ERR")
        episode_log.append({"error": error_msg})
    except (KeyError, ValueError, TypeError) as e:
        error_msg = f"Unexpected response format from environment: {e}"
        if verbose:
            log(error_msg, "ERR")
        episode_log.append({"error": error_msg})
    except Exception as e:
        if verbose:
            log(f"Error: {e}", "ERR")
        episode_log.append({"error": str(e)})

    # Get task info for name/difficulty
    try:
        tasks_resp = env_client.get("/tasks")
        tasks_resp.raise_for_status()
        task_info = {}
        for t in tasks_resp.json().get("tasks", []):
            if t["task_id"] == task_id:
                task_info = t
                break
    except Exception as e:
        # If we can't get task info, use defaults
        task_info = {}
        if verbose:
            log(f"Warning: Could not retrieve task info: {e}", "WARN")

    return {
        "task_id": task_id,
        "task_name": task_info.get("name", task_id),
        "difficulty": task_info.get("difficulty", "?"),
        "score": score,
        "steps_taken": steps_taken,
        "success": score >= task_info.get("passing_score", 0.6),
        "episode_log": episode_log,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run inference agent against SRE Incident Response environment"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Environment base URL")
    parser.add_argument("--max-steps", type=int, default=12, help="Max steps per episode")
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"],
                        help="Tasks to run (task1, task2, task3)")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit non-zero when not all tasks pass (default: exit 0 if script completes).",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        safe_print("WARN: OPENAI_API_KEY not set; using deterministic fallback policy (no OpenAI calls).")

    dash = "—" if UNICODE_OK else "-"
    safe_print(f"\n{HR_THICK*60}")
    safe_print(f"  SRE Incident Response {dash} Inference Evaluation")
    safe_print(f"{HR_THICK*60}")
    safe_print(f"  Model:    {args.model}")
    safe_print(f"  Env URL:  {args.base_url}")
    safe_print(f"  Tasks:    {', '.join(args.tasks)}")
    safe_print(f"  MaxSteps: {args.max_steps}")
    safe_print(f"{HR_THICK*60}")

    results = []
    start = time.time()

    server_proc: Optional[subprocess.Popen] = None
    try:
        # Verify environment is reachable; auto-start local server if needed.
        if not _wait_for_health(args.base_url, timeout_s=3.0) and _is_localhost_url(args.base_url):
            log("Environment not reachable; starting local server...", "WARN")
            server_proc = _start_local_server()

        if not _wait_for_health(args.base_url, timeout_s=20.0):
            x = "✗" if UNICODE_OK else "x"
            safe_print(f"\n  {x} Environment not reachable at {args.base_url}")
            sys.exit(1)

        with httpx.Client(base_url=args.base_url, timeout=30.0) as env_client:
            health = env_client.get("/health")
            health.raise_for_status()
            ok = "✓" if UNICODE_OK else "+"
            safe_print(f"\n  {ok} Environment healthy: {health.json()}")

            with httpx.Client(timeout=60.0) as llm_client:
                for task_id in args.tasks:
                    result = run_task(
                        env_client=env_client,
                        llm_client=llm_client,
                        task_id=task_id,
                        model=args.model,
                        max_steps=args.max_steps,
                        verbose=not args.quiet,
                    )
                    results.append(result)
                    time.sleep(0.5)  # Rate limiting courtesy
    finally:
        if server_proc is not None:
            _stop_local_server(server_proc)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    mean_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    passed = sum(1 for r in results if r["success"])

    safe_print(f"\n{HR_THICK*60}")
    safe_print("  INFERENCE RESULTS SUMMARY")
    safe_print(f"{HR_THICK*60}")
    safe_print(f"  {'Task':<35} {'Diff':<8} {'Score':<8} {'Steps':<7} {'Status'}")
    safe_print(f"  {HR_THIN*55}")
    for r in results:
        status = ("✓ PASS" if UNICODE_OK else "+ PASS") if r["success"] else ("✗ FAIL" if UNICODE_OK else "x FAIL")
        safe_print(
            f"  {r['task_name']:<35} {r['difficulty']:<8} "
            f"{r['score']:.4f}  {r['steps_taken']:<7} {status}"
        )
    safe_print(f"  {HR_THIN*55}")
    safe_print(f"  {'Mean Score':<35} {'':8} {mean_score:.4f}")
    safe_print(f"  Tasks passed: {passed}/{len(results)}")
    safe_print(f"  Elapsed: {elapsed:.1f}s")
    safe_print(f"{HR_THICK*60}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "model": args.model,
        "environment": "sre-incident-response",
        "results": results,
        "summary": {
            "mean_score": round(mean_score, 4),
            "tasks_passed": passed,
            "total_tasks": len(results),
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    try:
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            safe_print(f"  Results saved to {args.output}")
        else:
            # Always save a inference_results.json for reproducibility
            with open("inference_results.json", "w") as f:
                json.dump(output, f, indent=2)
            safe_print("  Results saved to inference_results.json")
    except Exception as e:
        safe_print(f"  WARN: Could not write results file: {e}")

    # Exit code:
    # - default: 0 if script ran to completion (so runners don't treat "failed tasks" as a crash)
    # - strict: 0 only if all tasks pass
    if args.strict_exit:
        sys.exit(0 if passed == len(results) else 1)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        safe_print("\nInterrupted.")
        raise
    except Exception as e:
        safe_print(f"FATAL: inference.py crashed: {e}")
        sys.exit(2)