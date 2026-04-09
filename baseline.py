#!/usr/bin/env python3
"""
Baseline Inference Script — SRE Incident Response OpenEnv
==========================================================
Runs a ReAct-style OpenAI agent against all three tasks and
reports reproducible baseline scores.

Usage:
    export OPENAI_API_KEY="sk-..."
    export OPENENV_BASE_URL="http://localhost:7860"  # or your HF Space URL
    python baseline.py

    # Run specific tasks:
    python baseline.py --tasks task1 task2

    # Use a different model:
    python baseline.py --model gpt-4o

Requirements:
    pip install openai httpx rich
"""

import os
import sys
import json
import re
import argparse
import time
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

def clamp_score_strict(score: float, eps: float = 1e-3) -> float:
    """
    Hackathon validator requirement: scores must be strictly within (0, 1).
    Clamp away from endpoints to avoid returning exactly 0.0 or 1.0.
    """
    try:
        s = float(score)
    except Exception:
        s = 0.0
    if s <= 0.0:
        return eps
    if s >= 1.0:
        return 1.0 - eps
    return s


def log(msg: str, level: str = "INFO"):
    prefix = {"INFO": "ℹ", "OK": "✓", "WARN": "⚠", "ERR": "✗"}.get(level, "•")
    print(f"  {prefix} {msg}")


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


def call_llm(client: httpx.Client, model: str, messages: list) -> str:
    """Call OpenAI chat completions API."""
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
        print(f"\n{'─'*60}")
        print(f"  Task: {task_id.upper()}")
        print(f"{'─'*60}")

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

            # Get action from LLM
            action_text = call_llm(llm_client, model, messages)
            conversation.append({"role": "assistant", "content": action_text})

            action_dict = parse_action(action_text)
            action_type = action_dict.get("action_type", "unknown")
            parameters = action_dict.get("parameters", {})

            if verbose:
                params_str = json.dumps(parameters) if parameters else "{}"
                print(f"  Step {step_num+1:2d}: {action_type}({params_str})", end="")

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
                print(f" → reward: {reward_str} | score: {current_score:.3f}")

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
            print(f"\n  {'─'*30}")
            log(f"Final score: {score:.4f}", "OK" if score >= 0.6 else "WARN")
            log(f"Steps taken: {steps_taken}", "INFO")
            if breakdown:
                log("Breakdown:", "INFO")
                for k, v in breakdown.items():
                    print(f"      {k}: {v:+.4f}")

    except Exception as e:
        if verbose:
            log(f"Error: {e}", "ERR")
        episode_log.append({"error": str(e)})

    # Get task info for name/difficulty
    tasks_resp = env_client.get("/tasks")
    task_info = {}
    if tasks_resp.status_code == 200:
        for t in tasks_resp.json().get("tasks", []):
            if t["task_id"] == task_id:
                task_info = t
                break

    final_score = clamp_score_strict(score)
    return {
        "task_id": task_id,
        "task_name": task_info.get("name", task_id),
        "difficulty": task_info.get("difficulty", "?"),
        "score": final_score,
        "steps_taken": steps_taken,
        "success": final_score >= task_info.get("passing_score", 0.6),
        "episode_log": episode_log,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline agent against SRE Incident Response environment"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Environment base URL")
    parser.add_argument("--max-steps", type=int, default=12, help="Max steps per episode")
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"],
                        help="Tasks to run (task1, task2, task3)")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  SRE Incident Response — Baseline Evaluation")
    print(f"{'═'*60}")
    print(f"  Model:    {args.model}")
    print(f"  Env URL:  {args.base_url}")
    print(f"  Tasks:    {', '.join(args.tasks)}")
    print(f"  MaxSteps: {args.max_steps}")
    print(f"{'═'*60}")

    # Verify environment is reachable
    with httpx.Client(base_url=args.base_url, timeout=30.0) as env_client:
        try:
            health = env_client.get("/health")
            health.raise_for_status()
            print(f"\n  ✓ Environment healthy: {health.json()}")
        except Exception as e:
            print(f"\n  ✗ Environment not reachable at {args.base_url}: {e}")
            sys.exit(1)

        results = []
        start = time.time()

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

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    mean_score = (
        sum(r["score"] for r in results) / len(results)
        if results
        else clamp_score_strict(0.0)
    )
    passed = sum(1 for r in results if r["success"])

    print(f"\n{'═'*60}")
    print(f"  BASELINE RESULTS SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Task':<35} {'Diff':<8} {'Score':<8} {'Steps':<7} {'Status'}")
    print(f"  {'─'*55}")
    for r in results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(
            f"  {r['task_name']:<35} {r['difficulty']:<8} "
            f"{r['score']:.4f}  {r['steps_taken']:<7} {status}"
        )
    print(f"  {'─'*55}")
    print(f"  {'Mean Score':<35} {'':8} {mean_score:.4f}")
    print(f"  Tasks passed: {passed}/{len(results)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'═'*60}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "model": args.model,
        "environment": "sre-incident-response",
        "results": results,
        "summary": {
            # Avoid rounding to 0.0/1.0; validator requires strict (0,1).
            "mean_score": clamp_score_strict(mean_score),
            "tasks_passed": passed,
            "total_tasks": len(results),
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to {args.output}")
    else:
        # Always save a baseline_results.json for reproducibility
        with open("baseline_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to baseline_results.json")

    # Exit code: 0 if all tasks pass, 1 otherwise
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
