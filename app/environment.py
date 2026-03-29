"""
Session manager for the SRE Incident Response environment.
Manages per-session state, step processing, and reward accumulation.
"""

import uuid
import copy
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

from app.models import (
    Observation, Action, Reward, StepResponse,
    StateResponse, GraderResponse,
)
from app.tasks.base import AVAILABLE_ACTIONS
from app.tasks import TASK_REGISTRY


class Session:
    """In-memory session state for one episode."""

    def __init__(self, task_id: str, seed: int = 42):
        self.session_id = str(uuid.uuid4())
        self.task_id = task_id
        self.step = 0
        self.done = False
        self.total_reward = 0.0
        self.action_history: list = []
        self.created_at = datetime.now(timezone.utc).isoformat()

        task = TASK_REGISTRY[task_id]
        self.world_state = task.initial_state(seed=seed)

    def to_state_response(self, grader_score: Optional[float] = None) -> StateResponse:
        return StateResponse(
            session_id=self.session_id,
            task_id=self.task_id,
            step=self.step,
            done=self.done,
            total_reward=round(self.total_reward, 4),
            world_state=_serialize_state(self.world_state),
            action_history=self.action_history,
            grader_score=grader_score,
        )


def _serialize_state(state: dict) -> dict:
    """Make world_state JSON-serializable (convert sets, etc.)."""
    result = {}
    for k, v in state.items():
        if k.startswith("_"):
            continue  # hide internal fields
        if isinstance(v, set):
            result[k] = list(v)
        elif isinstance(v, dict):
            result[k] = _serialize_state(v)
        else:
            result[k] = v
    return result


class EnvironmentManager:
    """Manages all active sessions."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def reset(self, task_id: str, seed: int = 42) -> Tuple[Observation, str]:
        """Start a new episode. Returns (initial_observation, session_id)."""
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY.keys())}")

        session = Session(task_id=task_id, seed=seed)
        self._sessions[session.session_id] = session

        task = TASK_REGISTRY[task_id]
        obs = task.get_observation(session.world_state, session.session_id, step=0)
        obs.message = (
            f"New episode started. Task: {task.name} (difficulty: {task.difficulty}). "
            f"Max steps: {task.max_steps}. Investigate the incident and resolve it."
        )
        return obs, session.session_id

    def step(self, session_id: str, action: Action) -> StepResponse:
        """Process one action and return (observation, reward, done, info)."""
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found. Call /reset first.")

        session = self._sessions[session_id]

        if session.done:
            task = TASK_REGISTRY[session.task_id]
            obs = task.get_observation(session.world_state, session_id, session.step)
            obs.message = "Episode already complete. Call /reset to start a new episode."
            return StepResponse(
                observation=obs,
                reward=Reward(value=0.0, cumulative=session.total_reward, breakdown={},
                              message="Episode already complete."),
                done=True,
                info={"episode_complete": True},
            )

        task = TASK_REGISTRY[session.task_id]
        session.step += 1

        # Check max steps
        if session.step > task.max_steps:
            session.done = True
            obs = task.get_observation(session.world_state, session_id, session.step)
            obs.message = f"Episode ended: max steps ({task.max_steps}) reached without resolution."
            timeout_reward = -0.10
            session.total_reward += timeout_reward
            return StepResponse(
                observation=obs,
                reward=Reward(
                    value=timeout_reward,
                    cumulative=round(session.total_reward, 4),
                    breakdown={"timeout_penalty": timeout_reward},
                    message="Max steps reached.",
                ),
                done=True,
                info={"timeout": True, "steps": session.step},
            )

        # Process action
        new_state, step_reward, done, message = task.process_action(
            action.action_type,
            action.parameters,
            session.world_state,
        )
        session.world_state = new_state
        session.done = done
        session.total_reward += step_reward

        # Record history
        session.action_history.append({
            "step": session.step,
            "action_type": action.action_type,
            "parameters": action.parameters,
            "reward": round(step_reward, 4),
            "message_preview": message[:120] if message else "",
        })

        # Build observation
        obs = task.get_observation(session.world_state, session_id, session.step)
        obs.message = message

        # Grade for info
        score, grade_breakdown = task.grade(session.world_state, session.action_history)

        reward_obj = Reward(
            value=round(step_reward, 4),
            cumulative=round(session.total_reward, 4),
            breakdown=_build_reward_breakdown(action.action_type, step_reward),
            message=f"Step reward: {step_reward:+.3f} | Cumulative: {session.total_reward:+.3f}",
        )

        return StepResponse(
            observation=obs,
            reward=reward_obj,
            done=done,
            info={
                "step": session.step,
                "max_steps": task.max_steps,
                "grader_score": score,
                "episode_complete": done,
            },
        )

    def get_state(self, session_id: str) -> StateResponse:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found.")
        session = self._sessions[session_id]
        task = TASK_REGISTRY[session.task_id]
        score, _ = task.grade(session.world_state, session.action_history)
        return session.to_state_response(grader_score=score)

    def grade(self, session_id: str) -> GraderResponse:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found.")
        session = self._sessions[session_id]
        task = TASK_REGISTRY[session.task_id]
        score, breakdown = task.grade(session.world_state, session.action_history)

        passing = score >= task.passing_score
        return GraderResponse(
            session_id=session_id,
            task_id=session.task_id,
            score=score,
            breakdown=breakdown,
            episode_complete=session.done,
            steps_taken=session.step,
            message=(
                f"Score: {score:.4f} | "
                f"{'PASS' if passing else 'FAIL'} "
                f"(threshold: {task.passing_score}) | "
                f"Steps: {session.step}/{task.max_steps}"
            ),
        )

    def cleanup_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def active_sessions(self) -> int:
        return len(self._sessions)


def _build_reward_breakdown(action_type: str, value: float) -> Dict[str, float]:
    if value > 0:
        return {f"{action_type}_reward": round(value, 4)}
    elif value < 0:
        return {f"{action_type}_penalty": round(value, 4)}
    return {}
