# server/firewatch_env_environment.py
# Phase 2 — Updated imports to use ServiceMetrics (replaces ServiceSnapshot).
# Three endpoint methods with hardcoded placeholder responses.
# Zero simulation logic. Full implementation added in Phase 7.
#
# Base class and import paths confirmed from official OpenEnv builder docs:
# https://meta-pytorch.org/OpenEnv/environment-builder/
#
# IMPORTANT: The dual-import pattern below is REQUIRED by OpenEnv.
# - Relative import (..models) works when running in-repo via PYTHONPATH=src:envs
# - Bare import (models) works when running in Docker via PYTHONPATH=/app/env
# Both paths must be present or the server will fail in one of the two contexts.

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Dual-import pattern — required for both in-repo and Docker execution
try:
    from ..models import FirewatchAction, SystemObservation, ServiceMetrics
except ImportError:
    from models import FirewatchAction, SystemObservation, ServiceMetrics


class FirewatchEnvironment(Environment):
    """
    SRE Incident Response RL Environment — Phase 2 stub.

    Simulates a microservice production system where an AI agent acts as
    an on-call SRE engineer, diagnosing and remediating incidents before
    the SLO error budget is exhausted.

    This stub returns hardcoded placeholder responses to pass openenv validate
    and confirm the server wires correctly. All three methods wrap their logic
    in try/except to guarantee the Space never returns a 500.
    """

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # reset() — initialise a new episode
    # ------------------------------------------------------------------

    def reset(self, difficulty: str = "easy", seed: int | None = None) -> SystemObservation:
        """
        Start a new incident episode.

        Args:
            difficulty: One of "easy", "medium", "hard".
            seed: Optional integer seed for reproducible episodes.
                  Same seed + difficulty always produces the same episode.

        Returns:
            SystemObservation with initial system state (all services healthy).
        """
        try:
            self._state = State(episode_id=str(uuid4()), step_count=0)

            # Phase 2 stub — hardcoded placeholder observation.
            # Phase 7 replaces this with generate_episode(difficulty, seed).
            return SystemObservation(
                services={
                    "auth-service": ServiceMetrics(
                        service_name="auth-service",
                        service_instance_id="auth-7d9f8b-xkp2m",
                        status="healthy",
                        http_server_error_rate=0.0,
                        http_server_request_duration_p99=0.12,
                        process_memory_utilization=0.35,
                        process_cpu_utilization=0.20,
                        restart_count=0,
                        recent_logs=[],
                    )
                },
                active_alerts=[],
                dependency_graph={"auth-service": []},
                slo_budget_remaining_pct=100.0,
                bad_customer_minutes=0.0,
                sim_time_elapsed_seconds=0,
                sim_tick=0,
                action_history=[],
                incident_declared=False,
                mttm_achieved_tick=None,
            )

        except Exception as exc:
            # Zero-crash policy — never let an exception propagate to HTTP layer.
            return SystemObservation(
                services={},
                active_alerts=[],
                dependency_graph={},
                slo_budget_remaining_pct=100.0,
                bad_customer_minutes=0.0,
                sim_time_elapsed_seconds=0,
                sim_tick=0,
                action_history=[{"action_type": "reset", "target_service": "", "feedback_string": f"reset error: {exc}"}],
                incident_declared=False,
                mttm_achieved_tick=None,
            )

    # ------------------------------------------------------------------
    # step() — execute one agent action
    # ------------------------------------------------------------------

    def step(self, action: FirewatchAction) -> SystemObservation:
        """
        Execute one agent action and advance the simulation by one tick.

        The simulation degrades autonomously each tick BEFORE the agent
        action is applied — time pressure is real, not cosmetic.

        Args:
            action: A FirewatchAction specifying what the agent wants to do.

        Returns:
            Updated SystemObservation after the tick and action.
            reward, done, and info are added by the app.py wrapper.
        """
        try:
            self._state = State(
                episode_id=self._state.episode_id,
                step_count=self._state.step_count + 1,
            )

            # Phase 2 stub — return placeholder observation.
            # Phase 7 replaces with full tick() + action handling + reward.
            return SystemObservation(
                services={},
                active_alerts=[],
                dependency_graph={},
                slo_budget_remaining_pct=95.0,
                bad_customer_minutes=0.5,
                sim_time_elapsed_seconds=30,
                sim_tick=self._state.step_count,
                action_history=[
                    {
                        "action_type": action.action_type,
                        "target_service": action.target_service or "",
                        "feedback_string": f"stub: {action.action_type} on {action.target_service}",
                    }
                ],
                incident_declared=action.action_type == "declare_resolved",
                mttm_achieved_tick=None,
            )

        except Exception as exc:
            return SystemObservation(
                services={},
                active_alerts=[],
                dependency_graph={},
                slo_budget_remaining_pct=0.0,
                bad_customer_minutes=0.0,
                sim_time_elapsed_seconds=0,
                sim_tick=self._state.step_count,
                action_history=[{"action_type": "step", "target_service": "", "feedback_string": f"step error: {exc}"}],
                incident_declared=False,
                mttm_achieved_tick=None,
            )

    # ------------------------------------------------------------------
    # state — read current episode metadata (property, no side effects)
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """
        Return current episode metadata.
        Read-only — does not advance the simulation or mutate any state.
        """
        return self._state