# server/firewatch_env_environment.py
# Phase 7 — Full OpenEnv Wiring & Server Integration.
#
# Wires all six components (models, config, simulation, actions, rewards)
# behind the OpenEnv step/reset/state API. This file is the integration
# point ONLY — it never defines simulation logic, reward calculations,
# or model definitions.
#
# Base class: openenv.core.env_server.interfaces.Environment
# HTTP wrapping: handled by create_app() in app.py
#
# The OpenEnv framework calls serialize_observation() which extracts
# done, reward, metadata from the returned Observation, placing them
# at the top level of the HTTP response. Our SystemObservation inherits
# from Observation, so these fields are available.

from __future__ import annotations

import random
import traceback
from collections import deque
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Dual-import pattern — required for both in-repo and Docker execution
try:
    from ..models import (
        FirewatchAction,
        SystemObservation,
        ServiceMetrics,
        Alert,
    )
    from ..simulation import ServiceMesh, generate_episode, FaultConfig, _count_blast_radius
    from ..actions import ActionHandler
    from ..rewards import RewardEngine, EpisodeResult, grade, build_info_dict, compute_premature_exit_penalty
    from ..config import (
        ALL_SERVICES,
        TASKS,
        SLO_BUDGET_BY_DIFFICULTY,
        SLO_BURN_RATE_BY_DIFFICULTY,
        SLO_BURN_RATE_MITIGATED_MULTIPLIER,
        USER_FACING_SERVICES,
        ESCALATE_INVESTIGATION_COST_MULTIPLIER,
        INVESTIGATION_ACTIONS,
        STATUS_THRESHOLD_DEGRADED_ERROR,
        SECONDS_PER_TICK,
    )
except ImportError:
    from models import (
        FirewatchAction,
        SystemObservation,
        ServiceMetrics,
        Alert,
    )
    from simulation import ServiceMesh, generate_episode, FaultConfig, _count_blast_radius
    from actions import ActionHandler
    from rewards import RewardEngine, EpisodeResult, grade, build_info_dict, compute_premature_exit_penalty
    from config import (
        ALL_SERVICES,
        TASKS,
        SLO_BUDGET_BY_DIFFICULTY,
        SLO_BURN_RATE_BY_DIFFICULTY,
        SLO_BURN_RATE_MITIGATED_MULTIPLIER,
        USER_FACING_SERVICES,
        ESCALATE_INVESTIGATION_COST_MULTIPLIER,
        INVESTIGATION_ACTIONS,
        STATUS_THRESHOLD_DEGRADED_ERROR,
        SECONDS_PER_TICK,
    )


def _build_observation(
    mesh: ServiceMesh,
    action_history: list[dict[str, str]],
    done: bool = False,
    reward: float | None = None,
    info: dict | None = None,
    user_impact_active: bool = True,
    current_slo_burn_rate: float = 1.5,
) -> SystemObservation:
    """Build a SystemObservation from current mesh state."""
    # Generate alerts from current service metrics
    alerts = _generate_alerts(mesh)

    return SystemObservation(
        services=dict(mesh.services),
        active_alerts=alerts,
        dependency_graph=mesh.dependency_graph,
        slo_budget_remaining_pct=round(mesh.slo_budget, 2),
        bad_customer_minutes=round(mesh.incident_metrics.bad_customer_minutes, 4),
        sim_time_elapsed_seconds=mesh.sim_time_seconds,
        sim_tick=mesh.tick_count,
        action_history=action_history[-10:],  # Last 10 actions
        incident_declared=False,
        mttm_achieved_tick=mesh.incident_metrics.mttm_achieved_tick,
        user_impact_active=user_impact_active,
        current_slo_burn_rate=current_slo_burn_rate,
        # OpenEnv Observation fields
        done=done,
        reward=reward,
        metadata=info or {},
    )


def _generate_alerts(mesh: ServiceMesh) -> list[Alert]:
    """Generate alerts based on current service metric thresholds."""
    alerts: list[Alert] = []
    for name, m in mesh.services.items():
        if m.http_server_error_rate >= 0.50:
            alerts.append(Alert(
                alert_id=uuid4().hex[:8],
                alertname="HighErrorRate",
                service_name=name,
                severity="critical",
                description=(
                    f"http_server_error_rate is {m.http_server_error_rate:.2f} "
                    f"(threshold: 0.05) on {name} for {mesh.tick_count} ticks"
                ),
                fired_at_tick=mesh.tick_count,
                metric_name="http_server_error_rate",
                metric_value=m.http_server_error_rate,
                threshold_value=0.05,
            ))
        elif m.http_server_error_rate >= 0.10:
            alerts.append(Alert(
                alert_id=uuid4().hex[:8],
                alertname="HighErrorRate",
                service_name=name,
                severity="warning",
                description=(
                    f"http_server_error_rate is {m.http_server_error_rate:.2f} "
                    f"(threshold: 0.05) on {name} for {mesh.tick_count} ticks"
                ),
                fired_at_tick=mesh.tick_count,
                metric_name="http_server_error_rate",
                metric_value=m.http_server_error_rate,
                threshold_value=0.05,
            ))

        if m.http_server_request_duration_p99 >= 2.0:
            alerts.append(Alert(
                alert_id=uuid4().hex[:8],
                alertname="HighLatency",
                service_name=name,
                severity="critical",
                description=(
                    f"http_server_request_duration_p99 is "
                    f"{m.http_server_request_duration_p99:.2f}s "
                    f"(threshold: 2.0s) on {name}"
                ),
                fired_at_tick=mesh.tick_count,
                metric_name="http_server_request_duration_p99",
                metric_value=m.http_server_request_duration_p99,
                threshold_value=2.0,
            ))
        elif m.http_server_request_duration_p99 >= 0.50:
            alerts.append(Alert(
                alert_id=uuid4().hex[:8],
                alertname="HighLatency",
                service_name=name,
                severity="warning",
                description=(
                    f"http_server_request_duration_p99 is "
                    f"{m.http_server_request_duration_p99:.2f}s "
                    f"(threshold: 0.5s) on {name}"
                ),
                fired_at_tick=mesh.tick_count,
                metric_name="http_server_request_duration_p99",
                metric_value=m.http_server_request_duration_p99,
                threshold_value=0.5,
            ))

        if m.process_memory_utilization >= 0.80:
            severity = "critical" if m.process_memory_utilization >= 0.95 else "warning"
            alerts.append(Alert(
                alert_id=uuid4().hex[:8],
                alertname="MemoryPressure",
                service_name=name,
                severity=severity,
                description=(
                    f"process_memory_utilization is "
                    f"{m.process_memory_utilization:.2f} "
                    f"(threshold: 0.80) on {name}"
                ),
                fired_at_tick=mesh.tick_count,
                metric_name="process_memory_utilization",
                metric_value=m.process_memory_utilization,
                threshold_value=0.80,
            ))

        if m.status == "down":
            alerts.append(Alert(
                alert_id=uuid4().hex[:8],
                alertname="ServiceDown",
                service_name=name,
                severity="page",
                description=f"{name} is DOWN",
                fired_at_tick=mesh.tick_count,
                metric_name="status",
                metric_value=1.0,
                threshold_value=0.0,
            ))

    return alerts


def _empty_observation(error_msg: str = "") -> SystemObservation:
    """Return a minimal valid observation for error cases."""
    return SystemObservation(
        services={},
        active_alerts=[],
        dependency_graph={},
        slo_budget_remaining_pct=100.0,
        bad_customer_minutes=0.0,
        sim_time_elapsed_seconds=0,
        sim_tick=0,
        action_history=(
            [{"action_type": "error", "target_service": "", "feedback_string": error_msg}]
            if error_msg else []
        ),
        incident_declared=False,
        mttm_achieved_tick=None,
        done=False,
        reward=None,
        metadata={"error": error_msg} if error_msg else {},
    )


class FirewatchEnvironment(Environment):
    """
    SRE Incident Response RL Environment — Phase 7 Full Integration.

    Wires all components behind the OpenEnv step/reset/state API:
    - ServiceMesh (simulation.py) — physics engine
    - FaultInjector (simulation.py) — procedural episode generation
    - ActionHandler (actions.py) — 10 action types → state mutations
    - RewardEngine (rewards.py) — outcome-based per-step rewards
    - Grader (rewards.py) — unified 4-component episode scoring

    Zero-crash policy: every public method wraps its logic in try/except.
    Invalid inputs return HTTP 200 with error info, never HTTP 500.
    """

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Stateless components (created once, reused across episodes)
        self._reward_engine = RewardEngine()
        self._action_handler = ActionHandler()

        # Per-episode state (set in reset)
        self._mesh: ServiceMesh | None = None
        self._fault_config: FaultConfig | None = None
        self._difficulty: str = "easy"
        self._episode_seed: int = 0
        self._episode_result = EpisodeResult()
        self._prev_obs: SystemObservation | None = None
        self._action_history: list[dict[str, str]] = []
        self._episode_done: bool = False
        self._task_id: str | None = None
        self._max_ticks: int = 20

    # ------------------------------------------------------------------
    # reset() — initialise a new episode
    # ------------------------------------------------------------------

    def reset(
        self,
        difficulty: str = "easy",
        seed: int | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> SystemObservation:
        """
        Start a new incident episode.

        Args:
            difficulty: One of "easy", "medium", "hard".
            seed: Optional integer seed for reproducible episodes.
                  Same seed + difficulty always produces the same episode.
            task_id: Optional explicit task ID for Phase 1+ task configs.
                     When provided, uses the specific TaskConfig directly.

        Returns:
            SystemObservation with initial system state.
        """
        try:
            # Generate deterministic seed if not provided
            if seed is None:
                seed = random.randint(0, 2**31 - 1)

            self._state = State(episode_id=str(uuid4()), step_count=0)
            self._difficulty = difficulty
            self._episode_seed = seed
            self._task_id = task_id

            # --- SPEC-04 §4: Fail-fast service validation ---
            # Verify all service references exist in ALL_SERVICES before generating.
            _task_cfg = None
            if task_id:
                _task_cfg = TASKS.get(task_id)
            if _task_cfg is None:
                for _t in TASKS.values():
                    if _t.difficulty == difficulty and _t.seed == seed:
                        _task_cfg = _t
                        break
            if _task_cfg is not None:
                _all_svc_set = set(ALL_SERVICES)
                _refs: set[str] = set()
                _refs.update(_task_cfg.services)
                if _task_cfg.fault_service:
                    _refs.add(_task_cfg.fault_service)
                if _task_cfg.secondary_fault_service:
                    _refs.add(_task_cfg.secondary_fault_service)
                _refs.update(_task_cfg.red_herrings)
                if _task_cfg.adversarial_logs:
                    for _adv in _task_cfg.adversarial_logs:
                        _adv_svc = _adv.get("service")
                        if _adv_svc:
                            _refs.add(_adv_svc)
                _refs.update(_task_cfg.task_metrics_schema.keys())
                for _svc_name in _refs:
                    if _svc_name and _svc_name not in _all_svc_set:
                        raise ValueError(
                            f"Task {_task_cfg.task_id} references unregistered "
                            f"service: {_svc_name}"
                        )

            # Generate episode (task_id enables SPEC-03 explicit task configs)
            self._mesh, self._fault_config = generate_episode(
                difficulty, seed, task_id=task_id
            )

            # Propagate initial fault so tick=0 observation shows degradation.
            # Without this, generate_episode() sets fault config but hasn't run
            # physics yet — all services appear healthy and a rational agent
            # will immediately declare_resolved.
            self._mesh.tick()

            # Reset stateful components
            self._reward_engine.reset()
            self._action_handler = ActionHandler()
            # Initialize with services_affected from fault config (PRD §11.3)
            # Root cause + all fault sources + downstream dependents = affected services
            affected = {self._fault_config.root_cause_service}
            # SPEC-01: include all fault sources from active_faults
            if self._mesh.active_faults:
                for fault in self._mesh.active_faults:
                    affected.add(fault.fault_service)
            # Add downstream dependents reachable via reverse dep graph
            queue = list(affected)
            visited = set(queue)
            for svc in queue:
                for other_svc, deps in self._mesh.dependency_graph.items():
                    if svc in deps and other_svc not in visited:
                        affected.add(other_svc)
                        queue.append(other_svc)
                        visited.add(other_svc)
            self._episode_result = EpisodeResult(
                services_affected=len(affected),
                _affected_services=affected,
            )
            # Set static episode-level counts for grade() (Fix 1 + Fix 3)
            self._episode_result.services_affected_static = _count_blast_radius(
                self._mesh, self._fault_config
            )
            self._episode_result.total_services_in_episode = len(self._mesh.services)
            self._episode_result.initial_slo_budget = SLO_BUDGET_BY_DIFFICULTY[difficulty]
            self._action_history = []
            self._episode_done = False

            # Look up max ticks — try task_id first, then seed-based, then legacy
            task_config = None
            if task_id:
                task_config = TASKS.get(task_id)
            if task_config is None:
                # Seed-based or legacy lookup
                for t in TASKS.values():
                    if t.difficulty == difficulty and t.seed == seed:
                        task_config = t
                        break
            if task_config is None:
                task_config = TASKS.get(f"task_{difficulty}")
            self._max_ticks = task_config.max_ticks if task_config else 20

            # Build initial observation
            obs = _build_observation(
                mesh=self._mesh,
                action_history=self._action_history,
                done=False,
                reward=None,
            )
            self._prev_obs = obs
            return obs

        except Exception as exc:
            return _empty_observation(f"reset error: {exc}")

    # ------------------------------------------------------------------
    # step() — execute one agent action
    # ------------------------------------------------------------------

    def step(
        self,
        action: FirewatchAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> SystemObservation:
        """
        Execute one agent action and advance the simulation by one tick.

        The simulation degrades autonomously each tick BEFORE the agent
        action is applied — time pressure is real, not cosmetic.

        Args:
            action: A FirewatchAction specifying what the agent wants to do.
            timeout_s: Optional timeout (unused, required by base class).

        Returns:
            SystemObservation with updated state, reward, done, and info.
        """
        try:
            if self._mesh is None or self._fault_config is None:
                return _empty_observation(
                    "No active episode. Call reset() first."
                )

            if self._episode_done:
                return _empty_observation(
                    "Episode already completed. Call reset() to start a new one."
                )

            self._state = State(
                episode_id=self._state.episode_id,
                step_count=self._state.step_count + 1,
            )

            # --- 1. mesh.tick() FIRST — autonomous degradation ---
            bcm_delta = self._mesh.tick()

            # --- SLO burn with mitigation shield ---
            # mesh.tick() already decremented slo_budget; undo it and reapply with shield.
            difficulty = self._difficulty
            base_burn = SLO_BURN_RATE_BY_DIFFICULTY[difficulty]

            # Check user-facing service health for mitigation shield
            user_impact_active = any(
                self._mesh.services[s].http_server_error_rate > STATUS_THRESHOLD_DEGRADED_ERROR
                for s in USER_FACING_SERVICES
                if s in self._mesh.services
            )

            burn_this_tick = (
                base_burn
                if user_impact_active
                else base_burn * SLO_BURN_RATE_MITIGATED_MULTIPLIER
            )

            # The mesh already applied degradation-scaled burn in tick().
            # Restore the mesh's slo_budget to what it was before tick() burned it,
            # then apply our mitigation-aware burn.
            degraded_count = sum(1 for m in self._mesh.services.values() if m.status != "healthy")
            total_services = len(self._mesh.services)
            if total_services > 0:
                old_burn = base_burn * (degraded_count / total_services)
                # Restore then apply new burn
                self._mesh.slo_budget = min(
                    SLO_BUDGET_BY_DIFFICULTY[difficulty],
                    self._mesh.slo_budget + old_burn
                )
                self._mesh.slo_budget = max(0.0, self._mesh.slo_budget - burn_this_tick)

            _current_slo_burn_rate = burn_this_tick

            # --- 2. Record metrics for action handler history ---
            self._action_handler.record_tick(self._mesh)

            # --- 3. Validate and apply action ---
            target = action.target_service
            action_valid = True
            wrong_action = False

            # Check if target is valid for actions that require it
            if action.action_type not in ("declare_resolved", "escalate"):
                if target is None:
                    action_valid = False
                elif target not in self._mesh.services:
                    action_valid = False

            if action_valid:
                feedback, wrong_action = self._action_handler.apply(
                    action, self._mesh, self._fault_config
                )
            else:
                if target is None and action.action_type not in ("declare_resolved", "escalate"):
                    feedback = (
                        f"Action '{action.action_type}' requires a target_service. "
                        f"No action taken."
                    )
                elif target is not None and target not in self._mesh.services:
                    feedback = (
                        f"Invalid target: '{target}' is not an active service "
                        f"in this episode. Active services: "
                        f"{list(self._mesh.services.keys())}. No action taken."
                    )
                else:
                    feedback = f"Invalid action: {action.action_type}. No action taken."

            # --- 4. Record action in history ---
            self._action_history.append({
                "action_type": action.action_type,
                "target_service": target or "",
                "feedback_string": feedback,
            })

            # --- 5. Handle declare_resolved (sets incident_declared) ---
            incident_declared = action.action_type == "declare_resolved"

            # --- Specialist discount: reduce SLO burn for investigation actions ---
            if (
                action.action_type in INVESTIGATION_ACTIONS
                and self._action_handler.specialist_active_ticks > 0
            ):
                # Give back the SLO that would have been discounted
                discount = burn_this_tick * (1.0 - ESCALATE_INVESTIGATION_COST_MULTIPLIER)
                self._mesh.slo_budget = min(
                    SLO_BUDGET_BY_DIFFICULTY[self._difficulty],
                    self._mesh.slo_budget + discount
                )
                self._action_handler.specialist_active_ticks -= 1

            # --- 6. Build next observation ---
            next_obs = _build_observation(
                mesh=self._mesh,
                action_history=self._action_history,
                done=False,  # Set below after checking termination
                reward=None,  # Set below after computing reward
                user_impact_active=user_impact_active,
                current_slo_burn_rate=_current_slo_burn_rate,
            )
            # Update incident_declared
            next_obs.incident_declared = incident_declared

            # --- 7. Compute reward ---
            if self._prev_obs is not None:
                reward, breakdown = self._reward_engine.compute(
                    self._prev_obs, action, next_obs,
                    action_valid, wrong_action,
                )
            else:
                reward = 0.0
                breakdown = {
                    "health_improvement": 0.0,
                    "slo_preservation": 0.0,
                    "mttm_bonus": 0.0,
                    "time_cost": 0.0,
                    "wrong_action_penalty": 0.0,
                    "slo_breach_penalty": 0.0,
                    "total": 0.0,
                }

            # --- Premature exit penalty ---
            if action.action_type == "declare_resolved":
                mttm_achieved = next_obs.mttm_achieved_tick is not None
                exit_penalty = compute_premature_exit_penalty(
                    next_obs,
                    tick_count=self._mesh.tick_count,
                    mttm_achieved=mttm_achieved,
                )
                if exit_penalty < 0.0:
                    reward += exit_penalty
                    breakdown["premature_exit_penalty"] = round(exit_penalty, 6)
                    breakdown["total"] = round(sum(v for k, v in breakdown.items() if k != "total"), 6)

            # --- 8. Update episode result ---
            self._episode_result.update(next_obs, wrong_action)

            # --- 9. Check termination conditions ---
            done = (
                self._mesh.slo_budget <= 0.0
                or self._mesh.tick_count >= self._max_ticks
                or incident_declared
            )

            # --- 10. Grade if done ---
            episode_score: float | None = None
            if done:
                episode_score = grade(
                    self._episode_result, self._difficulty,
                    task_id=self._task_id,
                )
                self._episode_done = True

            # --- 11. Build rich info dict ---
            info = build_info_dict(
                prev_obs=self._prev_obs or next_obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                reward_breakdown=breakdown,
                action_valid=action_valid,
                action_feedback=feedback,
                wrong_action=wrong_action,
                done=done,
                episode_result=self._episode_result if done else None,
                episode_score=episode_score,
                difficulty=self._difficulty,
            )

            # --- 12. Set done/reward on observation ---
            next_obs.done = done
            next_obs.reward = round(reward, 6)
            next_obs.episode_score = episode_score   # None unless done=True
            next_obs.metadata = info

            # --- 13. Update prev_obs ---
            self._prev_obs = next_obs

            return next_obs

        except Exception as exc:
            tb = traceback.format_exc()
            error_obs = _empty_observation(f"step error: {exc}")
            error_obs.done = False
            error_obs.reward = 0.0
            error_obs.metadata = {"error": str(exc), "traceback": tb}
            return error_obs

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