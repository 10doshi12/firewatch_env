# rewards.py
# Phase 6 — Reward Engine & Grader.
# Per-step reward computation + episode-level grading.
# All rewards derived from observable system outcomes only.
#
# This file defines:
#   1. RewardEngine — per-step reward with 6 components
#   2. EpisodeResult — running episode statistics tracker
#   3. grade() — unified 4-component scoring (0.0–1.0)
#   4. build_info_dict() — rich info dict for step() responses

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .models import SystemObservation, FirewatchAction
    from .config import (
        REWARD_WEIGHT_HEALTH,
        REWARD_WEIGHT_SLO,
        REWARD_MTTM_BONUS,
        REWARD_TIME_COST,
        REWARD_WRONG_ACTION_PENALTY,
        REWARD_SLO_BREACH_PENALTY,
        GRADER_WEIGHT_RECOVERY,
        GRADER_WEIGHT_SPEED,
        GRADER_WEIGHT_PRECISION,
        GRADER_WEIGHT_SLO,
        GRADER_WRONG_ACTION_PENALTY_PER_ACTION,
        GRADER_SPEED_MTTM_WEIGHT,
        GRADER_SPEED_BCM_WEIGHT,
        TASKS,
    )
except ImportError:
    from models import SystemObservation, FirewatchAction
    from config import (
        REWARD_WEIGHT_HEALTH,
        REWARD_WEIGHT_SLO,
        REWARD_MTTM_BONUS,
        REWARD_TIME_COST,
        REWARD_WRONG_ACTION_PENALTY,
        REWARD_SLO_BREACH_PENALTY,
        GRADER_WEIGHT_RECOVERY,
        GRADER_WEIGHT_SPEED,
        GRADER_WEIGHT_PRECISION,
        GRADER_WEIGHT_SLO,
        GRADER_WRONG_ACTION_PENALTY_PER_ACTION,
        GRADER_SPEED_MTTM_WEIGHT,
        GRADER_SPEED_BCM_WEIGHT,
        TASKS,
    )


# ==========================================================================
# RewardEngine — per-step reward computation
# ==========================================================================

class RewardEngine:
    """
    Computes per-step rewards from observable system outcomes.

    Six reward components:
      1. Health improvement — positive when mean error rate decreases
      2. SLO preservation — tracks budget depletion rate
      3. MTTM bonus — one-time reward when BCM delta hits zero
      4. Time cost — constant negative per step (urgency signal)
      5. Wrong action penalty — remediating a healthy service
      6. SLO breach penalty — terminal when budget exhausted
    """

    def __init__(self) -> None:
        self._mttm_bonus_given = False

    def reset(self) -> None:
        """Reset per-episode state."""
        self._mttm_bonus_given = False

    def compute(
        self,
        prev_obs: SystemObservation,
        action: FirewatchAction,
        next_obs: SystemObservation,
        action_valid: bool,
        wrong_action: bool,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute reward for a single step.

        Args:
            prev_obs: Observation before this step.
            action: Action taken.
            next_obs: Observation after this step.
            action_valid: Whether the action was accepted.
            wrong_action: Whether the agent remediated a healthy service.

        Returns:
            Tuple of (total_reward, breakdown_dict).
        """
        # 1. Health improvement: mean error rate decrease
        prev_mean = _mean_error_rate(prev_obs)
        next_mean = _mean_error_rate(next_obs)
        health_improvement = (prev_mean - next_mean) * REWARD_WEIGHT_HEALTH

        # 2. SLO preservation: budget change
        slo_delta = next_obs.slo_budget_remaining_pct - prev_obs.slo_budget_remaining_pct
        slo_preservation = slo_delta * REWARD_WEIGHT_SLO

        # 3. MTTM bonus: one-time when mttm_achieved_tick is first set
        mttm_bonus = 0.0
        if (
            next_obs.mttm_achieved_tick is not None
            and prev_obs.mttm_achieved_tick is None
            and not self._mttm_bonus_given
        ):
            mttm_bonus = REWARD_MTTM_BONUS
            self._mttm_bonus_given = True

        # 4. Time cost: constant negative every step
        time_cost = REWARD_TIME_COST

        # 5. Wrong action penalty
        wrong_penalty = REWARD_WRONG_ACTION_PENALTY if wrong_action else 0.0

        # 6. SLO breach terminal penalty
        slo_breach = 0.0
        if (
            next_obs.slo_budget_remaining_pct <= 0.0
            and prev_obs.slo_budget_remaining_pct > 0.0
        ):
            slo_breach = REWARD_SLO_BREACH_PENALTY

        total = (
            health_improvement
            + slo_preservation
            + mttm_bonus
            + time_cost
            + wrong_penalty
            + slo_breach
        )

        breakdown = {
            "health_improvement": round(health_improvement, 6),
            "slo_preservation": round(slo_preservation, 6),
            "mttm_bonus": round(mttm_bonus, 6),
            "time_cost": round(time_cost, 6),
            "wrong_action_penalty": round(wrong_penalty, 6),
            "slo_breach_penalty": round(slo_breach, 6),
            "total": round(total, 6),
        }

        return total, breakdown


# ==========================================================================
# EpisodeResult — running episode statistics
# ==========================================================================

@dataclass
class EpisodeResult:
    """Tracks statistics needed for episode grading."""

    services_affected: int = 0
    services_recovered: int = 0
    ticks_taken: int = 0
    mttm_ticks: int | None = None
    wrong_actions: int = 0
    total_actions: int = 0
    final_slo_budget: float = 100.0
    bad_customer_minutes: float = 0.0

    # Internal tracking
    _affected_services: set[str] = field(default_factory=set, repr=False)
    _recovered_services: set[str] = field(default_factory=set, repr=False)
    # Services ACTUALLY observed as degraded (status != healthy at some point)
    _observed_degraded: set[str] = field(default_factory=set, repr=False)

    def update(
        self,
        obs: SystemObservation,
        wrong_action: bool,
    ) -> None:
        """Update episode statistics after each step."""
        self.ticks_taken = obs.sim_tick

        # Track affected services (any that were degraded at any point)
        for name, metrics in obs.services.items():
            if metrics.status != "healthy":
                self._affected_services.add(name)
                self._observed_degraded.add(name)
            elif name in self._observed_degraded:
                # Only count as recovered if it was actually observed degraded
                self._recovered_services.add(name)

        self.services_affected = len(self._affected_services)
        self.services_recovered = len(self._recovered_services)

        # Track MTTM
        if obs.mttm_achieved_tick is not None and self.mttm_ticks is None:
            self.mttm_ticks = obs.mttm_achieved_tick

        # Track actions
        self.total_actions += 1
        if wrong_action:
            self.wrong_actions += 1

        # Update final values
        self.final_slo_budget = obs.slo_budget_remaining_pct
        self.bad_customer_minutes = obs.bad_customer_minutes

    def to_dict(self) -> dict:
        """Serialize for episode summary."""
        return {
            "services_affected": self.services_affected,
            "services_recovered": self.services_recovered,
            "ticks_taken": self.ticks_taken,
            "mttm_ticks": self.mttm_ticks,
            "wrong_actions": self.wrong_actions,
            "final_slo_budget": round(self.final_slo_budget, 2),
            "bad_customer_minutes": round(self.bad_customer_minutes, 2),
            "recovery_ratio": (
                round(self.services_recovered / self.services_affected, 3)
                if self.services_affected > 0
                else 0.0
            ),
        }


# ==========================================================================
# grade() — unified episode scoring
# ==========================================================================

def grade(episode_result: EpisodeResult, difficulty: str) -> float:
    """
    Compute final episode score using unified 4-component formula.

    Components (weights from config.py):
      - Recovery (40%): services_recovered / services_affected
      - Speed (25%): composite of MTTM and BCM scores
      - Precision (20%): penalized by wrong actions
      - SLO (15%): final budget remaining

    Args:
        episode_result: Completed episode statistics.
        difficulty: "easy", "medium", or "hard" — for max_ticks lookup.

    Returns:
        Float between 0.0 and 1.0.
    """
    er = episode_result
    task_key = f"task_{difficulty}"
    task = TASKS.get(task_key)
    if task is None:
        return 0.0

    max_ticks = task.max_ticks
    max_bcm = task.max_bad_customer_minutes

    # 1. Recovery (40%)
    if er.services_affected > 0:
        recovery = er.services_recovered / er.services_affected
    else:
        recovery = 1.0  # No affected services = perfect recovery

    # Penalize early exit without fix: if the agent gave up, assume worst case for BCM and SLO
    if recovery < 1.0 and er.ticks_taken < max_ticks:
        bcm_score = 0.0
        slo = 0.0
    else:
        # BCM score: total user impact relative to worst case
        bcm_score = max(0.0, 1.0 - (er.bad_customer_minutes / max_bcm))
        # SLO (15%) — budget remaining
        slo = max(0.0, min(1.0, er.final_slo_budget / 100.0))

    # 2. Speed (25%) — composite of MTTM + BCM
    # MTTM score: how quickly user impact was zeroed
    if er.mttm_ticks is not None:
        mttm_score = max(0.0, 1.0 - (er.mttm_ticks / max_ticks))
    else:
        mttm_score = 0.0

    speed = (
        GRADER_SPEED_MTTM_WEIGHT * mttm_score
        + GRADER_SPEED_BCM_WEIGHT * bcm_score
    )

    # 3. Precision (20%) — penalized by wrong actions
    precision = max(
        0.0, 1.0 - (er.wrong_actions * GRADER_WRONG_ACTION_PENALTY_PER_ACTION)
    )
    
    # False resolution penalty
    if recovery == 0.0:
        precision = 0.0  # doing nothing then exiting is inherently imprecise

    # Final weighted score
    score = (
        GRADER_WEIGHT_RECOVERY * recovery
        + GRADER_WEIGHT_SPEED * speed
        + GRADER_WEIGHT_PRECISION * precision
        + GRADER_WEIGHT_SLO * slo
    )

    return max(0.0, min(1.0, score))


# ==========================================================================
# Rich Info Dictionary Builder
# ==========================================================================

def build_info_dict(
    prev_obs: SystemObservation,
    next_obs: SystemObservation,
    action: FirewatchAction,
    reward: float,
    reward_breakdown: dict[str, float],
    action_valid: bool,
    action_feedback: str,
    wrong_action: bool,
    done: bool,
    episode_result: EpisodeResult | None = None,
    episode_score: float | None = None,
    difficulty: str = "easy",
) -> dict:
    """
    Build the rich info dictionary for step() responses.

    Contains both programmatic fields (for reward computation) and
    semantic fields (for LLM judge comprehension).
    """
    # --- Programmatic fields ---
    info: dict = {
        "reward": round(reward, 6),
        "reward_breakdown": reward_breakdown,
        "action_valid": action_valid,
        "action_feedback": action_feedback,
        "slo_budget_remaining_pct": round(next_obs.slo_budget_remaining_pct, 2),
        "bad_customer_minutes": round(next_obs.bad_customer_minutes, 2),
        "sim_time_elapsed_seconds": next_obs.sim_time_elapsed_seconds,
        "mttm_achieved": next_obs.mttm_achieved_tick is not None,
    }

    # --- Semantic fields (for LLM judge) ---

    # System state summary
    status_counts: dict[str, int] = {}
    for m in next_obs.services.values():
        status_counts[m.status] = status_counts.get(m.status, 0) + 1
    state_parts = []
    for status in ["down", "critical", "degraded", "healthy"]:
        count = status_counts.get(status, 0)
        if count > 0:
            state_parts.append(f"{count} {status}")
    info["system_state"] = ", ".join(state_parts) if state_parts else "unknown"

    # Degraded service names
    info["services_degraded"] = [
        name for name, m in next_obs.services.items()
        if m.status != "healthy"
    ]

    # Recovering services (error_rate improved this tick)
    recovering = []
    for name, m in next_obs.services.items():
        if name in prev_obs.services:
            prev_err = prev_obs.services[name].http_server_error_rate
            if m.http_server_error_rate < prev_err - 0.01:
                recovering.append(name)
    info["services_recovering"] = recovering

    # Semantic analysis narrative
    info["semantic_analysis"] = _build_semantic_analysis(
        action, action_feedback, wrong_action, action_valid,
        next_obs, prev_obs, recovering,
    )

    # Blast radius
    impacted = len(info["services_degraded"])
    downstream_at_risk = []
    for name in info["services_degraded"]:
        for svc, deps in next_obs.dependency_graph.items():
            if name in deps and svc not in info["services_degraded"]:
                downstream_at_risk.append(svc)
    info["blast_radius"] = {
        "services_impacted": impacted,
        "downstream_at_risk": list(set(downstream_at_risk)),
    }

    # Incident progress
    info["incident_progress"] = _assess_progress(next_obs, done)

    # Fixed simulation type string
    info["simulation_type"] = (
        "AIOps 2.0 incident response environment with OTel-compatible "
        "telemetry, autonomous cascade propagation, adversarial telemetry "
        "injection, and continuous MTTM/MTTR tracking"
    )

    # --- Episode end fields ---
    if done and episode_result is not None:
        info["episode_score"] = round(episode_score or 0.0, 4)
        info["episode_summary"] = episode_result.to_dict()

    return info


def _build_semantic_analysis(
    action: FirewatchAction,
    feedback: str,
    wrong_action: bool,
    action_valid: bool,
    next_obs: SystemObservation,
    prev_obs: SystemObservation,
    recovering: list[str],
) -> str:
    """Generate contextual narrative for the LLM judge."""
    parts: list[str] = []

    if not action_valid:
        parts.append(
            f"Agent attempted '{action.action_type}' but the action was "
            f"invalid. No system state was modified."
        )
    elif wrong_action:
        parts.append(
            f"Agent applied '{action.action_type}' to "
            f"'{action.target_service}' which was not significantly degraded. "
            f"This indicates premature remediation before sufficient "
            f"investigation. The actual root cause remains unaddressed."
        )
    elif action.action_type in ("fetch_logs", "get_metrics_detail", "trace_dependencies"):
        parts.append(
            f"Agent performed investigation: '{action.action_type}' on "
            f"'{action.target_service}'. This is an information-gathering "
            f"step that does not modify system state."
        )
    elif action.action_type in ("restart_service", "rollback_deploy", "revert_config", "scale_replicas", "circuit_break"):
        parts.append(
            f"Agent applied remediation: '{action.action_type}' to "
            f"'{action.target_service}'."
        )
        if recovering:
            parts.append(
                f"System health is improving — services recovering: "
                f"{recovering}."
            )
        else:
            parts.append(
                f"No immediate improvement observed. The remediation may "
                f"need time to take effect, or it may be targeting the "
                f"wrong service/fault type."
            )
    elif action.action_type == "declare_resolved":
        parts.append("Agent declared the incident resolved. Episode ending.")
    elif action.action_type == "escalate":
        parts.append(
            "Agent escalated the incident. This costs SLO budget but "
            "brings specialist attention."
        )

    # Overall state assessment
    degraded_count = sum(
        1 for m in next_obs.services.values() if m.status != "healthy"
    )
    total = len(next_obs.services)
    if degraded_count == 0:
        parts.append("All services are now healthy.")
    elif degraded_count == total:
        parts.append(
            "All services are degraded — situation is critical. "
            "Immediate action required."
        )
    else:
        parts.append(
            f"{degraded_count}/{total} services remain degraded."
        )

    return " ".join(parts)


def _assess_progress(obs: SystemObservation, done: bool) -> str:
    """Assess incident resolution progress."""
    if done:
        return "100% - resolved"

    if obs.mttm_achieved_tick is not None:
        return "75% - remediation in progress"

    degraded = sum(1 for m in obs.services.values() if m.status != "healthy")
    total = len(obs.services)

    if degraded == 0:
        return "100% - resolved"
    elif degraded < total * 0.3:
        return "75% - remediation in progress"
    elif obs.sim_tick > 0:
        return "25% - root cause identified"
    else:
        return "0%"


# ==========================================================================
# Helper
# ==========================================================================

def _mean_error_rate(obs: SystemObservation) -> float:
    """Compute mean error rate across all services in observation."""
    services = obs.services
    if not services:
        return 0.0
    return sum(m.http_server_error_rate for m in services.values()) / len(services)


# ==========================================================================
# Public API
# ==========================================================================

__all__ = [
    "RewardEngine",
    "EpisodeResult",
    "grade",
    "build_info_dict",
]
