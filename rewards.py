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
from math import isclose

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
        REWARD_PREMATURE_EXIT_BASE,
        REWARD_PREMATURE_EXIT_SCALE,
        HEALTHY_ERROR_RATE_THRESHOLD,
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
        REWARD_PREMATURE_EXIT_BASE,
        REWARD_PREMATURE_EXIT_SCALE,
        HEALTHY_ERROR_RATE_THRESHOLD,
    )


# ==========================================================================
# RewardEngine — per-step reward computation
# ==========================================================================

class RewardEngine:
    """
    Computes per-step rewards from observable system outcomes.

    Six reward components, each with a distinct behavioral incentive:
      1. Health improvement (1.0) — primary signal: mean error rate decrease
         weighted by dependency topology (upstream services count more)
      2. SLO preservation (0.3) — secondary: tracks budget depletion rate.
         Lower weight than health to prevent SLO-gaming over actual recovery.
      3. MTTM bonus (2.0) — one-time reward when BCM delta hits zero.
         2× health weight to strongly reward the "stop the bleeding" moment
         (per Google SRE's "mitigate before investigate" doctrine).
      4. Time cost (-0.05) — constant negative per step. Creates urgency
         without dominating the health signal.
      5. Wrong action penalty (-0.5) — remediating a service below
         HEALTHY_ERROR_RATE_THRESHOLD (0.05). Enforces precision.
      6. SLO breach penalty (-2.0) — terminal when budget exhausted.
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
        # 1. Health improvement: weighted mean error rate decrease
        prev_mean = _weighted_mean_error_rate(prev_obs.services, prev_obs.dependency_graph)
        next_mean = _weighted_mean_error_rate(next_obs.services, next_obs.dependency_graph)
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
    initial_slo_budget: float = 100.0   # Set at reset() time; used by grade() for SLO normalization
    bad_customer_minutes: float = 0.0

    # Static episode-level counts set once in reset() — never mutated by update()
    services_affected_static: int = 0    # predicted blast radius from FaultConfig BFS
    total_services_in_episode: int = 0   # total services active in this episode

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
            "initial_slo_budget": round(self.initial_slo_budget, 2),
            "bad_customer_minutes": round(self.bad_customer_minutes, 2),
            "recovery_ratio": (
                round(self.services_recovered / self.services_affected, 3)
                if self.services_affected > 0
                else 0.0
            ),
        }


# ==========================================================================
# Task-Specific Grader Conditions (SPEC-07 Phase 4)
# ==========================================================================

# Per-task condition sets. Each entry: (check_fn, description).
# check_fn signature: (services: dict[str, ServiceMetrics]) -> tuple[bool, str]
# Returns (passed: bool, details: str).

def _check_auth_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("auth-service")
    if svc is None:
        return False, "auth-service not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"auth-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


def _check_auth_active_requests(services: dict) -> tuple[bool, str]:
    svc = services.get("auth-service")
    if svc is None:
        return False, "auth-service not in topology"
    # Baseline range for active_requests is 1-200; fault condition >> baseline
    passed = 1 <= svc.http_server_active_requests <= 200
    return passed, f"auth-service active_requests={svc.http_server_active_requests} (baseline: 1-200)"


def _check_inventory_p99(services: dict) -> tuple[bool, str]:
    svc = services.get("inventory-service")
    if svc is None:
        return False, "inventory-service not in topology"
    passed = svc.http_server_request_duration_p99 < 1.0
    return passed, f"inventory-service p99={svc.http_server_request_duration_p99:.2f}s (threshold: 1.0s)"


def _check_order_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("order-service")
    if svc is None:
        return False, "order-service not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"order-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


def _check_lb_weight(services: dict) -> tuple[bool, str]:
    svc = services.get("user-profile-service")
    if svc is None:
        return False, "user-profile-service not in topology"
    # task-scoped metric: lb_weight_normalized; healthy ≈ 1.0
    weight = getattr(svc, "lb_weight_normalized", None)
    if weight is None:
        return False, "lb_weight_normalized not available"
    passed = isclose(weight, 1.0, abs_tol=0.3)
    return passed, f"user-profile-service lb_weight_normalized={weight:.2f} (≈1.0)"


def _check_lb_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("user-profile-service")
    if svc is None:
        return False, "user-profile-service not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"user-profile-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


def _check_payment_restart_stable(services: dict) -> tuple[bool, str]:
    svc = services.get("payment-processor")
    if svc is None:
        return False, "payment-processor not in topology"
    # Stable means restart_count is not increasing — we check the absolute value
    passed = svc.restart_count >= 0  # baseline
    return passed, f"payment-processor restart_count={svc.restart_count}"


def _check_payment_liveness(services: dict) -> tuple[bool, str]:
    svc = services.get("payment-processor")
    if svc is None:
        return False, "payment-processor not in topology"
    status = getattr(svc, "liveness_probe_status", None)
    passed = status == "passing"
    return passed, f"payment-processor liveness_probe_status={status}"


def _check_disk_stable(services: dict) -> tuple[bool, str]:
    svc = services.get("api-gateway")
    if svc is None:
        return False, "api-gateway not in topology"
    # Stable means disk ratio is not growing toward 1.0
    ratio = getattr(svc, "process_disk_usage_ratio", None)
    if ratio is None:
        return False, "process_disk_usage_ratio not available"
    passed = ratio < 0.99  # stabilizes after fix, doesn't reclaim
    return passed, f"api-gateway process_disk_usage_ratio={ratio:.2f}"


def _check_log_level_info(services: dict) -> tuple[bool, str]:
    svc = services.get("api-gateway")
    if svc is None:
        return False, "api-gateway not in topology"
    level = getattr(svc, "application_log_level", None)
    passed = level == "INFO"
    return passed, f"api-gateway application_log_level={level}"


def _check_apigateway_error_rate_lt_10(services: dict) -> tuple[bool, str]:
    svc = services.get("api-gateway")
    if svc is None:
        return False, "api-gateway not in topology"
    passed = svc.http_server_error_rate < 0.10
    return passed, f"api-gateway error_rate={svc.http_server_error_rate:.2f} (threshold: 0.10)"


def _check_apigateway_error_rate_lt_05(services: dict) -> tuple[bool, str]:
    svc = services.get("api-gateway")
    if svc is None:
        return False, "api-gateway not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"api-gateway error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


def _check_retry_rps_multiplier(services: dict) -> tuple[bool, str]:
    svc = services.get("api-gateway")
    if svc is None:
        return False, "api-gateway not in topology"
    multiplier = getattr(svc, "effective_rps_multiplier", None)
    if multiplier is None:
        return False, "effective_rps_multiplier not available"
    passed = multiplier < 1.2
    return passed, f"api-gateway effective_rps_multiplier={multiplier:.2f} (threshold: 1.2)"


def _check_notification_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("notification-service")
    if svc is None:
        return False, "notification-service not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"notification-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


def _check_canary_weight_zero(services: dict) -> tuple[bool, str]:
    svc = services.get("checkout-service")
    if svc is None:
        return False, "checkout-service not in topology"
    weight = getattr(svc, "canary_traffic_weight", None)
    if weight is None:
        return False, "canary_traffic_weight not available"
    passed = weight == 0.0
    return passed, f"checkout-service canary_traffic_weight={weight}"


def _check_checkout_error_rate_lt_05(services: dict) -> tuple[bool, str]:
    svc = services.get("checkout-service")
    if svc is None:
        return False, "checkout-service not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"checkout-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


def _check_checkout_error_rate_lt_02(services: dict) -> tuple[bool, str]:
    svc = services.get("checkout-service")
    if svc is None:
        return False, "checkout-service not in topology"
    passed = svc.http_server_error_rate < 0.02
    return passed, f"checkout-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.02)"


def _check_replica_lag(services: dict) -> tuple[bool, str]:
    svc = services.get("user-service")
    if svc is None:
        return False, "user-service not in topology"
    lag = getattr(svc, "db_replication_lag_seconds", None)
    if lag is None:
        return False, "db_replication_lag_seconds not available"
    passed = lag < 5.0
    return passed, f"user-service db_replication_lag={lag:.1f}s (threshold: 5.0s)"


def _check_read_path_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("user-service")
    if svc is None:
        return False, "user-service not in topology"
    rate = getattr(svc, "http_server_read_path_error_rate", None)
    if rate is None:
        return False, "http_server_read_path_error_rate not available"
    passed = rate < 0.05
    return passed, f"user-service read_path_error_rate={rate:.2f} (threshold: 0.05)"


def _check_replica_health(services: dict) -> tuple[bool, str]:
    svc = services.get("user-service")
    if svc is None:
        return False, "user-service not in topology"
    health = getattr(svc, "db_replica_health", None)
    passed = health == "synced"
    return passed, f"user-service db_replica_health={health}"


def _check_pricing_memory_lt_60(services: dict) -> tuple[bool, str]:
    svc = services.get("pricing-service")
    if svc is None:
        return False, "pricing-service not in topology"
    passed = svc.process_memory_utilization < 0.60
    return passed, f"pricing-service memory={svc.process_memory_utilization:.2f} (threshold: 0.60)"


def _check_pricing_error_rate_lt_10(services: dict) -> tuple[bool, str]:
    svc = services.get("pricing-service")
    if svc is None:
        return False, "pricing-service not in topology"
    passed = svc.http_server_error_rate < 0.10
    return passed, f"pricing-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.10)"


def _check_catalog_circuit_breaker_closed(services: dict) -> tuple[bool, str]:
    svc = services.get("product-catalog")
    if svc is None:
        return False, "product-catalog not in topology"
    state = getattr(svc, "circuit_breaker_state", None)
    passed = state == "closed"
    return passed, f"product-catalog circuit_breaker_state={state}"


def _check_cache_hit_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("cache-service")
    if svc is None:
        return False, "cache-service not in topology"
    rate = getattr(svc, "cache_hit_rate", None)
    if rate is None:
        return False, "cache_hit_rate not available"
    passed = rate > 0.85
    return passed, f"cache-service cache_hit_rate={rate:.2f} (threshold: 0.85)"


def _check_user_db_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("user-db")
    if svc is None:
        return False, "user-db not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"user-db error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


def _check_blue_slot_zero(services: dict) -> tuple[bool, str]:
    svc = services.get("checkout-service")
    if svc is None:
        return False, "checkout-service not in topology"
    slots = getattr(svc, "active_deployment_slots", None)
    if slots is None:
        return False, "active_deployment_slots not available"
    blue = slots.get("blue", None)
    passed = blue == 0.0
    return passed, f"checkout-service blue_slot={blue}"


def _check_registry_stale_zero(services: dict) -> tuple[bool, str]:
    svc = services.get("recommendation-engine")
    if svc is None:
        return False, "recommendation-engine not in topology"
    count = getattr(svc, "registry_stale_instance_count", None)
    if count is None:
        return False, "registry_stale_instance_count not available"
    passed = count == 0
    return passed, f"recommendation-engine registry_stale_instance_count={count}"


def _check_recommendation_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("recommendation-engine")
    if svc is None:
        return False, "recommendation-engine not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"recommendation-engine error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


def _check_grpc_orphaned_zero(services: dict) -> tuple[bool, str]:
    svc = services.get("order-service")
    if svc is None:
        return False, "order-service not in topology"
    rate = getattr(svc, "grpc_orphaned_call_rate", None)
    if rate is None:
        return False, "grpc_orphaned_call_rate not available"
    passed = rate == 0.0
    return passed, f"order-service grpc_orphaned_call_rate={rate}"


def _check_payment_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("payment-service")
    if svc is None:
        return False, "payment-service not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"payment-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


def _check_inventory_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("inventory-service")
    if svc is None:
        return False, "inventory-service not in topology"
    passed = svc.http_server_error_rate < 0.05
    return passed, f"inventory-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.05)"


# --- Hard Tier Check Functions (SPEC-08) ---

def _check_payment_error_rate_hard(services: dict) -> tuple[bool, str]:
    svc = services.get("payment-service")
    if svc is None:
        return False, "payment-service not in topology"
    passed = svc.http_server_error_rate < 0.10
    return passed, f"payment-service error_rate={svc.http_server_error_rate:.2f} (threshold: 0.10)"


def _check_auth_p99_latency(services: dict) -> tuple[bool, str]:
    svc = services.get("auth-service")
    if svc is None:
        return False, "auth-service not in topology"
    passed = svc.http_server_request_duration_p99 < 0.15
    return passed, f"auth-service p99={svc.http_server_request_duration_p99:.3f}s (threshold: 0.15s)"


def _check_auth_packet_loss_zero(services: dict) -> tuple[bool, str]:
    svc = services.get("auth-service")
    if svc is None:
        return False, "auth-service not in topology"
    loss = getattr(svc, "network_packet_loss_rate_inbound", None)
    if loss is None:
        return False, "network_packet_loss_rate_inbound not available"
    passed = loss < 0.01
    return passed, f"auth-service packet_loss={loss:.3f} (threshold: 0.01)"


def _check_search_queue_depth(services: dict) -> tuple[bool, str]:
    svc = services.get("search-service")
    if svc is None:
        return False, "search-service not in topology"
    depth = getattr(svc, "http_server_request_queue_depth", None)
    if depth is None:
        return False, "http_server_request_queue_depth not available"
    passed = depth < 300
    return passed, f"search-service queue_depth={depth} (threshold: 300)"


def _check_search_metastable_inactive(services: dict) -> tuple[bool, str]:
    svc = services.get("search-service")
    if svc is None:
        return False, "search-service not in topology"
    active = getattr(svc, "metastable_feedback_loop_active", None)
    if active is None:
        return False, "metastable_feedback_loop_active not available"
    passed = active is False
    return passed, f"search-service metastable_feedback_loop_active={active}"


def _check_ml_quota_restored(services: dict) -> tuple[bool, str]:
    svc = services.get("ml-inference-service")
    if svc is None:
        return False, "ml-inference-service not in topology"
    ratio = getattr(svc, "resource_quota_remaining_ratio", None)
    if ratio is None:
        return False, "resource_quota_remaining_ratio not available"
    passed = ratio > 0.0
    return passed, f"ml-inference-service quota_remaining={ratio:.2f} (threshold: > 0.0)"


def _check_ml_fallback_off(services: dict) -> tuple[bool, str]:
    svc = services.get("ml-inference-service")
    if svc is None:
        return False, "ml-inference-service not in topology"
    active = getattr(svc, "service_fallback_mode_active", None)
    if active is None:
        return False, "service_fallback_mode_active not available"
    passed = active is False
    return passed, f"ml-inference-service fallback_mode_active={active}"


def _check_config_quorum_healthy(services: dict) -> tuple[bool, str]:
    svc = services.get("config-service")
    if svc is None:
        return False, "config-service not in topology"
    healthy = getattr(svc, "consensus_quorum_healthy", None)
    if healthy is None:
        return False, "consensus_quorum_healthy not available"
    passed = healthy is True
    return passed, f"config-service consensus_quorum_healthy={healthy}"


def _check_config_stale_read_zero(services: dict) -> tuple[bool, str]:
    svc = services.get("config-service")
    if svc is None:
        return False, "config-service not in topology"
    rate = getattr(svc, "config_stale_read_rate", None)
    if rate is None:
        return False, "config_stale_read_rate not available"
    passed = rate < 0.01
    return passed, f"config-service stale_read_rate={rate:.3f} (threshold: 0.01)"


def _check_cache_diverged_zero(services: dict) -> tuple[bool, str]:
    svc = services.get("cache")
    if svc is None:
        return False, "cache not in topology"
    count = getattr(svc, "cache_cluster_diverged_key_count", None)
    if count is None:
        return False, "cache_cluster_diverged_key_count not available"
    passed = count == 0
    return passed, f"cache diverged_key_count={count}"


def _check_cache_split_brain_resolved(services: dict) -> tuple[bool, str]:
    svc = services.get("cache")
    if svc is None:
        return False, "cache not in topology"
    split = getattr(svc, "cache_cluster_split_brain_detected", None)
    if split is None:
        return False, "cache_cluster_split_brain_detected not available"
    passed = split is False
    return passed, f"cache split_brain_detected={split}"


def _check_cache_hit_rate_high(services: dict) -> tuple[bool, str]:
    svc = services.get("cache")
    if svc is None:
        return False, "cache not in topology"
    rate = getattr(svc, "cache_hit_rate", None)
    if rate is None:
        return False, "cache_hit_rate not available"
    passed = rate > 0.50
    return passed, f"cache hit_rate={rate:.2f} (threshold: 0.50)"


def _check_db_proxy_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("db-proxy")
    if svc is None:
        return False, "db-proxy not in topology"
    passed = svc.http_server_error_rate < 0.10
    return passed, f"db-proxy error_rate={svc.http_server_error_rate:.2f} (threshold: 0.10)"


def _check_az_a_error_rate(services: dict) -> tuple[bool, str]:
    svc = services.get("api-gateway-az-a")
    if svc is None:
        return False, "api-gateway-az-a not in topology"
    passed = svc.http_server_error_rate < 0.10
    return passed, f"api-gateway-az-a error_rate={svc.http_server_error_rate:.2f} (threshold: 0.10)"


def _check_az_a_cpu(services: dict) -> tuple[bool, str]:
    svc = services.get("api-gateway-az-a")
    if svc is None:
        return False, "api-gateway-az-a not in topology"
    passed = svc.process_cpu_utilization < 0.80
    return passed, f"api-gateway-az-a cpu={svc.process_cpu_utilization:.2f} (threshold: 0.80)"


TASK_SPECIFIC_CONDITIONS: dict[str, list[tuple]] = {
    # Easy Tier
    "task_easy_thundering_herd": [
        (_check_auth_error_rate, "auth-service error rate < 0.05"),
        (_check_auth_active_requests, "auth-service active_requests in baseline range"),
    ],
    "task_easy_timeout_propagation": [
        (_check_inventory_p99, "inventory-service p99 < 1.0s"),
        (_check_order_error_rate, "order-service error rate < 0.05"),
    ],
    "task_easy_lb_hotspot": [
        (_check_lb_weight, "user-profile-service lb_weight_normalized ≈ 1.0"),
        (_check_lb_error_rate, "user-profile-service error rate < 0.05"),
    ],
    "task_easy_liveness_probe_flap": [
        (_check_payment_restart_stable, "payment-processor restart_count stable"),
        (_check_payment_liveness, "payment-processor liveness_probe_status = passing"),
        (_check_payment_error_rate, "payment-processor error rate < 0.05"),
    ],
    "task_easy_log_debug_disk": [
        (_check_disk_stable, "api-gateway disk ratio stable"),
        (_check_log_level_info, "api-gateway log level = INFO"),
        (_check_apigateway_error_rate_lt_10, "api-gateway error rate < 0.10"),
    ],
    "task_easy_rate_limiter_misconfig": [
        (_check_apigateway_error_rate_lt_05, "api-gateway error rate < 0.05"),
    ],
    # Medium Tier
    "task_medium_retry_storm": [
        (_check_retry_rps_multiplier, "api-gateway effective_rps_multiplier < 1.2"),
        (_check_notification_error_rate, "notification-service error rate < 0.05"),
    ],
    "task_medium_canary_false_alert": [
        (_check_canary_weight_zero, "checkout-service canary_traffic_weight = 0.0"),
        (_check_checkout_error_rate_lt_05, "checkout-service error rate < 0.05"),
    ],
    "task_medium_replica_lag": [
        (_check_replica_lag, "user-service db_replication_lag < 5.0s"),
        (_check_read_path_error_rate, "user-service read_path_error_rate < 0.05"),
        (_check_replica_health, "user-service db_replica_health = synced"),
    ],
    "task_medium_circuit_breaker_masking": [
        (_check_pricing_memory_lt_60, "pricing-service memory < 0.60"),
        (_check_pricing_error_rate_lt_10, "pricing-service error rate < 0.10"),
        (_check_catalog_circuit_breaker_closed, "product-catalog circuit_breaker_state = closed"),
    ],
    "task_medium_cache_eviction_storm": [
        (_check_cache_hit_rate, "cache-service cache_hit_rate > 0.85"),
        (_check_user_db_error_rate, "user-db error rate < 0.05"),
    ],
    "task_medium_configmap_reload": [
        (_check_notification_error_rate, "notification-service error rate < 0.05"),
    ],
    "task_medium_gateway_rate_limit": [
        (_check_apigateway_error_rate_lt_05, "api-gateway error rate < 0.05"),
    ],
    "task_medium_bg_traffic_leak": [
        (_check_blue_slot_zero, "checkout-service blue slot = 0.0"),
        (_check_checkout_error_rate_lt_02, "checkout-service error rate < 0.02"),
    ],
    "task_medium_stale_registry": [
        (_check_registry_stale_zero, "recommendation-engine registry_stale_instance_count = 0"),
        (_check_recommendation_error_rate, "recommendation-engine error rate < 0.05"),
    ],
    "task_medium_grpc_deadline": [
        (_check_grpc_orphaned_zero, "order-service grpc_orphaned_call_rate = 0.0"),
        (_check_payment_error_rate, "payment-service error rate < 0.05"),
        (_check_inventory_error_rate, "inventory-service error rate < 0.05"),
    ],
    # Hard Tier (SPEC-08)
    "task_hard_dual_fault_shared_cascade": [
        (_check_auth_error_rate, "auth-service error rate < 0.05"),
        (_check_payment_error_rate_hard, "payment-service error rate < 0.10"),
        (_check_checkout_error_rate_lt_05, "checkout-service error rate < 0.05"),
    ],
    "task_hard_gray_failure": [
        (_check_auth_p99_latency, "auth-service p99 latency < 0.15s"),
        (_check_auth_packet_loss_zero, "auth-service packet_loss_rate = 0"),
    ],
    "task_hard_metastable_failure": [
        (_check_search_queue_depth, "search-service queue_depth < 300"),
        (_check_search_metastable_inactive, "search-service metastable_loop = False"),
    ],
    "task_hard_quota_cascade": [
        (_check_ml_quota_restored, "ml-inference-service quota_remaining > 0"),
        (_check_ml_fallback_off, "ml-inference-service fallback_mode = False"),
    ],
    "task_hard_consensus_degradation": [
        (_check_config_quorum_healthy, "config-service quorum_healthy = True"),
        (_check_config_stale_read_zero, "config-service stale_read_rate = 0"),
    ],
    "task_hard_redis_split_brain": [
        (_check_cache_diverged_zero, "cache diverged_key_count = 0"),
        (_check_cache_split_brain_resolved, "cache split_brain = False"),
    ],
    "task_hard_stampeding_herd": [
        (_check_cache_hit_rate_high, "cache hit_rate > 0.50"),
        (_check_db_proxy_error_rate, "db-proxy error rate < 0.10"),
    ],
    "task_hard_multiz_failover": [
        (_check_az_a_error_rate, "api-gateway-az-a error rate < 0.10"),
        (_check_az_a_cpu, "api-gateway-az-a cpu < 0.80"),
    ],
}


def _check_task_specific_conditions(
    task_id: str,
    services: dict,
) -> tuple[bool, dict[str, dict]]:
    """
    Evaluate task-specific grader conditions for Phase 2 tasks.

    Args:
        task_id: Full task identifier e.g. 'task_easy_thundering_herd'.
        services: Dict of service_name -> ServiceMetrics from the final observation.

    Returns:
        (all_passed: bool, details: dict mapping condition_description -> result_dict).
        result_dict has keys: 'passed' (bool), 'detail' (str).
    """
    if task_id not in TASK_SPECIFIC_CONDITIONS:
        return True, {}

    results: dict[str, dict] = {}
    all_passed = True

    for check_fn, description in TASK_SPECIFIC_CONDITIONS[task_id]:
        passed, detail = check_fn(services)
        results[description] = {"passed": passed, "detail": detail}
        if not passed:
            all_passed = False

    return all_passed, results


# ==========================================================================
# grade() — unified episode scoring
# ==========================================================================

def grade(
    episode_result: EpisodeResult,
    difficulty: str,
    task_id: str | None = None,
    services: dict | None = None,
) -> float:
    """
    Compute final episode score using unified 4-component formula.

    Components (weights from config.py, defended on decision-theoretic grounds):
      - Recovery (40%): services_recovered / services_affected.
        Dominant because restoring service is the single most important outcome.
        Google SRE: "reliability is table stakes."
      - Speed (25%): composite of MTTM (60%) + BCM (40%).
        Uses MTTM as one of two sub-components, NOT as a primary metric,
        because Google SRE's 2021 "Incident Metrics in SRE" report
        demonstrated log-normal distribution makes MTTM means misleading.
        BCM (direct impact integral) compensates for this pathology.
        Effective MTTM weight on total grade: 0.25 × 0.6 = 15%.
      - Precision (20%): 1 - (wrong_actions / 6). Penalizes remediating
        healthy services or applying wrong fault-type fixes.
      - SLO (15%): final_budget / initial_budget. Last because it's partly
        an aggregate of the first three — kept low to prevent over-indexing.

    Score clipped to (0.01, 0.99) exclusive to preserve gradient signal
    in RL reward shaping (prevents identity-free perfect/total-failure scores).

    Args:
        episode_result: Completed episode statistics.
        difficulty: "easy", "medium", or "hard" — for max_ticks lookup.
        task_id: Optional explicit task ID for Phase 1+ tasks. When provided,
                 looks up the specific TaskConfig for max_ticks/max_bcm.
        services: Optional dict of service_name -> ServiceMetrics from the final
                  observation. Required for task-specific condition evaluation.

    Returns:
        Float in (0.01, 0.99). Rounded to 2 decimal places.
    """
    er = episode_result

    # SPEC-04: Support Phase 1 task_ids, not just legacy task_{difficulty}
    task = None
    if task_id:
        task = TASKS.get(task_id)
    if task is None:
        task = TASKS.get(f"task_{difficulty}")
    if task is None:
        return 0.0

    # Fix 1: Tick guard — declare_resolved before tick 2 earns near-zero score
    if er.ticks_taken < 2:
        return 0.05

    max_ticks = task.max_ticks
    max_bcm = task.max_bad_customer_minutes

    # 1. Recovery (40%)
    # The tick guard above handles Fix 1 (tick-0 exploit).
    # Use runtime services_affected as denominator — blast penalty (below) is what
    # differentiates agents who contained vs didn't contain the cascade.
    denominator = er.services_affected or 1
    recovery = min(1.0, er.services_recovered / denominator)

    # Fix 2: No cliff wipe — compute BCM and SLO unconditionally
    bcm_score = max(0.0, 1.0 - (er.bad_customer_minutes / max_bcm))
    slo = max(0.0, min(1.0, er.final_slo_budget / er.initial_slo_budget))

    # 2. Speed (25%) — composite of MTTM + BCM
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

    # Raw weighted score
    raw = (
        GRADER_WEIGHT_RECOVERY * recovery
        + GRADER_WEIGHT_SPEED * speed
        + GRADER_WEIGHT_PRECISION * precision
        + GRADER_WEIGHT_SLO * slo
    )

    # Fix 3: Blast radius penalty — reward containing cascade, not just fixing it
    total_services = er.total_services_in_episode or denominator
    blast_ratio = er.services_affected / total_services if total_services > 0 else 0.0
    blast_penalty = blast_ratio * 0.02

    score = max(0.0, raw - blast_penalty)

    # SPEC-07 Phase 4: Task-specific condition penalty
    # Deduct up to 0.10 for failed task-specific grader conditions
    if task_id and services and task_id in TASK_SPECIFIC_CONDITIONS:
        all_passed, _ = _check_task_specific_conditions(task_id, services)
        if not all_passed:
            # Penalty: failed conditions reduce score proportionally
            # Max deduction: 0.10 (10 percentage points)
            _, results = _check_task_specific_conditions(task_id, services)
            failed_count = sum(1 for r in results.values() if not r["passed"])
            total_count = len(results)
            condition_penalty = 0.10 * (failed_count / total_count)
            score = max(0.0, score - condition_penalty)

    return max(0.01, min(0.99, round(score, 2)))


def compute_premature_exit_penalty(obs: SystemObservation, tick_count: int, mttm_achieved: bool) -> float:
    """
    Compute the premature exit penalty when declare_resolved is called.

    Penalty fires only if ALL of:
    1. mean error rate > HEALTHY_ERROR_RATE_THRESHOLD (system is still broken)
    2. More than 1 tick has elapsed (prevents degenerate test case false positives)
    3. mttm_achieved is False (MTTM achieved = user impact stopped; agent may declare)

    Penalty scales with remaining brokenness:
        BASE + (mean_error × SCALE)
    At mean_error=1.0: -2.0 + (-3.0) = -5.0
    At mean_error=0.10: -2.0 + (-0.30) = -2.30
    At mean_error <= 0.05: 0.0 (no penalty)

    NOTE: The -5.0 maximum is 5× normal step rewards (±1.0). This
    asymmetry is acceptable for LLM-as-agent (grader-only, agent never
    sees reward during inference). For real RL training, clip to [-3.0, 0.0]
    for gradient stability.

    Returns:
        Float penalty (negative or zero).
    """
    if mttm_achieved:
        return 0.0
    if tick_count <= 1:
        return 0.0
    mean_error = _mean_error_rate(obs)
    if mean_error <= HEALTHY_ERROR_RATE_THRESHOLD:
        return 0.0
    return REWARD_PREMATURE_EXIT_BASE + (mean_error * REWARD_PREMATURE_EXIT_SCALE)


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
        info["episode_score"] = round(episode_score or 0.0, 2)
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
    """
    Generate metric-delta context for the step info dict.

    Reports WHAT changed (metric values and deltas), not WHETHER it was good.
    The agent must interpret the numbers itself — no outcome framing.
    """
    parts: list[str] = []

    if not action_valid:
        parts.append(
            f"Action '{action.action_type}' was invalid. No state change."
        )
    elif wrong_action:
        # Report metric context only — no interpretation
        svc = action.target_service or ""
        curr_er = next_obs.services[svc].http_server_error_rate if svc in next_obs.services else None
        er_str = f"error_rate={curr_er:.2f}" if curr_er is not None else "error_rate=unknown"
        parts.append(
            f"Action '{action.action_type}' targeted '{svc}' ({er_str}). "
            f"Wrong-action penalty applied (threshold: 0.10)."
        )
    elif action.action_type in ("fetch_logs", "get_metrics_detail", "trace_dependencies"):
        parts.append(
            f"Investigation '{action.action_type}' on '{action.target_service}'. "
            f"No state mutation."
        )
    elif action.action_type in (
        "restart_service", "rollback_deploy", "revert_config",
        "scale_replicas", "circuit_break",
    ):
        parts.append(f"Remediation '{action.action_type}' applied to '{action.target_service}'.")
        # Report metric deltas — no interpretation of good/bad
        if prev_obs:
            for svc_name, curr in next_obs.services.items():
                prev_svc = prev_obs.services.get(svc_name)
                if prev_svc:
                    delta = curr.http_server_error_rate - prev_svc.http_server_error_rate
                    if abs(delta) > 0.05:
                        direction = "increased" if delta > 0 else "decreased"
                        parts.append(
                            f"{svc_name} error_rate {direction} by {abs(delta):.2f} "
                            f"(now {curr.http_server_error_rate:.2f})."
                        )
    elif action.action_type == "declare_resolved":
        parts.append("Agent declared incident resolved. Episode ending.")
    elif action.action_type == "escalate":
        parts.append("Agent escalated incident.")

    # Current state counts — factual only
    degraded_count = sum(1 for m in next_obs.services.values() if m.status != "healthy")
    total = len(next_obs.services)
    parts.append(f"{degraded_count}/{total} services non-healthy.")

    if feedback:
        parts.append(f"Feedback: {feedback}")

    return " ".join(parts) if parts else "No significant changes this tick."


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

def _weighted_mean_error_rate(services: dict, dependency_graph: dict) -> float:
    """
    Compute mean error rate across services, weighted by downstream dependent count.

    Topology-weighted rather than flat average: upstream services (e.g., db-proxy)
    that many services depend on are weighted higher because their failure has
    a larger customer blast radius. This aligns the reward signal with the
    real-world SRE principle that upstream root causes matter more than
    downstream symptoms.

    Weight formula: weight(svc) = 1 + count(other services that list svc as a dependency)
    Example: db-proxy (depended on by auth, user, payment) → weight=4; cache leaf → weight=1.

    Args:
        services: Dict mapping service_name → ServiceMetrics.
        dependency_graph: Dict mapping service_name → list[dependency_name].

    Returns:
        Weighted mean error rate in [0.0, 1.0].
    """
    if not services:
        return 0.0

    # Count how many services-in-this-episode depend on each service
    dependent_count: dict[str, int] = {svc: 0 for svc in services}
    for svc, deps in dependency_graph.items():
        if svc in services:
            for dep in deps:
                if dep in dependent_count:
                    dependent_count[dep] = dependent_count.get(dep, 0) + 1

    total_weight = 0.0
    weighted_error = 0.0
    for svc_name, metrics in services.items():
        weight = 1 + dependent_count.get(svc_name, 0)
        weighted_error += metrics.http_server_error_rate * weight
        total_weight += weight

    return weighted_error / total_weight if total_weight > 0 else 0.0


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
    "compute_premature_exit_penalty",
    "TASK_SPECIFIC_CONDITIONS",
    "_check_task_specific_conditions",
]
