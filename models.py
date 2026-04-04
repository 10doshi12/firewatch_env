# models.py
# Phase 1 stub — minimum typed models to pass openenv validate.
# All fields have explicit type annotations. No Any. No untyped fields.
# Phase 2 expands every model with full field specifications.

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stub sub-models
# Defined here so services and active_alerts are fully typed (no bare dict/list)
# ---------------------------------------------------------------------------

class ServiceSnapshot(BaseModel):
    """
    Minimal typed snapshot of one service's metrics.
    Expanded to full ServiceMetrics in Phase 2 with all OTel fields.
    """
    status: str = "healthy"
    http_server_error_rate: float = 0.0
    http_server_request_duration_p99: float = 0.1
    process_memory_utilization: float = 0.0
    process_cpu_utilization: float = 0.0
    restart_count: int = 0
    recent_logs: list[str] = Field(default_factory=list)


class AlertSnapshot(BaseModel):
    """
    Minimal typed alert entry following Prometheus Alertmanager conventions.
    Expanded to full Alert model in Phase 2.
    """
    alert_id: str
    alertname: str
    service_name: str
    severity: str
    description: str
    fired_at_tick: int = 0


# ---------------------------------------------------------------------------
# Core exported models
# ---------------------------------------------------------------------------

class FirewatchAction(BaseModel):
    """
    Agent action. action_type must be one of the 10 valid action strings.
    Literal constraint added in Phase 2 once all action types are confirmed.
    target_service is required for all actions except declare_resolved and escalate.
    """
    action_type: str
    target_service: str | None = None
    parameters: dict[str, str] = Field(default_factory=dict)


class SystemObservation(BaseModel):
    """
    Complete observable state of the simulated production environment.
    Returned by reset(), step(), and state().
    services is keyed by service_name.
    """
    services: dict[str, ServiceSnapshot] = Field(default_factory=dict)
    active_alerts: list[AlertSnapshot] = Field(default_factory=list)
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    slo_budget_remaining_pct: float = 100.0
    bad_customer_minutes: float = 0.0
    sim_time_elapsed_seconds: int = 0
    sim_tick: int = 0
    action_history: list[str] = Field(default_factory=list)
    incident_declared: bool = False
    mttm_achieved_tick: int | None = None


class ActionResult(BaseModel):
    """
    Structured result of an agent action.
    Included in the info dict returned by every step() call.
    """
    valid: bool
    feedback: str
    action_type: str = ""
    target_service: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FirewatchAction",
    "SystemObservation",
    "ActionResult",
    "ServiceSnapshot",
    "AlertSnapshot",
]