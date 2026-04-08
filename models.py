# models.py
# Phase 2 — All Pydantic models for FirewatchEnv.
# Every field has explicit type annotations. No Any (except FirewatchAction.parameters).
# Field names follow OpenTelemetry semantic conventions.
#
# Models defined here:
#   1. ServiceMetrics — per-service telemetry snapshot (21 OTel fields)
#   2. Alert — Prometheus Alertmanager-format alert
#   3. SystemObservation — complete observable state (returned by reset/step/state)
#   4. FirewatchAction — agent command with strict Literal action_type
#   5. ActionResult — structured result of an action
#   6. derive_status() — utility to compute status from metric thresholds

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# OpenEnv base types — provide done, reward, metadata fields
# required by the HTTP server's serialize_observation() and deserialize_action()
try:
    from openenv.core.env_server.types import (
        Observation as _ObservationBase,
        Action as _ActionBase,
    )
except ImportError:
    # Fallback for environments where openenv-core is not installed
    _ObservationBase = BaseModel  # type: ignore[assignment,misc]
    _ActionBase = BaseModel  # type: ignore[assignment,misc]

try:
    from .config import (
        STATUS_THRESHOLD_CRITICAL_ERROR,
        STATUS_THRESHOLD_CRITICAL_LATENCY,
        STATUS_THRESHOLD_DEGRADED_ERROR,
        STATUS_THRESHOLD_DEGRADED_LATENCY,
        STATUS_THRESHOLD_DOWN_ERROR,
        STATUS_THRESHOLD_DOWN_MEMORY,
    )
except ImportError:
    from config import (
        STATUS_THRESHOLD_CRITICAL_ERROR,
        STATUS_THRESHOLD_CRITICAL_LATENCY,
        STATUS_THRESHOLD_DEGRADED_ERROR,
        STATUS_THRESHOLD_DEGRADED_LATENCY,
        STATUS_THRESHOLD_DOWN_ERROR,
        STATUS_THRESHOLD_DOWN_MEMORY,
    )


# --------------------------------------------------------------------------
# Type aliases for readability
# --------------------------------------------------------------------------

ServiceStatus = Literal["healthy", "degraded", "critical", "down"]

AlertName = Literal[
    "HighErrorRate",
    "HighLatency",
    "MemoryPressure",
    "HighCPU",
    "ServiceDown",
    "RequestBacklog",
]

AlertSeverity = Literal["warning", "critical", "page"]

ActionType = Literal[
    # Investigation actions — reveal information, no state mutation
    "fetch_logs",
    "get_metrics_detail",
    "trace_dependencies",
    # Advanced diagnostic investigation actions (SPEC-9)
    "strace_process",
    "profiler_dump",
    "check_gc_pressure",
    "trace_distributed_request",
    "inspect_thread_pool",
    "inspect_commit_diff",
    # Remediation actions — mutate system state
    "restart_service",
    "rollback_deploy",
    "revert_config",
    "scale_replicas",
    "circuit_break",
    # Advanced remediation actions (SPEC-9)
    "traffic_shift",
    # Meta actions — episode control
    "declare_resolved",
    "escalate",
]


# --------------------------------------------------------------------------
# ServiceMetrics — per-service telemetry (replaces Phase 1 ServiceSnapshot)
# --------------------------------------------------------------------------

class ServiceMetrics(BaseModel):
    """
    Complete telemetry snapshot for one microservice.

    All metric field names follow OpenTelemetry semantic conventions.
    Underscore naming is the Pydantic convention; each field documents
    the corresponding OTel dot-notation name.

    Status is NOT auto-computed — the simulation sets it explicitly
    via derive_status() after mutating metrics each tick.
    """

    # --- Resource attributes (OTel resource) ---
    service_name: str = Field(
        ..., description="OTel: service.name. e.g. 'payment-service'"
    )
    service_version: str = Field(
        default="v1.0.0", description="OTel: service.version"
    )
    service_instance_id: str = Field(
        ..., description="OTel: service.instance.id. e.g. 'payment-7d9f8b-xkp2m'"
    )

    # --- Derived status ---
    status: ServiceStatus = Field(
        default="healthy",
        description="Derived from metric thresholds. Set by simulation via derive_status().",
    )

    # --- HTTP server metrics (OTel stable) ---
    http_server_request_duration_p99: float = Field(
        default=0.1,
        description="OTel: http.server.request.duration p99 bucket. Unit: seconds. Healthy: 0.05–0.5.",
    )
    http_server_error_rate: float = Field(
        default=0.0,
        description="Derived from OTel http.response.status_code 5xx ratio. Unit: ratio 0.0–1.0.",
    )
    http_server_active_requests: int = Field(
        default=50,
        description="OTel: http.server.active_requests. Unit: {request}. Normal: 1–200.",
    )

    # --- Process metrics (OTel) ---
    process_cpu_utilization: float = Field(
        default=0.15,
        description="OTel: process.cpu.utilization. Unit: ratio 0.0–1.0 (NOT percentage).",
    )
    process_memory_usage_bytes: int = Field(
        default=178257920,
        description="OTel: process.memory.usage. Unit: bytes. ~170MB default.",
    )
    process_memory_limit_bytes: int = Field(
        default=536870912,
        description="Container config, not OTel-emitted. Unit: bytes. 512MB default.",
    )
    process_memory_utilization: float = Field(
        default=0.33,
        description="Derived: usage_bytes / limit_bytes. Can exceed 1.0 before OOMKill.",
    )
    process_open_file_descriptors: int = Field(
        default=120,
        description="OTel: process.open_file_descriptor.count. High = connection exhaustion.",
    )

    # --- Runtime performance metrics (JVM/V8/Go runtime) ---
    runtime_gc_pause_duration_ms: float = Field(
        default=15.0,
        description=(
            "OTel: runtime.{language}.gc.pause.duration. "
            "Unit: milliseconds. Stop-the-world GC pause time. "
            "Healthy: <50ms. Critical: >500ms."
        ),
    )
    runtime_gc_count_per_second: float = Field(
        default=2.0,
        description=(
            "OTel: runtime.{language}.gc.count (rate). "
            "Unit: {gc}/s. GC cycles per second. "
            "Healthy: <5. Thrashing: >30."
        ),
    )
    runtime_jvm_threads_count: int = Field(
        default=50,
        description=(
            "OTel: runtime.jvm.threads.count. "
            "Unit: {thread}. Active threads. "
            "Saturated when == max_threads."
        ),
    )
    runtime_jvm_threads_max: int = Field(
        default=200,
        description=(
            "OTel: Configured max thread pool size. "
            "Saturation = threads_count >= threads_max."
        ),
    )
    runtime_thread_pool_queue_depth: int = Field(
        default=0,
        description=(
            "OTel-adjacent: Pending requests in thread pool queue. "
            "High value = backpressure, head-of-line blocking."
        ),
    )

    # --- Runtime / deployment metadata ---
    runtime_uptime_seconds: int = Field(
        default=86400,
        description="OTel: process.runtime.uptime. Resets to 0 on restart. 24h default.",
    )
    restart_count: int = Field(
        default=0,
        description="OTel-adjacent: k8s.container.restart_count. Increments on OOMKill.",
    )
    last_deployment_sha: str = Field(
        default="a3f9d21",
        description="Short git SHA of last deployment.",
    )
    last_deployment_age_seconds: int = Field(
        default=172800,
        description="Seconds since last deployment. Low = recent deploy = suspect for bad_deploy.",
    )
    last_config_revision: int = Field(
        default=1,
        description="Monotonically increasing config revision number.",
    )
    last_config_age_seconds: int = Field(
        default=259200,
        description="Seconds since last config change. Low = suspect for config_drift.",
    )

    # --- Logs (populated only after fetch_logs action) ---
    recent_logs: list[str] = Field(
        default_factory=list,
        description="Empty by default. Populated by fetch_logs action. Last 20 log lines.",
    )


# --------------------------------------------------------------------------
# Alert — Prometheus Alertmanager format
# --------------------------------------------------------------------------

class Alert(BaseModel):
    """
    Alert following Prometheus Alertmanager payload conventions.
    Generated by the simulation when metric thresholds are breached.
    Resolves automatically when metric returns below threshold.
    """

    alert_id: str = Field(
        ..., description="Short UUID. e.g. 'a1b2c3d4'"
    )
    alertname: AlertName = Field(
        ..., description="Human-readable alert name."
    )
    service_name: str = Field(
        ..., description="Which service triggered the alert."
    )
    severity: AlertSeverity = Field(
        ..., description="Severity level."
    )
    description: str = Field(
        ...,
        description=(
            "Human-readable description. Format: "
            "'<metric> is <value> (threshold: <threshold>) on <service> for <n> ticks'"
        ),
    )
    fired_at_tick: int = Field(
        ..., description="Simulation tick when the threshold was crossed."
    )
    metric_name: str = Field(
        ..., description="The OTel metric name that breached threshold."
    )
    metric_value: float = Field(
        ..., description="Current value at time of firing."
    )
    threshold_value: float = Field(
        ..., description="The configured threshold that was crossed."
    )


# --------------------------------------------------------------------------
# SystemObservation — complete observable state
# --------------------------------------------------------------------------

class SystemObservation(_ObservationBase):
    """
    Complete observable state returned by reset(), step(), and state().
    The agent receives this after every action.

    Inherits from openenv Observation which provides:
      - done: bool (episode terminated)
      - reward: float | None (step reward)
      - metadata: dict (additional info dict)
    """

    services: dict[str, ServiceMetrics] = Field(
        default_factory=dict,
        description="Per-service metrics keyed by service_name. Subset of full topology.",
    )
    active_alerts: list[Alert] = Field(
        default_factory=list,
        description="Currently firing alerts. Auto-resolve when metric recovers.",
    )
    dependency_graph: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Static topology for this episode. Does not change between ticks.",
    )
    slo_budget_remaining_pct: float = Field(
        default=100.0,
        description="Error budget %. Starts at 100.0, depletes per tick. 0.0 = episode over.",
    )
    bad_customer_minutes: float = Field(
        default=0.0,
        description="Cumulative user impact. Google SRE MTTM measurement.",
    )
    sim_time_elapsed_seconds: int = Field(
        default=0,
        description="Simulated seconds since episode start. 30s per tick.",
    )
    sim_tick: int = Field(
        default=0,
        description="Current tick number. Starts at 0 after reset().",
    )
    action_history: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Last 10 actions. Each entry: "
            "{action_type, target_service, feedback_string}."
        ),
    )
    incident_declared: bool = Field(
        default=False,
        description="True if agent called declare_resolved. Terminal condition.",
    )
    mttm_achieved_tick: int | None = Field(
        default=None,
        description="Tick when user impact first reached zero. None until achieved.",
    )
    episode_score: float | None = Field(
        default=None,
        description="Final grader score in (0.0, 1.0) exclusive. Set only when done=True.",
    )


# --------------------------------------------------------------------------
# FirewatchAction — agent command
# --------------------------------------------------------------------------

class FirewatchAction(_ActionBase):
    """
    Agent action. action_type is strictly validated against 10 allowed values.
    Unknown action_types are rejected with Pydantic ValidationError.
    The environment catches ValidationError and returns a graceful error response.

    Inherits from openenv Action which provides:
      - metadata: dict (additional action metadata)
    """

    action_type: ActionType = Field(
        ..., description="SRE command to execute."
    )
    target_service: str | None = Field(
        default=None,
        description="service_name to target. Required for all except declare_resolved/escalate.",
    )
    parameters: dict[str, object] = Field(
        default_factory=dict,
        description="Optional action params. e.g. {'memory_limit_mb': 1024} for scale_replicas.",
    )


# --------------------------------------------------------------------------
# ActionResult — structured action feedback
# --------------------------------------------------------------------------

class ActionResult(BaseModel):
    """
    Structured result of an agent action.
    Included in the info dict returned by every step() call.
    """

    valid: bool = Field(
        ..., description="Whether the action was valid and executed."
    )
    feedback: str = Field(
        ..., description="Human-readable feedback about what happened."
    )
    action_type: str = Field(
        default="", description="Echo of the action_type that was executed."
    )
    target_service: str | None = Field(
        default=None, description="Echo of the target_service."
    )


# --------------------------------------------------------------------------
# Status derivation utility
# --------------------------------------------------------------------------

def derive_status(
    error_rate: float,
    latency_p99: float,
    memory_utilization: float,
) -> ServiceStatus:
    """
    Compute service status from metric values.

    Applied in priority order: down → critical → degraded → healthy.
    Thresholds sourced from config.py (PRD §7.2).

    The simulation calls this after mutating metrics each tick to update
    the status field. It is NOT auto-computed on model access because the
    simulation needs explicit control over when status updates happen.
    """
    if (
        error_rate >= STATUS_THRESHOLD_DOWN_ERROR
        or memory_utilization >= STATUS_THRESHOLD_DOWN_MEMORY
    ):
        return "down"

    if (
        error_rate >= STATUS_THRESHOLD_CRITICAL_ERROR
        or latency_p99 >= STATUS_THRESHOLD_CRITICAL_LATENCY
    ):
        return "critical"

    if (
        error_rate >= STATUS_THRESHOLD_DEGRADED_ERROR
        or latency_p99 >= STATUS_THRESHOLD_DEGRADED_LATENCY
    ):
        return "degraded"

    return "healthy"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

__all__ = [
    "ServiceMetrics",
    "Alert",
    "SystemObservation",
    "FirewatchAction",
    "ActionResult",
    "ActionType",
    "AlertName",
    "AlertSeverity",
    "ServiceStatus",
    "derive_status",
]