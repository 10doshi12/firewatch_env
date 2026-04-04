# actions.py
# Phase 5 — Action Handler. Maps all 10 action types to ServiceMesh mutations.
# Returns structured feedback strings. Never crashes on any input.
#
# Import hierarchy: actions.py imports models.py, config.py, simulation.py

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from .models import FirewatchAction, ActionResult
    from .config import (
        HEALTHY_ERROR_RATE_THRESHOLD,
        FULL_DEPENDENCY_GRAPH,
        STATUS_THRESHOLD_DEGRADED_ERROR,
        SLO_BURN_RATE_BY_DIFFICULTY,
        SECONDS_PER_TICK,
    )
except ImportError:
    from models import FirewatchAction, ActionResult
    from config import (
        HEALTHY_ERROR_RATE_THRESHOLD,
        FULL_DEPENDENCY_GRAPH,
        STATUS_THRESHOLD_DEGRADED_ERROR,
        SLO_BURN_RATE_BY_DIFFICULTY,
        SECONDS_PER_TICK,
    )

if TYPE_CHECKING:
    from .simulation import ServiceMesh, FaultConfig


class ActionHandler:
    """
    Maps FirewatchAction commands to ServiceMesh state mutations.

    One primary method: apply() — takes an action, mesh, and fault config,
    returns a feedback string and a wrong_action flag.

    Design principles:
    - Investigation actions reveal info, never mutate state
    - Remediation on wrong service or wrong fault type = no effect on fault
    - Remediating a healthy service (error_rate < threshold) = wrong action
    - Never crashes on any input
    """

    def __init__(self) -> None:
        # Track metric history for get_metrics_detail (last 3 ticks)
        self._metric_history: dict[str, list[dict[str, float]]] = {}
        # Track active circuit breakers: {service_name: ticks_remaining}
        self._circuit_breakers: dict[str, int] = {}

    def record_tick(self, mesh: "ServiceMesh") -> None:
        """Record current metrics for history tracking. Call after each tick."""
        for name, m in mesh.services.items():
            if name not in self._metric_history:
                self._metric_history[name] = []
            self._metric_history[name].append({
                "error_rate": round(m.http_server_error_rate, 4),
                "latency_p99": round(m.http_server_request_duration_p99, 4),
                "memory_util": round(m.process_memory_utilization, 4),
                "cpu_util": round(m.process_cpu_utilization, 4),
            })
            # Keep only last 5 entries
            if len(self._metric_history[name]) > 5:
                self._metric_history[name] = self._metric_history[name][-5:]

        # Decrement circuit breakers
        expired = []
        for svc, ticks in self._circuit_breakers.items():
            self._circuit_breakers[svc] = ticks - 1
            if self._circuit_breakers[svc] <= 0:
                expired.append(svc)
        for svc in expired:
            del self._circuit_breakers[svc]

    def apply(
        self,
        action: FirewatchAction,
        mesh: "ServiceMesh",
        fault_config: "FaultConfig",
    ) -> tuple[str, bool]:
        """
        Apply an action to the service mesh.

        Returns:
            Tuple of (feedback_string, was_wrong_action).
            feedback_string is human-readable for agent and LLM judge.
            was_wrong_action is True if agent remediated a healthy service.
        """
        at = action.action_type
        target = action.target_service

        # --- Meta actions (no target required) ---
        if at == "declare_resolved":
            return self._declare_resolved(mesh)

        if at == "escalate":
            return self._escalate(mesh)

        # --- All other actions require target_service ---
        if target is None:
            return (
                f"Action '{at}' requires a target_service. No action taken.",
                False,
            )

        if target not in mesh.services:
            return (
                f"Invalid target: '{target}' is not an active service in this "
                f"episode. Active services: {list(mesh.services.keys())}. "
                f"No action taken.",
                False,
            )

        # --- Investigation actions ---
        if at == "fetch_logs":
            return self._fetch_logs(target, mesh, fault_config)

        if at == "get_metrics_detail":
            return self._get_metrics_detail(target, mesh)

        if at == "trace_dependencies":
            return self._trace_dependencies(target, mesh)

        # --- Remediation actions ---
        # Check for wrong action: remediating a healthy service
        target_metrics = mesh.services[target]
        is_wrong = target_metrics.http_server_error_rate < STATUS_THRESHOLD_DEGRADED_ERROR

        if at == "restart_service":
            return self._restart_service(target, mesh, fault_config, is_wrong)

        if at == "rollback_deploy":
            return self._rollback_deploy(target, mesh, fault_config, is_wrong)

        if at == "revert_config":
            return self._revert_config(target, mesh, fault_config, is_wrong)

        if at == "scale_replicas":
            return self._scale_replicas(target, mesh, fault_config, action, is_wrong)

        if at == "circuit_break":
            return self._circuit_break(target, mesh, fault_config, is_wrong)

        return (f"Unknown action type: {at}. No action taken.", False)

    # ------------------------------------------------------------------
    # Investigation actions
    # ------------------------------------------------------------------

    def _fetch_logs(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig"
    ) -> tuple[str, bool]:
        """Populate recent_logs on target service."""
        logs = mesh.get_logs_for_service(target)
        mesh.services[target].recent_logs = logs
        return (
            f"Fetched {len(logs)} log lines for {target}. "
            f"Review recent_logs in observation.",
            False,
        )

    def _get_metrics_detail(
        self, target: str, mesh: "ServiceMesh"
    ) -> tuple[str, bool]:
        """Return metric trend over last 3 ticks."""
        history = self._metric_history.get(target, [])
        svc = mesh.services[target]

        if len(history) < 2:
            return (
                f"{target}: error_rate={svc.http_server_error_rate:.4f}, "
                f"latency_p99={svc.http_server_request_duration_p99:.3f}s, "
                f"memory_util={svc.process_memory_utilization:.2f}, "
                f"cpu_util={svc.process_cpu_utilization:.2f}. "
                f"Insufficient history for trend analysis (need 2+ ticks).",
                False,
            )

        # Last 3 entries
        recent = history[-3:] if len(history) >= 3 else history

        error_trend = "→".join(f"{h['error_rate']:.2f}" for h in recent)
        latency_trend = "→".join(f"{h['latency_p99']:.2f}" for h in recent)
        memory_trend = "→".join(f"{h['memory_util']:.2f}" for h in recent)

        # Detect trend pattern
        errors = [h["error_rate"] for h in recent]
        if len(errors) >= 2:
            if errors[-1] > errors[0] * 1.2:
                pattern = "Pattern suggests active fault propagation, not transient spike."
            elif errors[-1] < errors[0] * 0.8:
                pattern = "Pattern suggests recovery in progress."
            else:
                pattern = "Metrics stable — no clear degradation trend."
        else:
            pattern = ""

        feedback = (
            f"{target}: error_rate trended {error_trend} over last "
            f"{len(recent)} ticks. latency_p99 trended {latency_trend}. "
            f"memory_utilization trended {memory_trend}. {pattern}"
        )
        return (feedback, False)

    def _trace_dependencies(
        self, target: str, mesh: "ServiceMesh"
    ) -> tuple[str, bool]:
        """Return upstream and downstream dependency chains."""
        graph = mesh.dependency_graph

        # Downstream: services that `target` calls
        downstream = graph.get(target, [])

        # Upstream: services that call `target`
        upstream = [
            svc for svc, deps in graph.items()
            if target in deps
        ]

        # Build extended chains
        downstream_detail = []
        for d in downstream:
            status = mesh.services[d].status if d in mesh.services else "unknown"
            downstream_detail.append(f"{d} (status: {status})")

        upstream_detail = []
        for u in upstream:
            status = mesh.services[u].status if u in mesh.services else "unknown"
            upstream_detail.append(f"{u} (status: {status})")

        feedback = (
            f"{target} dependency analysis: "
            f"Calls (downstream): [{', '.join(downstream_detail) or 'none'}]. "
            f"Called by (upstream): [{', '.join(upstream_detail) or 'none'}]. "
        )

        # Add insight about cascade direction
        svc = mesh.services[target]
        if svc.status != "healthy" and upstream:
            upstream_with_issues = [
                u for u in upstream
                if u in mesh.services and mesh.services[u].status != "healthy"
            ]
            if upstream_with_issues:
                feedback += (
                    f"Note: upstream services {upstream_with_issues} are also "
                    f"degraded — investigate whether {target} is a victim of "
                    f"upstream fault propagation."
                )

        return (feedback, False)

    # ------------------------------------------------------------------
    # Remediation actions
    # ------------------------------------------------------------------

    def _restart_service(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
        is_wrong: bool,
    ) -> tuple[str, bool]:
        """Restart target service."""
        svc = mesh.services[target]

        if is_wrong:
            return (
                f"Restarted {target} (status was {svc.status}, error_rate "
                f"{svc.http_server_error_rate:.4f}). Service was not significantly "
                f"degraded — this may be a premature remediation.",
                True,
            )

        if target == fc.root_cause_service and fc.fault_type == "oom":
            # Correct: restart temporarily fixes OOM
            svc.process_memory_utilization = 0.20
            svc.process_memory_usage_bytes = int(
                0.20 * svc.process_memory_limit_bytes
            )
            svc.http_server_error_rate = max(0.0, svc.http_server_error_rate - 0.5)
            svc.http_server_request_duration_p99 = max(0.1, svc.http_server_request_duration_p99 * 0.3)
            svc.runtime_uptime_seconds = 0
            svc.restart_count += 1
            svc.status = "degraded"
            # Note: does NOT set fault_halted — OOM will recur without scale_replicas
            return (
                f"Restarted {target}. Memory utilization reset to 20%. "
                f"Error rate reduced. Warning: OOM root cause not resolved — "
                f"memory will accumulate again. Consider scale_replicas to "
                f"increase memory limit.",
                False,
            )

        if target == fc.root_cause_service and fc.fault_type == "memory_leak":
            # Partially effective: restart resets memory but leak continues
            svc.process_memory_utilization = 0.25
            svc.process_memory_usage_bytes = int(
                0.25 * svc.process_memory_limit_bytes
            )
            svc.http_server_error_rate = max(0.0, svc.http_server_error_rate - 0.3)
            svc.http_server_request_duration_p99 = max(0.1, svc.http_server_request_duration_p99 * 0.5)
            svc.runtime_uptime_seconds = 0
            svc.restart_count += 1
            return (
                f"Restarted {target}. Memory reset temporarily. Warning: "
                f"memory leak will continue — this buys time but does not "
                f"fix the root cause.",
                False,
            )

        # Wrong remediation type for this fault (but service is degraded)
        svc.restart_count += 1
        svc.runtime_uptime_seconds = 0
        return (
            f"Restarted {target}. Service restarted but underlying issue "
            f"persists (fault type is not OOM). Restart has no effect on "
            f"the active fault.",
            False,
        )

    def _rollback_deploy(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
        is_wrong: bool,
    ) -> tuple[str, bool]:
        """Rollback deployment on target service."""
        svc = mesh.services[target]

        if is_wrong:
            return (
                f"Rolled back deployment on {target} (error_rate "
                f"{svc.http_server_error_rate:.4f}). Service was not "
                f"significantly degraded — unnecessary rollback.",
                True,
            )

        prev_sha = "".join(
            chr(ord("a") + (ord(c) - ord("0")) % 6) if c.isdigit() else c
            for c in svc.last_deployment_sha
        )[:7]

        if target == fc.root_cause_service and fc.fault_type == "bad_deploy":
            # Correct: halt fault progression
            mesh.fault_halted = True
            svc.last_deployment_sha = prev_sha
            svc.last_deployment_age_seconds = 172800  # Reset to old deploy age
            # Error rate starts declining
            svc.http_server_error_rate = max(0.0, svc.http_server_error_rate * 0.5)
            svc.http_server_request_duration_p99 = max(
                0.1, svc.http_server_request_duration_p99 * 0.5
            )
            return (
                f"Rollback initiated for {target}. Reverting to sha: "
                f"{prev_sha}. Error rate declining — fault progression halted.",
                False,
            )

        return (
            f"Rolled back deployment on {target} to sha: {prev_sha}. "
            f"However, the active fault is not a bad deployment — this "
            f"rollback had no effect on fault progression.",
            False,
        )

    def _revert_config(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
        is_wrong: bool,
    ) -> tuple[str, bool]:
        """Revert configuration on target service."""
        svc = mesh.services[target]

        if is_wrong:
            return (
                f"Reverted config on {target} (error_rate "
                f"{svc.http_server_error_rate:.4f}). Service was not "
                f"significantly degraded — unnecessary config revert.",
                True,
            )

        if target == fc.root_cause_service and fc.fault_type == "config_drift":
            # Correct: restore connection pool
            mesh.fault_halted = True
            svc.process_open_file_descriptors = 120  # Normal range
            svc.http_server_request_duration_p99 = max(
                0.1, svc.http_server_request_duration_p99 * 0.2
            )
            svc.http_server_error_rate = max(0.0, svc.http_server_error_rate * 0.4)
            svc.last_config_age_seconds = 0
            svc.last_config_revision += 1
            return (
                f"Config reverted for {target}. Connection pool restored "
                f"to default limits. Latency returning to normal.",
                False,
            )

        return (
            f"Reverted config on {target}. However, the active fault is "
            f"not a config drift issue — this had no effect on fault "
            f"progression.",
            False,
        )

    def _scale_replicas(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
        action: FirewatchAction,
        is_wrong: bool,
    ) -> tuple[str, bool]:
        """Scale replicas / increase memory limit for target service."""
        svc = mesh.services[target]

        if is_wrong:
            return (
                f"Scaled {target} (error_rate {svc.http_server_error_rate:.4f}). "
                f"Service was not significantly degraded — unnecessary scaling.",
                True,
            )

        # Get new memory limit from parameters or default to 2x
        new_limit_mb = action.parameters.get("memory_limit_mb")
        if new_limit_mb is None:
            new_limit_mb = (svc.process_memory_limit_bytes // (1024 * 1024)) * 2
        else:
            new_limit_mb = int(new_limit_mb)

        new_limit_bytes = new_limit_mb * 1024 * 1024

        if target == fc.root_cause_service and fc.fault_type in ("oom", "memory_leak"):
            # Correct: increase memory headroom
            svc.process_memory_limit_bytes = new_limit_bytes
            # Recalculate utilization with new limit
            svc.process_memory_utilization = (
                svc.process_memory_usage_bytes / svc.process_memory_limit_bytes
            )
            if fc.fault_type == "oom":
                mesh.fault_halted = True
            return (
                f"Scaled {target}: memory limit increased to {new_limit_mb}Mi. "
                f"Memory utilization dropped to "
                f"{svc.process_memory_utilization:.1%} with new headroom."
                + (" OOM risk eliminated." if fc.fault_type == "oom" else
                   " Memory leak continues but with more runway."),
                False,
            )

        # Wrong fault type
        svc.process_memory_limit_bytes = new_limit_bytes
        svc.process_memory_utilization = (
            svc.process_memory_usage_bytes / svc.process_memory_limit_bytes
        )
        return (
            f"Scaled {target}: memory limit increased to {new_limit_mb}Mi. "
            f"However, the active fault is not memory-related — this had "
            f"limited effect on fault progression.",
            False,
        )

    def _circuit_break(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
        is_wrong: bool,
    ) -> tuple[str, bool]:
        """Activate circuit breaker to stop cascade from target."""
        svc = mesh.services[target]

        if is_wrong:
            return (
                f"Circuit breaker activated for {target} (error_rate "
                f"{svc.http_server_error_rate:.4f}). Service was not "
                f"significantly degraded — unnecessary circuit break.",
                True,
            )

        # Register circuit breaker for 3 ticks
        self._circuit_breakers[target] = 3

        # Find services that depend on target and stabilize their error rates
        dependents = [
            svc_name for svc_name, deps in mesh.dependency_graph.items()
            if target in deps
        ]

        for dep_name in dependents:
            if dep_name in mesh.services:
                dep = mesh.services[dep_name]
                # Reduce cascaded error contribution
                dep.http_server_error_rate = max(
                    0.0, dep.http_server_error_rate * 0.5
                )

        dep_names = ", ".join(dependents) if dependents else "none"
        return (
            f"Circuit breaker activated for {target}. Traffic from "
            f"dependents halted for 3 ticks. Affected dependents: "
            f"[{dep_names}]. Cascade from {target} is contained but "
            f"underlying fault is NOT resolved.",
            False,
        )

    # ------------------------------------------------------------------
    # Meta actions
    # ------------------------------------------------------------------

    def _declare_resolved(
        self, mesh: "ServiceMesh"
    ) -> tuple[str, bool]:
        """Declare the incident resolved and trigger grader evaluation."""
        return (
            "Incident declared resolved. Evaluating episode...",
            False,
        )

    def _escalate(
        self, mesh: "ServiceMesh"
    ) -> tuple[str, bool]:
        """Escalate — costs 3 ticks of SLO budget."""
        # Burn 3x the normal SLO rate
        extra_burn = mesh.slo_burn_rate * 3.0
        mesh.slo_budget -= extra_burn
        mesh.slo_budget = max(0.0, mesh.slo_budget)
        return (
            f"Escalation initiated. Specialist team paged. Response "
            f"expected in 3 tick-equivalents. SLO budget cost: "
            f"{extra_burn:.1f}%. Remaining: {mesh.slo_budget:.1f}%.",
            False,
        )

    def is_circuit_broken(self, service_name: str) -> bool:
        """Check if a service has an active circuit breaker."""
        return service_name in self._circuit_breakers


# ==========================================================================
# Public API
# ==========================================================================

__all__ = [
    "ActionHandler",
]
