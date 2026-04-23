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
        SLO_BURN_RATE_BY_DIFFICULTY,
        SECONDS_PER_TICK,
        SYSCALL_PATTERNS,
        PROFILE_PATTERNS,
        GC_PRESSURE_PATTERNS,
        THREAD_POOL_PATTERNS,
        BAD_DEPLOY_DIFFS_TEMPLATES,
        TRAFFIC_SHIFT_LATENCY_PENALTY,
        TRAFFIC_SHIFT_MIN_DRAIN,
        TRAFFIC_SHIFT_MAX_DRAIN,
        ESCALATE_SPECIALIST_TICKS,
        ESCALATE_INVESTIGATION_COST_MULTIPLIER,
        INVESTIGATION_ACTIONS,
        ACTION_REGISTRY,
    )
except ImportError:
    from models import FirewatchAction, ActionResult
    from config import (
        HEALTHY_ERROR_RATE_THRESHOLD,
        FULL_DEPENDENCY_GRAPH,
        SLO_BURN_RATE_BY_DIFFICULTY,
        SECONDS_PER_TICK,
        SYSCALL_PATTERNS,
        PROFILE_PATTERNS,
        GC_PRESSURE_PATTERNS,
        THREAD_POOL_PATTERNS,
        BAD_DEPLOY_DIFFS_TEMPLATES,
        TRAFFIC_SHIFT_LATENCY_PENALTY,
        TRAFFIC_SHIFT_MIN_DRAIN,
        TRAFFIC_SHIFT_MAX_DRAIN,
        ESCALATE_SPECIALIST_TICKS,
        ESCALATE_INVESTIGATION_COST_MULTIPLIER,
        INVESTIGATION_ACTIONS,
        ACTION_REGISTRY,
    )

if TYPE_CHECKING:
    from .simulation import ServiceMesh, FaultConfig



# --------------------------------------------------------------------------
# Wrong-Action Guard (SPEC-02 §2)
# --------------------------------------------------------------------------

def is_wrong_action(
    action_name: str,
    target_service: str | None,
    mesh: "ServiceMesh",
) -> bool:
    """Determine if an action constitutes a wrong action (SPEC-02 §2).

    Returns True if the action is a guarded remediation targeting a
    non-existent or healthy service. Investigation and meta actions
    always return False.

    Uses the ACTION_REGISTRY to check:
    1. Category must be "Remediation" (otherwise not guarded)
    2. guard_applies must be True (False skips the check)
    3. Target service must exist in the mesh
    4. Target service error_rate must be >= ERROR_RATE_GUARD_THRESHOLD
    """
    action_def = ACTION_REGISTRY.get(action_name)
    if action_def is None:
        return False

    if action_def.category != "Remediation":
        return False

    if not action_def.guard_applies:
        return False

    if target_service is None or target_service not in mesh.services:
        return True  # targeting a non-existent service is always wrong

    return mesh.services[target_service].http_server_error_rate < HEALTHY_ERROR_RATE_THRESHOLD


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
        self.specialist_active_ticks: int = 0

    def _halt_fault_on(
        self,
        mesh: "ServiceMesh",
        target: str,
        fault_type: str,
    ) -> None:
        """Halt the matching FaultState on a service (SPEC-01 §6).

        Iterates mesh.active_faults to find the first non-halted fault
        matching the target service and fault type, then marks it halted.
        Falls back to legacy mesh.fault_halted for backward compat.
        """
        if mesh.active_faults:
            for fault in mesh.active_faults:
                if (
                    fault.fault_service == target
                    and fault.fault_type == fault_type
                    and not fault.halted
                ):
                    fault.halted = True
                    fault.halted_at_tick = mesh.tick_count
                    break
        # Always set legacy flag for backward compat
        mesh.fault_halted = True
        # Counts how many remaining investigation actions get the specialist discount.
        # Set by escalate action. Decremented by environment.py when applying discount.

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
        # Check for wrong action via centralized guard (SPEC-02 §2)
        is_wrong = is_wrong_action(at, target, mesh)

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

        # --- Advanced diagnostic investigation actions (SPEC-9) ---
        if at == "strace_process":
            return self._strace_process(target, mesh, fault_config)

        if at == "profiler_dump":
            return self._profiler_dump(target, mesh, fault_config)

        if at == "check_gc_pressure":
            return self._check_gc_pressure(target, mesh, fault_config)

        if at == "trace_distributed_request":
            return self._trace_distributed_request(target, mesh)

        if at == "inspect_thread_pool":
            return self._inspect_thread_pool(target, mesh, fault_config)

        if at == "inspect_commit_diff":
            return self._inspect_commit_diff(target, mesh, fault_config)

        # --- Advanced remediation actions (SPEC-9) ---
        if at == "traffic_shift":
            return self._traffic_shift(target, mesh, fault_config, action.parameters)

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

    def _strace_process(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """
        Simulates strace syscall tracing.
        Returns syscall frequency distribution based on fault type.
        Differentiates kernel-level resource exhaustion from application bugs.
        """
        if target not in mesh.services:
            return (
                f"Invalid target: {target} is not an active service. No action taken.",
                False,
            )

        svc = mesh.services[target]

        if target == fc.root_cause_service:
            pattern = SYSCALL_PATTERNS.get(fc.fault_type, SYSCALL_PATTERNS["healthy"])
            context = f"(ROOT CAUSE: {fc.fault_type})"
        elif svc.status in ("degraded", "critical", "down"):
            pattern = SYSCALL_PATTERNS["healthy"]
            context = "(Cascaded degradation - normal syscall pattern)"
        else:
            pattern = SYSCALL_PATTERNS["healthy"]
            context = "(Healthy baseline)"

        syscall_lines = "\n".join(
            f"  {syscall:>12s}: {freq * 100:5.1f}%"
            for syscall, freq in sorted(pattern.items(), key=lambda x: -x[1])
        )

        interpretation = ""
        if fc.fault_type == "oom" and target == fc.root_cause_service:
            interpretation = (
                "\n[Analysis] High mmap/brk frequency indicates aggressive heap expansion. "
                "Process is requesting memory from kernel faster than it can be allocated. "
                "OOMKill likely imminent."
            )
        elif fc.fault_type == "config_drift" and target == fc.root_cause_service:
            interpretation = (
                "\n[Analysis] High accept() frequency with EMFILE errors. "
                "File descriptor exhaustion - connection pool misconfiguration."
            )

        feedback = (
            f"strace output for {target} (sampled over 1 tick):\n"
            f"{syscall_lines}\n"
            f"{context}{interpretation}"
        )
        return (feedback, False)

    def _profiler_dump(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """
        Simulates CPU profiler (pprof/py-spy flame graph).
        Categorizes CPU time into: user code, I/O wait, GC, kernel.
        """
        if target not in mesh.services:
            return (f"Invalid target: {target} is not an active service.", False)

        svc = mesh.services[target]

        if target == fc.root_cause_service:
            pattern = PROFILE_PATTERNS.get(fc.fault_type, PROFILE_PATTERNS["healthy"])
        elif svc.http_server_request_duration_p99 > 1.0:
            pattern = PROFILE_PATTERNS["network_partition"]
        else:
            pattern = PROFILE_PATTERNS["healthy"]

        profile_lines = "\n".join(
            f"  {category:>15s}: {pct * 100:5.1f}%"
            for category, pct in sorted(pattern.items(), key=lambda x: -x[1])
        )

        interpretation = ""
        if pattern.get("user_code_cpu", 0) > 0.70:
            interpretation = (
                "\n[Diagnosis] CPU-bound: Application code consuming majority of cycles. "
                "Likely infinite loop, hot regex, or algorithmic issue. "
                "Check recent deployments."
            )
        elif pattern.get("io_wait", 0) > 0.70:
            interpretation = (
                "\n[Diagnosis] I/O-bound: Process blocked waiting on network/disk. "
                "Not a local CPU issue - investigate upstream dependencies."
            )
        elif pattern.get("gc", 0) > 0.40:
            interpretation = (
                "\n[Diagnosis] GC-bound: Garbage collector consuming >40% of CPU. "
                "Memory leak or insufficient heap. Run check_gc_pressure for details."
            )

        feedback = (
            f"Profile dump for {target} (1-second sample):\n"
            f"{profile_lines}\n"
            f"Total samples: 10,000{interpretation}"
        )
        return (feedback, False)

    def _check_gc_pressure(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """
        Simulates runtime GC metrics (JVM/V8/Go).
        Shows GC pause time, cycle frequency, heap reclamation efficiency.
        Updates service's runtime_gc_* fields with measured values.
        """
        if target not in mesh.services:
            return (f"Invalid target: {target}.", False)

        svc = mesh.services[target]

        if (
            target == fc.root_cause_service
            and fc.fault_type in ("memory_leak", "oom")
        ):
            pattern = GC_PRESSURE_PATTERNS[fc.fault_type]
        elif svc.process_memory_utilization > 0.70:
            pattern = GC_PRESSURE_PATTERNS["memory_leak"]
        else:
            pattern = GC_PRESSURE_PATTERNS["healthy"]

        # Update live service metrics with GC data
        svc.runtime_gc_pause_duration_ms = pattern["gc_pause_time_ms"]
        svc.runtime_gc_count_per_second = pattern["gc_cycles_per_second"]

        feedback = (
            f"GC pressure analysis for {target}:\n"
            f"  GC cycles/sec:     {pattern['gc_cycles_per_second']:>6.1f}"
            f" ({_gc_rate_label(pattern['gc_cycles_per_second'])})\n"
            f"  Avg pause time:    {pattern['gc_pause_time_ms']:>6.1f}ms"
            f" ({_gc_pause_label(pattern['gc_pause_time_ms'])})\n"
            f"  Heap after GC:     {pattern['heap_after_gc_mb']:>6.0f}MB\n"
            f"  GC CPU usage:      {pattern['gc_cpu_percent']:>6.1f}%\n"
        )

        if pattern["gc_cycles_per_second"] > 30:
            feedback += (
                "\n[CRITICAL] GC thrashing detected. Runtime spending majority of time in "
                "garbage collection. Heap is not being reclaimed effectively — indicates "
                "memory leak or insufficient heap size."
            )
        elif pattern["gc_pause_time_ms"] > 500:
            feedback += (
                "\n[WARNING] Long stop-the-world pauses causing latency spikes. "
                "Consider heap tuning or switching to concurrent GC."
            )

        return (feedback, False)

    def _trace_distributed_request(
        self,
        target: str,
        mesh: "ServiceMesh",
    ) -> tuple[str, bool]:
        """
        Simulates distributed tracing (Jaeger/Zipkin/OTel).
        Traces request from api-gateway through the call chain to target,
        showing per-service span durations.
        Different from trace_dependencies (static topology) — this shows dynamic latency.
        """
        if target not in mesh.services:
            return (f"Invalid target: {target}.", False)

        def _get_downstream(service: str) -> set[str]:
            downstream: set[str] = set()
            for dep in mesh.dependency_graph.get(service, []):
                downstream.add(dep)
                downstream.update(_get_downstream(dep))
            return downstream

        trace_spans: list[dict] = []
        total_duration_ms = 0.0

        def _build_trace(current: str, depth: int = 0) -> None:
            nonlocal total_duration_ms
            if current not in mesh.services:
                return
            svc = mesh.services[current]
            duration_ms = svc.http_server_request_duration_p99 * 1000
            total_duration_ms += duration_ms
            trace_spans.append({
                "service": current,
                "duration_ms": duration_ms,
                "depth": depth,
                "status": svc.status,
            })
            for dep in mesh.dependency_graph.get(current, []):
                if dep == target or target in _get_downstream(dep):
                    _build_trace(dep, depth + 1)
                    break

        entry = "api-gateway" if "api-gateway" in mesh.services else target
        _build_trace(entry)

        if not trace_spans:
            return (f"distributed_trace: no services found in call chain for {target}.", False)

        bottleneck = max(trace_spans, key=lambda s: s["duration_ms"])

        trace_lines = []
        for span in trace_spans:
            indent = "  " * span["depth"]
            status_sym = (
                "⚠" if span["status"] in ("degraded", "critical")
                else ("✗" if span["status"] == "down" else "✓")
            )
            trace_lines.append(
                f"{indent}{status_sym} {span['service']:<20s} {span['duration_ms']:>7.1f}ms"
            )

        feedback = (
            f"Distributed trace to {target}:\n"
            + "\n".join(trace_lines)
            + f"\n\nTotal duration: {total_duration_ms:.1f}ms\n"
            + f"[Bottleneck] {bottleneck['service']} ({bottleneck['duration_ms']:.1f}ms, "
            + f"{bottleneck['duration_ms'] / total_duration_ms * 100:.1f}% of total)\n"
        )

        if bottleneck["service"] != target:
            feedback += (
                f"[Analysis] {target} is not the bottleneck. "
                f"Investigate {bottleneck['service']}."
            )

        return (feedback, False)

    def _inspect_thread_pool(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """
        Simulates thread pool / worker stats.
        Shows active threads, max capacity, queue depth, queue time.
        Updates service runtime_jvm_* fields with measured values.
        Saturation threshold: 200 active requests (PRD §7.2).
        """
        if target not in mesh.services:
            return (f"Invalid target: {target}.", False)

        svc = mesh.services[target]

        if svc.http_server_active_requests >= 180:
            pattern = THREAD_POOL_PATTERNS["saturated"]
        elif svc.http_server_active_requests >= 120:
            pattern = THREAD_POOL_PATTERNS["high_load"]
        else:
            pattern = THREAD_POOL_PATTERNS["healthy"]

        # Update live service metrics
        svc.runtime_jvm_threads_count = pattern["active_threads"]
        svc.runtime_jvm_threads_max = pattern["max_threads"]
        svc.runtime_thread_pool_queue_depth = pattern["queued_requests"]

        utilization_pct = (pattern["active_threads"] / pattern["max_threads"]) * 100

        feedback = (
            f"Thread pool stats for {target}:\n"
            f"  Active threads:    {pattern['active_threads']:>4d} / {pattern['max_threads']}"
            f" ({utilization_pct:.1f}% utilized)\n"
            f"  Queued requests:   {pattern['queued_requests']:>4d}\n"
            f"  Avg queue time:    {pattern['avg_queue_time_ms']:>4d}ms\n"
        )

        if pattern["active_threads"] == pattern["max_threads"]:
            feedback += (
                "\n[CRITICAL] Thread pool saturated. All workers busy. "
                "New requests queuing indefinitely. "
                "\n[Cause] Either:\n"
                "  1. Threads blocked on slow downstream I/O (check trace_distributed_request)\n"
                "  2. CPU-bound task monopolizing threads (check profiler_dump)\n"
                "  3. Need to scale replicas to increase total thread capacity"
            )
        elif pattern["queued_requests"] > 500:
            feedback += (
                "\n[WARNING] High request backlog. "
                "System approaching saturation. Consider scaling."
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

        if target == fc.root_cause_service and fc.fault_type == "network_partition":
            # Correct: restart re-establishes connections, halts partition
            self._halt_fault_on(mesh, target, "network_partition")
            svc.http_server_error_rate = max(0.0, svc.http_server_error_rate * 0.3)
            svc.http_server_request_duration_p99 = max(0.1, svc.http_server_request_duration_p99 * 0.2)
            svc.runtime_uptime_seconds = 0
            svc.restart_count += 1
            return (
                f"Restarted {target}. Network connections re-established. "
                f"Error rate declining — partition resolved.",
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
            self._halt_fault_on(mesh, target, "bad_deploy")
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
            self._halt_fault_on(mesh, target, "config_drift")
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
                self._halt_fault_on(mesh, target, "oom")
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

    def _traffic_shift(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
        parameters: dict,
    ) -> tuple[str, bool]:
        """
        Graceful traffic routing — drains percentage of traffic from target.
        Trades slight latency increase (cross-zone routing) for error rate reduction.
        Unlike circuit_break (full cutoff), this is a partial traffic shift.
        """
        if target not in mesh.services:
            return (f"Invalid target: {target}.", False)

        svc = mesh.services[target]

        # Extract drain percentage
        drain_pct = parameters.get("drain_percentage", 0.80)
        if not isinstance(drain_pct, (int, float)):
            return (
                f"Invalid drain_percentage: must be numeric. Got {type(drain_pct).__name__}.",
                False,
            )
        drain_pct = float(drain_pct)

        if drain_pct < TRAFFIC_SHIFT_MIN_DRAIN:
            return (
                f"drain_percentage too low ({drain_pct:.1%}). "
                f"Minimum {TRAFFIC_SHIFT_MIN_DRAIN:.1%} required for observable effect.",
                False,
            )

        clamped_note = ""
        if drain_pct > TRAFFIC_SHIFT_MAX_DRAIN:
            drain_pct = TRAFFIC_SHIFT_MAX_DRAIN
            clamped_note = f" (clamped to {TRAFFIC_SHIFT_MAX_DRAIN:.1%} max)"

        # wrong_action: use centralized guard (SPEC-02 §2)
        is_wrong = is_wrong_action("traffic_shift", target, mesh)

        # Mutations
        original_active = svc.http_server_active_requests
        svc.http_server_active_requests = int(original_active * (1.0 - drain_pct))

        original_error_rate = svc.http_server_error_rate
        svc.http_server_error_rate = round(original_error_rate * (1.0 - drain_pct), 4)

        original_latency = svc.http_server_request_duration_p99
        svc.http_server_request_duration_p99 = round(
            original_latency * TRAFFIC_SHIFT_LATENCY_PENALTY, 4
        )

        feedback = (
            f"Traffic shift applied to {target}:\n"
            f"  Drain percentage:  {drain_pct:.1%}{clamped_note}\n"
            f"  Active requests:   {original_active} → {svc.http_server_active_requests}"
            f" ({-drain_pct * 100:.1f}%)\n"
            f"  Error rate:        {original_error_rate:.3f} → {svc.http_server_error_rate:.3f}\n"
            f"  P99 latency:       {original_latency * 1000:.0f}ms → "
            f"{svc.http_server_request_duration_p99 * 1000:.0f}ms "
            f"(+{(TRAFFIC_SHIFT_LATENCY_PENALTY - 1) * 100:.0f}% cross-zone penalty)\n"
        )

        if is_wrong:
            feedback += (
                "\n[WARNING] Service error rate was below healthy threshold (0.10). "
                "Draining a healthy service wastes SLO budget."
            )
        else:
            feedback += (
                f"\n[Success] Blast radius contained. "
                f"{drain_pct * 100:.0f}% of traffic rerouted to backup capacity. "
                "Monitor for improvement."
            )

        return (feedback, is_wrong)

    def _inspect_commit_diff(
        self,
        target: str,
        mesh: "ServiceMesh",
        fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """
        Simulates git diff for recent deployment.
        Shows the code change that caused bad_deploy fault.
        Confirms hypothesis before executing rollback.
        """
        if target not in mesh.services:
            return (f"Invalid target: {target}.", False)

        svc = mesh.services[target]

        if (
            target == fc.root_cause_service
            and fc.fault_type == "bad_deploy"
        ):
            # Select diff template based on service metrics
            if svc.http_server_error_rate > 0.70:
                diff_key = "null_pointer"
            elif svc.process_cpu_utilization > 0.85:
                diff_key = "infinite_loop"
            else:
                diff_key = "timeout_removal"

            diff_text = BAD_DEPLOY_DIFFS_TEMPLATES.get(diff_key, "No diff available.")

            feedback = (
                f"Git diff for {target} deployment (SHA: {svc.last_deployment_sha}):\n"
                f"{diff_text}\n"
                f"[Analysis] Deployment {svc.last_deployment_age_seconds}s ago introduced "
                f"regression. Recommend rollback_deploy to previous stable version."
            )
        else:
            feedback = (
                f"Git diff for {target} (SHA: {svc.last_deployment_sha}):\n"
                f"No anomalous changes detected in recent deployments.\n"
                f"Last deploy: {svc.last_deployment_age_seconds}s ago (within normal range).\n"
                f"[Analysis] Deployment is not the root cause. Consider other fault types."
            )

        return (feedback, False)

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

    def _escalate(self, mesh: "ServiceMesh") -> tuple[str, bool]:
        """Grant specialist discount on next investigation actions."""
        self.specialist_active_ticks = ESCALATE_SPECIALIST_TICKS
        return (
            f"Escalation initiated. Specialist team paged. "
            f"Next {ESCALATE_SPECIALIST_TICKS} investigation actions will cost "
            f"{int(ESCALATE_INVESTIGATION_COST_MULTIPLIER * 100)}% of normal SLO budget.",
            False,
        )

    def is_circuit_broken(self, service_name: str) -> bool:
        """Check if a service has an active circuit breaker."""
        return service_name in self._circuit_breakers


def _gc_rate_label(rate: float) -> str:
    """Classify GC cycle rate into human-readable severity label."""
    if rate > 30:
        return "CRITICAL"
    if rate > 10:
        return "HIGH"
    return "normal"


def _gc_pause_label(pause_ms: float) -> str:
    """Classify GC pause duration into human-readable severity label."""
    if pause_ms > 500:
        return "CRITICAL"
    if pause_ms > 100:
        return "elevated"
    return "normal"


# ==========================================================================
# Public API
# ==========================================================================

__all__ = [
    "ActionHandler",
    "is_wrong_action",
]
