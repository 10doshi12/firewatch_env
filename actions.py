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
        LATENCY_GUARD_THRESHOLD,
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
        LATENCY_GUARD_THRESHOLD,
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
    """Determine if an action constitutes a wrong action (SPEC-02 §2, SPEC-06 §1).

    Returns True if the action is a guarded remediation targeting a
    non-existent or healthy service. Investigation and meta actions
    always return False.

    Uses the ACTION_REGISTRY to check:
    1. Category must be "Remediation" (otherwise not guarded)
    2. guard_applies must be True (False skips the check)
    3. Target service must exist in the mesh
    4. Service is considered healthy only when BOTH:
       - error_rate < HEALTHY_ERROR_RATE_THRESHOLD (0.05)
       - latency_p99 < LATENCY_GUARD_THRESHOLD (0.50s)
       This dual check fixes gray failure (H-R1) where error_rate ≈ 0%
       but latency is 8× baseline due to TCP retransmission masking.
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

    svc = mesh.services[target_service]
    service_is_healthy = (
        svc.http_server_error_rate < HEALTHY_ERROR_RATE_THRESHOLD
        and svc.http_server_request_duration_p99 < LATENCY_GUARD_THRESHOLD
    )
    return service_is_healthy


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

        # --- Phase 2 investigation actions (SPEC-05) ---
        if at == "inspect_network_policy":
            return self._inspect_network_policy(target, mesh, fault_config)

        if at == "inspect_quota_usage":
            return self._inspect_quota_usage(target, mesh, fault_config)

        if at == "inspect_consensus_state":
            return self._inspect_consensus_state(target, mesh, fault_config)

        if at == "inspect_cluster_topology":
            return self._inspect_cluster_topology(target, mesh, fault_config)

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

        # --- Advanced remediation actions (SPEC-9) ---
        if at == "traffic_shift":
            return self._traffic_shift(target, mesh, fault_config, action.parameters)

        # --- Phase 2 Easy tier remediation (SPEC-05) ---
        if at == "enable_connection_throttle":
            return self._enable_connection_throttle(target, mesh, fault_config, is_wrong)

        if at == "extend_timeout":
            return self._extend_timeout(target, mesh, fault_config, is_wrong)

        if at == "optimize_query":
            return self._optimize_query(target, mesh, fault_config, is_wrong)

        if at == "rebalance_load":
            return self._rebalance_load(target, mesh, fault_config, is_wrong)

        if at == "adjust_probe_timing":
            return self._adjust_probe_timing(target, mesh, fault_config, is_wrong)

        if at == "set_log_level":
            return self._set_log_level(target, mesh, fault_config, action, is_wrong)

        # --- Phase 2 Medium tier remediation (SPEC-05) ---
        if at == "disable_retries":
            return self._disable_retries(target, mesh, fault_config, is_wrong)

        if at == "configure_retry_backoff":
            return self._configure_retry_backoff(target, mesh, fault_config, is_wrong)

        if at == "rollback_canary":
            return self._rollback_canary(target, mesh, fault_config, is_wrong)

        if at == "promote_canary":
            return self._promote_canary(target, mesh, fault_config, is_wrong)

        if at == "redirect_reads_to_primary":
            return self._redirect_reads_to_primary(target, mesh, fault_config, is_wrong)

        if at == "force_replica_resync":
            return self._force_replica_resync(target, mesh, fault_config, is_wrong)

        if at == "evict_cache_by_pattern":
            return self._evict_cache_by_pattern(target, mesh, fault_config, is_wrong)

        if at == "increase_cache_memory":
            return self._increase_cache_memory(target, mesh, fault_config, is_wrong)

        if at == "complete_traffic_switch":
            return self._complete_traffic_switch(target, mesh, fault_config, action, is_wrong)

        if at == "deregister_stale_instances":
            return self._deregister_stale_instances(target, mesh, fault_config, is_wrong)

        if at == "enable_deadline_propagation":
            return self._enable_deadline_propagation(target, mesh, fault_config, is_wrong)

        # --- Phase 2 Hard tier remediation (SPEC-05) ---
        if at == "revert_network_policy":
            return self._revert_network_policy(target, mesh, fault_config)

        if at == "disable_fallback_mode":
            return self._disable_fallback_mode(target, mesh, fault_config, is_wrong)

        if at == "request_quota_increase":
            return self._request_quota_increase(target, mesh, fault_config, action, is_wrong)

        if at == "force_leader_election":
            return self._force_leader_election(target, mesh, fault_config, is_wrong)

        if at == "isolate_minority_nodes":
            return self._isolate_minority_nodes(target, mesh, fault_config, is_wrong)

        if at == "redirect_config_reads_to_majority":
            return self._redirect_config_reads_to_majority(target, mesh, fault_config, is_wrong)

        if at == "flush_diverged_keys":
            return self._flush_diverged_keys(target, mesh, fault_config, is_wrong)

        if at == "force_cluster_resync":
            return self._force_cluster_resync(target, mesh, fault_config, is_wrong)

        if at == "enable_cache_warming":
            return self._enable_cache_warming(target, mesh, fault_config, is_wrong)

        if at == "rate_limit_cache_misses":
            return self._rate_limit_cache_misses(target, mesh, fault_config, is_wrong)

        if at == "rebalance_az_traffic":
            return self._rebalance_az_traffic(target, mesh, fault_config)

        if at == "scale_az_capacity":
            return self._scale_az_capacity(target, mesh, fault_config)

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
    # Phase 2 Investigation actions (SPEC-05)
    # ------------------------------------------------------------------

    def _inspect_network_policy(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """Returns network policies affecting target. Does NOT modify state."""
        svc = mesh.services[target]
        if target == fc.root_cause_service and fc.fault_type == "network_partition":
            feedback = (
                f"Network policy inspection for {target}:\n"
                f'{{"rules": [{{"action": "drop", "match": "inbound", '
                f'"packet_loss_pct": 15.0, "affected_ports": [5432, 6379]}}], '
                f'"packet_loss_pct": 15.0, "affected_ports": [5432, 6379]}}\n'
                f"[Analysis] Active packet loss policy detected. Recommend revert_network_policy."
            )
        else:
            feedback = (
                f"Network policy inspection for {target}:\n"
                f'{{"rules": [], "packet_loss_pct": 0.0, "affected_ports": []}}\n'
                f"[Analysis] No anomalous network policies detected."
            )
        return (feedback, False)

    def _inspect_quota_usage(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """Returns quota utilization for target. Does NOT modify state."""
        svc = mesh.services[target]
        if target == fc.root_cause_service and fc.fault_type in ("config_drift", "bad_deploy"):
            feedback = (
                f"Quota usage for {target}:\n"
                f'{{"gpu_compute": {{"used": 95, "limit": 100, "remaining_ratio": 0.05}}, '
                f'"bandwidth": {{"used": 88, "limit": 100, "remaining_ratio": 0.12}}, '
                f'"db_connections": {{"used": 48, "limit": 50, "remaining_ratio": 0.04}}}}\n'
                f"[Analysis] Multiple quotas near exhaustion. Recommend request_quota_increase."
            )
        else:
            feedback = (
                f"Quota usage for {target}:\n"
                f'{{"gpu_compute": {{"used": 30, "limit": 100, "remaining_ratio": 0.70}}, '
                f'"bandwidth": {{"used": 25, "limit": 100, "remaining_ratio": 0.75}}, '
                f'"db_connections": {{"used": 10, "limit": 50, "remaining_ratio": 0.80}}}}\n'
                f"[Analysis] All quotas within normal range."
            )
        return (feedback, False)

    def _inspect_consensus_state(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """Returns consensus cluster state. Does NOT modify state."""
        svc = mesh.services[target]
        if target == fc.root_cause_service and fc.fault_type == "network_partition":
            feedback = (
                f"Consensus state for {target}:\n"
                f'{{"nodes": 5, "leader": "node-2", "term": 47, "quorum_healthy": false, '
                f'"healthy_node_count": 3, '
                f'"partition_status_per_node": {{"node-1": "majority", "node-2": "majority", '
                f'"node-3": "majority", "node-4": "minority", "node-5": "minority"}}, '
                f'"config_age_seconds": {{"node-1": 2, "node-4": 3600, "node-5": 3600}}}}\n'
                f"[Analysis] Split-brain detected. 2 minority nodes serving stale config."
            )
        else:
            feedback = (
                f"Consensus state for {target}:\n"
                f'{{"nodes": 5, "leader": "node-1", "term": 42, "quorum_healthy": true, '
                f'"healthy_node_count": 5, '
                f'"partition_status_per_node": {{}}, "config_age_seconds": {{}}}}\n'
                f"[Analysis] Consensus cluster healthy. No partition detected."
            )
        return (feedback, False)

    def _inspect_cluster_topology(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """Returns Redis cluster topology. Does NOT modify state."""
        svc = mesh.services[target]
        if target == fc.root_cause_service and fc.fault_type == "network_partition":
            feedback = (
                f"Cluster topology for {target}:\n"
                f'{{"nodes": 6, "slot_map": "0-8191:master-1, 8192-16383:master-2", '
                f'"split_brain_detected": true, "diverged_key_count": 1247, '
                f'"partition_duration_seconds": 180}}\n'
                f"[Analysis] Split-brain detected. 1247 diverged keys. "
                f"Recommend flush_diverged_keys + force_cluster_resync."
            )
        else:
            feedback = (
                f"Cluster topology for {target}:\n"
                f'{{"nodes": 6, "slot_map": "0-8191:master-1, 8192-16383:master-2", '
                f'"split_brain_detected": false, "diverged_key_count": 0, '
                f'"partition_duration_seconds": 0}}\n'
                f"[Analysis] Cluster topology healthy. No split-brain."
            )
        return (feedback, False)

    # ------------------------------------------------------------------
    # Phase 2 Easy Tier Remediation (SPEC-05)
    # ------------------------------------------------------------------

    def _enable_connection_throttle(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Rate-limit inbound connections to target service."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Connection throttle enabled on {target} (error_rate {svc.http_server_error_rate:.4f}). Service was not degraded — unnecessary throttle.", True)
        if target == fc.root_cause_service and fc.fault_type == "bad_deploy":
            self._halt_fault_on(mesh, target, "bad_deploy")
            svc.http_server_active_requests = max(50, svc.http_server_active_requests // 2)
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.4)
            return (f"Connection throttle enabled on {target}. Inbound rate capped. Active requests declining.", False)
        svc.http_server_active_requests = max(50, svc.http_server_active_requests // 2)
        return (f"Connection throttle enabled on {target}. Active requests reduced but underlying fault persists.", False)

    def _extend_timeout(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Increase downstream timeout budget on target."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Timeout extended on {target} (error_rate {svc.http_server_error_rate:.4f}). Service was not degraded — unnecessary.", True)
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
        return (f"Downstream timeout extended on {target} to 15000ms. Timeout-caused errors clearing.", False)

    def _optimize_query(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Optimize DB query on target, halting query slowness."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Query optimized on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded — unnecessary.", True)
        if target == fc.root_cause_service and fc.fault_type in ("config_drift", "bad_deploy"):
            self._halt_fault_on(mesh, target, fc.fault_type)
            svc.http_server_request_duration_p99 = 0.08
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
            return (f"Query plan optimized on {target}. p99 latency recovering. Slow query threshold no longer exceeded.", False)
        svc.http_server_request_duration_p99 = max(0.1, svc.http_server_request_duration_p99 * 0.5)
        return (f"Query optimized on {target} but underlying fault is not query-related.", False)

    def _rebalance_load(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Reset lb_weight_normalized across all replicas."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Load rebalanced on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded — unnecessary.", True)
        if target == fc.root_cause_service and fc.fault_type == "config_drift":
            self._halt_fault_on(mesh, target, "config_drift")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
            svc.process_cpu_utilization = min(0.30, svc.process_cpu_utilization)
            return (f"Load balancer weights reset on {target}. All replicas now receiving equal traffic share.", False)
        return (f"Load rebalanced on {target} but underlying fault is not load-related.", False)

    def _adjust_probe_timing(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Fix liveness probe timing to stop restart cycle."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Probe timing adjusted on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded — unnecessary.", True)
        if target == fc.root_cause_service and fc.fault_type == "bad_deploy":
            self._halt_fault_on(mesh, target, "bad_deploy")
            svc.restart_count = max(0, svc.restart_count)
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
            return (f"Liveness probe timing updated on {target}. initialDelaySeconds=10, timeoutSeconds=8. Restart cycle halted.", False)
        return (f"Probe timing adjusted on {target} but underlying fault is not probe-related.", False)

    def _set_log_level(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
        action: FirewatchAction, is_wrong: bool,
    ) -> tuple[str, bool]:
        """Set application log level on target."""
        svc = mesh.services[target]
        level = action.parameters.get("level", "INFO")
        if level not in ("DEBUG", "INFO", "WARN", "ERROR"):
            level = "INFO"
        if is_wrong:
            return (f"Log level set to {level} on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded — unnecessary.", True)
        if target == fc.root_cause_service and fc.fault_type == "config_drift":
            self._halt_fault_on(mesh, target, "config_drift")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
            return (f"Log level set to {level} on {target}. Write rate dropping. Disk utilization stabilizing.", False)
        return (f"Log level set to {level} on {target}. No effect on active fault.", False)

    # ------------------------------------------------------------------
    # Phase 2 Medium Tier Remediation (SPEC-05)
    # ------------------------------------------------------------------

    def _disable_retries(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Disable retries on target to break amplification loop."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Retries disabled on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.5)
        svc.http_server_active_requests = max(30, svc.http_server_active_requests // 2)
        return (f"Retries disabled on {target}. Amplification broken. Downstream load returning to baseline.", False)

    def _configure_retry_backoff(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Configure exponential backoff with jitter on target."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Retry backoff configured on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        return (f"Exponential backoff with jitter configured on {target}. Max retries: 3, base delay: 100ms, max delay: 10s.", False)

    def _rollback_canary(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Roll back canary deployment, routing all traffic to stable."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Canary rolled back on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service and fc.fault_type == "bad_deploy":
            self._halt_fault_on(mesh, target, "bad_deploy")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.2)
            return (f"Canary rolled back on {target}. Traffic: 100% stable. Canary receiving 0%.", False)
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.5)
        return (f"Canary rolled back on {target} but fault is not canary-related.", False)

    def _promote_canary(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Promote canary to 100% traffic. Catastrophic if canary is broken."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Canary promoted on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service and fc.fault_type == "bad_deploy":
            svc.http_server_error_rate = min(1.0, svc.http_server_error_rate + 0.40)
            return (f"Canary promoted on {target}. Traffic: 100% canary. Error rate surging — WRONG ACTION for broken canary.", False)
        return (f"Canary promoted on {target}. Traffic: 100% canary.", False)

    def _redirect_reads_to_primary(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Route all reads to primary DB, eliminating stale-read errors."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Reads redirected to primary on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.2)
        svc.http_server_request_duration_p99 = min(svc.http_server_request_duration_p99 * 1.4, 5.0)
        return (f"Reads redirected to primary on {target}. Replica reads suspended. DataConsistencyExceptions clearing.", False)

    def _force_replica_resync(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Trigger full replica resync from primary."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Replica resync on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service and fc.fault_type == "network_partition":
            self._halt_fault_on(mesh, target, "network_partition")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.4)
            return (f"Replica resync initiated on {target}. Full sync in progress. Estimated: 3 ticks.", False)
        return (f"Replica resync initiated on {target} but fault is not replication-related.", False)

    def _evict_cache_by_pattern(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Evict oversized/hot-key cache entries."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Cache eviction on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.5)
        return (f"Cache eviction by pattern complete on {target}. Oversized keys removed. Hit rate recovering.", False)

    def _increase_cache_memory(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Increase maxmemory allocation on cache."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Cache memory increased on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service and fc.fault_type == "config_drift":
            self._halt_fault_on(mesh, target, "config_drift")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.2)
            svc.process_memory_utilization = min(0.50, svc.process_memory_utilization * 0.6)
            svc.process_memory_usage_bytes = int(svc.process_memory_utilization * svc.process_memory_limit_bytes)
            return (f"Cache memory limit increased on {target}. Eviction pressure resolved.", False)
        return (f"Cache memory increased on {target} but fault is not memory-related.", False)

    def _complete_traffic_switch(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
        action: FirewatchAction, is_wrong: bool,
    ) -> tuple[str, bool]:
        """Force all traffic to specified blue/green slot."""
        svc = mesh.services[target]
        slot = action.parameters.get("slot", "blue")
        if slot not in ("blue", "green"):
            slot = "blue"
        if is_wrong:
            return (f"Traffic switched to {slot} on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service and fc.fault_type == "config_drift":
            self._halt_fault_on(mesh, target, "config_drift")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.2)
            other = "green" if slot == "blue" else "blue"
            return (f"Traffic fully switched to {slot} on {target}. {other} slot receiving 0% traffic.", False)
        return (f"Traffic switched to {slot} on {target} but fault is not deployment-related.", False)

    def _deregister_stale_instances(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Remove stale instances from service registry."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Stale instances deregistered from {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service and fc.fault_type == "config_drift":
            self._halt_fault_on(mesh, target, "config_drift")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.1)
            return (f"Stale instances deregistered from {target}. Dead instances removed. Registry healthy.", False)
        return (f"Stale instances deregistered from {target} but fault is not registry-related.", False)

    def _enable_deadline_propagation(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Enable gRPC deadline propagation to downstream calls."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Deadline propagation enabled on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service and fc.fault_type == "bad_deploy":
            self._halt_fault_on(mesh, target, "bad_deploy")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
            return (f"gRPC deadline propagation enabled on {target}. Orphaned calls cancelling. Downstream thread pools draining.", False)
        return (f"Deadline propagation enabled on {target} but fault is not deadline-related.", False)

    # ------------------------------------------------------------------
    # Phase 2 Hard Tier Remediation (SPEC-05) — Part 1
    # ------------------------------------------------------------------

    def _revert_network_policy(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """Remove last-applied network policy. guard_applies=False (gray failure)."""
        svc = mesh.services[target]
        if target == fc.root_cause_service and fc.fault_type == "network_partition":
            self._halt_fault_on(mesh, target, "network_partition")
            svc.http_server_request_duration_p99 = max(0.1, svc.http_server_request_duration_p99 * 0.3)
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
            return (f"Network policy reverted on {target}. Packet loss rule removed. TCP retransmit rate normalizing.", False)
        return (f"Network policy reverted on {target}. No active packet loss policy found.", False)

    def _disable_fallback_mode(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Force service to return errors instead of degraded fallback."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Fallback mode disabled on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        svc.process_cpu_utilization = max(0.15, svc.process_cpu_utilization * 0.5)
        return (f"Fallback mode disabled on {target}. Service returning errors instead of degraded fallback responses.", False)

    def _request_quota_increase(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
        action: FirewatchAction, is_wrong: bool,
    ) -> tuple[str, bool]:
        """Increase resource quota for specified dimension."""
        svc = mesh.services[target]
        resource = action.parameters.get("resource", "db_connections")
        if resource not in ("gpu_compute", "bandwidth", "db_connections"):
            resource = "db_connections"
        if is_wrong:
            return (f"Quota increase for {resource} on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service:
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.4)
            return (f"Quota increase approved for {resource} on {target}. quota_remaining_ratio increased by +0.30.", False)
        return (f"Quota increase for {resource} on {target}. No effect on active fault.", False)

    def _force_leader_election(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Trigger immediate leader re-election. Brief storm for 1 tick."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Leader election triggered on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        svc.http_server_error_rate = min(1.0, svc.http_server_error_rate + 0.05)
        return (f"Leader election triggered on {target}. Election in progress (1 tick). New leader elected after.", False)

    def _isolate_minority_nodes(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Remove minority-partition nodes from serving traffic."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Minority nodes isolated on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
        return (f"Minority nodes isolated on {target}. Stale reads eliminated. Nodes removed from serving pool.", False)

    def _redirect_config_reads_to_majority(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Pin config reads to majority-partition nodes."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Config reads pinned to majority on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
        return (f"Config reads pinned to majority partition on {target}. Minority nodes no longer serving reads.", False)

    def _flush_diverged_keys(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Flush keys with write conflicts from cache cluster."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Diverged keys flushed on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
        return (f"Diverged keys flushed on {target}. Conflicted keys removed. Consistency restored.", False)

    def _force_cluster_resync(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Force full cluster resync from canonical master set."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Cluster resync on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service and fc.fault_type == "network_partition":
            self._halt_fault_on(mesh, target, "network_partition")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
            return (f"Full cluster resync initiated on {target}. Estimated completion: 3-5 ticks.", False)
        return (f"Cluster resync initiated on {target} but fault is not partition-related.", False)

    # ------------------------------------------------------------------
    # Phase 2 Hard Tier Remediation (SPEC-05) — Part 2
    # ------------------------------------------------------------------

    def _enable_cache_warming(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Pre-populate cache with hot-key set."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Cache warming enabled on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        if target == fc.root_cause_service and fc.fault_type == "config_drift":
            self._halt_fault_on(mesh, target, "config_drift")
            svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.3)
            svc.http_server_request_duration_p99 = max(0.1, svc.http_server_request_duration_p99 * 0.4)
            return (f"Cache warming enabled on {target}. Hot keys pre-populated. Hit rate recovering.", False)
        return (f"Cache warming enabled on {target} but fault is not cache-related.", False)

    def _rate_limit_cache_misses(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig", is_wrong: bool,
    ) -> tuple[str, bool]:
        """Rate-limit backend queries triggered by cache misses."""
        svc = mesh.services[target]
        if is_wrong:
            return (f"Cache miss rate limit on {target} (error_rate {svc.http_server_error_rate:.4f}). Not degraded.", True)
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.5)
        svc.http_server_active_requests = max(30, svc.http_server_active_requests // 2)
        return (f"Cache miss rate limit applied on {target}. Backend query rate capped. Thundering herd contained.", False)

    def _rebalance_az_traffic(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """Rebalance traffic across availability zones. guard_applies=False."""
        svc = mesh.services[target]
        svc.http_server_error_rate = max(0.01, svc.http_server_error_rate * 0.4)
        svc.http_server_request_duration_p99 = max(0.1, svc.http_server_request_duration_p99 * 0.5)
        return (f"AZ traffic rebalanced for {target}. Cross-zone routing equalized. Latency normalizing.", False)

    def _scale_az_capacity(
        self, target: str, mesh: "ServiceMesh", fc: "FaultConfig",
    ) -> tuple[str, bool]:
        """Add capacity in underprovisioned AZ. guard_applies=False."""
        svc = mesh.services[target]
        svc.http_server_active_requests = max(30, svc.http_server_active_requests // 2)
        svc.process_cpu_utilization = max(0.15, svc.process_cpu_utilization * 0.5)
        return (f"AZ capacity scaled for {target}. New instances provisioning. Request backlog draining.", False)

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
