# test_advanced_actions.py
# Unit tests for SPEC-9 advanced diagnostic actions.
# Tests each of the 7 new action types in isolation.

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

try:
    from ..actions import ActionHandler
    from ..models import FirewatchAction, ServiceMetrics
    from ..simulation import generate_episode
except ImportError:
    from actions import ActionHandler
    from models import FirewatchAction, ServiceMetrics
    from simulation import generate_episode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def oom_episode():
    """Episode with OOM fault."""
    for seed in range(100):
        mesh, fc = generate_episode("easy", seed)
        if fc.fault_type == "oom":
            return mesh, fc
    pytest.skip("No OOM episode found in first 100 seeds")


@pytest.fixture
def memory_leak_episode():
    """Episode with memory_leak fault."""
    for seed in range(200):
        mesh, fc = generate_episode("medium", seed)
        if fc.fault_type == "memory_leak":
            return mesh, fc
    pytest.skip("No memory_leak episode found in first 200 seeds")


@pytest.fixture
def bad_deploy_episode():
    """Episode with bad_deploy fault."""
    for seed in range(100):
        mesh, fc = generate_episode("easy", seed)
        if fc.fault_type == "bad_deploy":
            return mesh, fc
    pytest.skip("No bad_deploy episode found in first 100 seeds")


@pytest.fixture
def handler():
    return ActionHandler()


# ---------------------------------------------------------------------------
# strace_process tests
# ---------------------------------------------------------------------------

class TestStraceProcess:
    def test_oom_root_cause_shows_mmap(self, handler, oom_episode):
        mesh, fc = oom_episode
        action = FirewatchAction(
            action_type="strace_process",
            target_service=fc.root_cause_service,
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "mmap" in feedback.lower()
        assert "%" in feedback
        assert "ROOT CAUSE" in feedback

    def test_healthy_service_shows_baseline(self, handler, oom_episode):
        mesh, fc = oom_episode
        healthy = [s for s, m in mesh.services.items()
                   if s != fc.root_cause_service and m.status == "healthy"]
        if not healthy:
            pytest.skip("No healthy service available")
        action = FirewatchAction(
            action_type="strace_process",
            target_service=healthy[0],
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "Healthy baseline" in feedback

    def test_invalid_target_returns_graceful_error(self, handler, oom_episode):
        mesh, fc = oom_episode
        action = FirewatchAction(
            action_type="strace_process",
            target_service="nonexistent-service",
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "not an active service" in feedback.lower() or "not found" in feedback.lower()

    def test_is_investigation_never_wrong(self, handler, oom_episode):
        mesh, fc = oom_episode
        for svc_name in mesh.services:
            action = FirewatchAction(
                action_type="strace_process",
                target_service=svc_name,
            )
            _, wrong = handler.apply(action, mesh, fc)
            assert wrong is False, f"strace_process should never set wrong_action, failed for {svc_name}"


# ---------------------------------------------------------------------------
# profiler_dump tests
# ---------------------------------------------------------------------------

class TestProfilerDump:
    def test_bad_deploy_shows_cpu_bound(self, handler, bad_deploy_episode):
        mesh, fc = bad_deploy_episode
        action = FirewatchAction(
            action_type="profiler_dump",
            target_service=fc.root_cause_service,
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "cpu" in feedback.lower()
        assert "%" in feedback

    def test_memory_leak_shows_gc_bound(self, handler, memory_leak_episode):
        mesh, fc = memory_leak_episode
        action = FirewatchAction(
            action_type="profiler_dump",
            target_service=fc.root_cause_service,
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "gc" in feedback.lower()

    def test_always_returns_false_wrong(self, handler, oom_episode):
        mesh, fc = oom_episode
        for svc_name in mesh.services:
            action = FirewatchAction(
                action_type="profiler_dump",
                target_service=svc_name,
            )
            _, wrong = handler.apply(action, mesh, fc)
            assert wrong is False


# ---------------------------------------------------------------------------
# check_gc_pressure tests
# ---------------------------------------------------------------------------

class TestCheckGCPressure:
    def test_memory_leak_shows_thrashing(self, handler, memory_leak_episode):
        mesh, fc = memory_leak_episode
        action = FirewatchAction(
            action_type="check_gc_pressure",
            target_service=fc.root_cause_service,
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "thrashing" in feedback.lower() or "critical" in feedback.lower()

    def test_updates_live_gc_metrics(self, handler, memory_leak_episode):
        mesh, fc = memory_leak_episode
        action = FirewatchAction(
            action_type="check_gc_pressure",
            target_service=fc.root_cause_service,
        )
        handler.apply(action, mesh, fc)
        svc = mesh.services[fc.root_cause_service]
        assert svc.runtime_gc_pause_duration_ms > 100
        assert svc.runtime_gc_count_per_second > 5

    def test_healthy_service_shows_normal(self, handler, bad_deploy_episode):
        mesh, fc = bad_deploy_episode
        healthy = [s for s, m in mesh.services.items()
                   if m.process_memory_utilization < 0.50 and s != fc.root_cause_service]
        if not healthy:
            pytest.skip("No healthy low-memory service available")
        action = FirewatchAction(
            action_type="check_gc_pressure",
            target_service=healthy[0],
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "normal" in feedback.lower() or "gc pressure" in feedback.lower()

    def test_always_returns_false_wrong(self, handler, oom_episode):
        mesh, fc = oom_episode
        for svc_name in mesh.services:
            action = FirewatchAction(
                action_type="check_gc_pressure",
                target_service=svc_name,
            )
            _, wrong = handler.apply(action, mesh, fc)
            assert wrong is False


# ---------------------------------------------------------------------------
# trace_distributed_request tests
# ---------------------------------------------------------------------------

class TestTraceDistributedRequest:
    def test_identifies_bottleneck(self, handler):
        mesh, fc = generate_episode("medium", 137)
        for _ in range(5):
            mesh.tick()
        # Use an active service from the episode (not hardcoded)
        target = list(mesh.services.keys())[0]
        action = FirewatchAction(
            action_type="trace_distributed_request",
            target_service=target,
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "bottleneck" in feedback.lower()

    def test_shows_duration_in_ms(self, handler):
        mesh, fc = generate_episode("easy", 42)
        action = FirewatchAction(
            action_type="trace_distributed_request",
            target_service=list(mesh.services.keys())[0],
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "ms" in feedback

    def test_invalid_target_graceful(self, handler):
        mesh, fc = generate_episode("easy", 42)
        action = FirewatchAction(
            action_type="trace_distributed_request",
            target_service="nonexistent",
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False


# ---------------------------------------------------------------------------
# inspect_thread_pool tests
# ---------------------------------------------------------------------------

class TestInspectThreadPool:
    def test_saturated_service_reports_saturation(self, handler):
        mesh, fc = generate_episode("easy", 42)
        target = list(mesh.services.keys())[0]
        mesh.services[target].http_server_active_requests = 190
        action = FirewatchAction(
            action_type="inspect_thread_pool",
            target_service=target,
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "saturated" in feedback.lower() or "200" in feedback

    def test_updates_live_thread_metrics(self, handler):
        mesh, fc = generate_episode("easy", 42)
        target = list(mesh.services.keys())[0]
        mesh.services[target].http_server_active_requests = 190
        action = FirewatchAction(
            action_type="inspect_thread_pool",
            target_service=target,
        )
        handler.apply(action, mesh, fc)
        svc = mesh.services[target]
        assert svc.runtime_jvm_threads_count == 200
        assert svc.runtime_jvm_threads_max == 200

    def test_healthy_service_reports_normal(self, handler):
        mesh, fc = generate_episode("easy", 42)
        target = list(mesh.services.keys())[0]
        mesh.services[target].http_server_active_requests = 30
        action = FirewatchAction(
            action_type="inspect_thread_pool",
            target_service=target,
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "50" in feedback or "healthy" in feedback.lower() or "12" in feedback

    def test_always_returns_false_wrong(self, handler):
        mesh, fc = generate_episode("easy", 42)
        for svc_name in mesh.services:
            action = FirewatchAction(
                action_type="inspect_thread_pool",
                target_service=svc_name,
            )
            _, wrong = handler.apply(action, mesh, fc)
            assert wrong is False


# ---------------------------------------------------------------------------
# traffic_shift tests
# ---------------------------------------------------------------------------

class TestTrafficShift:
    def test_valid_shift_reduces_error_rate(self, handler):
        mesh, fc = generate_episode("easy", 42)
        target = fc.root_cause_service
        mesh.services[target].http_server_error_rate = 0.60
        original_rate = mesh.services[target].http_server_error_rate
        action = FirewatchAction(
            action_type="traffic_shift",
            target_service=target,
            parameters={"drain_percentage": 0.80},
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        new_rate = mesh.services[target].http_server_error_rate
        assert new_rate < original_rate

    def test_adds_latency_penalty(self, handler):
        mesh, fc = generate_episode("easy", 42)
        target = fc.root_cause_service
        mesh.services[target].http_server_error_rate = 0.50
        mesh.services[target].http_server_request_duration_p99 = 1.0
        action = FirewatchAction(
            action_type="traffic_shift",
            target_service=target,
            parameters={"drain_percentage": 0.80},
        )
        handler.apply(action, mesh, fc)
        new_latency = mesh.services[target].http_server_request_duration_p99
        assert abs(new_latency - 1.15) < 0.01

    def test_healthy_service_is_wrong_action(self, handler):
        mesh, fc = generate_episode("easy", 42)
        healthy = [s for s, m in mesh.services.items() if m.http_server_error_rate < 0.10]
        if not healthy:
            pytest.skip("No healthy service")
        action = FirewatchAction(
            action_type="traffic_shift",
            target_service=healthy[0],
            parameters={"drain_percentage": 0.50},
        )
        _, wrong = handler.apply(action, mesh, fc)
        assert wrong is True

    def test_drain_below_minimum_rejected(self, handler):
        mesh, fc = generate_episode("easy", 42)
        target = fc.root_cause_service
        mesh.services[target].http_server_error_rate = 0.50
        action = FirewatchAction(
            action_type="traffic_shift",
            target_service=target,
            parameters={"drain_percentage": 0.02},
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert "too low" in feedback.lower() or "minimum" in feedback.lower()

    def test_drain_above_maximum_clamped(self, handler):
        mesh, fc = generate_episode("easy", 42)
        target = fc.root_cause_service
        mesh.services[target].http_server_error_rate = 0.50
        action = FirewatchAction(
            action_type="traffic_shift",
            target_service=target,
            parameters={"drain_percentage": 0.99},
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert "clamped" in feedback.lower() or "95" in feedback

    def test_missing_drain_percentage_uses_default(self, handler):
        mesh, fc = generate_episode("easy", 42)
        target = fc.root_cause_service
        mesh.services[target].http_server_error_rate = 0.50
        original_rate = mesh.services[target].http_server_error_rate
        action = FirewatchAction(
            action_type="traffic_shift",
            target_service=target,
            parameters={},
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        # Default is 0.80, so error rate should drop
        new_rate = mesh.services[target].http_server_error_rate
        assert new_rate < original_rate


# ---------------------------------------------------------------------------
# inspect_commit_diff tests
# ---------------------------------------------------------------------------

class TestInspectCommitDiff:
    def test_bad_deploy_root_cause_shows_diff(self, handler, bad_deploy_episode):
        mesh, fc = bad_deploy_episode
        action = FirewatchAction(
            action_type="inspect_commit_diff",
            target_service=fc.root_cause_service,
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "SHA" in feedback.upper() or "sha" in feedback.lower()
        assert "---" in feedback or "+++" in feedback or "Recommend rollback" in feedback

    def test_non_root_cause_shows_clean(self, handler, oom_episode):
        mesh, fc = oom_episode
        # OOM fault → inspect_commit_diff on root cause should show clean (not bad_deploy)
        action = FirewatchAction(
            action_type="inspect_commit_diff",
            target_service=fc.root_cause_service,
        )
        feedback, wrong = handler.apply(action, mesh, fc)
        assert wrong is False
        assert "not the root cause" in feedback.lower() or "no anomalous" in feedback.lower()

    def test_includes_deployment_sha(self, handler, bad_deploy_episode):
        mesh, fc = bad_deploy_episode
        action = FirewatchAction(
            action_type="inspect_commit_diff",
            target_service=fc.root_cause_service,
        )
        feedback, _ = handler.apply(action, mesh, fc)
        sha = mesh.services[fc.root_cause_service].last_deployment_sha
        assert sha in feedback

    def test_always_returns_false_wrong(self, handler, bad_deploy_episode):
        mesh, fc = bad_deploy_episode
        for svc_name in mesh.services:
            action = FirewatchAction(
                action_type="inspect_commit_diff",
                target_service=svc_name,
            )
            _, wrong = handler.apply(action, mesh, fc)
            assert wrong is False
