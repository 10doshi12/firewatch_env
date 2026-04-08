# test_full_episode_with_advanced.py
# Integration test: full episode using advanced diagnostic actions.
# Validates that new actions work correctly end-to-end via the environment.

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

try:
    from ..simulation import generate_episode
    from ..actions import ActionHandler
    from ..models import FirewatchAction
except ImportError:
    from simulation import generate_episode
    from actions import ActionHandler
    from models import FirewatchAction


def test_advanced_actions_in_episode_no_crash():
    """
    All 7 advanced actions can be called in sequence without error.
    Tests the full apply() dispatch path for each new action type.
    """
    mesh, fc = generate_episode("medium", 42)
    handler = ActionHandler()
    handler.record_tick(mesh)

    target = fc.root_cause_service
    new_action_types = [
        "strace_process",
        "profiler_dump",
        "check_gc_pressure",
        "inspect_thread_pool",
        "inspect_commit_diff",
        "trace_distributed_request",
    ]

    for action_type in new_action_types:
        action = FirewatchAction(action_type=action_type, target_service=target)
        feedback, wrong = handler.apply(action, mesh, fc)
        assert isinstance(feedback, str), f"{action_type} did not return a string"
        assert len(feedback) > 0, f"{action_type} returned empty feedback"
        assert wrong is False, f"{action_type} should be investigation-only"

    # traffic_shift separately — it's a remediation action
    mesh.services[target].http_server_error_rate = 0.50
    action = FirewatchAction(
        action_type="traffic_shift",
        target_service=target,
        parameters={"drain_percentage": 0.80},
    )
    feedback, wrong = handler.apply(action, mesh, fc)
    assert isinstance(feedback, str)
    assert len(feedback) > 0
    assert wrong is False  # service was degraded, so not wrong


def test_investigation_actions_do_not_mutate_error_rate():
    """
    All investigation actions must NOT change the service's http_server_error_rate.
    Only traffic_shift and other remediation actions should mutate state.
    """
    mesh, fc = generate_episode("easy", 42)
    handler = ActionHandler()
    handler.record_tick(mesh)

    target = fc.root_cause_service
    original_error_rate = mesh.services[target].http_server_error_rate

    investigation_actions = [
        "strace_process",
        "profiler_dump",
        "check_gc_pressure",
        "inspect_thread_pool",
        "inspect_commit_diff",
        "trace_distributed_request",
    ]

    for action_type in investigation_actions:
        action = FirewatchAction(action_type=action_type, target_service=target)
        handler.apply(action, mesh, fc)
        current_rate = mesh.services[target].http_server_error_rate
        assert current_rate == original_error_rate, (
            f"{action_type} mutated http_server_error_rate: "
            f"{original_error_rate} → {current_rate}"
        )


def test_traffic_shift_reduces_error_rate_in_episode():
    """
    traffic_shift must reduce error_rate on a degraded service.
    """
    mesh, fc = generate_episode("medium", 137)
    # Advance a few ticks so fault has propagated
    for _ in range(3):
        mesh.tick()

    handler = ActionHandler()
    target = fc.root_cause_service
    mesh.services[target].http_server_error_rate = 0.45

    before_rate = mesh.services[target].http_server_error_rate
    action = FirewatchAction(
        action_type="traffic_shift",
        target_service=target,
        parameters={"drain_percentage": 0.80},
    )
    feedback, wrong = handler.apply(action, mesh, fc)
    after_rate = mesh.services[target].http_server_error_rate

    assert wrong is False
    assert after_rate < before_rate
    assert "traffic" in feedback.lower() or "drain" in feedback.lower()


def test_all_17_actions_dispatchable():
    """
    All 17 action types dispatch without raising exceptions.
    """
    mesh, fc = generate_episode("hard", 256)
    handler = ActionHandler()
    handler.record_tick(mesh)

    target = fc.root_cause_service
    all_actions = [
        # Original 10
        ("fetch_logs", target),
        ("get_metrics_detail", target),
        ("trace_dependencies", target),
        ("restart_service", target),
        ("rollback_deploy", target),
        ("revert_config", target),
        ("scale_replicas", target),
        ("circuit_break", target),
        ("declare_resolved", None),
        ("escalate", None),
        # New SPEC-9 actions
        ("strace_process", target),
        ("profiler_dump", target),
        ("check_gc_pressure", target),
        ("trace_distributed_request", target),
        ("inspect_thread_pool", target),
        ("inspect_commit_diff", target),
        ("traffic_shift", target),
    ]

    for action_type, target_svc in all_actions:
        params = {"drain_percentage": 0.80} if action_type == "traffic_shift" else {}
        action = FirewatchAction(
            action_type=action_type,
            target_service=target_svc,
            parameters=params,
        )
        try:
            feedback, wrong = handler.apply(action, mesh, fc)
            assert isinstance(feedback, str), f"{action_type} feedback is not str"
        except Exception as e:
            pytest.fail(f"{action_type} raised an exception: {e}")
