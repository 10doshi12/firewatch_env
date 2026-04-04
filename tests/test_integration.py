# tests/test_integration.py
# Phase 7 — Integration tests for OpenEnv wiring.
# Validates the acceptance criteria from PRD §12.6.

from __future__ import annotations

import sys
import os

# Ensure the firewatch_env package root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FirewatchAction, SystemObservation
from simulation import generate_episode
from actions import ActionHandler
from rewards import RewardEngine, EpisodeResult, grade
from server.firewatch_env_environment import FirewatchEnvironment


# --------------------------------------------------------------------------
# Test 1: Deterministic reset
# Two calls to reset(easy, 42) return identical initial observations
# --------------------------------------------------------------------------

def test_reset_deterministic():
    """PRD §12.6: Two calls to reset(easy, 42) return byte-identical initial observations."""
    env1 = FirewatchEnvironment()
    env2 = FirewatchEnvironment()

    obs1 = env1.reset(difficulty="easy", seed=42)
    obs2 = env2.reset(difficulty="easy", seed=42)

    # Same services
    assert set(obs1.services.keys()) == set(obs2.services.keys()), \
        f"Service sets differ: {obs1.services.keys()} vs {obs2.services.keys()}"

    # Same metrics on each service
    for name in obs1.services:
        m1 = obs1.services[name]
        m2 = obs2.services[name]
        assert m1.http_server_error_rate == m2.http_server_error_rate, \
            f"Error rate mismatch on {name}: {m1.http_server_error_rate} vs {m2.http_server_error_rate}"
        assert m1.process_memory_utilization == m2.process_memory_utilization, \
            f"Memory util mismatch on {name}: {m1.process_memory_utilization} vs {m2.process_memory_utilization}"
        assert m1.http_server_request_duration_p99 == m2.http_server_request_duration_p99, \
            f"Latency mismatch on {name}"

    # Same dependency graph
    assert obs1.dependency_graph == obs2.dependency_graph

    # Same SLO budget
    assert obs1.slo_budget_remaining_pct == obs2.slo_budget_remaining_pct

    print("✓ test_reset_deterministic PASSED")


# --------------------------------------------------------------------------
# Test 2: Full episode flow
# reset → step(fetch_logs) → step(restart_service) → step(declare_resolved)
# --------------------------------------------------------------------------

def test_full_episode_flow():
    """PRD §12.6: Sequential calls complete without error."""
    env = FirewatchEnvironment()

    # Reset
    obs = env.reset(difficulty="easy", seed=42)
    assert obs.sim_tick == 0
    assert obs.slo_budget_remaining_pct == 100.0
    assert len(obs.services) > 0
    assert obs.done is False

    # Pick a service to investigate
    target = list(obs.services.keys())[0]

    # Step 1: fetch_logs
    action1 = FirewatchAction(action_type="fetch_logs", target_service=target)
    obs1 = env.step(action1)
    assert obs1.sim_tick == 1
    assert obs1.done is False
    assert obs1.reward is not None

    # Step 2: restart_service
    action2 = FirewatchAction(action_type="restart_service", target_service=target)
    obs2 = env.step(action2)
    assert obs2.sim_tick == 2
    assert obs2.done is False

    # Step 3: declare_resolved
    action3 = FirewatchAction(action_type="declare_resolved")
    obs3 = env.step(action3)
    assert obs3.done is True
    assert obs3.reward is not None
    # Episode score should be in metadata
    assert "episode_score" in obs3.metadata, \
        f"episode_score not in metadata: {list(obs3.metadata.keys())}"

    print("✓ test_full_episode_flow PASSED")


# --------------------------------------------------------------------------
# Test 3: Invalid action handling
# step() with invalid input returns valid response, not crash
# --------------------------------------------------------------------------

def test_invalid_action_graceful():
    """PRD §12.6: step() with invalid target returns HTTP 200 with error info."""
    env = FirewatchEnvironment()
    env.reset(difficulty="easy", seed=42)

    # Action with non-existent service
    action = FirewatchAction(
        action_type="fetch_logs",
        target_service="nonexistent-service",
    )
    obs = env.step(action)

    # Should not crash
    assert obs is not None
    assert obs.done is False
    # Should have error/invalid feedback in action history
    assert len(obs.action_history) > 0
    assert "Invalid target" in obs.action_history[-1].get("feedback_string", "") or \
           "not an active service" in obs.action_history[-1].get("feedback_string", "")

    print("✓ test_invalid_action_graceful PASSED")


# --------------------------------------------------------------------------
# Test 4: Wrong action produces negative reward
# --------------------------------------------------------------------------

def test_wrong_action_negative_reward():
    """Remediating a healthy service should produce a wrong-action penalty."""
    env = FirewatchEnvironment()
    obs = env.reset(difficulty="easy", seed=42)

    # Find a healthy service (not the root cause)
    # Run a few ticks first so we have some degradation
    noop_action = FirewatchAction(action_type="fetch_logs", target_service=list(obs.services.keys())[0])
    env.step(noop_action)
    env.step(noop_action)

    # Now pick a service with low error rate
    healthy_services = [
        name for name, m in env._mesh.services.items()
        if m.http_server_error_rate < 0.10
    ]

    if healthy_services:
        target = healthy_services[0]
        action = FirewatchAction(action_type="restart_service", target_service=target)
        obs = env.step(action)
        # Check for wrong action penalty in metadata
        breakdown = obs.metadata.get("reward_breakdown", {})
        assert breakdown.get("wrong_action_penalty", 0.0) < 0.0, \
            f"Expected negative wrong_action_penalty, got {breakdown}"
        print("✓ test_wrong_action_negative_reward PASSED")
    else:
        print("⚠ test_wrong_action_negative_reward SKIPPED (no healthy services found at this seed)")


# --------------------------------------------------------------------------
# Test 5: Grader appears in done info
# --------------------------------------------------------------------------

def test_grader_in_done_info():
    """PRD §12.6: episode_score appears in done=True step's info dict."""
    env = FirewatchEnvironment()
    env.reset(difficulty="easy", seed=42)

    # Immediately declare resolved (worst case agent)
    action = FirewatchAction(action_type="declare_resolved")
    obs = env.step(action)

    assert obs.done is True
    assert "episode_score" in obs.metadata
    score = obs.metadata["episode_score"]
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    # Zero-effort agent should score poorly
    assert score < 0.30, f"Zero-effort score too high: {score}"

    print("✓ test_grader_in_done_info PASSED")


# --------------------------------------------------------------------------
# Test 6: SLO breach terminates episode
# --------------------------------------------------------------------------

def test_slo_breach_terminates():
    """Running enough ticks to deplete SLO causes done=True."""
    env = FirewatchEnvironment()
    env.reset(difficulty="hard", seed=100)

    # Just do noop investigation actions until SLO runs out or max ticks
    target = list(env._mesh.services.keys())[0]
    done = False
    tick = 0
    while not done and tick < 50:
        action = FirewatchAction(action_type="fetch_logs", target_service=target)
        obs = env.step(action)
        done = obs.done
        tick += 1

    assert done is True, f"Episode did not terminate after {tick} ticks"
    # Hard difficulty with 40 max ticks should terminate
    assert tick <= 41, f"Episode took too many ticks: {tick}"

    print("✓ test_slo_breach_terminates PASSED")


# --------------------------------------------------------------------------
# Test 7: Score variance (different agent behaviors yield different scores)
# --------------------------------------------------------------------------

def test_score_variance():
    """Grader must produce meaningfully different scores for different behaviors."""
    # Zero-effort agent: immediately gives up
    env1 = FirewatchEnvironment()
    env1.reset(difficulty="easy", seed=42)
    obs_zero = env1.step(FirewatchAction(action_type="declare_resolved"))
    score_zero = obs_zero.metadata["episode_score"]

    # Active agent: investigates, lets fault develop, remediates, then resolves
    env2 = FirewatchEnvironment()
    obs2 = env2.reset(difficulty="easy", seed=42)
    root_cause = env2._fault_config.root_cause_service
    fault_type = env2._fault_config.fault_type

    # Let the fault develop for a few ticks with investigation
    for svc in list(obs2.services.keys()):
        env2.step(FirewatchAction(action_type="fetch_logs", target_service=svc))

    # Apply correct remediation based on fault type
    if fault_type == "oom":
        env2.step(FirewatchAction(action_type="scale_replicas", target_service=root_cause))
    elif fault_type == "bad_deploy":
        env2.step(FirewatchAction(action_type="rollback_deploy", target_service=root_cause))
    elif fault_type == "config_drift":
        env2.step(FirewatchAction(action_type="revert_config", target_service=root_cause))
    elif fault_type == "memory_leak":
        env2.step(FirewatchAction(action_type="restart_service", target_service=root_cause))
    elif fault_type == "network_partition":
        env2.step(FirewatchAction(action_type="restart_service", target_service=root_cause))

    # Let system recover for a few ticks
    for _ in range(3):
        env2.step(FirewatchAction(action_type="fetch_logs", target_service=root_cause))

    obs_active = env2.step(FirewatchAction(action_type="declare_resolved"))
    score_active = obs_active.metadata["episode_score"]

    # Active agent should score higher than zero-effort
    assert score_active > score_zero, \
        f"Active agent ({score_active:.4f}) should score higher than zero-effort ({score_zero:.4f})"

    print(f"✓ test_score_variance PASSED (zero={score_zero:.4f}, active={score_active:.4f})")


# --------------------------------------------------------------------------
# Test 8: No episode active -> graceful response
# --------------------------------------------------------------------------

def test_no_episode_step():
    """step() without prior reset() should return graceful error."""
    env = FirewatchEnvironment()
    action = FirewatchAction(action_type="fetch_logs", target_service="test")
    obs = env.step(action)

    assert obs is not None
    # Should have error info
    assert len(obs.action_history) > 0 or obs.metadata.get("error")

    print("✓ test_no_episode_step PASSED")


# --------------------------------------------------------------------------
# Run all tests
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_reset_deterministic,
        test_full_episode_flow,
        test_invalid_action_graceful,
        test_wrong_action_negative_reward,
        test_grader_in_done_info,
        test_slo_breach_terminates,
        test_score_variance,
        test_no_episode_step,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All Phase 7 acceptance criteria PASSED ✓")
    else:
        print(f"FAILED — {failed} test(s) need fixing")
    print(f"{'='*60}")
