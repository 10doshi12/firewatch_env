#!/usr/bin/env python3
"""
test_inference.py — Phase 8 acceptance tests for inference.py.
Tests stdout format compliance without making actual LLM calls.
"""

from __future__ import annotations

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    fmt_reward,
    fmt_done,
    fmt_success,
    fmt_score,
    fmt_rewards_list,
    fmt_action,
    summarize_observation,
    parse_llm_response,
    SYSTEM_PROMPT,
    SUCCESS_SCORE_THRESHOLD,
)
from models import FirewatchAction
from server.firewatch_env_environment import FirewatchEnvironment


def test_format_reward():
    """Reward formatted to exactly 2 decimal places."""
    assert fmt_reward(0.854) == "0.85"
    assert fmt_reward(0.0) == "0.00"
    assert fmt_reward(None) == "0.00"
    assert fmt_reward(-0.1) == "-0.10"
    assert fmt_reward(1.0) == "1.00"
    print("✓ test_format_reward PASSED")


def test_format_done():
    """done is lowercase true/false (not Python True/False)."""
    assert fmt_done(True) == "true"
    assert fmt_done(False) == "false"
    # Ensure it's not Python-style
    assert fmt_done(True) != "True"
    print("✓ test_format_done PASSED")


def test_format_success():
    """success is lowercase true/false."""
    assert fmt_success(True) == "true"
    assert fmt_success(False) == "false"
    print("✓ test_format_success PASSED")


def test_format_score():
    """score formatted to exactly 3 decimal places."""
    assert fmt_score(0.8234) == "0.823"
    assert fmt_score(0.0) == "0.000"
    assert fmt_score(1.0) == "1.000"
    print("✓ test_format_score PASSED")


def test_format_rewards_list():
    """rewards comma-separated with 2 decimal places."""
    assert fmt_rewards_list([0.0, 0.5, 0.85, -0.1]) == "0.00,0.50,0.85,-0.10"
    assert fmt_rewards_list([]) == ""
    assert fmt_rewards_list([1.0]) == "1.00"
    print("✓ test_format_rewards_list PASSED")


def test_format_action():
    """action formatted as action_type:target_service."""
    a1 = FirewatchAction(action_type="fetch_logs", target_service="auth-service")
    assert fmt_action(a1) == "fetch_logs:auth-service"

    a2 = FirewatchAction(action_type="declare_resolved")
    assert fmt_action(a2) == "declare_resolved"
    print("✓ test_format_action PASSED")


def test_parse_json_response():
    """Parse clean JSON response."""
    resp = '{"action_type": "restart_service", "target_service": "cache"}'
    action = parse_llm_response(resp, ["cache", "db"])
    assert action.action_type == "restart_service"
    assert action.target_service == "cache"
    print("✓ test_parse_json_response PASSED")


def test_parse_markdown_wrapped():
    """Parse JSON wrapped in markdown code blocks."""
    resp = '```json\n{"action_type": "fetch_logs", "target_service": "db"}\n```'
    action = parse_llm_response(resp, ["cache", "db"])
    assert action.action_type == "fetch_logs"
    assert action.target_service == "db"
    print("✓ test_parse_markdown_wrapped PASSED")


def test_parse_fallback():
    """Fallback to fetch_logs on unparseable response."""
    resp = "I think we should restart the auth service because of high latency"
    action = parse_llm_response(resp, ["auth-service", "db"])
    assert action.action_type == "fetch_logs"
    assert action.target_service == "auth-service"
    print("✓ test_parse_fallback PASSED")


def test_parse_with_extra_text():
    """Parse JSON embedded in explanation text."""
    resp = 'Based on the metrics, I recommend:\n\n{"action_type": "rollback_deploy", "target_service": "api-gateway"}\n\nThis should fix the issue.'
    action = parse_llm_response(resp, ["api-gateway"])
    assert action.action_type == "rollback_deploy"
    assert action.target_service == "api-gateway"
    print("✓ test_parse_with_extra_text PASSED")


def test_summarize_under_400_tokens():
    """Observation summary stays under 400 tokens (~1600 chars)."""
    env = FirewatchEnvironment()
    obs = env.reset(difficulty="hard", seed=256)

    # After a few ticks
    for _ in range(3):
        target = list(obs.services.keys())[0]
        obs = env.step(FirewatchAction(action_type="fetch_logs", target_service=target))

    history = [
        {"action_type": "fetch_logs", "target_service": "svc1", "feedback_string": "Fetched 5 logs"},
        {"action_type": "get_metrics_detail", "target_service": "svc2", "feedback_string": "Error rate trending up"},
        {"action_type": "restart_service", "target_service": "svc1", "feedback_string": "Restarted"},
    ]
    summary = summarize_observation(obs, history)

    # rough token estimate: 1 token ≈ 4 chars
    estimated_tokens = len(summary) / 4
    assert estimated_tokens < 400, f"Summary too long: ~{estimated_tokens:.0f} tokens ({len(summary)} chars)"
    print(f"✓ test_summarize_under_400_tokens PASSED (~{estimated_tokens:.0f} tokens)")


def test_stdout_format_compliance():
    """Full stdout output matches exact spec format."""
    env = FirewatchEnvironment()
    obs = env.reset(difficulty="easy", seed=42)

    target = list(obs.services.keys())[0]

    # Simulate one task run
    step_lines = []
    actions_taken = [
        FirewatchAction(action_type="fetch_logs", target_service=target),
        FirewatchAction(action_type="declare_resolved"),
    ]

    rewards = []
    for i, action in enumerate(actions_taken, 1):
        obs = env.step(action)
        reward = obs.reward or 0.0
        rewards.append(reward)
        line = f"[STEP] step={i} action={fmt_action(action)} reward={fmt_reward(reward)} done={fmt_done(obs.done)} error=null"
        step_lines.append(line)

    # Verify START line format
    start_line = "[START] task=task_easy env=firewatch-env model=test-model"
    assert re.match(r"^\[START\] task=\S+ env=\S+ model=\S+$", start_line), f"Bad START: {start_line}"

    # Verify STEP line format
    for line in step_lines:
        assert re.match(
            r"^\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d{2} done=(true|false) error=\S+$",
            line
        ), f"Bad STEP: {line}"

    # Verify END line format
    score = obs.metadata.get("episode_score", 0.0)
    success = score >= SUCCESS_SCORE_THRESHOLD
    end_line = f"[END] success={fmt_success(success)} steps={len(actions_taken)} score={fmt_score(score)} rewards={fmt_rewards_list(rewards)}"
    assert re.match(
        r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=(-?\d+\.\d{2},?)+$",
        end_line
    ), f"Bad END: {end_line}"

    print("✓ test_stdout_format_compliance PASSED")


def test_system_prompt_completeness():
    """System prompt contains all 10 action types."""
    action_types = [
        "fetch_logs", "get_metrics_detail", "trace_dependencies",
        "restart_service", "rollback_deploy", "revert_config",
        "scale_replicas", "circuit_break", "declare_resolved", "escalate",
    ]
    for at in action_types:
        assert at in SYSTEM_PROMPT, f"Missing action {at} in system prompt"
    print("✓ test_system_prompt_completeness PASSED")


if __name__ == "__main__":
    tests = [
        test_format_reward,
        test_format_done,
        test_format_success,
        test_format_score,
        test_format_rewards_list,
        test_format_action,
        test_parse_json_response,
        test_parse_markdown_wrapped,
        test_parse_fallback,
        test_parse_with_extra_text,
        test_summarize_under_400_tokens,
        test_stdout_format_compliance,
        test_system_prompt_completeness,
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
        print("All Phase 8 acceptance criteria PASSED ✓")
    else:
        print(f"FAILED — {failed} test(s) need fixing")
    print(f"{'='*60}")
