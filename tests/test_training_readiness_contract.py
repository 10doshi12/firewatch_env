from __future__ import annotations

import os
import sys
from typing import get_args

import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ActionType, FirewatchAction
from config import ACTION_REGISTRY, TASKS
from server import firewatch_env_environment as env_module
from server.app import app
from server.firewatch_env_environment import FirewatchEnvironment


def _load_openenv_yaml() -> dict:
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "openenv.yaml",
    )
    with open(spec_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_reset_resolves_task_by_grader_seed() -> None:
    """Evaluator grader_seed resets should select the matching registered task."""
    env = FirewatchEnvironment()

    env.reset(difficulty="easy", seed=315)

    assert env._task_id == "task_easy_quota_runaway"


def test_task_seed_matches_grader_seed_for_registered_tasks() -> None:
    """A single seed identity avoids ambiguous reset behavior for training."""
    mismatches = [
        task_id
        for task_id, task in TASKS.items()
        if task.seed != task.grader_seed
    ]

    assert mismatches == []


def test_done_grading_receives_final_services(monkeypatch) -> None:
    """Task-specific grader conditions need access to final service state."""
    captured: dict[str, object] = {}

    def fake_grade(episode_result, difficulty, task_id=None, services=None):
        captured["difficulty"] = difficulty
        captured["task_id"] = task_id
        captured["services"] = services
        return 0.42

    monkeypatch.setattr(env_module, "grade", fake_grade)

    env = FirewatchEnvironment()
    env.reset(difficulty="easy", seed=315)
    obs = env.step(FirewatchAction(action_type="declare_resolved"))

    assert obs.done is True
    assert captured["task_id"] == "task_easy_quota_runaway"
    assert isinstance(captured["services"], dict)
    assert captured["services"]


def test_openenv_yaml_lists_all_registered_tasks() -> None:
    spec = _load_openenv_yaml()
    yaml_tasks = {task["id"]: task for task in spec["tasks"]}

    assert set(yaml_tasks) == set(TASKS)
    for task_id, task in TASKS.items():
        assert yaml_tasks[task_id]["difficulty"] == task.difficulty
        assert yaml_tasks[task_id]["grader_seed"] == task.grader_seed


def test_legacy_tasks_are_not_in_active_spec() -> None:
    spec = _load_openenv_yaml()
    yaml_tasks = {task["id"] for task in spec["tasks"]}

    legacy_ids = {"task_easy", "task_medium", "task_hard"}
    assert legacy_ids.isdisjoint(TASKS)
    assert legacy_ids.isdisjoint(yaml_tasks)


def test_action_schemas_match_registry() -> None:
    spec = _load_openenv_yaml()
    yaml_actions = set(
        spec["action_space"]["properties"]["action_type"]["enum"]
    )
    model_actions = set(get_args(ActionType))

    assert yaml_actions == set(ACTION_REGISTRY)
    assert model_actions == set(ACTION_REGISTRY)


def test_http_step_preserves_rich_info() -> None:
    client = TestClient(app)

    reset_resp = client.post(
        "/reset",
        json={"difficulty": "easy", "seed": 315, "task_id": "task_easy_quota_runaway"},
    )
    assert reset_resp.status_code == 200

    step_resp = client.post(
        "/step",
        json={
            "action": {
                "action_type": "fetch_logs",
                "target_service": "notification-service",
            }
        },
    )
    assert step_resp.status_code == 200
    payload = step_resp.json()

    assert "info" in payload
    assert "action_feedback" in payload["info"]
    assert "reward_breakdown" in payload["info"]
    assert payload["info"]["action_valid"] is True


def test_scale_replicas_halts_memory_leak_fault() -> None:
    env = FirewatchEnvironment()
    env.reset(
        difficulty="easy",
        seed=178,
        task_id="task_easy_fail_slow_memleak",
    )

    obs = env.step(
        FirewatchAction(
            action_type="scale_replicas",
            target_service="payment-service",
        )
    )

    assert "Memory leak" in obs.action_history[-1]["feedback_string"]
    assert all(
        fault.halted
        for fault in env._mesh.active_faults
        if fault.fault_service == "payment-service"
        and fault.fault_type == "memory_leak"
    )
