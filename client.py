# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Firewatch Env Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FirewatchAction, SystemObservation


class FirewatchEnv(
    EnvClient[FirewatchAction, SystemObservation, State]
):
    """
    Client for the Firewatch Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with FirewatchEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(difficulty="easy", seed=42)
        ...     print(result.observation.sim_tick)
        ...
        ...     action = FirewatchAction(action_type="fetch_logs", target_service="auth-service")
        ...     result = client.step(action)
        ...     print(result.observation.slo_budget_remaining_pct)
    """

    def _step_payload(self, action: FirewatchAction) -> Dict:
        """
        Convert FirewatchAction to JSON payload for step message.

        Args:
            action: FirewatchAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
        }
        if action.target_service is not None:
            payload["target_service"] = action.target_service
        if action.parameters:
            payload["parameters"] = action.parameters
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[SystemObservation]:
        """
        Parse server response into StepResult[SystemObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SystemObservation
        """
        obs_data = payload.get("observation", {})
        observation = SystemObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
