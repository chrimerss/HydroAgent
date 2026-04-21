"""Unit tests for reward functions."""

import pytest

from hydrollm.reward import (
    extract_nse_history,
    count_invalid_tool_calls,
    hydro_trajectory_reward,
)


class TestExtractNSEHistory:
    """Test NSE extraction from trajectory messages."""

    def test_extract_from_tool_result(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "ok", "nse": 0.65}'},
            {"role": "tool", "content": '{"status": "ok", "nse": 0.78}'},
        ]
        history = extract_nse_history(trajectory)
        assert history == [0.65, 0.78]

    def test_extract_empty_trajectory(self):
        assert extract_nse_history([]) == []

    def test_extract_no_nse_messages(self):
        trajectory = [
            {"role": "assistant", "content": "Let me analyze the results."},
            {"role": "tool", "content": '{"status": "ok", "validated_params": {}}'},
        ]
        assert extract_nse_history(trajectory) == []

    def test_extract_with_errors(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "error", "nse": -1.0}'},
            {"role": "tool", "content": '{"status": "ok", "nse": 0.72}'},
        ]
        history = extract_nse_history(trajectory)
        assert history == [-1.0, 0.72]


class TestCountInvalidToolCalls:
    """Test counting invalid tool calls."""

    def test_no_errors(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "ok", "nse": 0.65}'},
        ]
        assert count_invalid_tool_calls(trajectory) == 0

    def test_with_errors(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "error", "message": "timeout"}'},
            {"role": "tool", "content": '{"status": "ok", "nse": 0.65}'},
            {"role": "tool", "content": '{"status": "error", "message": "bad params"}'},
        ]
        assert count_invalid_tool_calls(trajectory) == 2


class TestHydroTrajectoryReward:
    """Test the trajectory-level reward function."""

    def test_no_nse_produced(self):
        rewards = hydro_trajectory_reward(
            completions=["I couldn't run the model."],
            trajectory_inputs=[[]],
        )
        assert rewards == [-1.0]

    def test_basic_nse_reward(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "ok", "nse": 0.6}'},
        ]
        rewards = hydro_trajectory_reward(
            completions=["done"],
            trajectory_inputs=[trajectory],
        )
        # 0.6 (best NSE) - 0.02 (1 run efficiency) = 0.58
        assert rewards[0] == pytest.approx(0.58)

    def test_target_bonus(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "ok", "nse": 0.85}'},
        ]
        rewards = hydro_trajectory_reward(
            completions=["done"],
            trajectory_inputs=[trajectory],
        )
        # 0.85 + 0.5 (target bonus) - 0.02 (efficiency) = 1.33
        assert rewards[0] == pytest.approx(1.33)

    def test_improvement_bonus(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "ok", "nse": 0.3}'},
            {"role": "tool", "content": '{"status": "ok", "nse": 0.6}'},
        ]
        rewards = hydro_trajectory_reward(
            completions=["done"],
            trajectory_inputs=[trajectory],
        )
        # best=0.6 + 0.2 (improvement) - 0.04 (2 runs) = 0.76
        assert rewards[0] == pytest.approx(0.76)

    def test_no_improvement_no_bonus(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "ok", "nse": 0.6}'},
            {"role": "tool", "content": '{"status": "ok", "nse": 0.4}'},
        ]
        rewards = hydro_trajectory_reward(
            completions=["done"],
            trajectory_inputs=[trajectory],
        )
        # best=0.6 + 0 (no improvement: 0.4 < 0.6) - 0.04 = 0.56
        assert rewards[0] == pytest.approx(0.56)

    def test_error_penalty(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "error", "message": "timeout"}'},
            {"role": "tool", "content": '{"status": "ok", "nse": 0.5}'},
        ]
        rewards = hydro_trajectory_reward(
            completions=["done"],
            trajectory_inputs=[trajectory],
        )
        # Error message has no "nse" key → nse_history = [0.5] (only from 2nd msg)
        # reward = 0.5 (best NSE) - 0.5 (1 error penalty) - 0.02 (1 nse run) = -0.02
        assert rewards[0] == pytest.approx(-0.02)

    def test_nse_clipped(self):
        trajectory = [
            {"role": "tool", "content": '{"status": "ok", "nse": -5.0}'},
        ]
        rewards = hydro_trajectory_reward(
            completions=["done"],
            trajectory_inputs=[trajectory],
        )
        # Clipped to -1.0 - 0.02 = -1.02
        assert rewards[0] == pytest.approx(-1.02)

    def test_multiple_rollouts(self):
        traj1 = [{"role": "tool", "content": '{"status": "ok", "nse": 0.5}'}]
        traj2 = [{"role": "tool", "content": '{"status": "ok", "nse": 0.9}'}]

        rewards = hydro_trajectory_reward(
            completions=["done", "done"],
            trajectory_inputs=[traj1, traj2],
        )
        assert len(rewards) == 2
        assert rewards[1] > rewards[0]  # Higher NSE → higher reward
