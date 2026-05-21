"""Config loading and object construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from adversarial_queueing.algorithms.amq import AMQConfig
from adversarial_queueing.algorithms.nnq import NNQConfig
from adversarial_queueing.envs.polling import PollingConfig
from adversarial_queueing.envs.routing import RoutingConfig
from adversarial_queueing.envs.service_rate_control import ServiceRateControlConfig
from adversarial_queueing.evaluation.bvi_sensitivity import BVISensitivityConfig
from adversarial_queueing.evaluation.policy_grid import PolicyGridConfig
from adversarial_queueing.evaluation.rollout import EvaluationConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def build_service_rate_config(data: dict[str, Any]) -> ServiceRateControlConfig:
    env = data["env"]
    bvi = data.get("bvi", {})
    return ServiceRateControlConfig(
        lambda_arrival=float(env["lambda_arrival"]),
        mu_levels=tuple(float(x) for x in env["mu_levels"]),
        service_costs=tuple(float(x) for x in env["service_costs"]),
        gamma=float(env.get("gamma", 0.95)),
        q_congestion=float(env.get("q_congestion", 1.0)),
        attack_cost=float(env.get("attack_cost", 0.5)),
        defend_cost=float(env.get("defend_cost", 0.2)),
        initial_state=int(env.get("initial_state", 0)),
        uniformization_rate=(
            None
            if env.get("uniformization_rate") is None
            else float(env["uniformization_rate"])
        ),
        low_threshold=int(env.get("low_threshold", 5)),
        high_threshold=int(env.get("high_threshold", 15)),
        bvi_max_queue_length=int(bvi.get("max_queue_length", 20)),
        boundary_mode=str(bvi.get("boundary_mode", "clip")),
    )


def build_routing_config(data: dict[str, Any]) -> RoutingConfig:
    env = data["env"]
    bvi = data.get("bvi", {})
    return RoutingConfig(
        lambda_arrival=float(env["lambda_arrival"]),
        mu_rates=tuple(float(x) for x in env["mu_rates"]),
        gamma=float(env.get("gamma", 0.95)),
        attack_cost=float(env.get("attack_cost", 0.5)),
        defend_cost=float(env.get("defend_cost", 0.2)),
        congestion_cost=str(env.get("congestion_cost", "sum")),
        initial_state=(
            None
            if env.get("initial_state") is None
            else tuple(int(x) for x in env["initial_state"])
        ),
        uniformization_rate=(
            None
            if env.get("uniformization_rate") is None
            else float(env["uniformization_rate"])
        ),
        bvi_max_queue_length=int(bvi.get("max_queue_length", 20)),
        boundary_mode=str(bvi.get("boundary_mode", "clip")),
    )


def build_polling_config(data: dict[str, Any]) -> PollingConfig:
    env = data["env"]
    bvi = data.get("bvi", {})
    return PollingConfig(
        lambda_arrivals=tuple(float(x) for x in env["lambda_arrivals"]),
        mu_service=float(env["mu_service"]),
        gamma=float(env.get("gamma", 0.95)),
        attack_cost=float(env.get("attack_cost", 0.5)),
        defend_cost=float(env.get("defend_cost", 0.2)),
        switch_cost=float(env.get("switch_cost", 0.1)),
        queue_cost=str(env.get("queue_cost", "sum")),
        initial_queues=(
            None
            if env.get("initial_queues") is None
            else tuple(int(x) for x in env["initial_queues"])
        ),
        initial_position=int(env.get("initial_position", 0)),
        uniformization_rate=(
            None
            if env.get("uniformization_rate") is None
            else float(env["uniformization_rate"])
        ),
        bvi_max_queue_length=int(bvi.get("max_queue_length", 3)),
        boundary_mode=str(bvi.get("boundary_mode", "clip")),
    )


def build_amq_config(data: dict[str, Any]) -> AMQConfig:
    amq = data.get("amq", {})
    return AMQConfig(
        feature_set=str(amq.get("feature_set", "basic_quadratic")),
        total_steps=int(amq.get("total_steps", 100)),
        eta0=float(amq.get("eta0", 0.01)),
        learning_rate_schedule=str(amq.get("learning_rate_schedule", "constant")),
        decay_power=float(amq.get("decay_power", 0.6)),
        seed=int(amq.get("seed", 0)),
        log_interval=int(amq.get("log_interval", 10)),
        weight_clip=(
            None
            if amq.get("weight_clip") is None
            else float(amq["weight_clip"])
        ),
        exploring_starts_probability=float(amq.get("exploring_starts_probability", 0.0)),
        exploring_starts_max_queue_length=(
            None
            if amq.get("exploring_starts_max_queue_length") is None
            else int(amq["exploring_starts_max_queue_length"])
        ),
        fitted_calibration_passes=int(amq.get("fitted_calibration_passes", 0)),
        fitted_calibration_max_queue_length=(
            None
            if amq.get("fitted_calibration_max_queue_length") is None
            else int(amq["fitted_calibration_max_queue_length"])
        ),
        fitted_calibration_eta=(
            None
            if amq.get("fitted_calibration_eta") is None
            else float(amq["fitted_calibration_eta"])
        ),
        fitted_calibration_method=str(
            amq.get("fitted_calibration_method", "semi_gradient")
        ),
        fitted_calibration_l2=float(amq.get("fitted_calibration_l2", 0.0)),
    )


def build_nnq_config(data: dict[str, Any]) -> NNQConfig:
    nnq = data.get("nnq", {})
    return NNQConfig(
        hidden_size=int(nnq.get("hidden_size", 32)),
        learning_rate=float(nnq.get("learning_rate", 0.001)),
        total_steps=int(nnq.get("total_steps", 200)),
        batch_size=int(nnq.get("batch_size", 16)),
        replay_capacity=int(nnq.get("replay_capacity", 1000)),
        target_update_interval=int(nnq.get("target_update_interval", 50)),
        epsilon=float(nnq.get("epsilon", 0.2)),
        seed=int(nnq.get("seed", 0)),
        log_interval=int(nnq.get("log_interval", 20)),
        state_scale=float(nnq.get("state_scale", 10.0)),
        state_feature_set=str(nnq.get("state_feature_set", "env")),
        forced_defender_action_probability=float(
            nnq.get("forced_defender_action_probability", 0.0)
        ),
        forced_defender_action=(
            None
            if nnq.get("forced_defender_action") is None
            else int(nnq["forced_defender_action"])
        ),
        exploring_starts_probability=float(nnq.get("exploring_starts_probability", 0.0)),
        exploring_starts_max_queue_length=(
            None
            if nnq.get("exploring_starts_max_queue_length") is None
            else int(nnq["exploring_starts_max_queue_length"])
        ),
        backup_mode=str(nnq.get("backup_mode", "sampled")),
    )


def build_evaluation_config(data: dict[str, Any]) -> EvaluationConfig:
    evaluation = data.get("evaluation", {})
    return EvaluationConfig(
        num_episodes=int(evaluation.get("num_episodes", 5)),
        horizon=int(evaluation.get("horizon", 25)),
        seed=int(evaluation.get("seed", 0)),
        tail_threshold=int(evaluation.get("tail_threshold", 8)),
        boundary_state=(
            None
            if evaluation.get("boundary_state") is None
            else int(evaluation["boundary_state"])
        ),
    )


def build_policy_grid_config(data: dict[str, Any]) -> PolicyGridConfig:
    policy_grid = data.get("policy_grid", {})
    return PolicyGridConfig(
        max_state=int(policy_grid.get("max_state", 10)),
        high_probability_threshold=float(
            policy_grid.get("high_probability_threshold", 0.5)
        ),
    )


def build_bvi_sensitivity_config(data: dict[str, Any]) -> BVISensitivityConfig:
    sensitivity = data.get("bvi_sensitivity", {})
    return BVISensitivityConfig(
        max_queue_lengths=tuple(int(x) for x in sensitivity["max_queue_lengths"]),
        tolerance=float(sensitivity.get("tolerance", 1e-6)),
        max_iterations=int(sensitivity.get("max_iterations", 2000)),
        boundary_mode=str(sensitivity.get("boundary_mode", "clip")),
        eval_state=int(sensitivity.get("eval_state", 0)),
    )
