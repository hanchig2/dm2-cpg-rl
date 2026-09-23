# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import rsl_rl.runners.on_policy_runner as on_policy_runner_module

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg

from modules.dm2_actor_critic import (
    DM2ActorCriticRecurrent,
    DM2MessageActorCriticRecurrent,
    DM2StructuredMessageActorCriticRecurrent,
    DM2FrozenResidualActorCriticRecurrent,
)

from .rsl_rl_ppo_cfg import (
    CPGUnitreeA1FlatPPORunnerCfg,
    CPGUnitreeA1MixedPPORunnerCfg,
)


# OnPolicyRunner resolves the policy from its string class name.
# Register the local DM2 policy without modifying installed RSL-RL files.
on_policy_runner_module.DM2ActorCriticRecurrent = (
    DM2ActorCriticRecurrent
)
on_policy_runner_module.DM2MessageActorCriticRecurrent = (
    DM2MessageActorCriticRecurrent
)
on_policy_runner_module.DM2StructuredMessageActorCriticRecurrent = (
    DM2StructuredMessageActorCriticRecurrent
)
on_policy_runner_module.DM2FrozenResidualActorCriticRecurrent = (
    DM2FrozenResidualActorCriticRecurrent
)


@configclass
class DM2CPGUnitreeA1FlatPPORunnerCfg(
    CPGUnitreeA1FlatPPORunnerCfg
):
    """PPO configuration for flat-terrain DM2-CPG-RL."""

    experiment_name = "dm2_v7_yaw_balanced_cpg_unitree_a1_flat"

    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="DM2ActorCriticRecurrent",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

@configclass
class DM2CPGUnitreeA1MixedPPORunnerCfg(
    CPGUnitreeA1MixedPPORunnerCfg
):
    """V8 DM2 static mixed-terrain PPO configuration."""

    experiment_name = (
        "dm2_v8_static_mixed_cpg_unitree_a1"
    )

    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="DM2ActorCriticRecurrent",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

@configclass
class DM2MessageCPGUnitreeA1MixedPPORunnerCfg(
    DM2CPGUnitreeA1MixedPPORunnerCfg
):
    """V12 message-passing DM2 mixed-terrain configuration."""

    experiment_name = (
        "dm2_v12_message_static_mixed_cpg_unitree_a1"
    )

    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="DM2MessageActorCriticRecurrent",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )


@configclass
class DM2StructuredMessageCPGUnitreeA1MixedPPORunnerCfg(
    DM2MessageCPGUnitreeA1MixedPPORunnerCfg
):
    """V13 relation-preserving DM2 mixed-terrain configuration."""

    experiment_name = (
        "dm2_v13_structured_message_static_mixed_cpg_unitree_a1"
    )

    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name=(
            "DM2StructuredMessageActorCriticRecurrent"
        ),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )


@configclass
class DM2FrozenResidualCPGUnitreeA1MixedPPORunnerCfg(
    DM2MessageCPGUnitreeA1MixedPPORunnerCfg
):
    """V14 frozen V12 policy with structured residual coordination."""

    max_iterations = 150
    save_interval = 10
    experiment_name = (
        "dm2_v14_frozen_residual_static_mixed_cpg_unitree_a1"
    )

    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name=(
            "DM2FrozenResidualActorCriticRecurrent"
        ),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    def __post_init__(self):
        super().__post_init__()

        # Conservative optimization because the residual is initialized
        # on top of an already useful locomotion controller.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 1.0e-4
