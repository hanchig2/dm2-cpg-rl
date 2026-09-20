# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import rsl_rl.runners.on_policy_runner as on_policy_runner_module

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg

from modules.dm2_actor_critic import DM2ActorCriticRecurrent

from .rsl_rl_ppo_cfg import CPGUnitreeA1FlatPPORunnerCfg


# OnPolicyRunner resolves the policy from its string class name.
# Register the local DM2 policy without modifying installed RSL-RL files.
on_policy_runner_module.DM2ActorCriticRecurrent = (
    DM2ActorCriticRecurrent
)


@configclass
class DM2CPGUnitreeA1FlatPPORunnerCfg(
    CPGUnitreeA1FlatPPORunnerCfg
):
    """PPO configuration for flat-terrain DM2-CPG-RL."""

    experiment_name = "dm2_v2_cpg_unitree_a1_flat"

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
