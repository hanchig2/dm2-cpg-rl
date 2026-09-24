# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import rsl_rl.runners.on_policy_runner as on_policy_runner_module

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlPpoActorCriticRecurrentCfg,
)

from modules.dm2_actor_critic import (
    DM2ActorCriticRecurrent,
    DM2GraphStudentTeacherRecurrent,
    DM2MessageActorCriticRecurrent,
    DM2RecurrentGraphActorCriticRecurrent,
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
on_policy_runner_module.DM2RecurrentGraphActorCriticRecurrent = (
    DM2RecurrentGraphActorCriticRecurrent
)
on_policy_runner_module.DM2GraphStudentTeacherRecurrent = (
    DM2GraphStudentTeacherRecurrent
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
class DM2RecurrentGraphCPGUnitreeA1MixedPPORunnerCfg(
    DM2MessageCPGUnitreeA1MixedPPORunnerCfg
):
    """V15 relation-aware recurrent-graph DM2 configuration."""

    experiment_name = (
        "dm2_v15_recurrent_graph_static_mixed_cpg_unitree_a1"
    )

    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="DM2RecurrentGraphActorCriticRecurrent",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )


@configclass
class DM2GraphDistillationCPGUnitreeA1MixedRunnerCfg(
    DM2RecurrentGraphCPGUnitreeA1MixedPPORunnerCfg
):
    """Central-teacher distillation for the V15 graph student."""

    max_iterations = 100
    save_interval = 10

    experiment_name = (
        "dm2_v15_graph_distillation_static_mixed_cpg_unitree_a1"
    )

    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        class_name="DM2GraphStudentTeacherRecurrent",
        init_noise_std=0.1,
        student_hidden_dims=[256, 128],
        teacher_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=True,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1.0e-4,
    )


from modules.dm2_actor_critic import (
    DM2GraphOnlyStudentTeacherRecurrent,
)

on_policy_runner_module.DM2GraphOnlyStudentTeacherRecurrent = (
    DM2GraphOnlyStudentTeacherRecurrent
)


@configclass
class DM2GraphOnlyDistillationCPGUnitreeA1MixedRunnerCfg(
    DM2GraphDistillationCPGUnitreeA1MixedRunnerCfg
):
    """V16 conservative graph-only Central-teacher distillation."""

    max_iterations = 100
    save_interval = 10

    experiment_name = (
        "dm2_v16_graph_only_distillation_"
        "static_mixed_cpg_unitree_a1"
    )

    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        class_name=(
            "DM2GraphOnlyStudentTeacherRecurrent"
        ),
        init_noise_std=0.1,
        student_hidden_dims=[256, 128],
        teacher_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=True,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=2.0e-5,
    )


from modules.dm2_actor_critic import (
    DM2AnchoredGraphStudentTeacherRecurrent,
)

on_policy_runner_module.DM2AnchoredGraphStudentTeacherRecurrent = (
    DM2AnchoredGraphStudentTeacherRecurrent
)


@configclass
class DM2AnchoredGraphDistillationCPGUnitreeA1MixedRunnerCfg(
    DM2GraphOnlyDistillationCPGUnitreeA1MixedRunnerCfg
):
    """V17 bounded graph correction anchored to frozen V12."""

    max_iterations = 100
    save_interval = 10

    experiment_name = (
        "dm2_v17_anchored_graph_distillation_"
        "static_mixed_cpg_unitree_a1"
    )

    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        class_name=(
            "DM2AnchoredGraphStudentTeacherRecurrent"
        ),
        init_noise_std=0.1,
        student_hidden_dims=[256, 128],
        teacher_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=True,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=2.0e-5,
    )


import copy as _v18_copy

from modules.dm2_actor_critic import (
    DM2AnchoredResidualPPOActorCriticRecurrent,
)


on_policy_runner_module.DM2AnchoredResidualPPOActorCriticRecurrent = (
    DM2AnchoredResidualPPOActorCriticRecurrent
)


@configclass
class DM2AnchoredResidualPPOCPGUnitreeA1MixedRunnerCfg(
    DM2RecurrentGraphCPGUnitreeA1MixedPPORunnerCfg
):
    """V18 frozen-V12 anchored graph residual optimized by PPO."""

    max_iterations = 100
    save_interval = 10

    experiment_name = (
        "dm2_v18_anchored_residual_ppo_"
        "static_mixed_cpg_unitree_a1"
    )

    policy = _v18_copy.deepcopy(
        DM2RecurrentGraphCPGUnitreeA1MixedPPORunnerCfg().policy
    )

    policy.class_name = (
        "DM2AnchoredResidualPPOActorCriticRecurrent"
    )
    policy.init_noise_std = 0.1

    algorithm = _v18_copy.deepcopy(
        DM2RecurrentGraphCPGUnitreeA1MixedPPORunnerCfg().algorithm
    )

    algorithm.learning_rate = 5.0e-5
    algorithm.entropy_coef = 0.0
    algorithm.schedule = "fixed"
