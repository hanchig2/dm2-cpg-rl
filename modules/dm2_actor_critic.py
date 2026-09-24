from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation

from modules.dm2_memory import RecurrentGraphLegMemory, SharedLegMemory


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(resolve_nn_activation(activation))
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DM2ActorCriticRecurrent(ActorCritic):
    """Parameter-shared per-leg actor with a centralized critic."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 128],
        critic_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_legs: int = 4,
        local_obs_dim: int = 46,
        local_action_dim: int = 3,
        **kwargs,
    ):
        # We intentionally initialize nn.Module directly because the standard
        # ActorCritic constructor creates a centralized actor MLP.
        nn.Module.__init__(self)

        if kwargs:
            print(
                "DM2ActorCriticRecurrent got unexpected arguments, "
                f"which will be ignored: {list(kwargs.keys())}"
            )

        if rnn_type.lower() != "lstm":
            raise ValueError("DM2 currently supports only LSTM memory.")

        expected_actor_obs = num_legs * local_obs_dim
        expected_actions = num_legs * local_action_dim

        if num_actor_obs != expected_actor_obs:
            raise ValueError(
                f"Expected num_actor_obs={expected_actor_obs}, "
                f"got {num_actor_obs}."
            )

        if num_actions != expected_actions:
            raise ValueError(
                f"Expected num_actions={expected_actions}, "
                f"got {num_actions}."
            )

        self.num_legs = num_legs
        self.local_obs_dim = local_obs_dim
        self.local_action_dim = local_action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # Four legs use the same LSTM weights but maintain independent states.
        self.memory_a = SharedLegMemory(
            input_size=local_obs_dim,
            num_legs=num_legs,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # The critic retains one centralized recurrent state per environment.
        self.memory_c = Memory(
            input_size=num_critic_obs,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # This actor head is applied independently to every leg.
        self.actor = _build_mlp(
            input_dim=rnn_hidden_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=local_action_dim,
            activation=activation,
        )

        self.critic = _build_mlp(
            input_dim=rnn_hidden_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation=activation,
        )

        self.noise_std_type = noise_std_type

        if noise_std_type == "scalar":
            self.std = nn.Parameter(
                init_noise_std * torch.ones(num_actions)
            )
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown noise_std_type: {noise_std_type}."
            )

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"DM2 shared leg actor: {self.actor}")
        print(f"DM2 centralized critic: {self.critic}")
        print(f"DM2 actor memory: {self.memory_a.rnn}")
        print(f"DM2 critic memory: {self.memory_c.rnn}")

    def _compute_action_mean(
        self,
        packed_actor_features: torch.Tensor,
    ) -> torch.Tensor:
        """Map [..., 4H] features to CPG-ordered [..., 12] means."""

        expected_size = self.num_legs * self.rnn_hidden_dim

        if packed_actor_features.shape[-1] != expected_size:
            raise ValueError(
                f"Expected packed actor feature size {expected_size}, "
                f"got {packed_actor_features.shape[-1]}."
            )

        local_features = packed_actor_features.reshape(
            *packed_actor_features.shape[:-1],
            self.num_legs,
            self.rnn_hidden_dim,
        )

        local_action_means = self.actor(local_features)

        return torch.cat(
            (
                local_action_means[..., 0],
                local_action_means[..., 1],
                local_action_means[..., 2],
            ),
            dim=-1,
        )

    def update_distribution(
        self,
        packed_actor_features: torch.Tensor,
    ):
        mean = self._compute_action_mean(packed_actor_features)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)

        self.distribution = Normal(mean, std)

    def act(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states=None,
    ) -> torch.Tensor:
        actor_features = self.memory_a(
            observations,
            masks,
            hidden_states,
        )
        actor_features = actor_features.squeeze(0)

        self.update_distribution(actor_features)
        return self.distribution.sample()

    def act_inference(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        actor_features = self.memory_a(observations)
        actor_features = actor_features.squeeze(0)
        return self._compute_action_mean(actor_features)

    def evaluate(
        self,
        critic_observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states=None,
    ) -> torch.Tensor:
        critic_features = self.memory_c(
            critic_observations,
            masks,
            hidden_states,
        )
        return self.critic(critic_features.squeeze(0))

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def detach_hidden_states(self, dones=None):
        self.memory_a.detach_hidden_states(dones)
        self.memory_c.detach_hidden_states(dones)

    def get_hidden_states(self):
        return (
            self.memory_a.hidden_states,
            self.memory_c.hidden_states,
        )

class DM2MessageActorCriticRecurrent(
    DM2ActorCriticRecurrent
):
    """DM2 actor with low-bandwidth recurrent inter-leg messages.

    Each leg receives its own recurrent feature and the mean recurrent
    feature of the other legs. The actor remains parameter-shared.

    When a legacy DM2 checkpoint is loaded, the original actor weights
    are copied into the local-feature columns and all new message
    columns are initialized to zero. The initial deterministic policy
    is therefore identical to the legacy policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        legacy_first_layer = self.actor[0]

        if not isinstance(
            legacy_first_layer,
            nn.Linear,
        ):
            raise TypeError(
                "Expected actor[0] to be nn.Linear."
            )

        if (
            legacy_first_layer.in_features
            != self.rnn_hidden_dim
        ):
            raise ValueError(
                "Unexpected legacy actor input size: "
                f"{legacy_first_layer.in_features}."
            )

        message_first_layer = nn.Linear(
            2 * self.rnn_hidden_dim,
            legacy_first_layer.out_features,
            bias=legacy_first_layer.bias is not None,
            device=legacy_first_layer.weight.device,
            dtype=legacy_first_layer.weight.dtype,
        )

        with torch.no_grad():
            message_first_layer.weight.zero_()
            message_first_layer.weight[
                :,
                :self.rnn_hidden_dim,
            ].copy_(
                legacy_first_layer.weight
            )

            if legacy_first_layer.bias is not None:
                message_first_layer.bias.copy_(
                    legacy_first_layer.bias
                )

        self.actor[0] = message_first_layer

        print(
            "DM2 message actor first layer: "
            f"{self.actor[0]}"
        )

    def _compute_action_mean(
        self,
        packed_actor_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply shared actor using local and other-leg features."""

        expected_size = (
            self.num_legs * self.rnn_hidden_dim
        )

        if (
            packed_actor_features.shape[-1]
            != expected_size
        ):
            raise ValueError(
                "Expected packed actor feature size "
                f"{expected_size}, got "
                f"{packed_actor_features.shape[-1]}."
            )

        local_features = (
            packed_actor_features.reshape(
                *packed_actor_features.shape[:-1],
                self.num_legs,
                self.rnn_hidden_dim,
            )
        )

        if self.num_legs <= 1:
            other_leg_message = torch.zeros_like(
                local_features
            )
        else:
            summed_features = torch.sum(
                local_features,
                dim=-2,
                keepdim=True,
            )

            other_leg_message = (
                summed_features - local_features
            ) / float(self.num_legs - 1)

        actor_input = torch.cat(
            (
                local_features,
                other_leg_message,
            ),
            dim=-1,
        )

        local_action_means = self.actor(
            actor_input
        )

        return torch.cat(
            (
                local_action_means[..., 0],
                local_action_means[..., 1],
                local_action_means[..., 2],
            ),
            dim=-1,
        )

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        """Expand a legacy DM2 actor checkpoint without changing behavior."""

        adapted_state_dict = state_dict.copy()

        weight_key = "actor.0.weight"
        source_weight = adapted_state_dict.get(
            weight_key
        )
        target_weight = self.actor[0].weight

        legacy_shape = (
            target_weight.shape[0],
            self.rnn_hidden_dim,
        )

        if (
            source_weight is not None
            and tuple(source_weight.shape)
            == legacy_shape
            and tuple(target_weight.shape)
            == (
                target_weight.shape[0],
                2 * self.rnn_hidden_dim,
            )
        ):
            expanded_weight = source_weight.new_zeros(
                target_weight.shape
            )

            expanded_weight[
                :,
                :self.rnn_hidden_dim,
            ].copy_(source_weight)

            adapted_state_dict[weight_key] = (
                expanded_weight
            )

            print(
                "[INFO]: Expanded legacy "
                "actor.0.weight from "
                f"{tuple(source_weight.shape)} to "
                f"{tuple(expanded_weight.shape)}; "
                "message columns initialized to zero."
            )

        # RSL-RL's ActorCritic overrides load_state_dict with
        # the older (state_dict, strict) signature, so do not pass
        # PyTorch's newer assign argument through to the parent.
        return super().load_state_dict(
            adapted_state_dict,
            strict=strict,
        )


class DM2RecurrentGraphActorCriticRecurrent(
    DM2MessageActorCriticRecurrent
):
    """V15 DM2 with communication inside recurrent state updates.

    The V12 mean-message actor is retained. Its independent per-leg
    LSTM is replaced by a relation-aware recurrent graph memory whose
    initial update is exactly zero.
    """

    def __init__(
        self,
        *args,
        graph_update_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        previous_memory = self.memory_a

        graph_memory = RecurrentGraphLegMemory(
            input_size=self.local_obs_dim,
            num_legs=self.num_legs,
            num_layers=previous_memory.num_layers,
            hidden_size=self.rnn_hidden_dim,
            graph_update_scale=graph_update_scale,
        )

        graph_memory.rnn.load_state_dict(
            previous_memory.rnn.state_dict()
        )

        self.memory_a = graph_memory

        print(
            "DM2 recurrent graph memory: "
            f"{self.memory_a.rnn}"
        )
        print(
            "DM2 graph relation order: "
            "self, contralateral, ipsilateral, diagonal"
        )
        print(
            "DM2 graph update scale: "
            f"{self.memory_a.graph_update_scale}"
        )

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        """Load V12 checkpoints with zero-initialized graph layers."""

        adapted_state_dict = state_dict.copy()
        current_state_dict = self.state_dict()

        graph_prefixes = (
            "memory_a.graph_gate.",
            "memory_a.graph_candidate.",
        )

        added_graph_keys = []

        for key, value in current_state_dict.items():
            if (
                key.startswith(graph_prefixes)
                and key not in adapted_state_dict
            ):
                adapted_state_dict[key] = (
                    value.detach().clone()
                )
                added_graph_keys.append(key)

        if added_graph_keys:
            print(
                "[INFO]: Initialized "
                f"{len(added_graph_keys)} recurrent-graph "
                "checkpoint tensors from exact-parity defaults."
            )

        return super().load_state_dict(
            adapted_state_dict,
            strict=strict,
        )


from rsl_rl.modules import (
    ActorCriticRecurrent as _CentralActorCriticRecurrent,
)


class DM2GraphStudentTeacherRecurrent(nn.Module):
    """Central recurrent teacher with a V15 graph-DM2 student."""

    is_recurrent = True

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        student_hidden_dims: list[int] = [256, 128],
        teacher_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 0.1,
        teacher_recurrent: bool = True,
        **kwargs,
    ):
        super().__init__()

        if not teacher_recurrent:
            raise ValueError(
                "V15 requires the Central recurrent teacher."
            )

        # Accepted by Isaac Lab configs but handled internally here.
        kwargs.pop("noise_std_type", None)

        if kwargs:
            print(
                "DM2GraphStudentTeacherRecurrent got "
                "unexpected arguments, which will be ignored: "
                f"{list(kwargs.keys())}"
            )

        self.distillation_action_noise_std = float(
            init_noise_std
        )

        self.student = (
            DM2RecurrentGraphActorCriticRecurrent(
                num_actor_obs=num_student_obs,
                num_critic_obs=num_teacher_obs,
                num_actions=num_actions,
                actor_hidden_dims=student_hidden_dims,
                critic_hidden_dims=student_hidden_dims,
                activation=activation,
                rnn_type=rnn_type,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_num_layers=rnn_num_layers,
                init_noise_std=init_noise_std,
            )
        )

        self.teacher = _CentralActorCriticRecurrent(
            num_actor_obs=num_teacher_obs,
            num_critic_obs=num_teacher_obs,
            num_actions=num_actions,
            actor_hidden_dims=teacher_hidden_dims,
            critic_hidden_dims=teacher_hidden_dims,
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            init_noise_std=init_noise_std,
        )

        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        self.teacher.eval()

        self.loaded_teacher = False
        self.loaded_student = False

        print(
            "V15 distillation student: "
            "DM2RecurrentGraphActorCriticRecurrent"
        )
        print(
            "V15 distillation teacher: "
            "ActorCriticRecurrent"
        )

    def train(self, mode: bool = True):
        super().train(mode)

        # The teacher must remain deterministic and frozen.
        self.teacher.eval()
        return self

    def act(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Sample actions from the student during data collection."""

        return self.student.act(observations)

    def act_inference(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return deterministic student actions for imitation loss."""

        return self.student.act_inference(
            observations
        )

    def evaluate(
        self,
        teacher_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return deterministic Central-teacher actions."""

        with torch.no_grad():
            return self.teacher.act_inference(
                teacher_observations
            )

    def reset(
        self,
        dones=None,
        hidden_states=None,
    ):
        """Reset or restore student and teacher actor memories."""

        if dones is None:
            if hidden_states is None:
                student_hidden = None
                teacher_hidden = None
            else:
                if (
                    not isinstance(
                        hidden_states,
                        (tuple, list),
                    )
                    or len(hidden_states) != 2
                ):
                    raise ValueError(
                        "Expected (student, teacher) "
                        "hidden states."
                    )

                student_hidden, teacher_hidden = (
                    hidden_states
                )

            self.student.memory_a.hidden_states = (
                student_hidden
            )
            self.teacher.memory_a.hidden_states = (
                teacher_hidden
            )
            return

        self.student.memory_a.reset(dones)
        self.teacher.memory_a.reset(dones)

    def detach_hidden_states(
        self,
        dones=None,
    ):
        self.student.memory_a.detach_hidden_states(
            dones
        )
        self.teacher.memory_a.detach_hidden_states(
            dones
        )

    def get_hidden_states(self):
        return (
            self.student.memory_a.hidden_states,
            self.teacher.memory_a.hidden_states,
        )

    def load_student_state_dict(
        self,
        state_dict,
    ):
        """Load V12 into the V15 student with exact parity."""

        self.student.load_state_dict(
            state_dict,
            strict=True,
        )

        # The V12 PPO checkpoint contains its exploration-noise
        # parameter. Distillation should retain the configured low
        # rollout noise while preserving the deterministic V12 mean.
        with torch.no_grad():
            if self.student.noise_std_type == "scalar":
                self.student.std.fill_(
                    self.distillation_action_noise_std
                )
            else:
                self.student.log_std.fill_(
                    torch.log(
                        torch.tensor(
                            self.distillation_action_noise_std,
                            device=self.student.log_std.device,
                            dtype=self.student.log_std.dtype,
                        )
                    )
                )

        self.loaded_student = True

        print(
            "[INFO]: Loaded V12 checkpoint into "
            "the V15 recurrent-graph student."
        )

    @property
    def action_std(self):
        """Expose student noise without requiring an action distribution."""
        if self.student.noise_std_type == "scalar":
            return self.student.std

        return torch.exp(
            self.student.log_std
        )

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
    ):
        """Load either a Central teacher or a saved distillation run."""

        # Central PPO checkpoint.
        if (
            any(
                key.startswith("actor.")
                for key in state_dict
            )
            and any(
                key.startswith("memory_a.")
                for key in state_dict
            )
        ):
            self.teacher.load_state_dict(
                state_dict,
                strict=strict,
            )

            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)

            self.teacher.eval()
            self.loaded_teacher = True

            print(
                "[INFO]: Loaded and froze the "
                "Central recurrent teacher."
            )

            # False tells OnPolicyRunner this is an RL-to-
            # distillation initialization rather than resumption.
            return False

        # V15 distillation checkpoint.
        if (
            any(
                key.startswith("student.")
                for key in state_dict
            )
            and any(
                key.startswith("teacher.")
                for key in state_dict
            )
        ):
            nn.Module.load_state_dict(
                self,
                state_dict,
                strict=strict,
            )

            self.loaded_teacher = True
            self.loaded_student = True

            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)

            self.teacher.eval()

            return True

        raise ValueError(
            "Checkpoint is neither a Central PPO checkpoint "
            "nor a V15 distillation checkpoint."
        )


class DM2GraphOnlyStudentTeacherRecurrent(
    DM2GraphStudentTeacherRecurrent
):
    """V16: frozen V12 backbone with graph-only distillation.

    Only the recurrent graph gate and candidate parameters are
    trainable. The inherited V12 actor, actor LSTM, critic, and action
    noise remain frozen.
    """

    GRAPH_UPDATE_SCALE = 0.03

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.student.memory_a.graph_update_scale = (
            self.GRAPH_UPDATE_SCALE
        )

        self._freeze_student_backbone()

        print(
            "V16 graph-only update scale: "
            f"{self.GRAPH_UPDATE_SCALE}"
        )

    def _freeze_student_backbone(self):
        trainable_prefixes = (
            "memory_a.graph_gate.",
            "memory_a.graph_candidate.",
        )

        for name, parameter in (
            self.student.named_parameters()
        ):
            parameter.requires_grad_(
                name.startswith(
                    trainable_prefixes
                )
            )

        trainable_names = [
            name
            for name, parameter in (
                self.student.named_parameters()
            )
            if parameter.requires_grad
        ]

        expected_names = {
            "memory_a.graph_gate.weight",
            "memory_a.graph_gate.bias",
            "memory_a.graph_candidate.weight",
            "memory_a.graph_candidate.bias",
        }

        if set(trainable_names) != expected_names:
            raise RuntimeError(
                "Unexpected V16 trainable parameters: "
                f"{trainable_names}"
            )

        trainable_count = sum(
            parameter.numel()
            for parameter in self.student.parameters()
            if parameter.requires_grad
        )

        frozen_count = sum(
            parameter.numel()
            for parameter in self.student.parameters()
            if not parameter.requires_grad
        )

        print(
            "V16 trainable graph parameters: "
            f"{trainable_names}"
        )
        print(
            "V16 trainable/frozen student parameters: "
            f"{trainable_count}/{frozen_count}"
        )

    def load_student_state_dict(
        self,
        state_dict,
    ):
        super().load_student_state_dict(
            state_dict
        )

        self.student.memory_a.graph_update_scale = (
            self.GRAPH_UPDATE_SCALE
        )

        self._freeze_student_backbone()

        print(
            "[INFO]: Froze the V12 backbone; "
            "only recurrent graph parameters are trainable."
        )

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
    ):
        result = super().load_state_dict(
            state_dict,
            strict=strict,
        )

        self.student.memory_a.graph_update_scale = (
            self.GRAPH_UPDATE_SCALE
        )

        self._freeze_student_backbone()

        return result


class DM2AnchoredGraphStudentTeacherRecurrent(
    DM2GraphOnlyStudentTeacherRecurrent
):
    """V17: bounded graph correction anchored to frozen V12.

    Three recurrent policies run in parallel:

    1. graph student: only graph gate/candidate parameters train;
    2. V12 anchor: exact frozen V12 policy;
    3. Central teacher: exact frozen Central recurrent policy.

    The deployed student action is bounded to +/- 0.05 relative to
    the V12 anchor. During collection, the Central action is blended
    conservatively toward the V12 action and stored as the native
    RSL-RL distillation target.

    The effective target is algebraically equivalent, up to an
    overall constant loss scale, to:

        MSE(student, blended_target)
        + anchor_weight * MSE(student, V12_anchor)
    """

    TEACHER_BLEND_BETA = 0.25
    TEACHER_DELTA_CLIP = 0.25
    MAX_ACTION_RESIDUAL = 0.05
    ANCHOR_LOSS_WEIGHT = 0.10

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        student_hidden_dims: list[int] = [256, 128],
        teacher_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 0.1,
        teacher_recurrent: bool = True,
        **kwargs,
    ):
        super().__init__(
            num_student_obs=num_student_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            init_noise_std=init_noise_std,
            teacher_recurrent=teacher_recurrent,
            **kwargs,
        )

        self.anchor = DM2MessageActorCriticRecurrent(
            num_actor_obs=num_student_obs,
            num_critic_obs=num_teacher_obs,
            num_actions=num_actions,
            actor_hidden_dims=student_hidden_dims,
            critic_hidden_dims=student_hidden_dims,
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            init_noise_std=init_noise_std,
        )

        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

        self.anchor.eval()
        self.loaded_anchor = False
        self._last_anchor_actions = None

        print(
            "V17 teacher blend beta: "
            f"{self.TEACHER_BLEND_BETA}"
        )
        print(
            "V17 teacher delta clip: "
            f"{self.TEACHER_DELTA_CLIP}"
        )
        print(
            "V17 maximum action residual: "
            f"{self.MAX_ACTION_RESIDUAL}"
        )
        print(
            "V17 anchor-loss weight: "
            f"{self.ANCHOR_LOSS_WEIGHT}"
        )

    def train(self, mode: bool = True):
        super().train(mode)

        # Both reference policies must remain deterministic.
        self.teacher.eval()
        self.anchor.eval()
        return self

    def _anchor_actions(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            anchor_actions = (
                self.anchor.act_inference(
                    observations
                )
            )

        self._last_anchor_actions = (
            anchor_actions.detach()
        )

        return anchor_actions

    def _bound_student_actions(
        self,
        raw_student_actions: torch.Tensor,
        anchor_actions: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.MAX_ACTION_RESIDUAL

        residual = scale * torch.tanh(
            (
                raw_student_actions
                - anchor_actions
            )
            / scale
        )

        return anchor_actions + residual

    def _set_student_distribution(
        self,
        action_mean: torch.Tensor,
    ):
        if self.student.noise_std_type == "scalar":
            action_std = (
                self.student.std.expand_as(
                    action_mean
                )
            )
        else:
            action_std = torch.exp(
                self.student.log_std
            ).expand_as(
                action_mean
            )

        self.student.distribution = (
            torch.distributions.Normal(
                action_mean,
                action_std,
            )
        )

    def act(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Sample around the bounded anchored student mean."""

        anchor_actions = self._anchor_actions(
            observations
        )

        raw_student_actions = (
            self.student.act_inference(
                observations
            )
        )

        bounded_actions = (
            self._bound_student_actions(
                raw_student_actions,
                anchor_actions,
            )
        )

        self._set_student_distribution(
            bounded_actions
        )

        return self.student.distribution.sample()

    def act_inference(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded deterministic graph-student actions."""

        anchor_actions = self._anchor_actions(
            observations
        )

        raw_student_actions = (
            self.student.act_inference(
                observations
            )
        )

        return self._bound_student_actions(
            raw_student_actions,
            anchor_actions,
        )

    def evaluate(
        self,
        teacher_observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return the anchored Central/V12 blended target."""

        if self._last_anchor_actions is None:
            raise RuntimeError(
                "V17 evaluate() requires act() to be "
                "called first for the same transition."
            )

        central_actions = super().evaluate(
            teacher_observations
        )

        anchor_actions = (
            self._last_anchor_actions
        )

        teacher_delta = torch.clamp(
            central_actions - anchor_actions,
            min=-self.TEACHER_DELTA_CLIP,
            max=self.TEACHER_DELTA_CLIP,
        )

        teacher_correction = torch.clamp(
            self.TEACHER_BLEND_BETA
            * teacher_delta,
            min=-self.MAX_ACTION_RESIDUAL,
            max=self.MAX_ACTION_RESIDUAL,
        )

        blended_target = (
            anchor_actions
            + teacher_correction
        )

        # For squared error, minimizing
        #
        #   ||a - blended||^2
        #   + lambda ||a - anchor||^2
        #
        # has the same optimum as MSE to this effective target.
        effective_target = (
            blended_target
            + self.ANCHOR_LOSS_WEIGHT
            * anchor_actions
        ) / (
            1.0
            + self.ANCHOR_LOSS_WEIGHT
        )

        return effective_target.detach()

    def reset(
        self,
        dones=None,
        hidden_states=None,
    ):
        """Reset/restore student, anchor, and teacher memories."""

        self._last_anchor_actions = None

        if dones is None:
            if hidden_states is None:
                student_hidden = None
                anchor_hidden = None
                teacher_hidden = None
            else:
                if (
                    not isinstance(
                        hidden_states,
                        (tuple, list),
                    )
                    or len(hidden_states) != 2
                ):
                    raise ValueError(
                        "Expected "
                        "((student, anchor), teacher) "
                        "hidden states."
                    )

                student_bundle, teacher_hidden = (
                    hidden_states
                )

                if (
                    not isinstance(
                        student_bundle,
                        (tuple, list),
                    )
                    or len(student_bundle) != 2
                ):
                    raise ValueError(
                        "Expected the first hidden-state "
                        "entry to be (student, anchor)."
                    )

                student_hidden, anchor_hidden = (
                    student_bundle
                )

            self.student.memory_a.hidden_states = (
                student_hidden
            )
            self.anchor.memory_a.hidden_states = (
                anchor_hidden
            )
            self.teacher.memory_a.hidden_states = (
                teacher_hidden
            )
            return

        self.student.memory_a.reset(dones)
        self.anchor.memory_a.reset(dones)
        self.teacher.memory_a.reset(dones)

    def detach_hidden_states(
        self,
        dones=None,
    ):
        self.student.memory_a.detach_hidden_states(
            dones
        )
        self.anchor.memory_a.detach_hidden_states(
            dones
        )
        self.teacher.memory_a.detach_hidden_states(
            dones
        )

    def get_hidden_states(self):
        return (
            (
                self.student.memory_a.hidden_states,
                self.anchor.memory_a.hidden_states,
            ),
            self.teacher.memory_a.hidden_states,
        )

    def load_student_state_dict(
        self,
        state_dict,
    ):
        """Load the same V12 checkpoint into student and anchor."""

        super().load_student_state_dict(
            state_dict
        )

        self.anchor.load_state_dict(
            state_dict,
            strict=True,
        )

        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

        self.anchor.eval()
        self.loaded_anchor = True

        print(
            "[INFO]: Loaded and froze the exact "
            "V12 recurrent anchor."
        )

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
    ):
        result = super().load_state_dict(
            state_dict,
            strict=strict,
        )

        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

        self.anchor.eval()

        if any(
            key.startswith("anchor.")
            for key in state_dict
        ):
            self.loaded_anchor = True

        self._freeze_student_backbone()

        return result


class DM2AnchoredResidualPPOActorCriticRecurrent(
    DM2RecurrentGraphActorCriticRecurrent
):
    """Frozen V12 actor plus bounded graph residual trained by PPO.

    The deployed deterministic mean is

        anchor + residual_limit * tanh(
            (raw_graph_actor - anchor)
            / residual_limit
        )

    Only the recurrent graph tensors and centralized critic are
    trainable. The V12 actor, exact V12 anchor, and action noise
    remain frozen.
    """

    GRAPH_UPDATE_SCALE = 0.03
    MAX_ACTION_RESIDUAL = 0.08

    # Used by scripts/rsl_rl/train.py after runner.load().
    freeze_observation_normalizers = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 128],
        critic_hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 0.1,
        **kwargs,
    ):
        if rnn_type.lower() != "lstm":
            raise ValueError(
                "V18 currently requires LSTM so its "
                "four-part actor hidden-state bundle "
                "has fixed semantics."
            )

        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            init_noise_std=init_noise_std,
            graph_update_scale=(
                self.GRAPH_UPDATE_SCALE
            ),
            **kwargs,
        )

        self.anchor = DM2MessageActorCriticRecurrent(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            init_noise_std=init_noise_std,
        )

        self._last_anchor_mean = None

        self._freeze_components()

        print(
            "V18 graph update scale: "
            f"{self.GRAPH_UPDATE_SCALE}"
        )
        print(
            "V18 maximum action residual: "
            f"{self.MAX_ACTION_RESIDUAL}"
        )
        print(
            "V18 action noise is fixed at: "
            f"{init_noise_std}"
        )
        print(
            "V18 actor hidden-state format: "
            "packed_h(student|anchor), "
            "packed_c(student|anchor)"
        )

    def _freeze_components(self):
        """Freeze anchor and V12 actor; train graph and critic."""

        for parameter in self.parameters():
            parameter.requires_grad_(False)

        for parameter in (
            self.memory_a.graph_gate.parameters()
        ):
            parameter.requires_grad_(True)

        for parameter in (
            self.memory_a.graph_candidate.parameters()
        ):
            parameter.requires_grad_(True)

        for parameter in self.memory_c.parameters():
            parameter.requires_grad_(True)

        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)

        if hasattr(self, "std"):
            self.std.requires_grad_(False)

        if hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)

        self.anchor.eval()

        trainable_names = [
            name
            for name, parameter
            in self.named_parameters()
            if parameter.requires_grad
        ]

        print("V18 trainable parameters:")

        for name in trainable_names:
            print(f"  {name}")

    def train(
        self,
        mode: bool = True,
    ):
        result = super().train(mode)

        # The frozen anchor must always remain deterministic.
        self.anchor.eval()

        return result

    def _split_actor_hidden_states(
        self,
        hidden_states,
    ):
        """Unpack student/anchor states from PPO storage layout.

        During online rollout, the four legs occupy the recurrent
        batch dimension and student/anchor features are concatenated:

            [layers, 4 * environments, 2 * hidden]

        RSL-RL storage folds the four legs into the feature dimension:

            [layers, trajectories, 4 * 2 * hidden]

        The stored feature order is interleaved by leg:

            S0, A0, S1, A1, S2, A2, S3, A3

        This method supports both representations and reconstructs
        the ordinary DM2 per-policy hidden representation.
        """

        if hidden_states is None:
            return None, None

        if not isinstance(
            hidden_states,
            (tuple, list),
        ):
            raise TypeError(
                "V18 actor hidden states must be "
                "a tuple/list containing packed h and c."
            )

        if len(hidden_states) != 2:
            raise ValueError(
                "Expected packed actor hidden states "
                "(combined_h, combined_c), got "
                f"{len(hidden_states)} entries."
            )

        def unpack_tensor(
            packed,
            state_name,
        ):
            width = packed.shape[-1]

            online_width = (
                2 * self.rnn_hidden_dim
            )

            storage_width = (
                self.num_legs
                * 2
                * self.rnn_hidden_dim
            )

            if width == online_width:
                # Online representation:
                # [..., student_hidden | anchor_hidden]
                student, anchor = torch.split(
                    packed,
                    self.rnn_hidden_dim,
                    dim=-1,
                )

                return (
                    student.contiguous(),
                    anchor.contiguous(),
                )

            if width == storage_width:
                # Stored representation:
                # [..., leg, student_or_anchor, hidden]
                prefix = packed.shape[:-1]

                interleaved = packed.reshape(
                    *prefix,
                    self.num_legs,
                    2,
                    self.rnn_hidden_dim,
                )

                student = (
                    interleaved[
                        ...,
                        :,
                        0,
                        :,
                    ]
                    .reshape(
                        *prefix,
                        self.num_legs
                        * self.rnn_hidden_dim,
                    )
                    .contiguous()
                )

                anchor = (
                    interleaved[
                        ...,
                        :,
                        1,
                        :,
                    ]
                    .reshape(
                        *prefix,
                        self.num_legs
                        * self.rnn_hidden_dim,
                    )
                    .contiguous()
                )

                return student, anchor

            raise ValueError(
                f"Unexpected packed {state_name} "
                f"width {width}; expected "
                f"{online_width} for online state "
                f"or {storage_width} for PPO storage."
            )

        (
            student_h,
            anchor_h,
        ) = unpack_tensor(
            hidden_states[0],
            "h",
        )

        (
            student_c,
            anchor_c,
        ) = unpack_tensor(
            hidden_states[1],
            "c",
        )

        student_hidden = (
            student_h,
            student_c,
        )

        anchor_hidden = (
            anchor_h,
            anchor_c,
        )

        return student_hidden, anchor_hidden

    def _bounded_mean(
        self,
        raw_mean: torch.Tensor,
        anchor_mean: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.MAX_ACTION_RESIDUAL

        return (
            anchor_mean
            + scale
            * torch.tanh(
                (raw_mean - anchor_mean)
                / scale
            )
        )

    def act(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states=None,
    ) -> torch.Tensor:
        """Sample from the bounded anchored PPO distribution."""

        (
            student_hidden,
            anchor_hidden,
        ) = self._split_actor_hidden_states(
            hidden_states
        )

        # Advance the trainable graph actor and form its raw
        # action distribution.
        super().act(
            observations,
            masks=masks,
            hidden_states=student_hidden,
        )

        raw_mean = self.action_mean
        raw_std = self.action_std

        # Advance the exact frozen V12 anchor using its own
        # recurrent state.
        with torch.no_grad():
            self.anchor.act(
                observations,
                masks=masks,
                hidden_states=anchor_hidden,
            )

            anchor_mean = (
                self.anchor.action_mean.detach()
            )

        bounded_mean = self._bounded_mean(
            raw_mean,
            anchor_mean,
        )

        self._last_anchor_mean = anchor_mean

        self.distribution = (
            torch.distributions.Normal(
                bounded_mean,
                raw_std,
            )
        )

        return self.distribution.sample()

    def act_inference(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return the deterministic bounded anchored mean."""

        raw_mean = super().act_inference(
            observations
        )

        with torch.no_grad():
            anchor_mean = (
                self.anchor.act_inference(
                    observations
                )
            ).detach()

        self._last_anchor_mean = anchor_mean

        return self._bounded_mean(
            raw_mean,
            anchor_mean,
        )

    def reset(
        self,
        dones=None,
    ):
        self._last_anchor_mean = None

        super().reset(dones)

        self.anchor.memory_a.reset(dones)

    def get_hidden_states(self):
        """Return PPO-compatible packed actor and critic LSTM states."""

        student_hidden = (
            self.memory_a.hidden_states
        )

        anchor_hidden = (
            self.anchor.memory_a.hidden_states
        )

        if (
            student_hidden is None
            and anchor_hidden is None
        ):
            actor_hidden = None
        else:
            if (
                student_hidden is None
                or anchor_hidden is None
            ):
                raise RuntimeError(
                    "Student and anchor recurrent "
                    "states became unsynchronized."
                )

            if (
                len(student_hidden) != 2
                or len(anchor_hidden) != 2
            ):
                raise RuntimeError(
                    "V18 expected LSTM (h,c) states."
                )

            combined_h = torch.cat(
                (
                    student_hidden[0],
                    anchor_hidden[0],
                ),
                dim=-1,
            )

            combined_c = torch.cat(
                (
                    student_hidden[1],
                    anchor_hidden[1],
                ),
                dim=-1,
            )

            actor_hidden = (
                combined_h,
                combined_c,
            )

        return (
            actor_hidden,
            self.memory_c.hidden_states,
        )

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        """Load either V12 warm start or a complete V18 checkpoint."""

        is_v18_checkpoint = any(
            key.startswith("anchor.")
            for key in state_dict
        )

        if is_v18_checkpoint:
            torch.nn.Module.load_state_dict(
                self,
                state_dict,
                strict=strict,
                assign=assign,
            )

            print(
                "[INFO]: Loaded complete V18 "
                "anchored-residual PPO checkpoint."
            )
        else:
            current = self.state_dict()

            # Start with a complete valid V18 state. This preserves
            # zero graph tensors and fixed 0.1 action noise.
            adapted = {
                key: value.detach().clone()
                for key, value in current.items()
            }

            ignored_noise = (
                "std",
                "log_std",
            )

            loaded_base = 0
            loaded_anchor = 0

            for key, value in state_dict.items():
                if (
                    key in current
                    and key not in ignored_noise
                    and tuple(current[key].shape)
                    == tuple(value.shape)
                ):
                    adapted[key] = (
                        value.detach().clone()
                    )
                    loaded_base += 1

                anchor_key = f"anchor.{key}"

                if (
                    anchor_key in current
                    and tuple(
                        current[anchor_key].shape
                    )
                    == tuple(value.shape)
                ):
                    adapted[anchor_key] = (
                        value.detach().clone()
                    )
                    loaded_anchor += 1

            torch.nn.Module.load_state_dict(
                self,
                adapted,
                strict=True,
                assign=assign,
            )

            print(
                "[INFO]: Loaded V12 into V18 "
                f"base tensors: {loaded_base}"
            )
            print(
                "[INFO]: Loaded V12 into frozen "
                f"anchor tensors: {loaded_anchor}"
            )
            print(
                "[INFO]: Preserved zero graph "
                "parameters and fixed 0.1 noise."
            )

        self.memory_a.graph_update_scale = (
            self.GRAPH_UPDATE_SCALE
        )

        self._freeze_components()

        # OnPolicyRunner uses this Boolean to determine that
        # both actor and critic normalizers should be loaded.
        return True
